import json
import copy
import random
import time
import logging
import os
import math
import subprocess
import tempfile
import wave
import shutil
import contextlib
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import pyrubberband as rb
import threading
from mutagen import File as MutagenFile

# --- Configuration ---
ANALYTICS_DIRECTORY = "analytics"
MIX_POINT_SECONDS = 30  # Start looking for a mix point 30 seconds before the end
CROSSFADE_SECONDS = 10
FALLBACK_THRESHOLD_SECONDS = 15
FALLBACK_TARGET_SECONDS = 10
# Use a smaller blocksize for lower latency, larger for more stability.
# 2048 is a good starting point.
BLOCK_SIZE = 2048
TEMPO_FRAME_SECONDS = 0.1
TEMPO_FRAME_OVERLAP = 0.5

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Global State ---
# A dictionary to hold the shared state between the DJ logic thread and the audio callback.
# This avoids using complex locks for this specific producer/consumer scenario.
player_state = {
    "deck_a": None,  # Holds the raw audio data (NumPy array) for the current track
    "deck_b": None,  # Holds the raw audio data for the incoming track
    "deck_a_info": None,
    "deck_b_info": None,
    "play_pos": 0,  # The current playback position in samples
    "fader": -1.0,  # Crossfader position: -1.0 is full Deck A, 1.0 is full Deck B
    "deck_b_start_pos": 0,  # The sample position when Deck B should start playing
    "samplerate": 44100,
    "channels": 2,
    "is_playing": True,
    "tracks_played_count": 0,
    "total_tracks_in_library": 0,
    "deck_a_cover": None,
    "deck_b_cover": None,
    "deck_a_cover_version": 0,
    "deck_b_cover_version": 0,
    "cover_version_counter": 0,
    "cover_art_cache": {},
    "tag_cache": {},
    "current_crossfade_duration": CROSSFADE_SECONDS,
    "precomputed_transition": None,
    "tempo_ramp_precompute_thread": None,
    "tempo_ramp_precompute_pair": None,
}

_rubberband_path = None
_rubberband_path_initialized = False

# --- Harmonic Mixing Logic (Circle of Fifths) ---
CIRCLE_OF_FIFTHS = {
    "C Major": 1,
    "A Minor": 1,
    "G Major": 2,
    "E Minor": 2,
    "D Major": 3,
    "B Minor": 3,
    "A Major": 4,
    "F# Minor": 4,
    "E Major": 5,
    "C# Minor": 5,
    "B Major": 6,
    "G# Minor": 6,
    "F# Major": 7,
    "D# Minor": 7,
    "C# Major": 8,
    "A# Minor": 8,
    "G# Major": 9,
    "F Minor": 9,
    "D# Major": 10,
    "C Minor": 10,
    "A# Major": 11,
    "G Minor": 11,
    "F Major": 12,
    "D Minor": 12,
}

_CANONICAL_KEY_MAP = {key.lower(): key for key in CIRCLE_OF_FIFTHS}


def normalize_key_label(key_str):
    if not isinstance(key_str, str):
        return None

    cleaned = key_str.strip()
    if not cleaned:
        return Nonec

    canonical = _CANONICAL_KEY_MAP.get(cleaned.lower())
    if canonical:
        return canonical

    # Fallback to title-casing (handles values like "c# minor") before checking again.
    title_cased = cleaned.title()
    return _CANONICAL_KEY_MAP.get(title_cased.lower())


def get_key_number(key_str):
    canonical = normalize_key_label(key_str)
    if not canonical:
        return 0

    return CIRCLE_OF_FIFTHS.get(canonical, 0)


def calculate_key_compatibility(key1, key2):
    num1, num2 = get_key_number(key1), get_key_number(key2)
    if num1 == 0 or num2 == 0:
        return 0
    distance = min(abs(num1 - num2), 12 - abs(num1 - num2))
    if distance == 0:
        return 100
    if distance == 1:
        return 75
    if distance == 2:
        return 50
    return 10


def load_library_from_analytics(directory=ANALYTICS_DIRECTORY):
    """Loads all per-track analytics files into a list of track dictionaries."""
    analytics_files = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".json"):
                    analytics_files.append(entry.path)
    except FileNotFoundError:
        logging.error(
            f"Analytics directory '{directory}' not found. Please run library_manager.py first."
        )
        return []
    except OSError as exc:
        logging.error(f"Failed to read analytics directory '{directory}': {exc}.")
        return []

    tracks = []
    for file_path in sorted(analytics_files):
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("path"):
                tracks.append(data)
            else:
                logging.warning(
                    f"Analytics file '{os.path.basename(file_path)}' missing required fields; skipping."
                )
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(
                f"Could not load analytics file '{os.path.basename(file_path)}': {exc}"
            )

    # Sanitize path fields (win -> posix)
    for track in tracks:
        track["path"] = track["path"].replace("\\", "/")
        # make it absolute if it's not already
        if not os.path.isabs(track["path"]):
            track["path"] = os.path.abspath(track["path"])

    return tracks


_find_best_next_track_call_count = 0


def find_best_next_track(current_track, available_tracks):
    """Finds the best harmonically and rhythmically compatible track."""
    global _find_best_next_track_call_count
    _find_best_next_track_call_count += 1

    if not available_tracks:
        return None

    # Collect all candidates with their scores
    scored_candidates = []
    for candidate in available_tracks:
        # randomly skip some candidates with same key to add variety
        if candidate["key"] == current_track["key"]:
            if random.random() < 0.3:
                continue
            if _find_best_next_track_call_count % 4 == 0:
                continue

        key_score = calculate_key_compatibility(current_track["key"], candidate["key"])
        bpm_diff = abs(current_track["bpm"] - candidate["bpm"])
        bpm_diff = max(0, bpm_diff - 5)
        bpm_score = max(0, 100 - (bpm_diff * 5))

        if _find_best_next_track_call_count % 1 == 0:
            total_score = (key_score * 0.95) - (bpm_score * 0.05)
        else:
            total_score = (key_score * 0.95) + (bpm_score * 0.05)
        scored_candidates.append((total_score, candidate))

    if not scored_candidates:
        return random.choice(available_tracks)

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:10]

    # Randomly select from top 10
    if top_candidates:
        candidate = random.choice(top_candidates)[1]
        logging.info(
            f"Selected next track, previous key: {current_track['key']}, bpm: {current_track['bpm']:.2f}; "
            f"next key: {candidate['key']}, bpm: {candidate['bpm']:.2f}; "
            f"score: {top_candidates[0][0]:.2f}"
        )
        return candidate

    # Fallback (though unlikely if available_tracks is not empty)
    return random.choice(available_tracks)


def get_cover_art(track_path):
    """Returns a dict with mime and data bytes for the track's cover art, or None."""
    cache = player_state["cover_art_cache"]
    if track_path in cache:
        return cache[track_path]

    full_path = os.path.abspath(track_path)
    cover = None
    try:
        audio_file = MutagenFile(full_path)
        if audio_file is not None and getattr(audio_file, "tags", None):
            tags = audio_file.tags
            picture = None

            if hasattr(tags, "getall"):
                images = tags.getall("APIC")
                if images:
                    picture = images[0]
            if picture is None:
                for key in tags.keys():
                    if str(key).startswith("APIC"):
                        picture = tags[key]
                        break

            if picture and getattr(picture, "data", None):
                cover = {
                    "mime": getattr(picture, "mime", "image/jpeg"),
                    "data": picture.data,
                }
    except Exception as exc:
        logging.debug(f"Cover art extraction failed for {track_path}: {exc}")

    cache[track_path] = cover
    return cover


def _extract_tag_value(tag_source, keys):
    for key in keys:
        value = None
        try:
            if hasattr(tag_source, "get"):
                value = tag_source.get(key)
        except Exception:
            value = None

        if value is None and hasattr(tag_source, "getall"):
            try:
                entries = tag_source.getall(key)
            except Exception:
                entries = None
            if entries:
                value = entries[0]

        if value is None:
            continue

        candidate = value
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None

        if candidate is None:
            continue

        if hasattr(candidate, "text"):
            texts = getattr(candidate, "text", None)
            candidate = texts[0] if texts else None
        elif hasattr(candidate, "value"):
            candidate = candidate.value

        if candidate is None:
            continue

        candidate = str(candidate).strip()
        if candidate:
            return candidate

    return None


def get_track_tags(track_path):
    """Returns cached title/artist/album tags for a track path."""
    cache = player_state["tag_cache"]
    if not track_path:
        return {"title": None, "artist": None, "album": None}

    if track_path in cache:
        return cache[track_path]

    tags = {"title": None, "artist": None, "album": None}
    full_path = os.path.abspath(track_path)

    for easy in (True, False):
        try:
            audio_file = MutagenFile(full_path, easy=easy)
        except Exception:
            audio_file = None

        if not audio_file:
            continue

        tag_source = getattr(audio_file, "tags", audio_file)
        if not tag_source:
            continue

        if tags["title"] is None:
            tags["title"] = _extract_tag_value(tag_source, ["title", "TIT2", "©nam"])
        if tags["artist"] is None:
            tags["artist"] = _extract_tag_value(tag_source, ["artist", "TPE1", "©ART"])
        if tags["album"] is None:
            tags["album"] = _extract_tag_value(tag_source, ["album", "TALB", "©alb"])

        if all(tags.values()):
            break

    cache[track_path] = tags
    return tags


def update_deck_cover(deck_key, track_info):
    cover = get_cover_art(track_info["path"])
    state_key = f"{deck_key}_cover"
    version_key = f"{deck_key}_cover_version"
    player_state[state_key] = cover
    player_state["cover_version_counter"] += 1
    player_state[version_key] = player_state["cover_version_counter"]


def _safe_bpm(value):
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        return None
    return bpm if bpm > 0 else None


def _clamp_ratio(value, minimum=0.5, maximum=2.0):
    return max(minimum, min(maximum, value))


def _get_rubberband_executable():
    global _rubberband_path, _rubberband_path_initialized
    if _rubberband_path_initialized:
        return _rubberband_path

    search_dirs = [
        os.path.abspath(os.path.dirname(__file__)),
        os.path.abspath(os.getcwd()),
    ]
    search_names = [
        "rubberband-r3.exe",
        "rubberband.exe",
        "rubberband-r3",
        "rubberband",
    ]

    for name in search_names:
        for base in search_dirs:
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                _rubberband_path = candidate
                _rubberband_path_initialized = True
                return _rubberband_path
        found = shutil.which(name)
        if found:
            _rubberband_path = found
            _rubberband_path_initialized = True
            return _rubberband_path

    _rubberband_path = None
    _rubberband_path_initialized = True
    return None


def _write_wav_int16(path, data, samplerate):
    if data.ndim == 1:
        data_to_write = data[:, np.newaxis]
    else:
        data_to_write = data

    clipped = np.clip(data_to_write, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16, copy=False)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(int16_data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(int16_data.tobytes())


def _read_wav_float32(path, expected_channels):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        frames = wf.getnframes()
        audio_bytes = wf.readframes(frames)

    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if channels > 0:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)

    if channels != expected_channels:
        if channels == 1 and expected_channels == 2:
            audio = np.repeat(audio, 2, axis=1)
        elif channels == 2 and expected_channels == 1:
            audio = audio[:, :1]
        else:
            raise ValueError(
                f"Unexpected channel count {channels}; expected {expected_channels}."
            )

    return (audio.astype(np.float32) / 32767.0).reshape(-1, expected_channels)


def _build_tempo_timemap(
    total_samples,
    samplerate,
    start_ratio,
    end_ratio,
    frame_seconds,
):
    if total_samples <= 1:
        return [(0, 0)], 1.0

    frame_samples = max(int(round(frame_seconds * samplerate)), 1)
    max_index = total_samples - 1
    positions = list(range(0, max_index, frame_samples))
    if positions[-1] != max_index:
        positions.append(max_index)

    mapping = []
    out_pos = 0.0
    last_target = -1

    for idx in range(len(positions) - 1):
        pos = positions[idx]
        progress = pos / max(max_index, 1)
        ratio = _clamp_ratio(start_ratio + (end_ratio - start_ratio) * progress)
        target = int(round(out_pos))
        if target <= last_target:
            target = last_target + 1
        mapping.append((pos, target))
        last_target = target

        next_pos = positions[idx + 1]
        step = max(next_pos - pos, 1)
        out_pos += step / ratio

    # Final anchor at end of segment
    final_pos = positions[-1]
    final_target = int(round(out_pos))
    if final_target <= last_target:
        final_target = last_target + 1
    mapping.append((final_pos, final_target))
    total_output_samples = max(final_target + 1, 1)

    return mapping, float(total_output_samples)


def _rubberband_tempo_ramp(
    segment,
    samplerate,
    start_ratio,
    end_ratio,
    frame_seconds,
):
    executable = _get_rubberband_executable()
    if not executable:
        return None

    squeeze = False
    if segment.ndim == 1:
        segment = segment[:, np.newaxis]
        squeeze = True

    mapping, total_output_samples = _build_tempo_timemap(
        len(segment), samplerate, start_ratio, end_ratio, frame_seconds
    )

    if not mapping:
        return segment[:, 0] if squeeze else segment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        input_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        output_path = tmp_out.name
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp_map:
        map_path = tmp_map.name
        for src, dst in mapping:
            tmp_map.write(f"{src} {dst}\n")

    try:
        _write_wav_int16(input_path, segment, samplerate)
        overall_ratio = total_output_samples / max(len(segment), 1)
        cmd = [
            executable,
            "--quiet",
            "--time",
            f"{overall_ratio:.8f}",
            "--timemap",
            map_path,
            input_path,
            output_path,
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = _read_wav_float32(output_path, segment.shape[1])
    finally:
        for path in (input_path, output_path, map_path):
            with contextlib.suppress(OSError):
                os.remove(path)

    if squeeze:
        return result[:, 0]

    return result


def _framewise_tempo_ramp(
    segment,
    samplerate,
    start_ratio,
    end_ratio,
    frame_seconds=TEMPO_FRAME_SECONDS,
    overlap=TEMPO_FRAME_OVERLAP,
) -> np.ndarray:
    if segment.size == 0:
        return segment

    try:
        processed = _rubberband_tempo_ramp(
            segment, samplerate, start_ratio, end_ratio, frame_seconds
        )
        if processed is not None:
            return processed
    except Exception as exc:
        logging.debug("Rubberband tempo ramp failed: %s", exc)

    return _framewise_tempo_ramp_overlap_add(
        segment,
        samplerate,
        start_ratio,
        end_ratio,
        frame_seconds,
        overlap,
    )


def _framewise_tempo_ramp_overlap_add(
    segment,
    samplerate,
    start_ratio,
    end_ratio,
    frame_seconds=TEMPO_FRAME_SECONDS,
    overlap=TEMPO_FRAME_OVERLAP,
):
    if segment.size == 0:
        return segment

    squeeze = False
    if segment.ndim == 1:
        segment = segment[:, np.newaxis]
        squeeze = True

    channels = segment.shape[1]
    frame_length = max(int(round(frame_seconds * samplerate)), 1)
    hop = max(int(round(frame_length * (1.0 - overlap))), 1)
    if hop >= frame_length:
        hop = max(frame_length - 1, 1)

    total_frames = max(1, int(math.ceil((len(segment) - frame_length) / hop)) + 1)

    frames_data = []
    frames_weight = []
    positions = []
    output_cursor = 0

    for idx in range(total_frames):
        start_idx = idx * hop
        end_idx = start_idx + frame_length

        frame = np.zeros((frame_length, channels), dtype=np.float32)
        if start_idx < len(segment):
            available = segment[start_idx : min(end_idx, len(segment))]
            frame[: len(available)] = available

        progress = idx / max(total_frames - 1, 1)
        ratio = _clamp_ratio(start_ratio + (end_ratio - start_ratio) * progress)

        stretched = rb.time_stretch(frame, samplerate, ratio).astype(
            np.float32, copy=False
        )
        window = np.hanning(len(stretched)).astype(np.float32)
        if not np.any(window):
            window = np.ones(len(stretched), dtype=np.float32)

        frames_data.append(stretched * window[:, None])
        frames_weight.append(window)
        positions.append(output_cursor)

        hop_out = max(1, int(len(stretched) * (1.0 - overlap)))
        output_cursor += hop_out

    total_length = positions[-1] + len(frames_data[-1])
    output = np.zeros((total_length, channels), dtype=np.float32)
    weight = np.zeros((total_length, 1), dtype=np.float32)

    for data, win, pos in zip(frames_data, frames_weight, positions):
        length = len(data)
        if pos + length > len(output):
            extra = pos + length - len(output)
            output = np.vstack([output, np.zeros((extra, channels), dtype=np.float32)])
            weight = np.vstack([weight, np.zeros((extra, 1), dtype=np.float32)])

        output[pos : pos + length] += data
        weight[pos : pos + length] += win[:, None]

    mask = weight.squeeze() > 1e-6
    if np.any(mask):
        output[mask] /= weight[mask]
    if np.any(~mask):
        output[~mask] = 0.0

    if squeeze:
        return output[:, 0]

    return output


def _generate_tempo_ramp_arrays(
    deck_a,
    deck_b_original,
    mix_start_sample,
    current_bpm,
    next_bpm,
    samplerate,
):
    crossfade_samples = max(1, int(CROSSFADE_SECONDS * samplerate))
    segment_a_start = max(
        0, min(mix_start_sample, max(len(deck_a) - crossfade_samples, 0))
    )
    segment_a_end = min(len(deck_a), segment_a_start + crossfade_samples)
    if segment_a_end <= segment_a_start:
        raise ValueError("Insufficient Deck A audio available for tempo ramp.")

    segment_b_end = min(len(deck_b_original), crossfade_samples)
    if segment_b_end <= 0:
        raise ValueError("Insufficient Deck B audio available for tempo ramp.")

    segment_a = deck_a[segment_a_start:segment_a_end]
    segment_b = deck_b_original[:segment_b_end]

    ramp_a = _framewise_tempo_ramp(
        segment_a,
        samplerate,
        start_ratio=1.0,
        end_ratio=_clamp_ratio(next_bpm / current_bpm),
    )
    ramp_b = _framewise_tempo_ramp(
        segment_b,
        samplerate,
        start_ratio=_clamp_ratio(current_bpm / next_bpm),
        end_ratio=1.0,
    )

    updated_deck_a = np.concatenate(
        (deck_a[:segment_a_start], ramp_a, deck_a[segment_a_end:]), axis=0
    ).astype(np.float32, copy=False)
    updated_deck_b = np.concatenate(
        (ramp_b, deck_b_original[segment_b_end:]), axis=0
    ).astype(np.float32, copy=False)

    fade_duration = max(len(ramp_a), len(ramp_b)) / samplerate

    return updated_deck_a, updated_deck_b, fade_duration, segment_a_start


def _estimate_mix_start_sample(track_info, deck_length, samplerate):
    track_duration_sec = deck_length / samplerate if samplerate else 0
    downbeats = sorted(track_info.get("downbeats", []) or [])
    window_start = max(0.0, track_duration_sec - MIX_POINT_SECONDS)
    ideal_mix_time = max(0.0, track_duration_sec - FALLBACK_TARGET_SECONDS)

    mix_time = None
    if downbeats:
        candidate_downbeats = [db for db in downbeats if db >= window_start]
        if candidate_downbeats:
            mix_time = min(candidate_downbeats, key=lambda db: abs(db - ideal_mix_time))
        else:
            earlier_downbeats = [db for db in downbeats if db < window_start]
            if earlier_downbeats:
                mix_time = earlier_downbeats[-1]

    if mix_time is None:
        mix_time = max(0.0, track_duration_sec - CROSSFADE_SECONDS - 2)

    mix_start_sample = int(round(mix_time * samplerate)) if samplerate else 0
    max_start = (
        max(0, deck_length - int(CROSSFADE_SECONDS * samplerate)) if samplerate else 0
    )
    mix_start_sample = max(0, min(mix_start_sample, max_start))
    return mix_start_sample, mix_time


def apply_tempo_ramp_to_transition(mix_start_sample, current_info, next_info):
    samplerate = player_state["samplerate"]
    deck_a = player_state.get("deck_a")
    if deck_a is None:
        raise ValueError("Deck A audio is not available for tempo ramping.")

    current_bpm = _safe_bpm(current_info.get("bpm"))
    next_bpm = _safe_bpm(next_info.get("bpm"))
    if not current_bpm or not next_bpm:
        raise ValueError("Missing BPM data for tempo ramp computation.")

    deck_b_original = load_and_prepare_track(next_info)
    updated_deck_a, updated_deck_b, fade_duration, _ = _generate_tempo_ramp_arrays(
        deck_a,
        deck_b_original,
        mix_start_sample,
        current_bpm,
        next_bpm,
        samplerate,
    )

    player_state["deck_a"] = updated_deck_a
    player_state["deck_b"] = updated_deck_b

    return fade_duration


def schedule_tempo_ramp_preprocessing(current_info, next_info):
    if not current_info or not next_info:
        player_state["precomputed_transition"] = None
        player_state["tempo_ramp_precompute_pair"] = None
        return

    deck_a = player_state.get("deck_a")
    samplerate = player_state.get("samplerate", 0)
    if deck_a is None or not samplerate:
        return

    from_path = current_info.get("path")
    to_path = next_info.get("path")
    if not from_path or not to_path:
        return

    pair = (from_path, to_path)

    existing_ready = player_state.get("precomputed_transition")
    if (
        existing_ready
        and existing_ready.get("status") == "ready"
        and existing_ready.get("from_path") == pair[0]
        and existing_ready.get("to_path") == pair[1]
    ):
        return

    active_thread = player_state.get("tempo_ramp_precompute_thread")
    if active_thread and active_thread.is_alive():
        if player_state.get("tempo_ramp_precompute_pair") == pair:
            return

    player_state["precomputed_transition"] = None
    player_state["tempo_ramp_precompute_pair"] = pair

    deck_a_snapshot = np.copy(deck_a)
    current_snapshot = {
        "path": from_path,
        "bpm": current_info.get("bpm"),
        "downbeats": list(current_info.get("downbeats", []) or []),
    }
    next_snapshot = copy.deepcopy(next_info)

    def _worker():
        try:
            current_bpm = _safe_bpm(current_snapshot.get("bpm"))
            next_bpm = _safe_bpm(next_snapshot.get("bpm"))
            if not current_bpm or not next_bpm:
                raise ValueError("Missing BPM data for tempo ramp computation.")

            deck_b_original = load_and_prepare_track(next_snapshot)
            mix_start_sample, mix_time_sec = _estimate_mix_start_sample(
                current_snapshot, len(deck_a_snapshot), samplerate
            )

            updated_deck_a, updated_deck_b, fade_duration, segment_a_start = (
                _generate_tempo_ramp_arrays(
                    deck_a_snapshot,
                    deck_b_original,
                    mix_start_sample,
                    current_bpm,
                    next_bpm,
                    samplerate,
                )
            )

            result = {
                "status": "ready",
                "from_path": pair[0],
                "to_path": pair[1],
                "deck_a_processed": updated_deck_a,
                "deck_b_processed": updated_deck_b,
                "mix_start_sample": mix_start_sample,
                "mix_time_sec": mix_time_sec,
                "fade_duration": fade_duration,
                "segment_a_start": segment_a_start,
                "prepared_at": time.time(),
            }
            logging.info(
                "Precomputed tempo ramp %s → %s (start %.2fs, fade %.2fs)",
                os.path.basename(pair[0]),
                os.path.basename(pair[1]),
                mix_time_sec if mix_time_sec is not None else -1,
                fade_duration,
            )
        except Exception as exc:
            logging.warning(
                "Tempo ramp precompute failed for %s → %s: %s",
                os.path.basename(pair[0]) if pair[0] else "unknown",
                os.path.basename(pair[1]) if pair[1] else "unknown",
                exc,
            )
            result = {
                "status": "error",
                "from_path": pair[0],
                "to_path": pair[1],
                "error": str(exc),
            }

        if (
            player_state.get("deck_a_info")
            and player_state["deck_a_info"].get("path") == pair[0]
            and player_state.get("deck_b_info")
            and player_state["deck_b_info"].get("path") == pair[1]
            and player_state.get("tempo_ramp_precompute_pair") == pair
        ):
            player_state["precomputed_transition"] = result

        if (
            player_state.get("tempo_ramp_precompute_thread")
            is threading.current_thread()
        ):
            player_state["tempo_ramp_precompute_thread"] = None

    worker_thread = threading.Thread(target=_worker, daemon=True)
    player_state["tempo_ramp_precompute_thread"] = worker_thread
    worker_thread.start()


# --- Audio Loading & Processing ---
def load_and_prepare_track(track_info, target_bpm=None):
    """Loads a track, prepares it for playback, and ensures it has a title."""
    logging.info(f"Loading: {os.path.basename(track_info['path'])}")

    # Ensure title exists
    if "title" not in track_info or not track_info["title"]:
        track_info["title"] = os.path.splitext(os.path.basename(track_info["path"]))[0]

    audio = AudioSegment.from_file(track_info["path"])
    audio = audio.set_frame_rate(player_state["samplerate"]).set_channels(
        player_state["channels"]
    )

    samples = (
        np.array(audio.get_array_of_samples())
        .reshape(-1, audio.channels)
        .astype(np.float32)
    )
    samples /= 2 ** (audio.sample_width * 8 - 1)  # Normalize to [-1.0, 1.0]

    if target_bpm and track_info["bpm"] != target_bpm:
        stretch_ratio = target_bpm / track_info["bpm"]
        logging.info(
            f"Stretching {track_info['bpm']}->{target_bpm} BPM (ratio: {stretch_ratio:.3f})"
        )
        # Corrected function call from stretch to time_stretch
        return rb.time_stretch(samples, player_state["samplerate"], stretch_ratio)

    return samples


# --- The Audio Callback (The "Chef") ---
def get_mixed_chunk(frames):
    """Returns the next mixed audio chunk as a float32 numpy array."""
    channels = player_state["channels"]
    if not player_state["is_playing"]:
        return np.zeros((frames, channels), dtype=np.float32)

    start = player_state["play_pos"]
    end = start + frames

    # Get audio chunk from Deck A
    chunk_a = np.zeros((frames, channels), dtype=np.float32)
    deck_a = player_state["deck_a"]
    if deck_a is not None:
        len_a = len(deck_a)
        if start < len_a:
            available_a = deck_a[start : min(end, len_a)]
            chunk_a[: len(available_a)] = available_a

    # Get audio chunk from Deck B
    chunk_b = np.zeros((frames, channels), dtype=np.float32)
    deck_b = player_state["deck_b"]
    if deck_b is not None:
        start_b_rel = start - player_state["deck_b_start_pos"]
        end_b_rel = end - player_state["deck_b_start_pos"]
        if end_b_rel > 0:
            len_b = len(deck_b)
            start_b_abs = max(0, start_b_rel)
            end_b_abs = min(len_b, end_b_rel)
            if start_b_abs < len_b:
                available_b = deck_b[start_b_abs:end_b_abs]
                offset = max(0, -start_b_rel)
                chunk_b[offset : offset + len(available_b)] = available_b

    # Mix using an equal-power crossfade curve for smooth volume transition
    fader = player_state["fader"]
    vol_a = np.cos((fader + 1) * 0.25 * np.pi)
    vol_b = np.sin((fader + 1) * 0.25 * np.pi)

    mixed_chunk = (chunk_a * vol_a) + (chunk_b * vol_b)

    player_state["play_pos"] += frames
    return mixed_chunk


def audio_callback(outdata, frames, time, status):
    if status:
        print(status, flush=True)

    if not player_state["is_playing"]:
        outdata.fill(0)
        return

    outdata[:] = get_mixed_chunk(frames)


def find_and_set_next_track(current_info, unplayed_tracks, library):
    """Finds the next track, updates deck_b_info, and returns the modified unplayed_tracks list."""
    if not unplayed_tracks:
        logging.info("--- ENTIRE LIBRARY PLAYED, RE-SHUFFLING ---")
        unplayed_tracks = [t for t in library if t["path"] != current_info["path"]]
        if not unplayed_tracks:
            logging.info("Only one song in library. Cannot re-shuffle.")
            player_state["is_playing"] = False
            return unplayed_tracks
        random.shuffle(unplayed_tracks)

    next_info = find_best_next_track(current_info, unplayed_tracks)
    if not next_info:
        logging.warning("Could not find a next track to pre-load. Stopping.")
        player_state["is_playing"] = False
        return unplayed_tracks

    logging.info("=" * 50)
    logging.info(f"🎧 Pre-selecting Next Up: {os.path.basename(next_info['path'])}")
    player_state["deck_b_info"] = next_info
    update_deck_cover("deck_b", next_info)

    # Return the updated list of unplayed tracks
    return [t for t in unplayed_tracks if t["path"] != next_info["path"]]


# --- The DJ "Brain" Thread ---
def run_dj_logic_headless():
    """
    The core DJ logic loop, designed to be run in a thread without direct audio output.
    It's controlled by the player_state and signals transitions.
    """
    library = load_library_from_analytics()
    if not library:
        logging.error(
            "Fatal Error: no analyzed tracks found. Please run library_manager.py to generate analytics."
        )
        player_state["is_playing"] = False
        return

    player_state["total_tracks_in_library"] = len(library)
    unplayed_tracks = library.copy()
    random.shuffle(unplayed_tracks)

    # Start with a track from the shuffled list
    first_track_info = unplayed_tracks.pop()
    player_state["deck_a_info"] = first_track_info
    player_state["deck_a"] = load_and_prepare_track(player_state["deck_a_info"])
    update_deck_cover("deck_a", player_state["deck_a_info"])
    player_state["tracks_played_count"] = 1
    logging.info(
        f"▶️ Starting with: {os.path.basename(player_state['deck_a_info']['path'])}"
    )

    # Immediately find the next track to populate the UI
    unplayed_tracks = find_and_set_next_track(
        player_state["deck_a_info"], unplayed_tracks, library
    )
    schedule_tempo_ramp_preprocessing(
        player_state["deck_a_info"], player_state.get("deck_b_info")
    )

    while player_state["is_playing"]:
        time.sleep(0.1)

        if player_state["deck_a"] is None:
            # This can happen if the initial load fails or at the very end.
            if not unplayed_tracks:
                logging.info("--- ALL TRACKS PLAYED, ENDING SESSION ---")
                player_state["is_playing"] = False
            else:
                # Attempt to load a new track if Deck A is unexpectedly empty
                next_info = unplayed_tracks.pop()
                player_state["deck_a_info"] = next_info
                player_state["deck_a"] = load_and_prepare_track(next_info)
                update_deck_cover("deck_a", next_info)
                player_state["tracks_played_count"] += 1
                logging.info(
                    f"▶️ Deck A was empty, starting new track: {os.path.basename(next_info['path'])}"
                )
                unplayed_tracks = find_and_set_next_track(
                    player_state["deck_a_info"], unplayed_tracks, library
                )
                schedule_tempo_ramp_preprocessing(
                    player_state["deck_a_info"], player_state.get("deck_b_info")
                )
            continue

        duration_samples = len(player_state["deck_a"])
        mix_point_samples = duration_samples - (
            MIX_POINT_SECONDS * player_state["samplerate"]
        )

        # If we are playing Deck A and are approaching the mix point...
        if (
            player_state["fader"] == -1.0
            and player_state["play_pos"] > mix_point_samples
        ):
            current_info = player_state["deck_a_info"]
            next_info = player_state["deck_b_info"]  # Already selected

            if not next_info:
                logging.warning("Next track info not found. Stopping.")
                player_state["is_playing"] = False
                continue

            current_bpm_val = _safe_bpm(current_info.get("bpm")) or 0.0
            next_bpm_val = _safe_bpm(next_info.get("bpm")) or current_bpm_val
            logging.info(
                f"Mixing from {current_bpm_val:.2f} BPM to {next_bpm_val:.2f} BPM..."
            )

            track_duration_sec = duration_samples / player_state["samplerate"]
            precomputed = player_state.get("precomputed_transition")
            precomputed_valid = (
                precomputed
                and precomputed.get("status") == "ready"
                and precomputed.get("from_path")
                and precomputed.get("to_path")
                and current_info.get("path") == precomputed.get("from_path")
                and next_info.get("path") == precomputed.get("to_path")
            )

            if precomputed_valid:
                mix_start_sample = precomputed.get("mix_start_sample", 0)
                mix_time_sec = precomputed.get("mix_time_sec")
                logging.info(
                    "Using precomputed tempo ramp (start %.2fs, fade %.2fs).",
                    (
                        mix_time_sec
                        if mix_time_sec is not None
                        else mix_start_sample / player_state["samplerate"]
                    ),
                    precomputed.get("fade_duration", CROSSFADE_SECONDS),
                )
            else:
                current_play_time_sec = (
                    player_state["play_pos"] / player_state["samplerate"]
                )
                downbeats = sorted(current_info.get("downbeats", []))
                mix_downbeat = None
                if downbeats:
                    ideal_mix_time = track_duration_sec - FALLBACK_TARGET_SECONDS
                    future_downbeats = [
                        db for db in downbeats if db > current_play_time_sec
                    ]

                    if future_downbeats:
                        mix_downbeat = min(
                            future_downbeats,
                            key=lambda db: abs(db - ideal_mix_time),
                        )
                        logging.info(
                            f"Found future downbeat at {mix_downbeat:.2f}s for mixing."
                        )
                    else:
                        logging.warning(
                            "No future downbeats found! Mixing immediately."
                        )
                else:
                    logging.info("No downbeat data; mixing based on time.")

                mix_start_sample = (
                    int(mix_downbeat * player_state["samplerate"])
                    if mix_downbeat
                    else int(
                        (track_duration_sec - CROSSFADE_SECONDS - 2)
                        * player_state["samplerate"]
                    )
                )

            mix_start_sample = max(player_state["play_pos"], mix_start_sample)

            player_state["deck_b_start_pos"] = mix_start_sample
            logging.info(
                f"Mixing on downbeat at {mix_start_sample / player_state['samplerate']:.2f}s"
            )

            if precomputed_valid:
                player_state["deck_a"] = precomputed["deck_a_processed"]
                player_state["deck_b"] = precomputed["deck_b_processed"]
                player_state["current_crossfade_duration"] = precomputed.get(
                    "fade_duration", CROSSFADE_SECONDS
                )
                player_state["precomputed_transition"] = None
            else:
                try:
                    fade_duration = apply_tempo_ramp_to_transition(
                        mix_start_sample, current_info, next_info
                    )
                    player_state["current_crossfade_duration"] = fade_duration
                    logging.info(
                        "Tempo ramp scheduled over %.2fs using 100ms frames.",
                        fade_duration,
                    )
                except Exception as exc:
                    logging.warning(
                        "Tempo ramp unavailable (%s); falling back to static stretch.",
                        exc,
                    )
                    target_bpm = current_bpm_val if current_bpm_val > 0 else None
                    if target_bpm:
                        player_state["deck_b"] = load_and_prepare_track(
                            next_info, target_bpm=target_bpm
                        )
                    else:
                        player_state["deck_b"] = load_and_prepare_track(next_info)
                    player_state["current_crossfade_duration"] = CROSSFADE_SECONDS

            # 3. Wait until the playhead reaches the mix point
            while player_state["play_pos"] < mix_start_sample:
                if not player_state["is_playing"]:
                    return  # Exit if playback is stopped externally
                time.sleep(0.01)

            # 4. Animate the crossfader
            logging.info("--- STARTING MIX ---")
            fade_duration = (
                player_state.get("current_crossfade_duration", CROSSFADE_SECONDS)
                or CROSSFADE_SECONDS
            )
            fade_start_time = time.time()
            while True:
                if not player_state["is_playing"]:
                    return  # Exit if playback is stopped
                linear_progress = min(
                    1.0,
                    (time.time() - fade_start_time) / fade_duration,
                )
                # Use an equal-power curve for smoother transition
                player_state["fader"] = -1.0 + 2.0 * linear_progress
                if linear_progress >= 1.0:
                    break
                time.sleep(0.01)
            logging.info("--- MIX COMPLETE ---")

            # 5. Transition state: Deck B becomes the new Deck A
            player_state["play_pos"] -= player_state["deck_b_start_pos"]
            player_state["deck_a"], player_state["deck_b"] = (
                player_state["deck_b"],
                None,
            )
            player_state["deck_a_info"], player_state["deck_b_info"] = (
                player_state["deck_b_info"],
                None,
            )
            player_state["deck_a_cover"] = player_state["deck_b_cover"]
            player_state["deck_a_cover_version"] = player_state["deck_b_cover_version"]
            player_state["deck_b_cover"] = None
            player_state["deck_b_cover_version"] = 0
            player_state["fader"] = -1.0
            player_state["current_crossfade_duration"] = CROSSFADE_SECONDS
            player_state["tracks_played_count"] += 1

            # Immediately find the *next* track for the new Deck A
            if player_state["is_playing"]:
                unplayed_tracks = find_and_set_next_track(
                    player_state["deck_a_info"], unplayed_tracks, library
                )


def initialize_player():
    """
    Loads the library and determines initial audio parameters without starting playback.
    """
    library = load_library_from_analytics()
    if not library:
        logging.error(
            "No analyzed tracks available. Please run library_manager.py before starting the player."
        )
        return False

    player_state["total_tracks_in_library"] = len(library)

    try:
        info = AudioSegment.from_file(library[0]["path"])
        player_state.update({"samplerate": info.frame_rate, "channels": info.channels})
        logging.info(
            f"Player initialized. Samplerate: {info.frame_rate}Hz, Channels: {info.channels}"
        )
        return True
    except Exception as e:
        logging.error(f"Error reading first library track for setup: {e}")

    return False


# --- Main Execution (for standalone playback) ---
if __name__ == "__main__":
    if not initialize_player():
        exit()

    # The original run_dj_logic is now the headless version
    dj_thread = threading.Thread(target=run_dj_logic_headless)

    try:
        with sd.OutputStream(
            samplerate=player_state["samplerate"],
            channels=player_state["channels"],
            callback=audio_callback,
            blocksize=BLOCK_SIZE,
        ) as stream:
            dj_thread.start()
            logging.info(
                f"Stream started. Sample Rate: {stream.samplerate}Hz. Press Ctrl+C to stop."
            )
            while dj_thread.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("\nStopping player...")
    except Exception as e:
        logging.error(f"An audio error occurred: {e}")
    finally:
        player_state["is_playing"] = False
        if dj_thread.is_alive():
            dj_thread.join()
        logging.info("Player stopped.")
