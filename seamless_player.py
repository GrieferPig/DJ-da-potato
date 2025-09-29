import json
import random
import time
import logging
import os
import math
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
}

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
        return None

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

        if _find_best_next_track_call_count % 7 == 0:
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

            # 1. Load the pre-selected next track onto Deck B
            logging.info(f"Mixing from {current_info['bpm']:.2f} BPM...")
            player_state["deck_b"] = load_and_prepare_track(
                next_info, target_bpm=current_info["bpm"]
            )

            # 2. Find the downbeat to align to
            current_play_time_sec = (
                player_state["play_pos"] / player_state["samplerate"]
            )
            track_duration_sec = duration_samples / player_state["samplerate"]
            downbeats = sorted(current_info.get("downbeats", []))
            mix_downbeat = None
            if downbeats:
                # Try to find a downbeat in the ideal mix-out window
                ideal_mix_time = track_duration_sec - FALLBACK_TARGET_SECONDS
                future_downbeats = [
                    db for db in downbeats if db > current_play_time_sec
                ]

                if future_downbeats:
                    # Find the downbeat closest to the ideal mix time
                    mix_downbeat = min(
                        future_downbeats, key=lambda db: abs(db - ideal_mix_time)
                    )
                    logging.info(
                        f"Found future downbeat at {mix_downbeat:.2f}s for mixing."
                    )
                else:
                    logging.warning("No future downbeats found! Mixing immediately.")
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
            # Ensure we don't try to mix in the past
            mix_start_sample = max(player_state["play_pos"], mix_start_sample)

            player_state["deck_b_start_pos"] = mix_start_sample
            logging.info(
                f"Mixing on downbeat at {mix_start_sample / player_state['samplerate']:.2f}s"
            )

            # 3. Wait until the playhead reaches the mix point
            while player_state["play_pos"] < mix_start_sample:
                if not player_state["is_playing"]:
                    return  # Exit if playback is stopped externally
                time.sleep(0.01)

            # 4. Animate the crossfader
            logging.info("--- STARTING MIX ---")
            fade_start_time = time.time()
            while True:
                if not player_state["is_playing"]:
                    return  # Exit if playback is stopped
                linear_progress = min(
                    1.0,
                    (time.time() - fade_start_time) / CROSSFADE_SECONDS,
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
