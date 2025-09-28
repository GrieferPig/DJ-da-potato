from flask import Flask, render_template, Response, jsonify, abort
from flask_socketio import SocketIO, emit
import seamless_player
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    engineio_options={"transports": ["websocket"]},
)

# --- Audio streaming buffer ---
AUDIO_HISTORY_SIZE = 64  # Keep a few seconds of recent audio for late joiners
audio_history = deque(maxlen=AUDIO_HISTORY_SIZE)
audio_sequence = 0
history_lock = threading.Lock()
audio_config = {}
config_lock = threading.Lock()
config_announced = False


def set_audio_config(sample_rate: int, channels: int, block_size: int) -> None:
    with config_lock:
        audio_config["sampleRate"] = sample_rate
        audio_config["channels"] = channels
        audio_config["blockSize"] = block_size


def get_audio_config_snapshot() -> dict:
    with config_lock:
        return dict(audio_config)


# --- Global state for web display ---
web_state = {
    "current_song": "None",
    "next_song": "None",
    "track_count": 0,
    "total_tracks": 0,
    "current_cover_version": 0,
    "next_cover_version": 0,
    "current_track_number": "--",
    "current_album": "Unknown Album",
    "current_artist": "Unknown Artist",
    "current_track_title": "Unknown Title",
    "next_track_number": "--",
    "next_album": "Unknown Album",
    "next_artist": "Unknown Artist",
    "next_track_title": "Unknown Title",
}


UNKNOWN_METADATA = {
    "track_number": "--",
    "album": "Unknown Album",
    "artist": "Unknown Artist",
    "track_title": "Unknown Title",
}


def parse_track_metadata(track_path: str | None) -> dict | None:
    if not track_path:
        return None

    stem = Path(track_path).stem
    parts = stem.split(" - ")
    if len(parts) < 3:
        return None

    prefix = parts[0].strip()
    artist = parts[1].strip()
    title = " - ".join(parts[2:]).strip()

    number = prefix[:2] if len(prefix) >= 2 and prefix[:2].isdigit() else ""

    return {
        "track_number": number,
        "artist": artist,
        "track_title": title or stem,
    }


def build_track_display_metadata(track_info: dict | None) -> dict:
    metadata = dict(UNKNOWN_METADATA)
    if not track_info:
        return metadata

    parsed = parse_track_metadata(track_info.get("path"))
    if parsed:
        for key, value in parsed.items():
            if value:
                metadata[key] = value

    tags = seamless_player.get_track_tags(track_info.get("path"))

    tag_title = tags.get("title") if tags else None
    extracted_artist_from_title = None
    if tag_title:
        # Sanitize title if it appears to contain artist info (e.g. "Artist - Title")
        sanitized_title = tag_title
        if " - " in tag_title:
            # Check if this looks like "Artist - Title" format
            parts = tag_title.split(" - ", 1)  # Split only on first " - "
            if len(parts) == 2:
                # If we also have an artist tag, and the first part matches the artist, use only the title part
                tag_artist_check = tags.get("artist") if tags else None
                if (
                    tag_artist_check
                    and parts[0].strip().lower() == tag_artist_check.strip().lower()
                ):
                    sanitized_title = parts[1].strip()
                    extracted_artist_from_title = parts[0].strip()
                # If no artist tag but first part looks like artist name (contains spaces or is short), use title part
                elif len(parts[0].strip()) < 50 and (
                    len(parts[1].strip()) > len(parts[0].strip())
                    or " " not in parts[0].strip()
                ):
                    sanitized_title = parts[1].strip()
                    extracted_artist_from_title = parts[0].strip()
        metadata["track_title"] = sanitized_title

    tag_artist = tags.get("artist") if tags else None
    # Prefer artist parsed from the title pattern over the tag's artist
    if extracted_artist_from_title:
        metadata["artist"] = extracted_artist_from_title
    elif tag_artist:
        metadata["artist"] = tag_artist

    tag_album = tags.get("album") if tags else None
    if tag_album:
        metadata["album"] = tag_album

    # Only use track_info title as fallback if metadata extraction failed AND the title doesn't look like a filename
    title = track_info.get("title")
    if (
        isinstance(title, str)
        and title.strip()
        and metadata["track_title"] == UNKNOWN_METADATA["track_title"]
        and " - " not in title  # Don't use titles that look like filenames
    ):
        metadata["track_title"] = title.strip()

    artist = track_info.get("artist")
    if (
        isinstance(artist, str)
        and artist.strip()
        and metadata["artist"] == UNKNOWN_METADATA["artist"]
    ):
        metadata["artist"] = artist.strip()

    album = track_info.get("album")
    if (
        isinstance(album, str)
        and album.strip()
        and metadata["album"] == UNKNOWN_METADATA["album"]
    ):
        metadata["album"] = album.strip()

    return metadata


def audio_producer():
    """Launches the DJ logic and broadcasts mixed audio chunks over Socket.IO."""
    global audio_sequence, config_announced

    dj_thread = threading.Thread(
        target=seamless_player.run_dj_logic_headless,
        daemon=True,
    )
    dj_thread.start()

    samplerate = seamless_player.player_state.get("samplerate", 44100)
    channels = seamless_player.player_state.get("channels", 2)
    frames = getattr(seamless_player, "BLOCK_SIZE", 2048)
    seconds_per_chunk = frames / float(samplerate) if samplerate else 0.05

    set_audio_config(samplerate, channels, frames)
    if not config_announced:
        socketio.emit("audio_config", get_audio_config_snapshot())
        config_announced = True

    while True:
        if not seamless_player.player_state.get("is_playing", False):
            break

        if seamless_player.player_state.get("deck_a") is None:
            time.sleep(0.05)
            continue

        loop_start = time.time()
        mixed_chunk = seamless_player.get_mixed_chunk(frames)
        clipped_chunk = np.clip(mixed_chunk, -1.0, 1.0)
        chunk_bytes = (clipped_chunk * 32767).astype(np.int16).tobytes()

        with history_lock:
            payload = {"seq": audio_sequence, "chunk": chunk_bytes}
            audio_history.append(payload)
            audio_sequence += 1

        socketio.emit("audio_chunk", payload)

        elapsed = time.time() - loop_start
        sleep_time = seconds_per_chunk - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    with history_lock:
        audio_history.clear()
        audio_sequence = 0
    config_announced = False


def state_updater():
    """Periodically updates the web state from the player state."""
    while True:
        deck_a_info = seamless_player.player_state.get("deck_a_info")
        deck_b_info = seamless_player.player_state.get("deck_b_info")

        if deck_a_info:
            current_meta = build_track_display_metadata(deck_a_info)
            web_state["current_song"] = current_meta["track_title"]
        else:
            current_meta = dict(UNKNOWN_METADATA)
            web_state["current_song"] = "None"

        if deck_b_info:
            next_meta = build_track_display_metadata(deck_b_info)
            web_state["next_song"] = next_meta["track_title"]
        else:
            next_meta = dict(UNKNOWN_METADATA)
            web_state["next_song"] = "Finding next track..."

        web_state["current_track_number"] = current_meta["track_number"]
        web_state["current_album"] = current_meta["album"]
        web_state["current_artist"] = current_meta["artist"]
        web_state["current_track_title"] = current_meta["track_title"]

        web_state["next_track_number"] = next_meta["track_number"]
        web_state["next_album"] = next_meta["album"]
        web_state["next_artist"] = next_meta["artist"]
        web_state["next_track_title"] = next_meta["track_title"]

        web_state["track_count"] = seamless_player.player_state.get(
            "tracks_played_count", 0
        )

        web_state["total_tracks"] = seamless_player.player_state.get(
            "total_tracks_in_library", 0
        )
        web_state["current_cover_version"] = seamless_player.player_state.get(
            "deck_a_cover_version", 0
        )
        web_state["next_cover_version"] = seamless_player.player_state.get(
            "deck_b_cover_version", 0
        )

        socketio.emit("track_state", dict(web_state))
        time.sleep(1)


@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/status")
def status():
    """Returns the current audio and player state."""
    return jsonify(
        {
            "config": get_audio_config_snapshot(),
            "state": dict(web_state),
        }
    )


@app.route("/cover/<deck>")
def cover(deck: str):
    deck = deck.lower()
    if deck == "current":
        cover_data = seamless_player.player_state.get("deck_a_cover")
        version = seamless_player.player_state.get("deck_a_cover_version", 0)
    elif deck == "next":
        cover_data = seamless_player.player_state.get("deck_b_cover")
        version = seamless_player.player_state.get("deck_b_cover_version", 0)
    else:
        abort(404)

    if not cover_data or not cover_data.get("data"):
        return Response(status=204)

    response = Response(
        cover_data["data"], mimetype=cover_data.get("mime", "image/jpeg")
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["ETag"] = str(version)
    return response


@socketio.on("connect")
def handle_connect():
    """Send configuration, recent audio, and current state to new clients."""
    snapshot = get_audio_config_snapshot()
    if snapshot:
        emit("audio_config", snapshot)

    with history_lock:
        history_snapshot = list(audio_history)

    for payload in history_snapshot:
        emit("audio_chunk", payload)

    emit("track_state", dict(web_state))


if __name__ == "__main__":
    # Initialize the player (loads library, etc.)
    if not seamless_player.initialize_player():
        raise SystemExit(1)

    # Start the DJ logic in a background thread
    dj_thread = threading.Thread(target=audio_producer, daemon=True)
    dj_thread.start()

    # Start the state updater in a background thread
    state_thread = threading.Thread(target=state_updater, daemon=True)
    state_thread.start()

    print("Starting Flask Socket.IO server...")
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)
