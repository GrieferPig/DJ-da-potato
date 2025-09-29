from flask import Flask, render_template, Response, jsonify, abort, send_from_directory
from flask_socketio import SocketIO, emit
import seamless_player
import threading
import time
from collections import deque
from pathlib import Path
import queue
import numpy as np

try:
    import lameenc
except ImportError:  # pragma: no cover - MP3 streaming optional without encoder
    lameenc = None

try:
    from eventlet.queue import (  # type: ignore[import]
        Empty as EventletQueueEmpty,
        Full as EventletQueueFull,
        LightQueue as EventletQueue,
    )
except ImportError:  # pragma: no cover - eventlet optional
    EventletQueueEmpty = None
    EventletQueueFull = None
    EventletQueue = None


app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    engineio_options={"transports": ["websocket"]},
)

SERVER_START_TIME = time.time()

# --- Audio streaming buffer ---
AUDIO_HISTORY_SIZE = 64  # Keep a few seconds of recent audio for late joiners
audio_history = deque(maxlen=AUDIO_HISTORY_SIZE)
audio_sequence = 0
history_lock = threading.Lock()
audio_config = {}
config_lock = threading.Lock()

audio_chunk_queue = queue.Queue(maxsize=128)
audio_encoder_thread = None
encoder_thread_lock = threading.Lock()
forwarder_lock = threading.Lock()
forwarder_started = False

QUEUE_EMPTY_EXCEPTIONS = (
    (queue.Empty,) if EventletQueueEmpty is None else (queue.Empty, EventletQueueEmpty)
)
QUEUE_FULL_EXCEPTIONS = (
    (queue.Full,) if EventletQueueFull is None else (queue.Full, EventletQueueFull)
)

MP3_CLIENT_QUEUE_SIZE = 256
mp3_stream_clients = set()
mp3_clients_lock = threading.Lock()


def _create_mp3_client_queue(maxsize: int) -> queue.Queue:
    if EventletQueue is not None:
        return EventletQueue(maxsize)
    return queue.Queue(maxsize=maxsize)


def _cooperative_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    try:
        socketio.sleep(seconds)
    except RuntimeError:
        time.sleep(seconds)


def _enqueue_mp3_message(target_queue: queue.Queue, message: dict) -> bool:
    try:
        target_queue.put_nowait(message)
        return True
    except QUEUE_FULL_EXCEPTIONS:
        try:
            target_queue.get_nowait()
        except QUEUE_EMPTY_EXCEPTIONS:
            pass
        try:
            target_queue.put_nowait(message)
            return True
        except QUEUE_FULL_EXCEPTIONS:
            return False
    except Exception:
        return False


def register_mp3_stream_client() -> queue.Queue:
    client_queue: queue.Queue = _create_mp3_client_queue(MP3_CLIENT_QUEUE_SIZE)
    with mp3_clients_lock:
        mp3_stream_clients.add(client_queue)

    snapshot = get_audio_config_snapshot()
    if snapshot:
        _enqueue_mp3_message(client_queue, {"type": "config", "data": snapshot})

    with history_lock:
        history_snapshot = list(audio_history)

    for payload in history_snapshot:
        _enqueue_mp3_message(client_queue, {"type": "chunk", "data": payload})

    return client_queue


def unregister_mp3_stream_client(client_queue: queue.Queue) -> None:
    with mp3_clients_lock:
        mp3_stream_clients.discard(client_queue)

    try:
        while True:
            client_queue.get_nowait()
    except QUEUE_EMPTY_EXCEPTIONS:
        pass


def broadcast_to_mp3_clients(message: dict) -> None:
    if not mp3_stream_clients:
        return

    with mp3_clients_lock:
        clients_snapshot = list(mp3_stream_clients)

    if not clients_snapshot:
        return

    stale_clients = []
    for client_queue in clients_snapshot:
        if not _enqueue_mp3_message(client_queue, message):
            stale_clients.append(client_queue)

    if stale_clients:
        with mp3_clients_lock:
            for client_queue in stale_clients:
                mp3_stream_clients.discard(client_queue)


def create_mp3_encoder(config: dict | None):
    if lameenc is None:
        return None

    config = config or {}
    sample_rate = int(config.get("sampleRate") or 44100)
    channels = int(config.get("channels") or 2)

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_bit_rate(192)
    encoder.set_channels(max(1, min(channels, 2)))
    encoder.set_quality(2)
    return encoder


def mp3_stream_generator():
    client_queue = register_mp3_stream_client()
    encoder = create_mp3_encoder(get_audio_config_snapshot())
    encoded_since_config = False

    try:
        while True:
            try:
                message = client_queue.get_nowait()
            except QUEUE_EMPTY_EXCEPTIONS:
                _cooperative_sleep(0.01)
                continue

            if not message:
                continue

            msg_type = message.get("type")

            if msg_type == "config":
                if encoder and encoded_since_config:
                    try:
                        final_bytes = encoder.flush()
                    except Exception:
                        final_bytes = b""
                    if final_bytes:
                        yield final_bytes
                encoder = create_mp3_encoder(message.get("data"))
                encoded_since_config = False

            elif msg_type == "chunk":
                if not encoder:
                    continue

                chunk_payload = message.get("data") or {}
                chunk_bytes = chunk_payload.get("chunk")
                if not chunk_bytes:
                    continue

                try:
                    mp3_bytes = encoder.encode(chunk_bytes)
                except Exception:
                    mp3_bytes = b""

                if mp3_bytes:
                    yield mp3_bytes
                    encoded_since_config = True

            elif msg_type == "stop":
                if encoder:
                    try:
                        final_bytes = encoder.flush()
                    except Exception:
                        final_bytes = b""
                    if final_bytes:
                        yield final_bytes
                break

    finally:
        unregister_mp3_stream_client(client_queue)


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
    "server_started_at": SERVER_START_TIME,
    "uptime_seconds": 0,
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

    # Do a final check whether the title contains a hyphen and split it if it looks like "Artist - Title"
    # edit: what a patchwork but it works for now
    if " - " in metadata["track_title"]:
        parts = metadata["track_title"].split(" - ", 1)
        if len(parts) == 2:
            metadata["artist"] = parts[0].strip()
            metadata["track_title"] = parts[1].strip()

    return metadata


def audio_encoder_worker():
    """Mixes audio chunks and places encoded PCM data onto a queue."""
    global audio_sequence, audio_encoder_thread

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
    snapshot = get_audio_config_snapshot()
    if snapshot:
        audio_chunk_queue.put({"type": "config", "data": snapshot})

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

        payload = {"seq": audio_sequence, "chunk": chunk_bytes}
        audio_chunk_queue.put({"type": "chunk", "data": payload})
        audio_sequence += 1

        elapsed = time.time() - loop_start
        sleep_time = seconds_per_chunk - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    with history_lock:
        audio_history.clear()
        audio_sequence = 0

    audio_chunk_queue.put({"type": "stop"})

    with encoder_thread_lock:
        audio_encoder_thread = None


def audio_forwarder():
    """Reads encoded audio messages from the queue and forwards via Socket.IO."""
    global forwarder_started
    while True:
        try:
            message = audio_chunk_queue.get_nowait()
        except QUEUE_EMPTY_EXCEPTIONS:
            socketio.sleep(0.01)
            continue

        if message is None:
            socketio.sleep(0.01)
            continue

        msg_type = message.get("type")
        data = message.get("data")

        if msg_type == "config" and data:
            broadcast_to_mp3_clients(message)
            socketio.emit("audio_config", data)
        elif msg_type == "chunk" and data:
            with history_lock:
                audio_history.append(data)
            broadcast_to_mp3_clients(message)
            socketio.emit("audio_chunk", data)
        elif msg_type == "stop":
            broadcast_to_mp3_clients(message)
            with forwarder_lock:
                forwarder_started = False
            break

        socketio.sleep(0)


def state_updater():
    """Periodically updates the web state from the player state."""
    while True:
        now = time.time()
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

        web_state["server_started_at"] = SERVER_START_TIME
        web_state["uptime_seconds"] = max(0, int(now - SERVER_START_TIME))

        socketio.emit("track_state", dict(web_state))
        socketio.sleep(1)


@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Serves the site favicon from the static directory."""
    return send_from_directory(
        app.static_folder,
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/mp3")
def stream_mp3():
    if lameenc is None:
        abort(
            503, description="MP3 streaming unavailable: lameenc encoder not installed."
        )

    response = Response(mp3_stream_generator(), mimetype="audio/mpeg")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Connection"] = "keep-alive"
    response.direct_passthrough = True
    return response


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


# --- Application Initialization ---


# This function will start our background threads.
# It's safe to use socketio.start_background_task in both environments.
def start_background_tasks():
    global audio_encoder_thread, forwarder_started

    with encoder_thread_lock:
        if audio_encoder_thread is None or not audio_encoder_thread.is_alive():
            audio_encoder_thread = threading.Thread(
                target=audio_encoder_worker,
                daemon=True,
            )
            audio_encoder_thread.start()

    with forwarder_lock:
        if not forwarder_started:
            socketio.start_background_task(target=audio_forwarder)
            forwarder_started = True

    socketio.start_background_task(target=state_updater)


# Always initialize the player, regardless of environment.
if not seamless_player.initialize_player():
    raise SystemExit(
        "Fatal: Could not initialize the seamless player. Is the library built?"
    )

import os

# Check if we're running under Gunicorn by looking for its environment variable.
IS_GUNICORN = "GUNICORN_PID" in os.environ

if IS_GUNICORN:
    # --- PRODUCTION MODE ---
    # Gunicorn is running the app. It will handle the web server part.
    # We just need to make sure our background tasks get started.
    print("Gunicorn environment detected. Starting background tasks...")
    start_background_tasks()
else:
    # --- DEVELOPMENT MODE ---
    # The script is being run directly (e.g., `python app.py`).
    # We need to start the background tasks AND the development server.
    if __name__ == "__main__":
        print(
            "Development environment detected. Starting background tasks and dev server..."
        )
        start_background_tasks()
        socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)
