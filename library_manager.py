import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from audio_analyzer import analyze_song
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
MUSIC_DIRECTORY = "music"
ANALYTICS_DIRECTORY = "analytics"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff"}

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try to import tqdm (interpreting 'tqlm' request as 'tqdm'); provide fallback if unavailable.
try:
    from tqdm import tqdm
except ImportError:  # fallback no-op

    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None):
            self.iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def update(self, n=1):
            pass


def find_music_files(directory):
    """Finds all supported audio files in a directory."""
    music_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                music_files.append(os.path.join(root, file))
    return music_files


def _ensure_analytics_dir(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        logging.error(f"Failed to create analytics directory '{directory}': {exc}")
        raise


def _normalize_track_identifier(file_path, music_dir):
    base_dir = os.path.abspath(music_dir)
    absolute_path = os.path.abspath(file_path)
    relative = os.path.relpath(absolute_path, base_dir)
    return relative.replace(os.sep, "/").lower()


def _analytics_file_path(file_path, music_dir, analytics_dir):
    normalized = _normalize_track_identifier(file_path, music_dir)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return os.path.join(analytics_dir, f"{digest}.json")


def build_library(music_dir, analytics_dir=ANALYTICS_DIRECTORY, show_progress=True):
    """
    Scans a directory for audio files, analyzes them in parallel,
    and saves the metadata to a JSON file.
    Optionally displays a tqdm progress bar (show_progress=True).
    """
    logging.info(f"Scanning for music in '{music_dir}'...")

    if not os.path.isdir(music_dir):
        logging.error(
            f"Music directory not found: '{music_dir}'. Please create it and add audio files."
        )
        return

    _ensure_analytics_dir(analytics_dir)

    all_music_files = find_music_files(music_dir)
    if not all_music_files:
        logging.warning(f"No music files found in '{music_dir}'.")
        return

    files_to_process = []
    skipped_files = 0
    for file_path in all_music_files:
        analytics_path = _analytics_file_path(file_path, music_dir, analytics_dir)
        if os.path.exists(analytics_path):
            skipped_files += 1
            logging.info(
                f"Skipping already analyzed track: {os.path.basename(file_path)}"
            )
            continue
        files_to_process.append((file_path, analytics_path))

    if not files_to_process:
        logging.info(
            "All tracks already analyzed. No new analytics files need to be generated."
        )
        return

    logging.info(
        f"Found {len(all_music_files)} music files, {len(files_to_process)} pending analysis (skipped {skipped_files})."
    )

    processed_count = 0
    failed_files = []

    def _handle_result(file_path, analytics_path, song_data):
        nonlocal processed_count
        if not song_data:
            failed_files.append(file_path)
            logging.warning(f"Failed to analyze: {os.path.basename(file_path)}")
            return

        normalized = _normalize_track_identifier(file_path, music_dir)
        song_data.setdefault("path", file_path)
        song_data.setdefault("relative_path", normalized)
        song_data["track_id"] = os.path.splitext(os.path.basename(analytics_path))[0]
        song_data["analyzed_at"] = datetime.now(timezone.utc).isoformat()

        try:
            with open(analytics_path, "w", encoding="utf-8") as analytics_file:
                json.dump(song_data, analytics_file, indent=2)
            processed_count += 1
            logging.info(
                f"Successfully analyzed: {os.path.basename(file_path)} -> {os.path.basename(analytics_path)}"
            )
        except IOError as exc:
            failed_files.append(file_path)
            logging.error(
                f"Failed to write analytics for {os.path.basename(file_path)}: {exc}"
            )

    # Use ProcessPoolExecutor to run analysis in parallel, which is much faster
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_file = {
            executor.submit(analyze_song, file_path): (file_path, analytics_path)
            for (file_path, analytics_path) in files_to_process
        }
        # Replace enumerate(as_completed(...)) loop with progress-enabled version
        if show_progress:
            with tqdm(
                total=len(files_to_process), desc="Analyzing", unit="file"
            ) as pbar:
                for future in as_completed(future_to_file):
                    file_path, analytics_path = future_to_file[future]
                    try:
                        song_data = future.result()
                        _handle_result(file_path, analytics_path, song_data)
                    except Exception as e:
                        failed_files.append(file_path)
                        logging.error(f"Error processing {file_path}: {e}")
                    finally:
                        pbar.update(1)
        else:
            for i, future in enumerate(as_completed(future_to_file)):
                file_path, analytics_path = future_to_file[future]
                try:
                    song_data = future.result()
                    _handle_result(file_path, analytics_path, song_data)
                except Exception as e:
                    failed_files.append(file_path)
                    logging.error(f"Error processing {file_path}: {e}")

    logging.info(
        f"Analysis complete. Generated {processed_count} analytics files (skipped {skipped_files}, failures {len(set(failed_files))})."
    )
    if failed_files:
        logging.warning(
            "Failed tracks: "
            + ", ".join(os.path.basename(path) for path in sorted(set(failed_files)))
        )


if __name__ == "__main__":
    build_library(MUSIC_DIRECTORY, ANALYTICS_DIRECTORY, show_progress=True)
