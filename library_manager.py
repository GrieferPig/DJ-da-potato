import os
import json
import logging
from audio_analyzer import analyze_song
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
MUSIC_DIRECTORY = "music"
LIBRARY_FILE = "music_library.json"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff"}

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_music_files(directory):
    """Finds all supported audio files in a directory."""
    music_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                music_files.append(os.path.join(root, file))
    return music_files


def build_library(music_dir, library_path):
    """
    Scans a directory for audio files, analyzes them in parallel,
    and saves the metadata to a JSON file.
    """
    logging.info(f"Scanning for music in '{music_dir}'...")

    if not os.path.isdir(music_dir):
        logging.error(
            f"Music directory not found: '{music_dir}'. Please create it and add audio files."
        )
        return

    files_to_process = find_music_files(music_dir)
    if not files_to_process:
        logging.warning(f"No music files found in '{music_dir}'.")
        return

    logging.info(f"Found {len(files_to_process)} music files. Starting analysis...")

    all_song_data = []

    # Use ProcessPoolExecutor to run analysis in parallel, which is much faster
    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(analyze_song, file_path): file_path
            for file_path in files_to_process
        }

        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                song_data = future.result()
                if song_data:
                    all_song_data.append(song_data)
                    logging.info(
                        f"({i+1}/{len(files_to_process)}) Successfully analyzed: {os.path.basename(file_path)}"
                    )
                else:
                    logging.warning(
                        f"({i+1}/{len(files_to_process)}) Failed to analyze: {os.path.basename(file_path)}"
                    )
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

    logging.info(f"Analysis complete. Saving library to '{library_path}'...")

    try:
        with open(library_path, "w") as f:
            json.dump(all_song_data, f, indent=4)
        logging.info(
            f"Successfully saved music library with {len(all_song_data)} tracks."
        )
    except IOError as e:
        logging.error(f"Failed to write to library file: {e}")


if __name__ == "__main__":
    build_library(MUSIC_DIRECTORY, LIBRARY_FILE)
