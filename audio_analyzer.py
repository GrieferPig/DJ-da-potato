import logging
import time
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mode
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

# Set up logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
FPS = 100  # madmom processes audio at 100 frames per second
PEAK_HEIGHT = 0.5  # Threshold for peak detection
PEAK_DISTANCE = 5  # Minimum distance between peaks in frames


def analyze_song(audio_file_path):
    """
    Analyzes a single audio file to extract key, BPM, and downbeat timestamps.

    This function takes the path to an audio file, processes it using the madmom library
    to find its musical characteristics, and returns them in a structured dictionary.

    Args:
        audio_file_path (str): The full path to the audio file.

    Returns:
        dict: A dictionary containing the song's metadata:
              - 'path': The original file path.
              - 'key': The detected musical key (e.g., 'C Major').
              - 'bpm': The estimated beats per minute.
              - 'beats_per_bar': The estimated time signature (e.g., 4).
              - 'downbeats': A list of timestamps (in seconds) for each downbeat.
    """
    try:
        start_time = time.time()
        logging.info(f"Analyzing: {audio_file_path}")

        # 1. Process beats and downbeats
        beat_proc = RNNDownBeatProcessor()
        beat_downbeat_probs = beat_proc(audio_file_path)

        # 2. Process key detection
        key_proc = CNNKeyRecognitionProcessor()
        key_probs = key_proc(audio_file_path)

        # --- Key Extraction ---
        pred_vector = key_probs[0]
        predicted_key_label = key_prediction_to_label(pred_vector)
        logging.info(f"Detected Key: {predicted_key_label}")

        # --- BPM and Beat Extraction ---
        beat_probs = beat_downbeat_probs[:, 0]
        downbeat_probs = beat_downbeat_probs[:, 1]

        # Find peaks in beat probabilities to estimate BPM
        beat_peak_indices, _ = find_peaks(
            beat_probs, height=PEAK_HEIGHT, distance=PEAK_DISTANCE
        )

        if len(beat_peak_indices) < 2:
            logging.warning(
                f"Not enough beat peaks found for {audio_file_path}. Skipping BPM calculation."
            )
            return None

        beat_peak_distances = np.diff(beat_peak_indices)

        # Filter out outlier distances to get a more stable BPM
        if len(beat_peak_distances) == 0:
            logging.warning(
                f"Could not calculate beat distances for {audio_file_path}. Skipping."
            )
            return None

        the_mode = mode(beat_peak_distances, keepdims=True).mode[0]
        error_margin = 0.20 * the_mode  # Increased margin for more flexibility
        lower_bound = the_mode - error_margin
        upper_bound = the_mode + error_margin

        filtered_distances = beat_peak_distances[
            (beat_peak_distances >= lower_bound) & (beat_peak_distances <= upper_bound)
        ]

        if len(filtered_distances) == 0:
            logging.warning(
                f"No stable beat pattern found for {audio_file_path}. Using unfiltered distances."
            )
            avg_distance = np.mean(beat_peak_distances)
        else:
            avg_distance = np.mean(filtered_distances)

        # Calculate BPM
        bpm = 60 / (avg_distance * (1 / FPS))

        # Adjust BPM to a typical musical range
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        estimated_bpm = round(bpm)
        logging.info(f"Estimated BPM: {estimated_bpm}")

        # --- Downbeat and Time Signature Extraction ---
        downbeat_peak_indices, _ = find_peaks(
            downbeat_probs, height=PEAK_HEIGHT, distance=PEAK_DISTANCE * 2
        )
        downbeat_timestamps = (downbeat_peak_indices / FPS).tolist()

        beats_per_bar = 4  # Default to 4/4
        if len(downbeat_peak_indices) > 1:
            avg_downbeat_distance = np.mean(np.diff(downbeat_peak_indices))
            beats_per_bar = round(avg_downbeat_distance / avg_distance)
            # Ensure a sensible time signature
            if beats_per_bar not in [2, 3, 4, 5, 6, 7]:
                beats_per_bar = 4

        logging.info(f"Estimated Beats Per Bar: {beats_per_bar}")

        end_time = time.time()
        logging.info(f"Finished analysis in {end_time - start_time:.2f} seconds.")

        return {
            "path": audio_file_path,
            "key": predicted_key_label,
            "bpm": estimated_bpm,
            "beats_per_bar": beats_per_bar,
            "downbeats": downbeat_timestamps,
        }

    except Exception as e:
        logging.error(f"Could not process {audio_file_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage: python audio_analyzer.py /path/to/your/song.mp3
    if len(sys.argv) < 2:
        print("Usage: python audio_analyzer.py <audio_file_path>")
        sys.exit(1)

    print("Analyzing {sys.argv[1]}...")
    file_path = sys.argv[1]
    song_data = analyze_song(file_path)
    if song_data:
        import json

        print(json.dumps(song_data, indent=2))
