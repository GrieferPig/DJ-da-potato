"""Utility script to pick a random track and report its best harmonic match."""

import argparse
import random
import sys
import os
import json
import logging
from typing import Any, Dict, Optional

# Replicated from seamless_player.py
ANALYTICS_DIRECTORY = "analytics"

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


# Add global counter for tracking find_best_next_track calls
_find_best_next_track_call_count = 0


def find_best_next_track(current_track, available_tracks):
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
        bpm_diff = max(0, bpm_diff - 5)  # Cap at 20 BPM difference
        bpm_score = max(0, 100 - (bpm_diff * 5))  # Score based on BPM proximity
        # Weighted score
        if _find_best_next_track_call_count % 7 == 0:
            total_score = (key_score * 0.95) - (bpm_score * 0.05)
        else:
            total_score = (key_score * 0.95) + (bpm_score * 0.05)
        scored_candidates.append((total_score, candidate))

    if not scored_candidates:
        # Fallback (though unlikely if available_tracks is not empty)
        return random.choice(available_tracks)

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Get top 10 (or all if fewer)
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


Track = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selects a random track from the analyzed library, finds the best match "
            "using seamless_player's harmonic mixing logic, and prints their BPM, "
            "keys, and compatibility score."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible selection.",
    )
    parser.add_argument(
        "--analytics-dir",
        default=ANALYTICS_DIRECTORY,
        help="Path to the analytics directory (defaults to seamless_player.ANALYTICS_DIRECTORY).",
    )
    return parser.parse_args()


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_track_name(track: Track) -> str:
    title = track.get("title") or track.get("path") or "<unknown>"
    artist = track.get("artist")
    if artist:
        return f"{title} — {artist}"
    return str(title)


def format_key_value(key_value: Any) -> str:
    canonical = normalize_key_label(key_value)
    if canonical:
        return canonical
    if isinstance(key_value, str) and key_value.strip():
        return key_value.strip()
    return "Unknown"


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    library = load_library_from_analytics(directory=args.analytics_dir)

    if len(library) < 2:
        print(
            "Need at least two analyzed tracks to evaluate matches. "
            f"Found {len(library)} in '{args.analytics_dir}'."
        )
        sys.exit(1)

    # Perform 10 consecutive selections
    available_tracks = library.copy()
    random.shuffle(available_tracks)
    current_track = available_tracks.pop()  # Start with a random track

    print("Starting with initial random track:")
    print(f"  Title: {format_track_name(current_track)}")
    print(f"  BPM: {coerce_float(current_track.get('bpm')) or 'Unknown'}")
    print(f"  Key: {format_key_value(current_track.get('key'))}")
    print()

    for i in range(10):
        if not available_tracks:
            print(f"Only {i+1} selections possible; no more tracks available.")
            break

        next_track = find_best_next_track(current_track, available_tracks)
        if not next_track:
            print(f"No suitable match found at selection {i+1}.")
            break

        print(f"Selection {i+1}:")
        print(f"  Title: {format_track_name(next_track)}")
        print(f"  BPM: {next_track.get('bpm') or 'Unknown'}")
        print(f"  Key: {format_key_value(next_track.get('key'))}")
        print()

        # Update for next iteration
        current_track = next_track
        available_tracks = [
            t for t in available_tracks if t["path"] != next_track["path"]
        ]


if __name__ == "__main__":
    main()
