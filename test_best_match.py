"""Utility script to pick a random track and report its best harmonic match."""

import argparse
import random
import sys
from typing import Any, Dict, Optional

from seamless_player import (
    ANALYTICS_DIRECTORY,
    calculate_key_compatibility,
    find_best_next_track,
    load_library_from_analytics,
    normalize_key_label,
)

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


def compute_match_score(current: Track, candidate: Track) -> Dict[str, Optional[float]]:
    current_bpm = coerce_float(current.get("bpm"))
    candidate_bpm = coerce_float(candidate.get("bpm"))
    current_key = normalize_key_label(current.get("key"))
    candidate_key = normalize_key_label(candidate.get("key"))
    key_score = calculate_key_compatibility(current_key, candidate_key)

    if current_bpm is not None and candidate_bpm is not None:
        bpm_diff = abs(current_bpm - candidate_bpm)
        bpm_score = max(0.0, 100.0 - (bpm_diff * 5.0))
    else:
        bpm_diff = None
        bpm_score = 0.0

    match_score = (key_score * 0.65) + (bpm_score * 0.35)

    return {
        "current_bpm": current_bpm,
        "candidate_bpm": candidate_bpm,
        "current_key": current_key,
        "candidate_key": candidate_key,
        "key_score": key_score,
        "bpm_score": bpm_score,
        "bpm_diff": bpm_diff,
        "match_score": match_score,
    }


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

    current_track = random.choice(library)
    remaining_tracks = [
        track for track in library if track.get("path") != current_track.get("path")
    ]

    if not remaining_tracks:
        print("Only one unique track available; cannot compute a match.")
        sys.exit(1)

    best_match = find_best_next_track(current_track, remaining_tracks)

    if not best_match:
        print("No suitable match found using the current algorithm.")
        sys.exit(1)

    metrics = compute_match_score(current_track, best_match)

    print("Selected Track:")
    print(f"  Title: {format_track_name(current_track)}")
    print(
        f"  BPM: {metrics['current_bpm'] if metrics['current_bpm'] is not None else 'Unknown'}"
    )
    print(
        "  Key: "
        f"{format_key_value(metrics['current_key'] if metrics['current_key'] else current_track.get('key'))}"
    )
    print()
    print("Best Match:")
    print(f"  Title: {format_track_name(best_match)}")
    print(
        f"  BPM: {metrics['candidate_bpm'] if metrics['candidate_bpm'] is not None else 'Unknown'}"
    )
    print(
        "  Key: "
        f"{format_key_value(metrics['candidate_key'] if metrics['candidate_key'] else best_match.get('key'))}"
    )
    print()
    print("Compatibility:")
    bpm_diff = metrics["bpm_diff"]
    bpm_diff_text = f"{bpm_diff:.2f}" if bpm_diff is not None else "Unknown"
    print(f"  Key Score: {metrics['key_score']:.2f}")
    print(f"  BPM Score: {metrics['bpm_score']:.2f} (Δ BPM: {bpm_diff_text})")
    print(f"  Match Score: {metrics['match_score']:.2f}")


if __name__ == "__main__":
    main()
