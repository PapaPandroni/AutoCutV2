#!/usr/bin/env python3
"""
AutoCut Demo Test Script

Quick way to test AutoCut with all your media files.
Creates real beat-synced videos you can watch!

Usage:
  python3 test_autocut_demo.py                    # Use all videos + random music
  python3 test_autocut_demo.py --audio song.mp3   # Specify music
  python3 test_autocut_demo.py --videos 5         # Limit to 5 videos
  python3 test_autocut_demo.py --pattern dramatic # Use specific pattern
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.clip_assembler import assemble_clips
from src.utils import SUPPORTED_VIDEO_FORMATS


def find_all_video_files(directory: str) -> list:
    """Find all supported video files in directory using enhanced format support."""
    video_files = []
    search_patterns = []

    # Create search patterns for all supported formats (case-insensitive)
    for ext in SUPPORTED_VIDEO_FORMATS:
        # Add both lowercase and uppercase variants
        search_patterns.append(f"{directory}/*{ext}")
        search_patterns.append(f"{directory}/*{ext.upper()}")

    # Search for all patterns
    for pattern in search_patterns:
        found_files = glob.glob(pattern)
        video_files.extend(found_files)

    # Remove duplicates and sort
    return sorted(set(video_files))


def main():
    parser = argparse.ArgumentParser(
        description="AutoCut Demo - Create beat-synced videos",
    )
    parser.add_argument("--audio", help="Specific audio file to use")
    parser.add_argument(
        "--videos",
        type=int,
        help="Number of videos to use (default: all)",
    )
    parser.add_argument(
        "--pattern",
        choices=["energetic", "balanced", "dramatic", "buildup"],
        default="balanced",
        help="Cutting pattern to use",
    )
    parser.add_argument("--output", help="Output filename (default: auto-generated)")

    args = parser.parse_args()

    # Find all video files using enhanced format support

    video_files = find_all_video_files("test_media")
    if not video_files:
        return False

    # Limit videos if requested
    if args.videos:
        video_files = video_files[: args.videos]

    # Group by file extension for better display
    format_counts = {}
    for vf in video_files:
        ext = Path(vf).suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1

    for _i, vf in enumerate(video_files, 1):
        name = Path(vf).name
        ext = Path(vf).suffix.upper()

    # Find audio file
    if args.audio:
        if not os.path.exists(args.audio):
            return False
        audio_file = args.audio
    else:
        # Find any supported audio file
        audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(f"test_media/{ext}"))

        if not audio_files:
            return False
        audio_file = audio_files[0]  # Use first one

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = int(time.time())
        output_file = f"output/autocut_demo_{args.pattern}_{timestamp}.mp4"

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Progress callback
    def progress_callback(step, progress):
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

    try:
        start_time = time.time()

        result_path = assemble_clips(
            video_files=video_files,
            audio_file=audio_file,
            output_path=output_file,
            pattern=args.pattern,
            progress_callback=progress_callback,
        )

        elapsed = time.time() - start_time

        # Show file info
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB

            # Check for timeline JSON
            timeline_json = result_path.replace(".mp4", "_timeline.json")
            if os.path.exists(timeline_json):
                pass

        return True

    except Exception as e:
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
