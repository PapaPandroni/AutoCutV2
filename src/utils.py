"""
Utility Functions for AutoCut

Common helper functions used across multiple modules.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Public API - explicitly exported symbols
__all__ = [
    # Constants
    "DEFAULT_CONFIG",
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_VIDEO_FORMATS",
    # Codec settings (re-exported from clip_assembler)
    "detect_optimal_codec_settings",
    # Utility / validation functions
    "ensure_output_directory",
    "filter_valid_video_files",
    "find_all_video_files",
    "format_duration",
    "get_config_value",
    "get_file_size_mb",
    "safe_filename",
    "setup_logging",
    "validate_audio_file",
    "validate_input_files",
    "validate_video_file",
]

# Import codec settings function from clip_assembler
try:
    from .clip_assembler import detect_optimal_codec_settings
except ImportError:
    # Fallback for direct execution
    try:
        from clip_assembler import detect_optimal_codec_settings
    except ImportError:
        # Define a fallback function if clip_assembler is not available
        def detect_optimal_codec_settings():
            import os

            return (
                {
                    "codec": "libx264",
                    "audio_codec": "aac",
                    "threads": os.cpu_count() or 4,
                },
                ["-preset", "ultrafast", "-crf", "23"],
            )


# Supported file formats - comprehensive list for modern video processing
SUPPORTED_VIDEO_FORMATS = {
    # Standard formats (original)
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".m4v",
    # Modern web formats
    ".webm",  # WebM - increasingly common for web content
    ".ogv",  # Ogg Video - open source video format
    # Mobile and device formats
    ".3gp",
    ".3g2",  # 3GPP - mobile phone recordings
    ".mp4v",  # MPEG-4 Video
    # Professional/broadcast formats
    ".mts",
    ".m2ts",  # MPEG Transport Stream - camcorder formats
    ".ts",  # Transport Stream
    ".vob",  # DVD Video Object
    ".divx",  # DivX format
    ".xvid",  # Xvid format
    # Additional container formats
    ".asf",  # Advanced Systems Format
    ".rm",
    ".rmvb",  # RealMedia formats
    ".f4v",  # Flash Video
    ".swf",  # Shockwave Flash (video)
}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration for AutoCut.

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("autocut.log")],
    )
    return logging.getLogger("autocut")


def validate_video_file(file_path: str) -> bool:
    """Validate that a file is a supported video format.

    Args:
        file_path: Path to the video file

    Returns:
        True if file exists and is a supported video format
    """
    if not Path(file_path).exists():
        return False

    file_extension = Path(file_path).suffix.lower()
    return file_extension in SUPPORTED_VIDEO_FORMATS


def validate_audio_file(file_path: str) -> bool:
    """Validate that a file is a supported audio format.

    Args:
        file_path: Path to the audio file

    Returns:
        True if file exists and is a supported audio format
    """
    if not Path(file_path).exists():
        return False

    file_extension = Path(file_path).suffix.lower()
    return file_extension in SUPPORTED_AUDIO_FORMATS


def validate_input_files(video_files: List[str], audio_file: str) -> List[str]:
    """Validate all input files and return list of errors.

    Args:
        video_files: List of video file paths
        audio_file: Path to audio file

    Returns:
        List of error messages (empty if all files are valid)
    """
    errors = []

    if not video_files:
        errors.append("No video files provided")
    else:
        for i, video_file in enumerate(video_files):
            if not validate_video_file(video_file):
                errors.append(
                    f"Video file {i + 1} is invalid or unsupported: {video_file}",
                )

    if not validate_audio_file(audio_file):
        errors.append(f"Audio file is invalid or unsupported: {audio_file}")

    return errors


def ensure_output_directory(output_path: str) -> str:
    """Ensure output directory exists and return absolute path.

    Args:
        output_path: Desired output file path

    Returns:
        Absolute path to output file

    Raises:
        OSError: If directory cannot be created
    """
    output_path = Path(output_path).resolve()
    output_dir = output_path.parent

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_path


def format_duration(seconds: float) -> str:
    """Format duration in seconds as MM:SS string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Safe filename suitable for filesystem
    """
    # Characters that are problematic in filenames
    invalid_chars = '<>:"/\\|?*'

    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, "_")

    # Remove any leading/trailing whitespace and dots
    safe_name = safe_name.strip(". ")

    # Ensure filename is not empty
    if not safe_name:
        safe_name = "untitled"

    return safe_name

    # Note: File size logging removed - not implemented in current version


# Legacy function for backward compatibility


# Smart transcoding cache to avoid re-processing identical files
_TRANSCODING_CACHE: Dict[str, Dict[str, Any]] = {}
_TRANSCODING_CACHE_TIMEOUT: int = 3600  # 1 hour


# Legacy function for backward compatibility


# Enhanced hardware detection cache for performance
_HARDWARE_DETECTION_CACHE: Optional[
    Tuple[Dict[str, Any], List[str], Dict[str, Any]]
] = None
_CACHE_TIMESTAMP: Optional[float] = None
_CACHE_TIMEOUT: int = 300  # 5 minutes


# Legacy validation function for backward compatibility


def filter_valid_video_files(file_list: List[str]) -> List[str]:
    """
    Filter out invalid video files that cause loading failures.

    CRITICAL FIX: Removes macOS resource fork files and other system files
    that can cause RuntimeError: "No video clips could be loaded successfully".

    Filters out:
    - macOS resource fork files (._filename)
    - macOS system files (.DS_Store)
    - Windows thumbnail files (Thumbs.db)
    - Empty or non-existent files
    - Files with invalid extensions despite glob matching

    Args:
        file_list: List of potential video file paths

    Returns:
        List of valid video file paths
    """
    valid_files = []
    filtered_count = 0

    for file_path in file_list:
        try:
            # Get filename from path
            filename = Path(file_path).name

            # Skip macOS resource fork files
            if filename.startswith("._"):
                filtered_count += 1
                continue

            # Skip common system files
            if filename in [".DS_Store", "Thumbs.db", "desktop.ini"]:
                filtered_count += 1
                continue

            # Skip hidden files (additional safety)
            if filename.startswith("."):
                filtered_count += 1
                continue

            # Check file exists and has size > 0
            if not Path(file_path).exists():
                filtered_count += 1
                continue

            if Path(file_path).stat().st_size == 0:
                filtered_count += 1
                continue

            # File passed all checks
            valid_files.append(file_path)

        except OSError:
            # Skip files we can't access
            filtered_count += 1
            continue

    # Log filtering results if any files were filtered
    if filtered_count > 0:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Filtered out {filtered_count} invalid/system files from video collection"
        )

    return valid_files


def find_all_video_files(directory: str) -> List[str]:
    """
    Find all supported video files in directory using enhanced format support.

    CRITICAL FIX: Now filters out macOS resource fork files and other system files
    that can cause "video loading failures" when the system tries to process them.

    Args:
        directory: Directory to search for video files

    Returns:
        List of valid video file paths, sorted and deduplicated
    """
    video_files = []
    search_patterns = []

    # Create search patterns for all supported formats (case-insensitive)
    for ext in SUPPORTED_VIDEO_FORMATS:
        # Add both lowercase and uppercase variants
        search_patterns.append(f"{directory}/*{ext}")
        search_patterns.append(f"{directory}/*{ext.upper()}")

    # Search for all patterns
    for pattern in search_patterns:
        found_files = [str(p) for p in Path().glob(pattern)]
        video_files.extend(found_files)

    # CRITICAL FIX: Filter out invalid files that cause loading failures
    filtered_files = filter_valid_video_files(video_files)

    # Remove duplicates and sort
    return sorted(set(filtered_files))


# Configuration defaults
DEFAULT_CONFIG = {
    "min_clip_duration": 0.5,  # Minimum clip duration in seconds (technical limit)
    "max_clip_duration": 8.0,  # Maximum clip duration in seconds (UX limit)
    "min_scene_beats": 1.0,  # Minimum scene duration in beats (musical logic)
    "scene_threshold": 30.0,  # Scene detection sensitivity
    "transition_duration": 0.5,  # Crossfade duration in seconds
    "output_quality": "high",  # Output quality ('low', 'medium', 'high')
    "temp_dir": "temp",  # Temporary files directory
}


def get_config_value(key: str, default=None):
    """Get configuration value with fallback to default."""
    return DEFAULT_CONFIG.get(key, default)
