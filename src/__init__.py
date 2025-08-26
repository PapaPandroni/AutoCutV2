"""
AutoCut - Automatic Beat-Synced Video Highlight Generator

A desktop application that automatically creates beat-synced highlight videos
from raw footage and music without requiring video editing knowledge.
"""

__version__ = "2.0.0"
__author__ = "AutoCut Team"

# Core module imports
from .audio_analyzer import analyze_audio
from .clip_assembler import assemble_clips, render_video

# Core utilities
from .core.exceptions import (
    AutoCutError,
    TranscodingError,
    ValidationError,
    VideoProcessingError,
    iPhoneCompatibilityError,
)
from .gui import main as gui_main

# Hardware detection
from .hardware.detection import (
    HardwareDetector,
    detect_optimal_codec_settings,
    detect_optimal_codec_settings_enhanced,
)

# Legacy utils functions that are still needed
from .utils import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    ensure_output_directory,
    format_duration,
    get_file_size_mb,
    safe_filename,
    setup_logging,
    validate_audio_file,
    validate_input_files,
    validate_video_file,
)
from .video.codec_detection import CodecDetector, detect_video_codec
from .video.transcoding import (
    TranscodingService,
    preprocess_video_if_needed,
    test_moviepy_h265_compatibility,
    transcode_hevc_to_h264,
)

# Import key refactored functionality for backwards compatibility
# Video validation and processing
from .video.validation import ValidationResult, VideoValidator
from .video_analyzer import analyze_video_file

# Expose key classes and functions at package level for easy access
__all__ = [
    "SUPPORTED_AUDIO_FORMATS",
    # Legacy utilities
    "SUPPORTED_VIDEO_FORMATS",
    # Exceptions
    "AutoCutError",
    "CodecDetector",
    # Hardware detection
    "HardwareDetector",
    "TranscodingError",
    "TranscodingService",
    "ValidationError",
    "ValidationResult",
    "VideoProcessingError",
    # Video processing
    "VideoValidator",
    # Core functions
    "analyze_audio",
    "analyze_video_file",
    "assemble_clips",
    "detect_optimal_codec_settings",
    "detect_optimal_codec_settings_enhanced",
    "detect_video_codec",
    "ensure_output_directory",
    "format_duration",
    "get_file_size_mb",
    "gui_main",
    "iPhoneCompatibilityError",
    "preprocess_video_if_needed",
    "render_video",
    "safe_filename",
    "setup_logging",
    "test_moviepy_h265_compatibility",
    "transcode_hevc_to_h264",
    "validate_audio_file",
    "validate_input_files",
    "validate_video_file",
]
