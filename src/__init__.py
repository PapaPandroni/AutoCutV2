"""
AutoCut - Automatic Beat-Synced Video Highlight Generator

A desktop application that automatically creates beat-synced highlight videos
from raw footage and music without requiring video editing knowledge.
"""

__version__ = "2.0.0"
__author__ = "AutoCut Team"

# Core module imports
from .audio_analyzer import analyze_audio
from .video_analyzer import analyze_video_file
from .clip_assembler import assemble_clips, render_video
from .gui import main as gui_main

# Import key refactored functionality for backwards compatibility
# Video validation and processing
from .video.validation import VideoValidator, ValidationResult
from .video.codec_detection import detect_video_codec, CodecDetector
from .video.transcoding import (
    transcode_hevc_to_h264,
    preprocess_video_if_needed,
    test_moviepy_h265_compatibility,
    TranscodingService
)

# Hardware detection
from .hardware.detection import (
    detect_optimal_codec_settings,
    detect_optimal_codec_settings_enhanced,
    HardwareDetector
)

# Core utilities
from .core.exceptions import (
    AutoCutError,
    VideoProcessingError,
    iPhoneCompatibilityError,
    ValidationError,
    TranscodingError
)

# Legacy utils functions that are still needed
from .utils import (
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    validate_video_file,
    validate_audio_file,
    validate_input_files,
    setup_logging,
    format_duration,
    get_file_size_mb,
    safe_filename,
    ensure_output_directory
)

# Expose key classes and functions at package level for easy access
__all__ = [
    # Core functions
    'analyze_audio',
    'analyze_video_file', 
    'assemble_clips',
    'render_video',
    'gui_main',
    
    # Video processing
    'VideoValidator',
    'ValidationResult',
    'CodecDetector',
    'detect_video_codec',
    'TranscodingService',
    'transcode_hevc_to_h264',
    'preprocess_video_if_needed',
    'test_moviepy_h265_compatibility',
    
    # Hardware detection
    'HardwareDetector',
    'detect_optimal_codec_settings',
    'detect_optimal_codec_settings_enhanced',
    
    # Exceptions
    'AutoCutError',
    'VideoProcessingError',
    'iPhoneCompatibilityError', 
    'ValidationError',
    'TranscodingError',
    
    # Legacy utilities
    'SUPPORTED_VIDEO_FORMATS',
    'SUPPORTED_AUDIO_FORMATS',
    'validate_video_file',
    'validate_audio_file',
    'validate_input_files',
    'setup_logging',
    'format_duration',
    'get_file_size_mb',
    'safe_filename',
    'ensure_output_directory'
]