"""
Video processing module for AutoCut V2.

This module contains video-related functionality including:
- Video validation and compatibility checking
- Video codec detection and analysis
- Video transcoding and preprocessing
- Video format normalization
- Timeline rendering and clip assembly
- Video encoding and optimization

Modules:
    validation: Unified video validation system
    codec_detection: Video codec detection and analysis
    transcoding: H.265/HEVC to H.264 transcoding pipeline
    preprocessing: Video preprocessing and format handling
    format_analyzer: Video format analysis and canvas optimization
    normalization: Video format normalization pipeline
    timeline_renderer: Timeline management and rendering coordination
    encoder: Video encoding with hardware acceleration
"""

# Import enhanced VideoChunk for unified usage
from .assembly.clip_selector import VideoChunk

# Import extracted video processing classes
try:
    from .encoder import (
        VideoEncoder,
        detect_optimal_codec_settings,
        detect_optimal_codec_settings_with_diagnostics,
    )
    from .format_analyzer import VideoFormatAnalyzer
    from .normalization import VideoNormalizationPipeline
    from .timeline_renderer import ClipTimeline, TimelineRenderer
except ImportError as e:
    pass

# Version information
__version__ = "2.0.0"
__author__ = "AutoCut Development Team"

__all__ = [
    "ClipTimeline",
    "TimelineRenderer",
    "VideoChunk",
    "VideoEncoder",
    "VideoFormatAnalyzer",
    "VideoNormalizationPipeline",
    "detect_optimal_codec_settings",
    "detect_optimal_codec_settings_with_diagnostics",
]
