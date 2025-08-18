"""
Video processing module for AutoCut V2.

This module contains video-related functionality including:
- Video validation and compatibility checking
- Video codec detection and analysis
- Video transcoding and preprocessing
- Video format normalization

Modules:
    validation: Unified video validation system
    codec_detection: Video codec detection and analysis
    transcoding: H.265/HEVC to H.264 transcoding pipeline
    preprocessing: Video preprocessing and format handling
"""

# Import enhanced VideoChunk for unified usage
from .assembly.clip_selector import VideoChunk

# Version information
__version__ = "2.0.0"
__author__ = "AutoCut Development Team"

__all__ = ['VideoChunk']
