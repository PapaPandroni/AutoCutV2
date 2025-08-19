"""
FFmpeg-based video processing module for AutoCut V2.

This module provides memory-efficient video processing using FFmpeg
instead of MoviePy for improved performance and reduced memory usage.

Key Features:
- Streaming video analysis without loading entire clips
- Apple Silicon hardware acceleration support
- Memory-efficient frame processing
- Analysis result caching system
"""

from .streaming_analyzer import StreamingVideoAnalyzer
from .hardware_accelerator import M2HardwareAccelerator  
from .memory_monitor import MemoryMonitor
from .cache_manager import AnalysisCacheManager

__all__ = [
    'StreamingVideoAnalyzer',
    'M2HardwareAccelerator', 
    'MemoryMonitor',
    'AnalysisCacheManager'
]