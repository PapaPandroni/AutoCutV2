"""Video Loading System for AutoCut V2.

This module provides a unified video loading system that consolidates
8 different loading strategies from the original god module into a
clean, maintainable architecture.

The system provides:
- Strategy pattern for different loading approaches
- Resource management and caching
- Error recovery and compatibility handling
- Memory-efficient processing

Key Components:
- VideoLoader: Main interface for video loading
- LoadingStrategies: Sequential, Parallel, and advanced strategies
- ResourceManager: Memory and resource lifecycle management
- VideoCache: Thread-safe caching system

Usage:
    >>> from src.video.loading import VideoLoader
    >>> loader = VideoLoader()
    >>> clips = loader.load_clips(video_paths, clip_specs)
"""

from .strategies import (
    VideoLoadingStrategy,
    SequentialLoader,
    ParallelLoader,
    RobustLoader,
    UnifiedVideoLoader,
    ClipSpec,
    LoadedClip,
    LoadingStrategyType,
)
from .resource_manager import (
    VideoResourceManager,
    MemoryMonitor,
    ResourceAllocation,
)
from .cache import (
    VideoCache,
    CacheEntry,
)

# Main interface
VideoLoader = UnifiedVideoLoader

__all__ = [
    # Main interface
    "VideoLoader",
    "UnifiedVideoLoader",
    # Core data types
    "ClipSpec",
    "LoadedClip",
    "LoadingStrategyType",
    # Loading strategies
    "VideoLoadingStrategy",
    "SequentialLoader",
    "ParallelLoader",
    "RobustLoader",
    # Resource management
    "VideoResourceManager",
    "MemoryMonitor",
    "ResourceAllocation",
    # Caching
    "VideoCache",
    "CacheEntry",
]
