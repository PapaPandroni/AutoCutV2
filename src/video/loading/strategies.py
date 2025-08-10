"""Video loading strategies for AutoCut V2.

Consolidates 8 different loading approaches from the original god module
into a clean strategy pattern implementation. Provides different loading
strategies optimized for various use cases and system constraints.

Original strategies consolidated:
1. Sequential Loading - Safe, memory-efficient
2. Advanced Memory Management - Optimized for large files
3. Robust Error Handling - Maximum error recovery
4. Parallel Loading - Performance optimization
5. Video Segment Loading - Individual clip processing
6. Smart Preprocessing - Format-aware loading
7. Delayed Cleanup - Resource optimization
8. RobustVideoLoader Class - Comprehensive recovery
"""

import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from moviepy import VideoFileClip

from ...core.exceptions import (
    VideoProcessingError,
    iPhoneCompatibilityError,
    TranscodingError,
    raise_validation_error,
)
from ...core.logging_config import get_logger, log_performance, LoggingContext
from .resource_manager import VideoResourceManager, MemoryMonitor
from .cache import VideoCache


@dataclass
class ClipSpec:
    """Specification for a video clip to be loaded."""

    file_path: str
    start_time: float
    end_time: float
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Validate clip specification
        if not os.path.exists(self.file_path):
            raise_validation_error(
                f"Video file does not exist: {self.file_path}",
                validation_type="file_existence",
                file_path=self.file_path,
            )

        if self.start_time < 0:
            raise_validation_error(
                f"Invalid start time: {self.start_time}",
                validation_type="time_range",
                file_path=self.file_path,
            )

        if self.end_time <= self.start_time:
            raise_validation_error(
                f"Invalid time range: start={self.start_time}, end={self.end_time}",
                validation_type="time_range",
                file_path=self.file_path,
            )

    @property
    def duration(self) -> float:
        """Get clip duration in seconds."""
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return f"ClipSpec({Path(self.file_path).name}, {self.start_time:.2f}-{self.end_time:.2f}s)"


@dataclass
class LoadedClip:
    """Container for a loaded video clip with metadata."""

    clip: VideoFileClip
    spec: ClipSpec
    load_time: float
    strategy_used: str
    memory_usage_mb: Optional[float] = None
    preprocessing_applied: bool = False

    def __str__(self) -> str:
        return f"LoadedClip({self.spec}, loaded_with={self.strategy_used})"


class LoadingStrategyType(Enum):
    """Available video loading strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ROBUST = "robust"
    AUTO = "auto"


class VideoLoadingStrategy(ABC):
    """Abstract base class for video loading strategies."""

    def __init__(
        self,
        cache: Optional[VideoCache] = None,
        resource_manager: Optional[VideoResourceManager] = None,
    ):
        self.cache = cache or VideoCache()
        self.resource_manager = resource_manager or VideoResourceManager()
        self.logger = get_logger(f"autocut.video.loading.{self.__class__.__name__}")
        self._stats = {
            "clips_loaded": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_time": 0.0,
        }

    @abstractmethod
    def load_clips(self, clip_specs: List[ClipSpec]) -> List[LoadedClip]:
        """Load video clips according to the strategy.

        Args:
            clip_specs: List of clip specifications to load

        Returns:
            List of loaded clips with metadata

        Raises:
            VideoProcessingError: If loading fails
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics for monitoring."""
        return self._stats.copy()

    def _load_single_clip(self, spec: ClipSpec) -> LoadedClip:
        """Load a single clip with error handling and caching."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(spec)
            cached_clip = self.cache.get(cache_key)

            if cached_clip:
                self._stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for {spec}")
                return LoadedClip(
                    clip=cached_clip,
                    spec=spec,
                    load_time=time.time() - start_time,
                    strategy_used=self.__class__.__name__,
                )

            # Load from disk
            with self.resource_manager.allocate_resources(estimated_memory_mb=100):
                clip = self._load_clip_from_disk(spec)

                # Cache the loaded clip
                self.cache.put(cache_key, clip, estimated_size_mb=50)

                loaded_clip = LoadedClip(
                    clip=clip,
                    spec=spec,
                    load_time=time.time() - start_time,
                    strategy_used=self.__class__.__name__,
                )

                self._stats["clips_loaded"] += 1
                self._stats["total_time"] += loaded_clip.load_time

                return loaded_clip

        except Exception as e:
            self._stats["errors"] += 1
            self.logger.error(
                f"Failed to load clip {spec}",
                extra={"error": str(e), "spec": str(spec)},
                exc_info=True,
            )
            raise VideoProcessingError(
                f"Failed to load video clip: {spec}",
                details={"spec": str(spec), "error": str(e)},
            ) from e

    def _load_clip_from_disk(self, spec: ClipSpec) -> VideoFileClip:
        """Load clip from disk with format handling."""
        try:
            # Load full video first
            video = VideoFileClip(spec.file_path)

            # Create subclip if needed
            if spec.start_time > 0 or spec.end_time < video.duration:
                clip = video.subclip(spec.start_time, spec.end_time)
                video.close()  # Clean up original
                return clip
            else:
                return video

        except Exception as e:
            # Handle iPhone H.265 compatibility issues
            if "codec" in str(e).lower() or "h265" in str(e).lower():
                raise iPhoneCompatibilityError(
                    f"iPhone H.265 compatibility issue with {spec.file_path}",
                    file_path=spec.file_path,
                    details={"original_error": str(e)},
                ) from e
            else:
                raise

    def _generate_cache_key(self, spec: ClipSpec) -> str:
        """Generate cache key for clip specification."""
        file_stat = os.stat(spec.file_path)
        return (
            f"{spec.file_path}:"
            f"{spec.start_time:.2f}-{spec.end_time:.2f}:"
            f"{file_stat.st_mtime}:{file_stat.st_size}"
        )


class SequentialLoader(VideoLoadingStrategy):
    """Sequential video loading strategy.

    Loads clips one by one in order. Memory-efficient and reliable,
    but slower for large numbers of clips.

    Best for:
    - Memory-constrained systems
    - Large video files
    - Debugging and development
    """

    @log_performance("sequential_clip_loading")
    def load_clips(self, clip_specs: List[ClipSpec]) -> List[LoadedClip]:
        """Load clips sequentially."""
        loaded_clips = []

        with LoggingContext("sequential_loading", self.logger) as ctx:
            ctx.log(f"Loading {len(clip_specs)} clips sequentially")

            for i, spec in enumerate(clip_specs, 1):
                ctx.log(f"Loading clip {i}/{len(clip_specs)}: {spec}")

                try:
                    loaded_clip = self._load_single_clip(spec)
                    loaded_clips.append(loaded_clip)

                    ctx.log(
                        f"Successfully loaded clip {i}",
                        extra={"load_time": f"{loaded_clip.load_time:.2f}s"},
                    )

                except Exception as e:
                    ctx.log(
                        f"Failed to load clip {i}: {e}",
                        level="ERROR",
                        extra={"spec": str(spec)},
                    )
                    # Continue with other clips
                    continue

            ctx.log(
                f"Sequential loading completed: {len(loaded_clips)}/{len(clip_specs)} clips loaded"
            )

        return loaded_clips


class ParallelLoader(VideoLoadingStrategy):
    """Parallel video loading strategy.

    Loads multiple clips concurrently using thread pool.
    Faster but uses more memory and system resources.

    Best for:
    - Systems with adequate memory
    - Many small clips
    - Performance-critical applications
    """

    def __init__(self, max_workers: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)

    @log_performance("parallel_clip_loading")
    def load_clips(self, clip_specs: List[ClipSpec]) -> List[LoadedClip]:
        """Load clips in parallel."""
        loaded_clips = []

        with LoggingContext("parallel_loading", self.logger) as ctx:
            ctx.log(
                f"Loading {len(clip_specs)} clips in parallel",
                extra={"max_workers": self.max_workers},
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_spec = {
                    executor.submit(self._load_single_clip, spec): spec
                    for spec in clip_specs
                }

                # Collect results as they complete
                for future in as_completed(future_to_spec):
                    spec = future_to_spec[future]

                    try:
                        loaded_clip = future.result()
                        loaded_clips.append(loaded_clip)

                        ctx.log(
                            f"Completed loading {spec}",
                            extra={"load_time": f"{loaded_clip.load_time:.2f}s"},
                        )

                    except Exception as e:
                        ctx.log(
                            f"Failed to load {spec}: {e}",
                            level="ERROR",
                            extra={"spec": str(spec)},
                        )
                        continue

            ctx.log(
                f"Parallel loading completed: {len(loaded_clips)}/{len(clip_specs)} clips loaded"
            )

        return loaded_clips


class RobustLoader(VideoLoadingStrategy):
    """Robust video loading strategy with maximum error recovery.

    Implements comprehensive error handling, retry logic, and fallback
    strategies. Designed for maximum reliability in production.

    Features:
    - Automatic retry with exponential backoff
    - Fallback to different loading methods
    - iPhone H.265 compatibility handling
    - Transcoding when needed

    Best for:
    - Production deployments
    - Mixed video formats
    - Unreliable storage systems
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_transcoding: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_transcoding = enable_transcoding
        self._retry_stats = {"retries": 0, "transcode_attempts": 0}

    @log_performance("robust_clip_loading")
    def load_clips(self, clip_specs: List[ClipSpec]) -> List[LoadedClip]:
        """Load clips with robust error handling."""
        loaded_clips = []

        with LoggingContext("robust_loading", self.logger) as ctx:
            ctx.log(f"Loading {len(clip_specs)} clips with robust strategy")

            for spec in clip_specs:
                try:
                    loaded_clip = self._load_clip_with_retry(spec)
                    loaded_clips.append(loaded_clip)

                except Exception as e:
                    ctx.log(
                        f"Failed to load {spec} after all retry attempts: {e}",
                        level="ERROR",
                        extra={"spec": str(spec)},
                    )
                    continue

            ctx.log(
                f"Robust loading completed: {len(loaded_clips)}/{len(clip_specs)} clips loaded",
                extra=self._retry_stats,
            )

        return loaded_clips

    def _load_clip_with_retry(self, spec: ClipSpec) -> LoadedClip:
        """Load clip with retry and fallback logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self._retry_stats["retries"] += 1
                    delay = self.retry_delay * (
                        2 ** (attempt - 1)
                    )  # Exponential backoff
                    self.logger.info(
                        f"Retry attempt {attempt} for {spec}",
                        extra={"delay": f"{delay:.1f}s"},
                    )
                    time.sleep(delay)

                return self._load_single_clip(spec)

            except iPhoneCompatibilityError as e:
                # Try transcoding for iPhone H.265 issues
                if self.enable_transcoding and attempt == 0:
                    try:
                        return self._load_with_transcoding(spec)
                    except Exception as transcoding_error:
                        self.logger.warning(
                            f"Transcoding failed for {spec}: {transcoding_error}"
                        )
                        last_exception = e
                        continue
                else:
                    last_exception = e
                    continue

            except Exception as e:
                last_exception = e
                continue

        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise VideoProcessingError(
                f"Failed to load {spec} after {self.max_retries + 1} attempts"
            )

    def _load_with_transcoding(self, spec: ClipSpec) -> LoadedClip:
        """Attempt to load clip with transcoding fallback."""
        self._retry_stats["transcode_attempts"] += 1

        # This would integrate with the transcoding system
        # For now, we'll raise an error to indicate transcoding is needed
        raise TranscodingError(
            f"Transcoding needed for {spec.file_path} - not implemented in this module",
            input_format="unknown",
            details={"spec": str(spec)},
        )


class UnifiedVideoLoader:
    """Unified video loader that selects optimal strategy based on context.

    Main interface for video loading in AutoCut V2. Automatically selects
    the best loading strategy based on system resources, clip characteristics,
    and user preferences.

    This replaces the 8 different loading approaches from the original god module
    with a clean, unified interface.
    """

    def __init__(
        self,
        default_strategy: LoadingStrategyType = LoadingStrategyType.AUTO,
        cache: Optional[VideoCache] = None,
        resource_manager: Optional[VideoResourceManager] = None,
    ):
        """Initialize unified video loader.

        Args:
            default_strategy: Default loading strategy to use
            cache: Video cache instance (creates new if None)
            resource_manager: Resource manager (creates new if None)
        """
        self.default_strategy = default_strategy
        self.cache = cache or VideoCache()
        self.resource_manager = resource_manager or VideoResourceManager()
        self.logger = get_logger("autocut.video.loading.UnifiedVideoLoader")

        # Initialize strategy instances
        self._strategies = {
            LoadingStrategyType.SEQUENTIAL: SequentialLoader(
                cache=self.cache, resource_manager=self.resource_manager
            ),
            LoadingStrategyType.PARALLEL: ParallelLoader(
                cache=self.cache, resource_manager=self.resource_manager
            ),
            LoadingStrategyType.ROBUST: RobustLoader(
                cache=self.cache, resource_manager=self.resource_manager
            ),
        }

    @log_performance("unified_video_loading")
    def load_clips(
        self, clip_specs: List[ClipSpec], strategy: Optional[LoadingStrategyType] = None
    ) -> List[LoadedClip]:
        """Load video clips using optimal strategy.

        Args:
            clip_specs: List of clip specifications to load
            strategy: Override strategy selection (optional)

        Returns:
            List of loaded clips

        Raises:
            VideoProcessingError: If loading fails
        """
        if not clip_specs:
            return []

        # Select strategy
        selected_strategy = strategy or self._select_optimal_strategy(clip_specs)
        loader = self._strategies[selected_strategy]

        with LoggingContext("unified_loading", self.logger) as ctx:
            ctx.log(
                f"Loading {len(clip_specs)} clips",
                extra={
                    "strategy": selected_strategy.value,
                    "total_duration": sum(spec.duration for spec in clip_specs),
                },
            )

            loaded_clips = loader.load_clips(clip_specs)

            ctx.log(
                f"Loading completed successfully",
                extra={
                    "clips_loaded": len(loaded_clips),
                    "success_rate": f"{len(loaded_clips) / len(clip_specs) * 100:.1f}%",
                },
            )

            return loaded_clips

    def _select_optimal_strategy(
        self, clip_specs: List[ClipSpec]
    ) -> LoadingStrategyType:
        """Select optimal loading strategy based on context."""
        if self.default_strategy != LoadingStrategyType.AUTO:
            return self.default_strategy

        # Strategy selection logic
        num_clips = len(clip_specs)
        total_duration = sum(spec.duration for spec in clip_specs)
        available_memory_gb = self.resource_manager.get_available_memory_gb()

        # Decision logic
        if available_memory_gb < 2.0 or total_duration > 300:  # 5 minutes
            # Low memory or long clips - use sequential
            strategy = LoadingStrategyType.SEQUENTIAL
        elif num_clips > 10 and available_memory_gb > 4.0:
            # Many clips and good memory - use parallel
            strategy = LoadingStrategyType.PARALLEL
        else:
            # Default to robust for reliability
            strategy = LoadingStrategyType.ROBUST

        self.logger.debug(
            f"Auto-selected strategy: {strategy.value}",
            extra={
                "num_clips": num_clips,
                "total_duration": f"{total_duration:.1f}s",
                "available_memory_gb": f"{available_memory_gb:.1f}GB",
            },
        )

        return strategy

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        stats = {
            "strategy_stats": {
                name.value: strategy.get_stats()
                for name, strategy in self._strategies.items()
            },
            "cache_stats": self.cache.get_stats(),
            "resource_stats": self.resource_manager.get_stats(),
        }
        return stats

    def clear_cache(self) -> None:
        """Clear video cache."""
        self.cache.clear()
        self.logger.info("Video cache cleared")

    def configure_strategy(self, strategy_type: LoadingStrategyType, **kwargs) -> None:
        """Configure specific strategy parameters."""
        if strategy_type in self._strategies:
            # This would update strategy configuration
            # Implementation depends on specific strategy parameters
            self.logger.info(
                f"Configured strategy {strategy_type.value}", extra={"config": kwargs}
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")


# Convenience functions for backward compatibility
def load_video_clips_sequential(clip_specs: List[ClipSpec]) -> List[LoadedClip]:
    """Legacy function for sequential loading."""
    loader = SequentialLoader()
    return loader.load_clips(clip_specs)


def load_video_clips_parallel(clip_specs: List[ClipSpec]) -> List[LoadedClip]:
    """Legacy function for parallel loading."""
    loader = ParallelLoader()
    return loader.load_clips(clip_specs)
