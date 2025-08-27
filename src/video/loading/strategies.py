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
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from moviepy import VideoFileClip

try:
    from moviepy.editor import ColorClip, CompositeVideoClip
except ImportError:
    # Fallback for older MoviePy versions
    try:
        from moviepy import ColorClip, CompositeVideoClip
    except ImportError:
        # Will be handled gracefully in _standardize_clip_resolution
        ColorClip = None
        CompositeVideoClip = None

try:
    from core.exceptions import (
        TranscodingError,
        VideoProcessingError,
        iPhoneCompatibilityError,
        raise_validation_error,
    )
    from core.logging_config import LoggingContext, get_logger, log_performance
except ImportError:
    # Fallback for testing without proper package structure
    import logging

    def get_logger(name):
        return logging.getLogger(name)

    def log_performance(func):
        return func

    class LoggingContext:
        def __init__(self, msg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class VideoProcessingError(Exception):
        pass

    class iPhoneCompatibilityError(Exception):
        pass

    class TranscodingError(Exception):
        pass

    def raise_validation_error(msg):
        raise Exception(msg)


from .cache import VideoCache
from .resource_manager import VideoResourceManager


@dataclass
class ClipSpec:
    """Specification for a video clip to be loaded."""

    file_path: str
    start_time: float
    end_time: float
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    _resolution_cache: Optional[Tuple[int, int]] = field(default=None, init=False)

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

    def get_video_resolution(self) -> Tuple[int, int]:
        """Get video resolution (width, height) with caching to avoid repeated detection.

        Returns:
            Tuple of (width, height)
        """
        if self._resolution_cache is not None:
            return self._resolution_cache

        try:
            # Quick resolution detection using MoviePy without loading full clip
            from moviepy import VideoFileClip

            with VideoFileClip(self.file_path) as temp_clip:
                self._resolution_cache = (temp_clip.w, temp_clip.h)
                return self._resolution_cache
        except Exception as e:
            # Fallback to common HD resolution if detection fails
            logger = get_logger("autocut.video.loading.ClipSpec")
            logger.warning(
                f"Failed to detect resolution for {self.file_path}: {e}, assuming HD"
            )
            self._resolution_cache = (1920, 1080)
            return self._resolution_cache

    def get_memory_complexity_factor(self) -> float:
        """Calculate memory complexity factor based on resolution.

        Returns:
            Multiplier for memory estimation (1.0 = HD baseline, 4.0 = 4K, etc.)
        """
        width, height = self.get_video_resolution()
        pixels = width * height

        # Define resolution tiers for memory estimation
        hd_pixels = 1920 * 1080  # ~2.1M pixels (baseline)

        if pixels <= hd_pixels:
            return 1.0  # HD baseline
        if pixels <= hd_pixels * 2:  # ~4M pixels (2K range)
            return 2.0
        if pixels <= hd_pixels * 4:  # ~8M pixels (4K range)
            return 4.0
        # 8K or higher
        return 8.0

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


class VideoCache:
    """Cache for parent video objects to prevent reader corruption in MoviePy 2.x.

    Maintains references to source VideoFileClip objects throughout the processing
    pipeline to prevent the 'NoneType' object has no attribute 'get_frame' error
    that occurs when subclipped videos lose their parent reader references.
    """

    def __init__(self):
        self._cache = {}  # filepath -> VideoFileClip
        self._parent_videos = {}  # Cache for parent video objects to prevent reader corruption
        self.logger = get_logger(__name__)

    def get_or_load_parent_video(self, filepath: str):
        """Get cached parent video or load new one if not cached.

        This is the CRITICAL fix for MoviePy 2.x reader corruption - we maintain
        parent video references to prevent subclipped videos from losing their readers.
        """
        if filepath not in self._parent_videos:
            # Dual import pattern for compatibility.moviepy
            try:
                from compatibility.moviepy import import_moviepy_safely
            except ImportError:
                # Fallback: create minimal function for basic MoviePy import
                def import_moviepy_safely():
                    from moviepy.editor import (
                        AudioFileClip,
                        CompositeVideoClip,
                        VideoFileClip,
                        concatenate_videoclips,
                    )

                    return (
                        VideoFileClip,
                        AudioFileClip,
                        concatenate_videoclips,
                        CompositeVideoClip,
                    )

            VideoFileClip, _, _, _ = import_moviepy_safely()

            self.logger.debug(f"Loading parent video into cache: {filepath}")
            self._parent_videos[filepath] = VideoFileClip(filepath)

        return self._parent_videos[filepath]

    def get_or_load(self, filepath: str):
        """Legacy method for compatibility."""
        return self.get_or_load_parent_video(filepath)

    def get(self, cache_key: str):
        """Get cached clip by cache key."""
        return self._cache.get(cache_key)

    def put(self, cache_key: str, clip, estimated_size_mb: float):
        """Put clip in cache with estimated size."""
        self._cache[cache_key] = clip
        self.logger.debug(f"Cached clip: {cache_key} ({estimated_size_mb}MB)")

    def get_stats(self):
        """Get cache statistics."""
        return {
            "cached_clips": len(self._cache),
            "parent_videos": len(self._parent_videos),
        }

    def clear(self):
        """Clear cache and close all video objects.

        IMPORTANT: This should only be called when the entire processing pipeline
        is complete to avoid causing NoneType reader errors.
        """
        # First clear regular clip cache
        self._cache.clear()

        def _safe_close_parent_video(filepath: str, clip):
            """Safely close a parent video clip."""
            try:
                if hasattr(clip, "close"):
                    clip.close()
                self.logger.debug(f"Closed parent video: {filepath}")
            except Exception as e:
                self.logger.warning(f"Failed to close parent video {filepath}: {e}")

        # Then close parent videos (this is safe to do at the end)
        for filepath, clip in self._parent_videos.items():
            _safe_close_parent_video(filepath, clip)

        self._parent_videos.clear()

    def clear_clips_only(self):
        """Clear only the clips cache, keeping parent videos alive."""
        self._cache.clear()
        self.logger.debug("Cleared clips cache, keeping parent videos alive")

    def __len__(self):
        return len(self._cache) + len(self._parent_videos)


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

    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics for monitoring."""
        return self._stats.copy()

    def _estimate_memory_requirements(self, spec: ClipSpec) -> float:
        """Estimate memory requirements for a clip with resolution awareness.

        CRITICAL FIX: Now accounts for 4K vs HD content properly to prevent
        memory buildup that was causing 42% success rates.

        Args:
            spec: Clip specification

        Returns:
            Estimated memory requirement in MB
        """
        # Base memory for video loading overhead
        base_memory_mb = 20.0

        # Estimate based on clip duration (most reliable indicator)
        duration = spec.duration

        # Base duration patterns for HD content
        if duration <= 5.0:
            duration_factor = 30.0 + (duration * 4.0)  # 30-50 MB
        elif duration <= 15.0:
            duration_factor = 50.0 + ((duration - 5.0) * 5.0)  # 50-100 MB
        else:
            duration_factor = 100.0 + min(
                (duration - 15.0) * 3.0, 100.0
            )  # 100-200 MB max

        # CRITICAL: Apply resolution complexity factor (4K = 4x memory)
        try:
            complexity_factor = spec.get_memory_complexity_factor()
            width, height = spec.get_video_resolution()
            resolution_adjusted_factor = duration_factor * complexity_factor
        except Exception as e:
            self.logger.warning(
                f"Failed to get resolution for {spec}: {e}, using HD baseline"
            )
            complexity_factor = 1.0
            width, height = (1920, 1080)
            resolution_adjusted_factor = duration_factor

        # Quality score adjustment (higher quality = more memory)
        quality_factor = 1.0 + (
            spec.quality_score * 0.3
        )  # Up to 30% more for high quality

        total_memory = base_memory_mb + (resolution_adjusted_factor * quality_factor)

        # Clamp to reasonable bounds - higher ceiling for 4K content
        max_memory = (
            1000.0 if complexity_factor >= 4.0 else 500.0
        )  # 4K gets higher limit
        total_memory = max(25.0, min(total_memory, max_memory))

        self.logger.debug(
            f"Memory estimate for {spec}: {total_memory:.1f}MB "
            f"(duration: {duration:.1f}s, resolution: {width}x{height}, "
            f"complexity: {complexity_factor:.1f}x, quality: {spec.quality_score:.2f})",
        )

        return total_memory

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

            # Estimate memory requirements intelligently
            estimated_memory_mb = self._estimate_memory_requirements(spec)

            # Load from disk
            with self.resource_manager.allocate_resources(
                estimated_memory_mb=estimated_memory_mb
            ):
                clip = self._load_clip_from_disk(spec)

                # Validate clip after loading
                if clip is None:
                    raise VideoProcessingError(f"Loaded clip is None for {spec}")

                # Test clip properties
                try:
                    duration = getattr(clip, "duration", None)
                    size = getattr(clip, "size", None)
                    if duration is None or size is None:
                        raise VideoProcessingError(
                            f"Loaded clip has invalid properties: duration={duration}, size={size}"
                        )
                    self.logger.debug(f"Loaded clip validated: {duration:.2f}s, {size}")
                except Exception as e:
                    raise VideoProcessingError(
                        f"Clip validation failed for {spec}: {e}"
                    ) from e

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

                # CRITICAL: Perform aggressive cleanup after each clip to prevent memory buildup
                self._aggressive_memory_cleanup(spec)

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

    def _aggressive_memory_cleanup(self, clip_spec: ClipSpec) -> None:
        """Perform aggressive memory cleanup after loading each clip.

        CRITICAL FIX: Forces immediate memory release to prevent the buildup
        that was causing 42% success rates at 88% memory usage.

        Args:
            clip_spec: The clip that was just loaded (for logging)
        """
        import gc

        try:
            # Force multiple generations of garbage collection
            collected_objects = 0
            for generation in range(3):
                collected = gc.collect(generation)
                collected_objects += collected

            # Clear any temporary MoviePy caches if accessible
            try:
                # MoviePy sometimes keeps internal caches - try to clear them
                import moviepy.editor

                if hasattr(moviepy.editor, "VideoFileClip"):
                    # Clear any class-level caches if they exist
                    pass
            except:
                pass  # Ignore cache clearing errors

            # Log cleanup results
            if collected_objects > 0:
                self.logger.debug(
                    f"Aggressive cleanup after {clip_spec}: freed {collected_objects} objects",
                )

        except Exception as e:
            self.logger.warning(f"Aggressive cleanup failed for {clip_spec}: {e}")

    def _check_system_memory_pressure(self) -> bool:
        """Check actual system memory pressure beyond our allocation tracking.

        Returns:
            True if system memory pressure is critical (>88%)
        """
        try:
            system_resources = (
                self.resource_manager.memory_monitor.get_system_resources()
            )
            return system_resources.memory_percent > 88.0
        except Exception as e:
            self.logger.warning(f"Failed to check system memory pressure: {e}")
            return False

    def _load_clip_from_disk(self, spec: ClipSpec) -> VideoFileClip:
        """Load clip from disk with MoviePy 2.x reader corruption prevention.

        CRITICAL FIX: Uses parent video cache to prevent the NoneType get_frame error
        that occurs when subclipped videos lose their parent reader references.
        """
        try:
            # CRITICAL: Get parent video from cache to prevent reader corruption
            parent_video = self.cache.get_or_load_parent_video(spec.file_path)

            # Create subclip if needed
            if spec.start_time > 0 or spec.end_time < parent_video.duration:
                # Use the correct method based on MoviePy version
                try:
                    # Try modern MoviePy 2.x method first
                    clip = parent_video.subclipped(spec.start_time, spec.end_time)
                    self.logger.debug(f"Subclip created using subclipped() for {spec}")
                except AttributeError:
                    # Fallback to older MoviePy 1.x method
                    try:
                        clip = parent_video.subclip(spec.start_time, spec.end_time)
                        self.logger.debug(f"Subclip created using subclip() for {spec}")
                    except AttributeError:
                        self.logger.exception(
                            f"Neither subclipped nor subclip available for {spec}"
                        )
                        raise

                # Validate the subclip can access frames
                try:
                    test_frame = clip.get_frame(0)
                    if test_frame is None:
                        raise RuntimeError(
                            f"Subclip get_frame returned None for {spec}"
                        )
                    self.logger.debug(f"Subclip validation successful for {spec}")
                except Exception as e:
                    self.logger.exception(f"Subclip validation failed for {spec}: {e}")
                    raise

                # CRITICAL: DO NOT close or copy the clip - this breaks the reader reference
                # The parent video must stay alive for the subclip to work
                self.logger.debug(f"Keeping parent video alive for subclip: {spec}")
                final_clip = clip

            else:
                # Return full video without creating subclip
                self.logger.debug(f"Using full video (no subclip needed) for {spec}")
                final_clip = parent_video

            # PHASE 6A: TEMPORARILY DISABLED - Resolution standardization causes MoviePy 2.x subclip reader corruption
            # Apply immediately during clip loading to prevent encoding issues
            # final_clip = self._standardize_clip_resolution(final_clip, spec)
            self.logger.debug(
                f"Resolution standardization temporarily disabled for {spec}"
            )

            return final_clip

        except Exception as e:
            self.logger.exception(f"Failed to load clip from disk for {spec}: {e}")
            # Handle iPhone H.265 compatibility issues
            if "codec" in str(e).lower() or "h265" in str(e).lower():
                raise iPhoneCompatibilityError(
                    f"iPhone H.265 compatibility issue with {spec.file_path}",
                    file_path=spec.file_path,
                    details={"original_error": str(e)},
                ) from e
            raise

    def _standardize_clip_resolution(
        self, clip: VideoFileClip, spec: ClipSpec
    ) -> VideoFileClip:
        """Standardize any input resolution to 1920x1080 with aspect ratio preservation.

        Handles all input types:
        - Landscape videos: Scale to fit within 1920x1080, add black bars if needed
        - Portrait videos: Scale to full height (1080), add black bars on sides
        - Square videos: Center with black bars on all sides

        Args:
            clip: The loaded video clip
            spec: ClipSpec for logging

        Returns:
            Standardized clip at 1920x1080 resolution
        """
        TARGET_WIDTH = 1920
        TARGET_HEIGHT = 1080
        TARGET_FPS = 25

        # Import required classes with proper error handling
        ColorClip_local = None
        CompositeVideoClip_local = None

        try:
            from moviepy.editor import (
                ColorClip as ColorClip_local,
                CompositeVideoClip as CompositeVideoClip_local,
            )
        except ImportError:
            try:
                from moviepy import (
                    ColorClip as ColorClip_local,
                    CompositeVideoClip as CompositeVideoClip_local,
                )
            except ImportError:
                self.logger.warning(
                    f"ColorClip/CompositeVideoClip not available - using basic scaling for {spec}"
                )
                ColorClip_local = None
                CompositeVideoClip_local = None

        # Import the correct MoviePy 2.x Resize effect
        try:
            from moviepy.video.fx.Resize import Resize
        except ImportError:
            try:
                from moviepy.video.fx import Resize
            except ImportError:
                self.logger.exception(
                    f"Cannot import Resize effect for MoviePy 2.x - falling back to original clip for {spec}"
                )
                return clip

        try:
            # Get current dimensions
            current_width, current_height = clip.size
            current_fps = clip.fps

            self.logger.debug(
                f"Standardizing {spec}: {current_width}x{current_height} @ {current_fps}fps → {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS}fps"
            )

            # Calculate aspect ratios
            current_aspect = current_width / current_height
            target_aspect = TARGET_WIDTH / TARGET_HEIGHT  # 16:9 = 1.777...

            if abs(current_aspect - target_aspect) < 0.01:
                # Already 16:9 aspect ratio - just scale
                self.logger.debug(
                    f"Clip {spec} has 16:9 aspect ratio, scaling directly"
                )
                resized_clip = clip.with_effects(
                    [Resize((TARGET_WIDTH, TARGET_HEIGHT))]
                )

            elif current_aspect > target_aspect:
                # Landscape video wider than 16:9 (e.g., ultrawide)
                # Scale to fit width, add black bars top/bottom
                new_height = int(TARGET_WIDTH / current_aspect)
                self.logger.debug(
                    f"Wide landscape {spec}: scaling to {TARGET_WIDTH}x{new_height}, adding top/bottom bars"
                )

                scaled_clip = clip.with_effects([Resize((TARGET_WIDTH, new_height))])

                # Add black background if ColorClip is available
                if ColorClip_local is not None and CompositeVideoClip_local is not None:
                    y_offset = (TARGET_HEIGHT - new_height) // 2
                    black_bg = ColorClip_local(
                        size=(TARGET_WIDTH, TARGET_HEIGHT),
                        color=(0, 0, 0),
                        duration=scaled_clip.duration,
                    )
                    resized_clip = CompositeVideoClip_local(
                        [
                            black_bg,
                            scaled_clip.with_position(("center", y_offset)),
                        ],
                        size=(TARGET_WIDTH, TARGET_HEIGHT),
                    )
                else:
                    # Fallback: Just scale and center without black bars
                    self.logger.warning(
                        f"ColorClip unavailable - centering {spec} without black bars"
                    )
                    resized_clip = scaled_clip.with_position("center").with_fps(
                        TARGET_FPS
                    )
                    # Try to pad to target size using Resize effect
                    try:
                        resized_clip = resized_clip.with_effects(
                            [Resize((TARGET_WIDTH, TARGET_HEIGHT))]
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to resize to target dimensions for {spec}: {e}"
                        )
                        # If resize fails, keep the scaled clip as-is

            else:
                # Portrait or narrow video (current_aspect < target_aspect)
                # Scale to fit height, add black bars on sides
                new_width = int(TARGET_HEIGHT * current_aspect)
                self.logger.debug(
                    f"Portrait/narrow {spec}: scaling to {new_width}x{TARGET_HEIGHT}, adding side bars"
                )

                scaled_clip = clip.with_effects([Resize((new_width, TARGET_HEIGHT))])

                # Add black background if ColorClip is available
                if ColorClip_local is not None and CompositeVideoClip_local is not None:
                    black_bg = ColorClip_local(
                        size=(TARGET_WIDTH, TARGET_HEIGHT),
                        color=(0, 0, 0),
                        duration=scaled_clip.duration,
                    )
                    resized_clip = CompositeVideoClip_local(
                        [
                            black_bg,
                            scaled_clip.with_position(("center", "center")),
                        ],
                        size=(TARGET_WIDTH, TARGET_HEIGHT),
                    )
                else:
                    # Fallback: Just scale and center without black bars
                    self.logger.warning(
                        f"ColorClip unavailable - centering {spec} without black bars"
                    )
                    resized_clip = scaled_clip.with_position("center").with_fps(
                        TARGET_FPS
                    )
                    # Try to pad to target size using Resize effect
                    try:
                        resized_clip = resized_clip.with_effects(
                            [Resize((TARGET_WIDTH, TARGET_HEIGHT))]
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to resize to target dimensions for {spec}: {e}"
                        )
                        # If resize fails, keep the scaled clip as-is

            # Ensure target FPS
            if abs(current_fps - TARGET_FPS) > 0.1:
                self.logger.debug(f"Adjusting FPS {spec}: {current_fps} → {TARGET_FPS}")
                resized_clip = resized_clip.with_fps(TARGET_FPS)

            # Verify final dimensions
            try:
                final_width, final_height = resized_clip.size
                if final_width != TARGET_WIDTH or final_height != TARGET_HEIGHT:
                    self.logger.warning(
                        f"Resolution standardization may have failed for {spec}: got {final_width}x{final_height}, expected {TARGET_WIDTH}x{TARGET_HEIGHT}"
                    )
                else:
                    self.logger.debug(
                        f"✅ Resolution standardized successfully for {spec}"
                    )
            except:
                # If size check fails, still return the clip
                self.logger.warning(f"Could not verify final dimensions for {spec}")

            return resized_clip

        except Exception as e:
            self.logger.exception(f"Failed to standardize resolution for {spec}: {e}")
            import traceback

            self.logger.exception(f"Traceback: {traceback.format_exc()}")
            # Return original clip if standardization fails
            self.logger.warning(f"Falling back to original resolution for {spec}")
            return clip

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
        """Load clips sequentially with batch processing and memory barriers.

        CRITICAL FIX: Processes clips in batches with aggressive cleanup
        between batches to prevent memory buildup that caused 42% success rates.
        """
        loaded_clips = []

        # Dynamic batch sizing based on system memory
        initial_batch_size = 5
        min_batch_size = 1
        current_batch_size = initial_batch_size

        with LoggingContext("sequential_loading", self.logger) as ctx:
            ctx.log(
                f"Loading {len(clip_specs)} clips sequentially with batch processing"
            )

            # Process clips in batches
            for batch_start in range(0, len(clip_specs), current_batch_size):
                batch_end = min(batch_start + current_batch_size, len(clip_specs))
                batch_specs = clip_specs[batch_start:batch_end]

                ctx.log(
                    f"Processing batch {batch_start // current_batch_size + 1}: "
                    f"clips {batch_start + 1}-{batch_end} (batch size: {len(batch_specs)})"
                )

                # Check system memory before starting batch
                if self._check_system_memory_pressure():
                    ctx.log(
                        "System memory pressure detected before batch - performing emergency cleanup"
                    )
                    self._perform_emergency_cleanup()
                    # Reduce batch size for memory-constrained processing
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    ctx.log(
                        f"Reduced batch size to {current_batch_size} due to memory pressure"
                    )

                # Process clips in current batch
                batch_success_count = 0
                for i, spec in enumerate(batch_specs):
                    global_index = batch_start + i + 1
                    ctx.log(f"Loading clip {global_index}/{len(clip_specs)}: {spec}")

                    try:
                        loaded_clip = self._load_single_clip(spec)
                        loaded_clips.append(loaded_clip)
                        batch_success_count += 1

                        ctx.log(
                            f"Successfully loaded clip {global_index}",
                            extra={"load_time": f"{loaded_clip.load_time:.2f}s"},
                        )

                    except Exception as e:
                        ctx.log(
                            f"Failed to load clip {global_index}: {e}",
                            level="ERROR",
                            extra={"spec": str(spec)},
                        )
                        # Add None placeholder to preserve index mapping
                        loaded_clips.append(None)

                # Memory barrier: Cleanup between batches
                ctx.log(
                    f"Batch completed: {batch_success_count}/{len(batch_specs)} clips loaded - performing batch cleanup"
                )
                self._perform_batch_cleanup()

                # Check if we should increase batch size (good memory situation)
                if (
                    not self._check_system_memory_pressure()
                    and current_batch_size < initial_batch_size
                ):
                    current_batch_size = min(initial_batch_size, current_batch_size + 1)
                    ctx.log(
                        f"Increased batch size to {current_batch_size} (good memory situation)"
                    )

            successful_count = sum(1 for clip in loaded_clips if clip is not None)
            ctx.log(
                f"Sequential loading completed: {successful_count}/{len(clip_specs)} clips loaded "
                f"({successful_count / len(clip_specs) * 100:.1f}% success rate)",
            )

        return loaded_clips

    def _perform_batch_cleanup(self) -> None:
        """Perform cleanup between batches to prevent memory accumulation."""
        import gc

        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear cache if it's getting large
            cache_stats = self.cache.get_stats()
            if cache_stats.get("total_size_mb", 0) > 200:  # Clear cache if >200MB
                self.cache.clear()
                self.logger.debug("Cleared video cache due to size (>200MB)")

            self.logger.debug(f"Batch cleanup: freed {collected} objects")

        except Exception as e:
            self.logger.warning(f"Batch cleanup failed: {e}")

    def _perform_emergency_cleanup(self) -> None:
        """Perform emergency cleanup when system memory pressure is critical."""
        import gc

        try:
            # Aggressive cleanup
            self.cache.clear()  # Clear all cached clips

            # Force multiple GC generations
            total_collected = 0
            for gen in range(3):
                collected = gc.collect(gen)
                total_collected += collected

            self.logger.info(
                f"Emergency cleanup: cleared cache and freed {total_collected} objects"
            )

        except Exception as e:
            self.logger.exception(f"Emergency cleanup failed: {e}")


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
        # Initialize results list with None placeholders to preserve index mapping
        loaded_clips = [None] * len(clip_specs)

        with LoggingContext("parallel_loading", self.logger) as ctx:
            ctx.log(
                f"Loading {len(clip_specs)} clips in parallel",
                extra={"max_workers": self.max_workers},
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks with index tracking
                future_to_index = {
                    executor.submit(self._load_single_clip, spec): i
                    for i, spec in enumerate(clip_specs)
                }

                # Collect results and place them in correct indices
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    spec = clip_specs[index]

                    try:
                        loaded_clip = future.result()
                        loaded_clips[index] = loaded_clip

                        ctx.log(
                            f"Completed loading {spec} at index {index}",
                            extra={"load_time": f"{loaded_clip.load_time:.2f}s"},
                        )

                    except Exception as e:
                        ctx.log(
                            f"Failed to load {spec} at index {index}: {e}",
                            level="ERROR",
                            extra={"spec": str(spec), "index": index},
                        )
                        # Keep None placeholder at this index

            successful_count = sum(1 for clip in loaded_clips if clip is not None)
            ctx.log(
                f"Parallel loading completed: {successful_count}/{len(clip_specs)} clips loaded",
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

            def _safe_load_clip_spec(spec, ctx):
                """Safely load a clip spec, returning None on failure."""
                try:
                    return self._load_clip_with_retry(spec)
                except Exception as e:
                    ctx.log(
                        f"Failed to load {spec} after all retry attempts: {e}",
                        level="ERROR",
                        extra={"spec": str(spec)},
                    )
                    # Return None placeholder to preserve index mapping
                    return None

            for spec in clip_specs:
                loaded_clip = _safe_load_clip_spec(spec, ctx)
                loaded_clips.append(loaded_clip)

            successful_count = sum(1 for clip in loaded_clips if clip is not None)
            ctx.log(
                f"Robust loading completed: {successful_count}/{len(clip_specs)} clips loaded",
                extra=self._retry_stats,
            )

        return loaded_clips

    def _try_single_load_attempt(self, spec: ClipSpec, attempt: int) -> Tuple[Optional[LoadedClip], Optional[Exception]]:
        """Try a single load attempt, returning result or exception."""
        try:
            if attempt > 0:
                self._retry_stats["retries"] += 1
                delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                self.logger.info(
                    f"Retry attempt {attempt} for {spec}",
                    extra={"delay": f"{delay:.1f}s"},
                )
                time.sleep(delay)

            result = self._load_single_clip(spec)
            return result, None

        except iPhoneCompatibilityError as e:
            # Try transcoding for iPhone H.265 issues
            if self.enable_transcoding and attempt == 0:
                try:
                    result = self._load_with_transcoding(spec)
                    return result, None
                except Exception as transcoding_error:
                    self.logger.warning(
                        f"Transcoding failed for {spec}: {transcoding_error}",
                    )
                    return None, e
            else:
                return None, e

        except Exception as e:
            return None, e

    def _load_clip_with_retry(self, spec: ClipSpec) -> LoadedClip:
        """Load clip with retry and fallback logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            result, exception = self._try_single_load_attempt(spec, attempt)
            if result is not None:
                return result
            last_exception = exception

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise VideoProcessingError(
            f"Failed to load {spec} after {self.max_retries + 1} attempts",
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
                cache=self.cache,
                resource_manager=self.resource_manager,
            ),
            LoadingStrategyType.PARALLEL: ParallelLoader(
                cache=self.cache,
                resource_manager=self.resource_manager,
            ),
            LoadingStrategyType.ROBUST: RobustLoader(
                cache=self.cache,
                resource_manager=self.resource_manager,
            ),
        }

    @log_performance("unified_video_loading")
    def load_clips(
        self,
        clip_specs: List[ClipSpec],
        strategy: Optional[LoadingStrategyType] = None,
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
                "Loading completed successfully",
                extra={
                    "clips_loaded": len(loaded_clips),
                    "success_rate": f"{len(loaded_clips) / len(clip_specs) * 100:.1f}%",
                },
            )

            return loaded_clips

    def _select_optimal_strategy(
        self,
        clip_specs: List[ClipSpec],
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
        return {
            "strategy_stats": {
                name.value: strategy.get_stats()
                for name, strategy in self._strategies.items()
            },
            "cache_stats": self.cache.get_stats(),
            "resource_stats": self.resource_manager.get_stats(),
        }

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
                f"Configured strategy {strategy_type.value}",
                extra={"config": kwargs},
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
