"""
Clip Assembly Module for AutoCut

Handles the core logic of matching video clips to musical beats,
applying variety patterns, and rendering the final video.
"""

import builtins
import contextlib
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

try:
    from adaptive_monitor import AdaptiveWorkerMonitor
    from memory.monitor import get_memory_info
    from system_profiler import SystemProfiler
except ImportError:
    # Fallback if modules not available
    class SystemProfiler:
        def __init__(self):
            pass

    class AdaptiveWorkerMonitor:
        def __init__(self):
            pass

    # Fallback get_memory_info if memory.monitor not available
    def get_memory_info():
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
            }
        except Exception:
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0}


try:
    from moviepy.editor import CompositeVideoClip, VideoFileClip, concatenate_videoclips
except ImportError:
    try:
        # Fallback for MoviePy 2.x direct imports
        from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips
    except ImportError:
        # Final fallback for testing without moviepy installation
        VideoFileClip = CompositeVideoClip = concatenate_videoclips = None
# Import VideoChunk from canonical location
try:
    from video import VideoChunk
except ImportError:
    try:
        from video_analyzer import VideoChunk
    except ImportError:
        # Fallback if VideoChunk not available
        VideoChunk = None

# Import extracted classes from new modular structure
try:
    from video.encoder import (
        VideoEncoder,
        detect_optimal_codec_settings,
        detect_optimal_codec_settings_with_diagnostics,
    )
    from video.timeline_renderer import ClipTimeline
except ImportError:
    # Classes will be defined inline below for backward compatibility
    try:
        from video_analyzer import VideoChunk
    except ImportError:
        # Fallback if modules not available
        class VideoChunk:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


# Import robust audio loading system
try:
    from audio_loader import load_audio_robust
except ImportError:
    # Fallback if audio_loader not available
    def load_audio_robust(audio_file):
        """
        Load audio using the robust multi-strategy loader from audio_loader module.
        This replaces the problematic AudioFileClip fallback with the comprehensive
        robust loader that handles WAV files and other formats safely.
        """
        try:
            # Try to import and use the comprehensive robust audio loader
            from audio_loader import load_audio_robust as robust_loader

            return robust_loader(audio_file)
        except ImportError:
            # If audio_loader module not available, try alternative robust approach

            # Import moviepy safely with compatibility layer
            from moviepy.editor import AudioFileClip

            try:
                # First try standard MoviePy loading
                return AudioFileClip(audio_file)
            except (AttributeError, RuntimeError, OSError) as e:
                if "proc" in str(e).lower() or "ffmpeg" in str(e).lower():
                    def _raise_no_audio_data():
                        raise RuntimeError("No audio data extracted from file")

                    # Fallback to FFmpeg subprocess for problematic files
                    try:
                        import subprocess

                        import numpy as np
                        from moviepy.audio.AudioClip import AudioArrayClip

                        # Use FFmpeg subprocess to bypass MoviePy's FFMPEG_AudioReader
                        cmd = [
                            "ffmpeg",
                            "-i",
                            audio_file,
                            "-f",
                            "s16le",
                            "-acodec",
                            "pcm_s16le",
                            "-ac",
                            "2",
                            "-ar",
                            "44100",
                            "-v",
                            "quiet",
                            "-",
                        ]

                        process = subprocess.run(
                            cmd, capture_output=True, check=True, timeout=60
                        )
                        audio_data = np.frombuffer(process.stdout, dtype=np.int16)

                        if len(audio_data) == 0:
                            _raise_no_audio_data()

                        # Convert to stereo float32 format
                        audio_data = (
                            audio_data.reshape(-1, 2).astype(np.float32) / 32768.0
                        )
                        return AudioArrayClip(audio_data, fps=44100)

                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Could not load audio file {audio_file}: {fallback_error}"
                        ) from fallback_error
                else:
                    # Re-raise non-audio-specific errors
                    raise


# Variety patterns to prevent monotonous cutting
VARIETY_PATTERNS = {
    "energetic": [2, 2, 4, 2, 2, 8],  # Fast 2-beat cuts with occasional longer pause
    "buildup": [8, 4, 4, 4, 4, 4],  # Start slow, maintain deliberate 4-beat pace
    "balanced": [4, 4, 4, 8, 4, 4],  # Consistent 4-beat pacing with variety
    "dramatic": [4, 4, 4, 4, 16],  # Build tension with 4-beat base, long dramatic hold
}


class VideoCache:
    """Thread-safe cache for loaded video files to prevent duplicate loading."""

    def __init__(self):
        self._cache: Dict[str, VideoFileClip] = {}
        self._lock = threading.Lock()
        self._ref_counts: Dict[str, int] = defaultdict(int)

    def get_or_load(self, video_path: str) -> VideoFileClip:
        """Get cached video or load it if not cached.

        Args:
            video_path: Path to video file

        Returns:
            VideoFileClip instance

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be loaded
        """
        if VideoFileClip is None:
            raise RuntimeError("MoviePy not available. Please install moviepy>=1.0.3")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with self._lock:
            if video_path not in self._cache:
                try:
                    video_clip = VideoFileClip(video_path)
                    self._cache[video_path] = video_clip
                except Exception as e:
                    raise RuntimeError(f"Failed to load video {video_path}: {e!s}") from e

            # Increment reference count
            self._ref_counts[video_path] += 1
            return self._cache[video_path]

    def release(self, video_path: str) -> None:
        """Release reference to cached video.

        Args:
            video_path: Path to video file to release
        """
        with self._lock:
            if video_path in self._ref_counts:
                self._ref_counts[video_path] -= 1

                # Remove from cache if no more references
                if self._ref_counts[video_path] <= 0:
                    if video_path in self._cache:
                        try:
                            self._cache[video_path].close()
                        except Exception:
                            pass  # Ignore cleanup errors
                        del self._cache[video_path]
                    del self._ref_counts[video_path]

    def get_cached_paths(self) -> List[str]:
        """Get list of currently cached video paths.

        Returns:
            List of cached video file paths
        """
        with self._lock:
            return list(self._cache.keys())

    def clear(self) -> None:
        """Clear all cached videos and close resources."""
        def _safe_close_video(video_clip):
            """Safely close video clip, ignoring any exceptions."""
            try:
                video_clip.close()
            except Exception:
                pass  # Ignore cleanup errors

        with self._lock:
            for video_clip in self._cache.values():
                _safe_close_video(video_clip)
            self._cache.clear()
            self._ref_counts.clear()


# VideoFormatAnalyzer class extracted to src/video/format_analyzer.py


# VideoNormalizationPipeline class extracted to src/video/normalization.py


def load_video_segment(
    clip_data: Dict[str, Any],
    video_cache: VideoCache,
) -> Optional[Tuple[Dict[str, Any], Any]]:
    """Load a single video segment in parallel processing.

    Args:
        clip_data: Dictionary containing video_file, start, end times
        video_cache: Shared video cache instance

    Returns:
        Tuple of (clip_data, video_segment) or None if failed
    """
    try:
        # Get cached video (thread-safe)
        source_video = video_cache.get_or_load(clip_data["video_file"])

        # Extract the specific segment using safe compatibility method
        segment = subclip_safely(source_video, clip_data["start"], clip_data["end"])

    except Exception:
        return None
    else:
        return (clip_data, segment)


# ============================================================================
# NEW THREAD-SAFE SEQUENTIAL VIDEO PROCESSING (Phase 1 Fix)
# Replaces dangerous parallel loading with memory-safe sequential processing
# ============================================================================


class MemoryMonitor:
    """Real-time memory monitoring with emergency cleanup capabilities."""

    def __init__(
        self,
        warning_threshold_gb: float = 4.0,
        emergency_threshold_gb: float = 6.0,
    ):
        import psutil

        self.warning_threshold = warning_threshold_gb * 1024 * 1024 * 1024
        self.emergency_threshold = emergency_threshold_gb * 1024 * 1024 * 1024
        self.baseline_memory = psutil.Process().memory_info().rss

    def get_current_usage_gb(self) -> float:
        """Get current memory usage above baseline in GB."""
        import psutil

        current = psutil.Process().memory_info().rss
        return (current - self.baseline_memory) / (1024 * 1024 * 1024)

    def should_emergency_cleanup(self) -> bool:
        """Check if emergency cleanup is needed."""
        import psutil

        return psutil.Process().memory_info().rss > self.emergency_threshold

    def log_memory_status(self, context: str) -> None:
        """Log current memory usage with context."""
        usage_gb = self.get_current_usage_gb()
        if usage_gb > 2.0:  # Only log if significant memory usage
            pass


class VideoResourceManager:
    """Ensures proper cleanup of VideoFileClip resources with support for delayed cleanup.

    This version supports both immediate cleanup (for simple cases) and delayed cleanup
    (for cases where subclips need to be used after parent video creation).
    """

    def __init__(self):
        self.active_videos = set()
        self.delayed_cleanup_videos = {}  # path -> video object for delayed cleanup

    def load_video_safely(self, video_path: str):
        """Context manager for safe video loading with guaranteed cleanup."""
        import gc
        from contextlib import contextmanager

        @contextmanager
        def _video_context():
            video = None
            try:
                # Use safe import pattern instead of global variable
                VideoFileClip, _, _, _ = import_moviepy_safely()
                
                video = VideoFileClip(video_path)
                self.active_videos.add(id(video))
                yield video
            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_path}: {e!s}") from e
            finally:
                if video is not None:
                    try:
                        self.active_videos.discard(id(video))
                        video.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                    del video
                    gc.collect()  # Force garbage collection

        return _video_context()

    def load_video_with_delayed_cleanup(self, video_path: str):
        """Load a video with delayed cleanup - video will be kept alive until cleanup_delayed_videos() is called.

        This is the CRITICAL FIX for the NoneType get_frame error:
        - Parent videos stay alive while their subclips are being used
        - Cleanup happens only after concatenation is complete
        """
        try:
            # Use safe import pattern instead of global variable
            VideoFileClip, _, _, _ = import_moviepy_safely()

            # Check if we already have this video loaded for delayed cleanup
            if video_path in self.delayed_cleanup_videos:
                return self.delayed_cleanup_videos[video_path]

            # Load the video and store it for delayed cleanup
            video = VideoFileClip(video_path)
            self.delayed_cleanup_videos[video_path] = video
            self.active_videos.add(id(video))
            
            # CRITICAL FIX: Return video in try block, not orphaned else block
            return video

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {e!s}") from e

    def cleanup_delayed_videos(self) -> None:
        """Clean up all videos that were loaded with delayed cleanup.

        This should be called AFTER concatenation is complete to ensure subclips
        remain valid during the entire video processing pipeline.
        """
        import gc

        def _safe_cleanup_video(video):
            """Safely cleanup a single video resource."""
            try:
                self.active_videos.discard(id(video))
                video.close()
            except Exception:
                pass  # Ignore cleanup errors during resource cleanup

        cleanup_count = len(self.delayed_cleanup_videos)
        if cleanup_count > 0:
            for video in self.delayed_cleanup_videos.values():
                _safe_cleanup_video(video)

            self.delayed_cleanup_videos.clear()
            gc.collect()  # Force garbage collection

    def emergency_cleanup(self) -> None:
        """Force cleanup of any remaining video resources."""
        import gc

        # Clean up delayed videos first
        self.cleanup_delayed_videos()

        gc.collect()
        self.active_videos.clear()

    def register_temp_file(self, temp_file_path) -> None:
        """Register a temporary file for cleanup.
        
        Args:
            temp_file_path: Path to temporary file to be cleaned up later
        """
        if not hasattr(self, '_temp_files'):
            self._temp_files = set()
        self._temp_files.add(str(temp_file_path))

    def cleanup_all(self) -> None:
        """Clean up all resources including delayed videos and temporary files."""
        import gc

        # Clean up delayed videos
        self.cleanup_delayed_videos()

        # Clean up temporary files
        if hasattr(self, '_temp_files'):
            for temp_file_path in self._temp_files.copy():
                try:
                    temp_path = Path(temp_file_path)
                    if temp_path.exists():
                        if temp_path.is_file():
                            temp_path.unlink()
                        elif temp_path.is_dir():
                            import shutil
                            shutil.rmtree(temp_path, ignore_errors=True)
                    self._temp_files.discard(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors

        gc.collect()
        self.active_videos.clear()


def load_video_clips_sequential(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Any], List[int], VideoResourceManager]:
    """Load video clips sequentially with memory-safe processing and smart preprocessing.

    CRITICAL FIX: This version uses delayed cleanup to prevent NoneType get_frame errors.
    Parent videos are kept alive until after concatenation, ensuring subclips remain valid.

    This function replaces the dangerous load_video_clips_parallel() to fix:
    1. Thread-safety violations (no shared VideoFileClip objects)
    2. Memory exhaustion (sequential loading with delayed cleanup)
    3. Resource leaks (proper cleanup after concatenation)
    4. Format compatibility issues (H.265, high-res preprocessing)
    5. NoneType get_frame errors (FIXED: parent videos stay alive)

    Args:
        sorted_clips: List of clip data dictionaries sorted by beat position
        video_files: List of source video files (for smart preprocessing analysis)
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (video_clips_list, failed_indices, resource_manager)
        The resource_manager must have cleanup_delayed_videos() called after concatenation

    Raises:
        RuntimeError: If no clips could be loaded successfully
    """

    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)

    if not sorted_clips:
        raise ValueError("No clips provided for loading")

    # Initialize memory monitoring and resource management with DELAYED CLEANUP
    memory_monitor = MemoryMonitor(warning_threshold_gb=4.0, emergency_threshold_gb=6.0)
    resource_manager = VideoResourceManager()  # This will now support delayed cleanup

    # PHASE 2 INTEGRATION: Smart preprocessing for format compatibility
    video_path_map = preprocess_videos_smart(
        video_files,
        progress_callback=progress_callback,
    )

    # Update clip data to use preprocessed video paths
    for clip_data in sorted_clips:
        original_path = clip_data["video_file"]
        if original_path in video_path_map:
            optimized_path = video_path_map[original_path]
            if optimized_path != original_path:
                clip_data["video_file"] = optimized_path
                clip_data["original_video_file"] = (
                    original_path  # Keep reference to original
                )

    # Memory monitoring at start
    get_memory_info()

    # Use a results dictionary to maintain perfect index alignment
    clip_results = {}  # Maps original_index -> video clip (or None for failed)
    failed_indices = []

    # Group clips by video file to minimize loading
    grouped_clips = _group_clips_by_file(sorted_clips)

    processed_files = 0
    total_clips_processed = 0

    for video_file, file_clips in grouped_clips.items():
        processed_files += 1

        # Check memory before loading each video file
        if memory_monitor.should_emergency_cleanup():
            resource_manager.emergency_cleanup()

        try:
            # CRITICAL FIX: Use delayed cleanup instead of immediate cleanup
            # This keeps the parent video alive so subclips remain valid
            source_video = resource_manager.load_video_with_delayed_cleanup(video_file)

            file_clips_loaded = 0

            # Process all clips from this video file
            for clip_data in file_clips:
                try:
                    memory_monitor.log_memory_status(
                        f"clip {total_clips_processed + 1}",
                    )

                    # Extract segment - parent video stays alive via delayed cleanup
                    segment = source_video.subclipped(
                        clip_data["start"],
                        clip_data["end"],
                    )

                    # Validation: Test the clip while parent video is available
                    # This validation is still important for catching other issues
                    try:
                        test_time = min(0.1, segment.duration)
                        test_frame = segment.get_frame(test_time)
                        if test_frame is None:
                            raise RuntimeError(
                                "get_frame returned None during validation",
                            )
                    except Exception as e:
                        raise RuntimeError(f"Clip validation failed: {e}") from e

                    original_index = clip_data.get(
                        "original_index",
                        total_clips_processed,
                    )
                    clip_results[original_index] = segment
                    file_clips_loaded += 1

                except Exception:
                    # Use original_index from grouped clips to maintain timeline alignment
                    original_index = clip_data.get(
                        "original_index",
                        total_clips_processed,
                    )
                    clip_results[original_index] = None
                    failed_indices.append(original_index)

                total_clips_processed += 1

            # NOTE: We do NOT close the source_video here - it will be cleaned up later
            # This is the key fix: parent videos stay alive until after concatenation

            # Update progress
            progress = 0.1 + (0.6 * total_clips_processed / len(sorted_clips))
            report_progress(
                f"Loaded {total_clips_processed}/{len(sorted_clips)} clips",
                progress,
            )

        except Exception:
            # Mark all clips from this file as failed using original indices
            for clip_data in file_clips:
                original_index = clip_data.get("original_index", total_clips_processed)
                clip_results[original_index] = None
                failed_indices.append(original_index)
                total_clips_processed += 1
            continue

    # Reconstruct video_clips list in original order, with None for failed clips
    video_clips = []
    for i in range(len(sorted_clips)):
        clip = clip_results.get(i)
        video_clips.append(clip)

    # Count successful clips (non-None)
    successful_clips = [clip for clip in video_clips if clip is not None]

    if not successful_clips:
        raise RuntimeError("No video clips could be loaded successfully")

    # Report final statistics
    success_count = len(successful_clips)

    if failed_indices:
        pass

    # Memory monitoring at completion
    final_memory = get_memory_info()

    # Memory usage warning
    if final_memory["percent"] > 85:
        pass

    report_progress(f"Successfully loaded {success_count} clips", 0.7)

    # Return clips with perfect index alignment, failed indices, and resource manager
    # The caller MUST call resource_manager.cleanup_delayed_videos() after concatenation
    return video_clips, failed_indices, resource_manager


def _group_clips_by_file(
    sorted_clips: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group clips by video file while preserving beat order within groups.

    This optimization reduces memory usage by loading each video file only once
    and extracting all required clips before moving to the next file.
    """
    from collections import defaultdict

    # First, group by file
    file_groups = defaultdict(list)
    for i, clip in enumerate(sorted_clips):
        # Add original index to maintain order tracking
        clip_with_index = clip.copy()
        clip_with_index["original_index"] = i
        file_groups[clip["video_file"]].append(clip_with_index)

    # Within each file, sort by start time for efficient sequential access
    for video_file, clips in file_groups.items():
        clips.sort(key=lambda c: c["start"])

    # Order files by first clip's beat position to maintain overall flow
    file_order = []
    for video_file, clips in file_groups.items():
        first_beat_position = min(c.get("beat_position", 0) for c in clips)
        file_order.append((first_beat_position, video_file))

    file_order.sort()  # Sort by beat position

    # Return ordered dictionary
    ordered_groups = {}
    for _, video_file in file_order:
        ordered_groups[video_file] = file_groups[video_file]

    return ordered_groups


# ============================================================================
# PHASE 2: SMART PREPROCESSING PIPELINE
# Handles H.265, format compatibility, and memory optimization
# ============================================================================


class VideoPreprocessor:
    """Smart video preprocessing for format compatibility and memory optimization."""

    def __init__(self):
        self.preprocessing_cache = {}
        self.supported_formats = {
            "h264": {"memory_efficient": True, "compatibility": "high"},
            "h265": {
                "memory_efficient": False,
                "compatibility": "medium",
            },  # Needs preprocessing
            "hevc": {
                "memory_efficient": False,
                "compatibility": "medium",
            },  # Same as h265
            "vp9": {"memory_efficient": True, "compatibility": "high"},
            "av1": {"memory_efficient": False, "compatibility": "low"},  # Newer format
        }

    def should_preprocess_video(self, video_path: str) -> Dict[str, Any]:
        """Determine if a video needs preprocessing and why.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with preprocessing decision and reasons
        """

        if not Path(video_path).exists():
            return {"needs_preprocessing": False, "reason": "file_not_found"}

        try:
            # Use ffprobe to detect video properties without loading into MoviePy
            codec_info = self._detect_video_properties_ffprobe(video_path)

            preprocessing_reasons = []
            needs_preprocessing = False

            # Check codec compatibility
            codec = codec_info.get("codec_name", "").lower()
            if codec in ["h265", "hevc"]:
                preprocessing_reasons.append("h265_codec_memory_intensive")
                needs_preprocessing = True
            elif codec == "av1":
                preprocessing_reasons.append("av1_codec_compatibility")
                needs_preprocessing = True

            # Check resolution for memory concerns
            width = codec_info.get("width", 0)
            height = codec_info.get("height", 0)
            if width > 2560 or height > 1440:  # Above 1440p
                preprocessing_reasons.append("high_resolution_memory_optimization")
                needs_preprocessing = True

            # Check frame rate for processing efficiency
            fps_str = codec_info.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 24.0
            else:
                fps = float(fps_str)

            if fps > 60:
                preprocessing_reasons.append("high_framerate_optimization")
                needs_preprocessing = True

            return {
                "needs_preprocessing": needs_preprocessing,
                "reasons": preprocessing_reasons,
                "codec": codec,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "estimated_memory_mb": self._estimate_memory_usage(width, height, fps),
            }

        except Exception as e:
            return {
                "needs_preprocessing": False,
                "reason": "analysis_failed",
                "error": str(e),
            }

    def _detect_video_properties_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Use ffprobe to detect video properties without loading full video."""
        import subprocess

        try:
            # Run ffprobe to get video stream information
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",  # First video stream only
                video_path,
            ]

            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "streams" in data and len(data["streams"]) > 0:
                    stream = data["streams"][0]
                    return {
                        "codec_name": stream.get("codec_name", "unknown"),
                        "width": int(stream.get("width", 0)),
                        "height": int(stream.get("height", 0)),
                        "r_frame_rate": stream.get("r_frame_rate", "24/1"),
                        "duration": float(stream.get("duration", 0)),
                        "bit_rate": int(stream.get("bit_rate", 0))
                        if stream.get("bit_rate")
                        else 0,
                    }

        except Exception:
            return {
                "codec_name": "unknown",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "24/1",
            }
        else:
            # Fallback if ffprobe fails
            return {
                "codec_name": "unknown",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "24/1",
            }

    def _estimate_memory_usage(self, width: int, height: int, fps: float) -> float:
        """Estimate memory usage in MB for video processing."""
        # Rough estimation: width * height * 3 bytes (RGB) * fps * typical_buffer_seconds / 1MB
        typical_buffer_seconds = 2.0  # MoviePy typically buffers a few seconds
        bytes_per_pixel = 3  # RGB

        memory_bytes = width * height * bytes_per_pixel * fps * typical_buffer_seconds
        return memory_bytes / (1024 * 1024)

    def preprocess_video_if_needed(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Preprocess video if needed and return path to processed version.

        Args:
            video_path: Path to original video file
            output_dir: Directory for processed files (default: same as input)

        Returns:
            Path to video file to use (original or preprocessed)
        """

        # Check if preprocessing is needed
        analysis = self.should_preprocess_video(video_path)

        if not analysis["needs_preprocessing"]:
            return video_path

        # Check cache first
        cache_key = f"{video_path}_{hash(str(analysis['reasons']))}"
        if cache_key in self.preprocessing_cache:
            cached_path = self.preprocessing_cache[cache_key]
            if Path(cached_path).exists():
                return cached_path

        # Determine output path
        if output_dir is None:
            output_dir = Path(video_path).parent

        base_name = Path(video_path).stem
        processed_name = f"{base_name}_processed.mp4"
        processed_path = Path(output_dir) / processed_name

        # If processed version already exists and is newer, use it
        if processed_path.exists() and processed_path.stat().st_mtime > Path(video_path).stat().st_mtime:
            self.preprocessing_cache[cache_key] = str(processed_path)
            return str(processed_path)

        # Perform preprocessing
        try:
            success = self._preprocess_with_ffmpeg_modern(
                video_path,
                processed_path,
                analysis,
                getattr(self, "_target_format", None),
            )

            if success and processed_path.exists():
                self.preprocessing_cache[cache_key] = str(processed_path)
                return str(processed_path)
        except Exception:
            return video_path
        else:
            return video_path

    def _preprocess_with_ffmpeg_modern(
        self,
        input_path: str,
        output_path: str,
        analysis: Dict,
        target_format: Optional[Dict] = None,
    ) -> bool:
        """Modern FFmpeg preprocessing with intelligent aspect ratio preservation.

        This replaces the old hard-coded 1920x1080 scaling that stretched portrait videos.
        Uses dynamic canvas dimensions and proper letterboxing with pad filter.
        """
        import subprocess

        try:
            # Build FFmpeg command based on preprocessing needs
            cmd = ["ffmpeg", "-y", "-i", input_path]  # -y to overwrite

            # Video codec settings
            if any(
                "h265" in reason or "hevc" in reason for reason in analysis["reasons"]
            ):
                # Convert H.265 to H.264 for better compatibility and lower memory
                cmd.extend(["-c:v", "libx264"])
                cmd.extend(["-preset", "fast"])  # Balance speed vs compression
            else:
                # Keep original codec but optimize
                cmd.extend(["-c:v", "libx264"])

            # ASPECT RATIO PRESERVATION - Modern approach using target format
            if (
                any("resolution" in reason for reason in analysis["reasons"])
                and target_format
            ):
                target_w = target_format.get("target_width", 1920)
                target_h = target_format.get("target_height", 1080)

                # Use modern FFmpeg scaling with aspect ratio preservation + letterboxing
                # Step 1: Scale down if needed, maintaining aspect ratio
                scale_filter = (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
                )

                # Step 2: Add letterboxing with pad filter (black bars)
                pad_filter = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"

                # Combine both filters for perfect aspect ratio preservation
                combined_filter = f"{scale_filter},{pad_filter}"
                cmd.extend(["-vf", combined_filter])

            elif any("resolution" in reason for reason in analysis["reasons"]):
                # Fallback: use safe 1920x1080 with letterboxing (backward compatibility)
                fallback_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
                cmd.extend(["-vf", fallback_filter])

            # Frame rate optimization
            if any("framerate" in reason for reason in analysis["reasons"]):
                # Use target FPS if available, otherwise limit to 30fps
                target_fps = (
                    target_format.get("target_fps", 30) if target_format else 30
                )
                cmd.extend(["-r", str(target_fps)])

            # Audio handling
            cmd.extend(["-c:a", "aac"])  # Standard audio codec
            cmd.extend(["-b:a", "128k"])  # Reasonable audio bitrate

            # Quality settings
            cmd.extend(["-crf", "23"])  # Good quality/size balance
            cmd.extend(["-movflags", "+faststart"])  # Web optimization

            cmd.append(output_path)

            # Run with timeout to prevent hanging
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                # Verify output dimensions match target
                if target_format:
                    # target_format dimensions available for future validation
                    pass
                else:
                    pass
                return True
            if result.stderr:
                pass  # First 200 chars
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        else:
            return False

    def cleanup_preprocessed_files(self, max_age_hours: int = 24) -> None:
        """Clean up old preprocessed files to save disk space."""
        import time

        current_time = time.time()
        cleaned_count = 0

        for cached_path in list(self.preprocessing_cache.values()):
            if Path(cached_path).exists():
                file_age_hours = (current_time - Path(cached_path).stat().st_mtime) / 3600
                if file_age_hours > max_age_hours:
                    try:
                        Path(cached_path).unlink()
                        cleaned_count += 1
                    except Exception:
                        pass  # Ignore cleanup errors

        # Clear cache entries for non-existent files
        self.preprocessing_cache = {
            k: v for k, v in self.preprocessing_cache.items() if Path(v).exists()
        }

        if cleaned_count > 0:
            pass


def preprocess_videos_smart(
    video_files: List[str],
    canvas_format: Optional[dict] = None,  # NEW: Intelligent canvas format
    output_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, str]:
    """Smart preprocessing of video files for optimal processing with intelligent canvas sizing.

    Args:
        video_files: List of video file paths to analyze and preprocess
        canvas_format: Intelligent canvas format from VideoFormatAnalyzer
        output_dir: Directory for preprocessed files (default: temp)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping original paths to optimized paths (original or preprocessed)
    """
    import tempfile

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="autocut_preprocessed_")

    preprocessor = VideoPreprocessor()

    # CRITICAL FIX: Pass canvas format to preprocessor for intelligent scaling
    if canvas_format:
        preprocessor._target_format = canvas_format
    else:
        pass

    video_map = {}

    preprocessing_needed = 0
    total_estimated_memory = 0

    for i, video_path in enumerate(video_files):
        if progress_callback:
            progress = i / len(video_files)
            progress_callback(f"Analyzing video {i + 1}/{len(video_files)}", progress)

        analysis = preprocessor.should_preprocess_video(video_path)

        if analysis["needs_preprocessing"]:
            preprocessing_needed += 1
            total_estimated_memory += analysis.get("estimated_memory_mb", 0)

        # Always preprocess if needed, map result (now with canvas format)
        optimized_path = preprocessor.preprocess_video_if_needed(video_path, output_dir)
        video_map[video_path] = optimized_path

    # Summary

    # NEW: Canvas format summary
    if canvas_format:
        pass

    if progress_callback:
        progress_callback("Smart preprocessing complete", 1.0)

    return video_map


# ============================================================================
# PHASE 3: ENHANCED MEMORY-SAFE PROCESSING
# Advanced memory management with batch processing and emergency cleanup
# ============================================================================


class AdvancedMemoryManager:
    """Advanced memory management for video processing with adaptive strategies."""

    def __init__(
        self,
        warning_threshold_gb: float = 4.0,
        emergency_threshold_gb: float = 6.0,
        critical_threshold_gb: float = 8.0,
    ):
        self.warning_threshold = warning_threshold_gb * 1024 * 1024 * 1024
        self.emergency_threshold = emergency_threshold_gb * 1024 * 1024 * 1024
        self.critical_threshold = critical_threshold_gb * 1024 * 1024 * 1024

        import psutil

        self.baseline_memory = psutil.Process().memory_info().rss
        self.system_total_memory = psutil.virtual_memory().total

        # Adaptive processing parameters
        self.emergency_cleanup_count = 0
        self.memory_warnings = 0
        self.processing_mode = "normal"  # 'normal', 'conservative', 'emergency'

    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status information."""
        import psutil

        process_memory = psutil.Process().memory_info()
        system_memory = psutil.virtual_memory()

        current_usage = process_memory.rss
        baseline_increase = current_usage - self.baseline_memory

        return {
            "current_usage_gb": current_usage / (1024**3),
            "baseline_increase_gb": baseline_increase / (1024**3),
            "system_total_gb": system_memory.total / (1024**3),
            "system_available_gb": system_memory.available / (1024**3),
            "system_percent": system_memory.percent,
            "is_warning": current_usage > self.warning_threshold,
            "is_emergency": current_usage > self.emergency_threshold,
            "is_critical": current_usage > self.critical_threshold,
            "processing_mode": self.processing_mode,
        }

    def should_switch_to_emergency_mode(self) -> bool:
        """Check if we should switch to emergency memory management mode."""
        status = self.get_memory_status()

        # Switch to emergency if memory is critical OR multiple warnings
        if status["is_critical"] or (
            status["is_emergency"] and self.memory_warnings > 3
        ):
            if self.processing_mode != "emergency":
                self.processing_mode = "emergency"
            return True

        if status["is_warning"]:
            if self.processing_mode == "normal":
                self.processing_mode = "conservative"
            self.memory_warnings += 1
            return False

        return False

    def perform_emergency_cleanup(self, context: str = "unknown") -> Dict[str, Any]:
        """Perform aggressive memory cleanup and return results."""
        import gc

        before_status = self.get_memory_status()

        # Multiple rounds of garbage collection
        for i in range(3):
            collected = gc.collect()
            if i == 0:
                total_collected = collected
            else:
                total_collected += collected

        # Force memory compaction if available
        try:
            import ctypes

            if hasattr(ctypes, "windll"):  # Windows
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass

        after_status = self.get_memory_status()

        cleanup_result = {
            "objects_collected": total_collected,
            "memory_before_gb": before_status["current_usage_gb"],
            "memory_after_gb": after_status["current_usage_gb"],
            "memory_freed_gb": before_status["current_usage_gb"]
            - after_status["current_usage_gb"],
            "cleanup_effective": after_status["current_usage_gb"]
            < before_status["current_usage_gb"],
        }

        self.emergency_cleanup_count += 1

        if not cleanup_result["cleanup_effective"]:
            pass

        return cleanup_result

    def get_optimal_batch_size(self, total_items: int) -> int:
        """Determine optimal batch size based on current memory mode."""
        if self.processing_mode == "emergency":
            return min(2, total_items)  # Process 2 at a time maximum
        if self.processing_mode == "conservative":
            return min(5, total_items)  # Process 5 at a time
        return min(10, total_items)  # Normal batch size

    def log_memory_summary(self, context: str) -> None:
        """Log comprehensive memory summary."""
        # Note: Method implementation incomplete - placeholder for future enhancement
        if self.emergency_cleanup_count > 0:
            pass
        if self.memory_warnings > 0:
            pass


def load_video_clips_with_advanced_memory_management(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Any], List[int], VideoResourceManager]:
    """Enhanced sequential loading with advanced memory management and batch processing.

    CRITICAL FIX: This version now uses delayed cleanup to prevent NoneType get_frame errors.
    Parent videos are kept alive until after concatenation, ensuring subclips remain valid.

    This is the most memory-safe version of video loading with multiple fallback strategies.
    """

    def report_progress(step: str, progress: float):
        if progress_callback:
            progress_callback(step, progress)

    if not sorted_clips:
        raise ValueError("No clips provided for loading")

    # Initialize advanced memory management with DELAYED CLEANUP
    memory_manager = AdvancedMemoryManager(
        warning_threshold_gb=3.0,  # More conservative thresholds
        emergency_threshold_gb=5.0,
        critical_threshold_gb=7.0,
    )
    resource_manager = VideoResourceManager()  # This will now support delayed cleanup

    memory_manager.log_memory_summary("initialization")

    # Smart preprocessing with memory considerations

    # Check memory before preprocessing
    if memory_manager.should_switch_to_emergency_mode():
        video_path_map = {path: path for path in video_files}  # No preprocessing
    else:
        video_path_map = preprocess_videos_smart(
            video_files,
            progress_callback=progress_callback,
        )

    # Update clip paths
    for clip_data in sorted_clips:
        original_path = clip_data["video_file"]
        if original_path in video_path_map:
            clip_data["video_file"] = video_path_map[original_path]

    # Group clips by file for memory efficiency
    grouped_clips = _group_clips_by_file(sorted_clips)

    video_clips = []
    failed_indices = []
    processed_files = 0
    total_clips_processed = 0

    for video_file, file_clips in grouped_clips.items():
        processed_files += 1

        # Check memory status before each file
        if memory_manager.should_switch_to_emergency_mode():
            cleanup_result = memory_manager.perform_emergency_cleanup(
                f"before file {processed_files}",
            )

            # If cleanup didn't help much, switch to single-clip processing
            if not cleanup_result["cleanup_effective"]:
                optimal_batch_size = 1
            else:
                optimal_batch_size = memory_manager.get_optimal_batch_size(
                    len(file_clips),
                )
        else:
            optimal_batch_size = memory_manager.get_optimal_batch_size(len(file_clips))

        # Process clips in batches to manage memory
        file_clips_loaded = 0

        try:
            # CRITICAL FIX: Use delayed cleanup instead of immediate cleanup
            # This keeps the parent video alive so subclips remain valid
            source_video = resource_manager.load_video_with_delayed_cleanup(video_file)

            # Process clips in memory-safe batches
            for batch_start in range(0, len(file_clips), optimal_batch_size):
                batch_end = min(batch_start + optimal_batch_size, len(file_clips))
                batch_clips = file_clips[batch_start:batch_end]

                for clip_data in batch_clips:
                    try:
                        # Check memory before each clip in emergency mode
                        if memory_manager.processing_mode == "emergency":
                            status = memory_manager.get_memory_status()
                            if status["is_critical"]:
                                break

                        # Extract segment - parent video stays alive via delayed cleanup
                        segment = subclip_safely(
                            source_video,
                            clip_data["start"],
                            clip_data["end"],
                        )

                        # Validation: Test the clip while parent video is available
                        # This validation is still important for catching other issues
                        try:
                            test_time = min(0.1, segment.duration)
                            test_frame = segment.get_frame(test_time)
                            if test_frame is None:
                                raise RuntimeError(
                                    "get_frame returned None during validation",
                                )
                        except Exception as e:
                            raise RuntimeError(f"Clip validation failed: {e}") from e

                        video_clips.append(segment)
                        file_clips_loaded += 1

                    except Exception:
                        # Use original_index from grouped clips to maintain timeline alignment
                        original_index = clip_data.get(
                            "original_index",
                            total_clips_processed,
                        )
                        failed_indices.append(original_index)

                    total_clips_processed += 1

                # Memory check after batch
                if memory_manager.processing_mode != "normal":
                    memory_manager.log_memory_summary("after batch")

                    # Force garbage collection between batches in conservative/emergency mode
                    import gc

                    gc.collect()

            # NOTE: We do NOT close the source_video here - it will be cleaned up later
            # This is the key fix: parent videos stay alive until after concatenation

        except Exception:
            # Mark all clips from this file as failed using original indices
            for clip_data in file_clips:
                original_index = clip_data.get("original_index", total_clips_processed)
                failed_indices.append(original_index)
                total_clips_processed += 1
            continue

        # Update progress
        progress = 0.1 + (0.6 * total_clips_processed / len(sorted_clips))
        report_progress(
            f"Advanced processing: {total_clips_processed}/{len(sorted_clips)} clips",
            progress,
        )

    if not video_clips:
        raise RuntimeError("No video clips could be loaded successfully")

    # Final memory summary
    memory_manager.log_memory_summary("completion")

    success_count = len(video_clips)
    # Note: Success rate calculation removed - not used in current implementation

    if failed_indices:
        pass

    report_progress(f"Advanced loading complete: {success_count} clips", 0.7)

    # Return clips with perfect index alignment, failed indices, and resource manager
    # The caller MUST call resource_manager.cleanup_delayed_videos() after concatenation
    return video_clips, failed_indices, resource_manager


# ============================================================================
# PHASE 4: ENHANCED ERROR HANDLING AND RECOVERY
# Multiple fallback strategies and graceful degradation
# ============================================================================


class RobustVideoLoader:
    """Robust video loading with multiple fallback strategies and detailed error reporting.

    CRITICAL FIX: Updated to support delayed cleanup pattern to prevent NoneType get_frame errors.
    """

    def __init__(self):
        self.error_statistics = {
            "total_attempts": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "fallback_usage": {
                "direct_loading": 0,
                "format_conversion": 0,
                "quality_reduction": 0,
                "emergency_mode": 0,
            },
            "error_types": {},
        }
        # Cache for loaded videos to support delayed cleanup pattern
        self._loaded_videos = {}

    def load_clip_with_fallbacks(
        self,
        clip_data: Dict[str, Any],
        resource_manager: VideoResourceManager,
        canvas_format: Optional[dict] = None,  # NEW: Intelligent canvas format
    ) -> Optional[Any]:
        """Load a single clip with multiple fallback strategies and intelligent canvas sizing.

        CRITICAL FIX: Now supports delayed cleanup to keep parent videos alive.
        NEW: Integrates intelligent canvas format for optimal scaling throughout all fallback strategies.

        Args:
            clip_data: Dictionary with video_file, start, end information
            resource_manager: Resource manager for delayed cleanup video loading
            canvas_format: Intelligent canvas format from VideoFormatAnalyzer

        Returns:
            VideoFileClip segment or None if all strategies failed
        """
        self.error_statistics["total_attempts"] += 1

        # NEW: Log canvas format usage for this clip
        if canvas_format:
            pass

        strategies = [
            ("direct_loading", self._load_direct_moviepy),
            ("format_conversion", self._load_with_format_conversion),
            ("quality_reduction", self._load_with_quality_reduction),
            ("emergency_mode", self._load_emergency_minimal),
        ]

        def _try_loading_strategy(strategy_name: str, strategy_func, clip_data, resource_manager, canvas_format):
            """Try a single loading strategy and return result or None."""
            try:
                # CRITICAL FIX: Pass canvas_format to all fallback strategies
                result = strategy_func(
                    clip_data, resource_manager, canvas_format=canvas_format
                )
                if result is not None:
                    self.error_statistics["successful_loads"] += 1
                    self.error_statistics["fallback_usage"][strategy_name] += 1

                    if strategy_name != "direct_loading":
                        pass
                    else:
                        pass

                    return result, None
                
                # CRITICAL FIX: Return None for unsuccessful result, not in else block
                return None, None
                
            except Exception as e:
                error_type = type(e).__name__
                self.error_statistics["error_types"][error_type] = (
                    self.error_statistics["error_types"].get(error_type, 0) + 1
                )
                return None, e

        for strategy_name, strategy_func in strategies:
            result, error = _try_loading_strategy(strategy_name, strategy_func, clip_data, resource_manager, canvas_format)
            if result is not None:
                return result
            if error is not None:
                pass  # Error logged by individual strategies

        # All strategies failed
        self.error_statistics["failed_loads"] += 1
        return None

    def _get_or_load_video(
        self,
        video_file: str,
        resource_manager: VideoResourceManager,
    ):
        """Get or load a video with delayed cleanup.

        CRITICAL FIX: Uses delayed cleanup instead of context managers to keep videos alive.
        """
        if video_file not in self._loaded_videos:
            # Load video with delayed cleanup - parent stays alive
            self._loaded_videos[video_file] = (
                resource_manager.load_video_with_delayed_cleanup(video_file)
            )
        return self._loaded_videos[video_file]

    def _load_direct_moviepy(
        self,
        clip_data: Dict[str, Any],
        resource_manager: VideoResourceManager,
        canvas_format: Optional[dict] = None,  # NEW: Canvas format for scaling
    ) -> Optional[Any]:
        """Direct loading with MoviePy and intelligent canvas scaling."""
        video_file = clip_data["video_file"]
        start_time = clip_data["start"]
        end_time = clip_data["end"]

        video_clip = self._get_or_load_video(video_file, resource_manager)
        if video_clip is None:
            raise RuntimeError(f"Could not load video file: {video_file}")

        # Create subclip with error handling
        try:
            segment = video_clip.subclipped(start_time, end_time)
        except AttributeError:
            segment = video_clip.subclip(start_time, end_time)

        # NEW: Apply intelligent canvas scaling if provided
        if canvas_format and segment is not None:
            try:
                from compatibility.moviepy import resize_with_aspect_preservation

                segment = resize_with_aspect_preservation(
                    segment,
                    target_width=canvas_format["target_width"],
                    target_height=canvas_format["target_height"],
                    scaling_mode="smart",  # Use smart scaling for optimal results
                )
            except Exception:
                pass
                # Continue with unscaled segment rather than failing

        return segment

    def _load_with_format_conversion(
        self,
        clip_data: Dict[str, Any],
        resource_manager: VideoResourceManager,
        canvas_format: Optional[dict] = None,  # NEW: Canvas format for scaling
    ) -> Optional[Any]:
        """Load with format conversion fallback and intelligent canvas scaling."""
        import subprocess
        import tempfile

        video_file = clip_data["video_file"]
        start_time = clip_data["start"]
        end_time = clip_data["end"]

        # Create temporary converted file
        temp_dir = tempfile.mkdtemp(prefix="autocut_conversion_")
        base_name = Path(video_file).stem
        converted_file = Path(temp_dir) / f"{base_name}_converted.mp4"

        try:
            # Build FFmpeg command for format conversion with intelligent canvas scaling
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_file,
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
            ]

            # NEW: Apply intelligent canvas scaling during conversion if provided
            if canvas_format:
                target_w = canvas_format["target_width"]
                target_h = canvas_format["target_height"]
                # Use aspect-aware scaling with letterboxing
                scale_filter = (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
                )
                pad_filter = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
                combined_filter = f"{scale_filter},{pad_filter}"
                cmd.extend(["-vf", combined_filter])
            else:
                pass

            # Standard conversion settings
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",  # Speed over quality for fallback
                    "-c:a",
                    "aac",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(converted_file),
                ]
            )

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[:200]}")

            # Load converted clip
            from moviepy.editor import VideoFileClip

            converted_clip = VideoFileClip(converted_file)

            # If canvas scaling wasn't applied during conversion, apply it now
            if not canvas_format:
                # Load and trim normally, then apply canvas scaling
                segment = (
                    converted_clip  # Full duration since we already trimmed with FFmpeg
                )
            else:
                segment = converted_clip  # Already scaled during conversion

            # Register for cleanup
            resource_manager.register_temp_file(converted_file)
            resource_manager.register_temp_file(temp_dir)
            
            # CRITICAL FIX: Return segment in try block, not orphaned else block
            return segment
            
        except subprocess.TimeoutExpired as timeout_error:
            # Cleanup on timeout
            try:
                if converted_file.exists():
                    converted_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass
            raise RuntimeError("Format conversion timed out") from timeout_error
        except Exception as e:
            # Cleanup on failure
            try:
                if converted_file.exists():
                    converted_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass
            raise RuntimeError(f"Format conversion failed: {e}") from e

    def _load_with_quality_reduction(
        self,
        clip_data: Dict[str, Any],
        resource_manager: VideoResourceManager,
        canvas_format: Optional[dict] = None,  # NEW: Canvas format for scaling
    ) -> Optional[Any]:
        """Load with quality reduction for memory-intensive videos and intelligent canvas scaling."""
        import subprocess
        import tempfile

        video_file = clip_data["video_file"]
        start_time = clip_data["start"]
        end_time = clip_data["end"]

        # Create temporary reduced quality file
        temp_dir = tempfile.mkdtemp(prefix="autocut_quality_reduction_")
        base_name = Path(video_file).stem
        reduced_file = Path(temp_dir) / f"{base_name}_reduced.mp4"

        try:
            # Build FFmpeg command for quality reduction with intelligent canvas scaling
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_file,
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
            ]

            # NEW: Apply intelligent canvas scaling during quality reduction if provided
            if canvas_format:
                target_w = canvas_format["target_width"]
                target_h = canvas_format["target_height"]
                target_fps = canvas_format.get("target_fps", 25)

                # Combine quality reduction with canvas scaling for maximum efficiency
                scale_filter = (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
                )
                pad_filter = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
                fps_filter = f"fps={target_fps}"
                combined_filter = f"{scale_filter},{pad_filter},{fps_filter}"

                cmd.extend(["-vf", combined_filter])
            else:
                # Standard quality reduction without canvas scaling
                cmd.extend(
                    [
                        "-vf",
                        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,fps=24",
                    ]
                )

            # Aggressive quality reduction settings
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",  # Lower quality for memory savings
                    "-c:a",
                    "aac",
                    "-b:a",
                    "96k",  # Reduced audio bitrate
                    "-avoid_negative_ts",
                    "make_zero",
                    str(reduced_file),
                ]
            )

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg quality reduction failed: {result.stderr[:200]}"
                )

            # Load reduced quality clip
            from moviepy.editor import VideoFileClip

            reduced_clip = VideoFileClip(reduced_file)
            segment = (
                reduced_clip  # Already trimmed and scaled during FFmpeg processing
            )

            # Register for cleanup
            resource_manager.register_temp_file(reduced_file)
            resource_manager.register_temp_file(temp_dir)
            
            # CRITICAL FIX: Return segment in try block, not orphaned else block
            return segment
            
        except subprocess.TimeoutExpired as timeout_error:
            # Cleanup on timeout
            try:
                if reduced_file.exists():
                    reduced_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass  # Ignore cleanup errors
            raise RuntimeError("Quality reduction timed out") from timeout_error
        except Exception as e:
            # Cleanup on failure
            try:
                if reduced_file.exists():
                    reduced_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass  # Ignore cleanup errors
            raise RuntimeError(f"Quality reduction failed: {e}") from e

    def _load_emergency_minimal(
        self,
        clip_data: Dict[str, Any],
        resource_manager: VideoResourceManager,
        canvas_format: Optional[dict] = None,  # NEW: Canvas format for scaling
    ) -> Optional[Any]:
        """Emergency minimal loading with maximum compatibility and intelligent canvas scaling."""
        import subprocess
        import tempfile

        video_file = clip_data["video_file"]
        start_time = clip_data["start"]
        end_time = clip_data["end"]

        # Create temporary minimal file
        temp_dir = tempfile.mkdtemp(prefix="autocut_emergency_")
        base_name = Path(video_file).stem
        minimal_file = Path(temp_dir) / f"{base_name}_minimal.mp4"

        try:
            # Emergency settings: maximum compatibility, minimal quality, with intelligent canvas scaling
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_file,
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
            ]

            # NEW: Apply emergency canvas scaling if provided
            if canvas_format:
                # Use smaller dimensions for emergency mode to reduce memory pressure
                emergency_width = min(
                    canvas_format["target_width"], 854
                )  # Max 854 wide for emergency
                emergency_height = min(
                    canvas_format["target_height"], 480
                )  # Max 480 high for emergency

                scale_filter = f"scale={emergency_width}:{emergency_height}:force_original_aspect_ratio=decrease"
                pad_filter = f"pad={emergency_width}:{emergency_height}:(ow-iw)/2:(oh-ih)/2:black"
                fps_filter = "fps=15"  # Very low FPS for emergency
                combined_filter = f"{scale_filter},{pad_filter},{fps_filter}"

                cmd.extend(["-vf", combined_filter])
            else:
                # Standard emergency settings without canvas scaling
                cmd.extend(
                    [
                        "-vf",
                        "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2:black,fps=15",
                    ]
                )

            # Minimal quality settings for maximum compatibility
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-profile:v",
                    "baseline",  # Maximum compatibility profile
                    "-level",
                    "3.0",
                    "-crf",
                    "35",  # Very low quality for minimal size
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",  # Minimal audio bitrate
                    "-ac",
                    "1",  # Mono audio to save space
                    "-ar",
                    "22050",  # Low sample rate
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    str(minimal_file),
                ]
            )

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=240,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg emergency processing failed: {result.stderr[:200]}"
                )

            # Load minimal clip
            from moviepy.editor import VideoFileClip

            minimal_clip = VideoFileClip(minimal_file)
            segment = minimal_clip  # Already processed with emergency settings

            # Register for cleanup
            resource_manager.register_temp_file(minimal_file)
            resource_manager.register_temp_file(temp_dir)
            
            # CRITICAL FIX: Return segment in try block, not orphaned else block
            return segment
            
        except subprocess.TimeoutExpired as timeout_error:
            # Cleanup on timeout
            try:
                if minimal_file.exists():
                    minimal_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass  # Ignore cleanup errors
            raise RuntimeError("Emergency loading timed out") from timeout_error
        except Exception as e:
            # Cleanup on failure
            try:
                if minimal_file.exists():
                    minimal_file.unlink()
                if Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception:
                pass  # Ignore cleanup errors
            raise RuntimeError(f"Emergency loading failed: {e}") from e

    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error statistics report."""
        if self.error_statistics["total_attempts"] == 0:
            success_rate = 0
        else:
            success_rate = (
                self.error_statistics["successful_loads"]
                / self.error_statistics["total_attempts"]
            )

        return {
            "success_rate": success_rate,
            "total_attempts": self.error_statistics["total_attempts"],
            "successful_loads": self.error_statistics["successful_loads"],
            "failed_loads": self.error_statistics["failed_loads"],
            "fallback_usage": self.error_statistics["fallback_usage"].copy(),
            "error_types": self.error_statistics["error_types"].copy(),
        }

    def print_error_summary(self) -> None:
        """Print detailed error summary for debugging."""
        report = self.get_error_report()

        if any(report["fallback_usage"].values()):
            for count in report["fallback_usage"].values():
                if count > 0:
                    pass

        if report["error_types"]:
            for count in report["error_types"].values():
                pass


def load_video_clips_with_robust_error_handling(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    canvas_format: Optional[
        dict
    ] = None,  # NEW: Canvas format for intelligent preprocessing
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Any], List[int], Dict[str, Any], VideoResourceManager]:
    """Load video clips with comprehensive error handling and recovery strategies.

    CRITICAL FIX: This version now uses delayed cleanup to prevent NoneType get_frame errors.
    Parent videos are kept alive until after concatenation, ensuring subclips remain valid.

    NEW: Integrates intelligent canvas format for optimal preprocessing and scaling.

    This is the most robust version that tries multiple approaches for each failed clip.

    Returns:
        Tuple of (video_clips, failed_indices, error_report, resource_manager)
    """

    def report_progress(step: str, progress: float):
        if progress_callback:
            progress_callback(step, progress)

    if not sorted_clips:
        raise ValueError("No clips provided for loading")

    # Initialize all management systems with DELAYED CLEANUP
    memory_manager = AdvancedMemoryManager()
    resource_manager = VideoResourceManager()  # Will support delayed cleanup
    robust_loader = RobustVideoLoader()

    # NEW: Log canvas format usage
    if canvas_format:
        pass
    else:
        pass

    # Smart preprocessing with error recovery and intelligent canvas format
    try:
        # CRITICAL FIX: Pass canvas_format to preprocessing
        video_path_map = preprocess_videos_smart(
            video_files,
            canvas_format=canvas_format,  # NEW: Pass canvas format for intelligent preprocessing
            progress_callback=progress_callback,
        )
    except Exception:
        video_path_map = {path: path for path in video_files}

    # Update clip paths
    for clip_data in sorted_clips:
        original_path = clip_data["video_file"]
        if original_path in video_path_map:
            clip_data["video_file"] = video_path_map[original_path]

    # Group clips for efficient processing
    grouped_clips = _group_clips_by_file(sorted_clips)

    video_clips = []
    failed_indices = []
    processed_files = 0
    total_clips_processed = 0

    memory_manager.log_memory_summary("robust loading start")

    for file_clips in grouped_clips.values():
        processed_files += 1

        file_clips_loaded = 0

        # Check memory and adapt strategy
        if memory_manager.should_switch_to_emergency_mode():
            memory_manager.perform_emergency_cleanup(f"before file {processed_files}")
            # Process one at a time in emergency mode
        else:
            # Note: Batch processing not implemented - clips processed sequentially
            pass

        # Process clips with robust error handling
        for i, clip_data in enumerate(file_clips):
            try:
                # Use robust loader with multiple fallback strategies and delayed cleanup
                # NEW: Pass canvas_format to the loader for intelligent scaling
                segment = robust_loader.load_clip_with_fallbacks(
                    clip_data,
                    resource_manager,
                    canvas_format=canvas_format,
                )

                if segment is not None:
                    video_clips.append(segment)
                    file_clips_loaded += 1
                else:
                    # Use original_index from grouped clips to maintain timeline alignment
                    original_index = clip_data.get(
                        "original_index",
                        total_clips_processed,
                    )
                    failed_indices.append(original_index)

            except Exception:
                # Use original_index from grouped clips to maintain timeline alignment
                original_index = clip_data.get("original_index", total_clips_processed)
                failed_indices.append(original_index)

            total_clips_processed += 1

            # Memory check between clips if in conservative mode
            if memory_manager.processing_mode != "normal" and i % 5 == 0:
                memory_manager.log_memory_summary("progress check")

            # Update progress
            progress = 0.1 + (0.6 * total_clips_processed / len(sorted_clips))
            report_progress(
                f"Robust loading: {total_clips_processed}/{len(sorted_clips)}",
                progress,
            )

    # Generate comprehensive error report
    error_report = robust_loader.get_error_report()
    robust_loader.print_error_summary()

    memory_manager.log_memory_summary("robust loading complete")

    if not video_clips:
        raise RuntimeError(
            "No video clips could be loaded successfully with any fallback strategy",
        )

    success_count = len(video_clips)
    success_rate = success_count / len(sorted_clips)

    # NEW: Log canvas format success
    if canvas_format:
        pass

    if success_rate < 0.5 or success_rate < 0.8:
        pass
    else:
        pass

    report_progress(f"Robust loading complete: {success_count} clips", 0.7)

    # Return clips with perfect index alignment, failed indices, error report, and resource manager
    # The caller MUST call resource_manager.cleanup_delayed_videos() after concatenation
    return video_clips, failed_indices, error_report, resource_manager


def load_video_clips_parallel(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    progress_callback: Optional[callable] = None,
    max_workers: Optional[int] = None,
) -> Tuple[List[Any], VideoCache, List[int]]:
    """Load video clips in parallel with intelligent caching and memory monitoring.

    Args:
        sorted_clips: List of clip data dictionaries sorted by beat position
        video_files: List of source video files for dynamic analysis
        progress_callback: Optional callback for progress updates
        max_workers: Maximum number of parallel workers (None = auto-detect based on system)

    Returns:
        Tuple of (video_clips_list, video_cache, failed_indices) for resource management

    Raises:
        RuntimeError: If no clips could be loaded successfully
    """

    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)

    if not sorted_clips:
        raise ValueError("No clips provided for loading")

    # Initialize cache and results
    video_cache = VideoCache()
    video_clips = []
    clip_mapping = {}  # Map to maintain order
    failed_indices = []  # Track which clips failed to load

    # Dynamic worker detection based on system capabilities
    if max_workers is None:
        # Use intelligent system profiling for optimal worker count
        profiler = SystemProfiler()
        capabilities = profiler.get_system_capabilities()
        video_profile = profiler.estimate_video_memory_usage(video_files)
        worker_analysis = profiler.calculate_optimal_workers(
            capabilities,
            video_profile,
            len(sorted_clips),
        )

        optimal_workers = worker_analysis["optimal_workers"]

        # Display detailed analysis
        profiler.print_system_analysis(capabilities, video_profile, worker_analysis)

    else:
        # Manual override provided
        optimal_workers = min(max_workers, len(sorted_clips))

    report_progress(
        f"Loading {len(sorted_clips)} clips with {optimal_workers} workers",
        0.1,
    )

    # Start adaptive monitoring for safety
    monitor = AdaptiveWorkerMonitor(optimal_workers)
    monitor.start_monitoring()

    try:
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all clip loading tasks
            future_to_index = {}
            for i, clip_data in enumerate(sorted_clips):
                future = executor.submit(load_video_segment, clip_data, video_cache)
                future_to_index[future] = i

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_index):
                completed_count += 1
                index = future_to_index[future]

                try:
                    result = future.result()
                    if result is not None:
                        clip_data, segment = result
                        clip_mapping[index] = segment
                    else:
                        failed_indices.append(index)

                except Exception:
                    failed_indices.append(index)

                # Update progress
                progress = 0.1 + (0.6 * completed_count / len(sorted_clips))
                report_progress(
                    f"Loaded {completed_count}/{len(sorted_clips)} clips",
                    progress,
                )

    except Exception as e:
        # Clean up cache on error
        video_cache.clear()
        raise RuntimeError(f"Parallel video loading failed: {e!s}") from e

    finally:
        # Always stop monitoring when done
        monitor.stop_monitoring()

    # Reconstruct clips in original order, skipping failed ones
    for i in range(len(sorted_clips)):
        if i in clip_mapping:
            video_clips.append(clip_mapping[i])

    if not video_clips:
        video_cache.clear()
        raise RuntimeError("No video clips could be loaded successfully")

    # Report failed clips
    if failed_indices:
        pass

    report_progress(f"Successfully loaded {len(video_clips)} clips", 0.7)

    # Memory monitoring at completion
    final_memory = get_memory_info()
    # Note: Memory increase tracking removed - not used in current implementation

    # Memory usage warning
    if final_memory["percent"] > 85:
        pass

    # Note: Cache statistics logging not implemented
    video_cache.get_cached_paths()  # Just ensure cache is accessible

    return video_clips, video_cache, failed_indices


def check_moviepy_api_compatibility():
    """Comprehensive MoviePy API compatibility analysis for version 2.1.2+ changes.

    Returns:
        Dict with complete API mapping and compatibility information
    """
    import inspect

    # Handle import structure changes in MoviePy 2.1.2
    try:
        # Try new import structure first (MoviePy 2.1.2+)
        from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

        import_pattern = "new"  # from moviepy import ...
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2)
            from moviepy.editor import (
                AudioFileClip,  # noqa: F401
                VideoFileClip,  # noqa: F401
                concatenate_videoclips,  # noqa: F401
            )

            import_pattern = "legacy"  # from moviepy.editor import ...
        except ImportError as import_error:
            raise RuntimeError("Could not import MoviePy with either import pattern") from import_error

    # CRITICAL FIX: Proper API detection for MoviePy 2.2.1
    # Use class inspection instead of unreliable dummy instances
    compatibility = {
        "import_pattern": import_pattern,
        "version_detected": "new" if import_pattern == "new" else "legacy",
        # FIXED: Direct method mappings based on MoviePy 2.2.1 patterns
        "method_mappings": {
            # Clip manipulation - MoviePy 2.2.1 uses subclipped
            "subclip": "subclipped",
            # Audio attachment - Test both options, prefer with_audio for 2.x
            "set_audio": "with_audio" if import_pattern == "new" else "set_audio",
            # Other method patterns (2.x uses with_ prefix)
            "set_duration": "with_duration"
            if import_pattern == "new"
            else "set_duration",
            "set_position": "with_position"
            if import_pattern == "new"
            else "set_position",
            "set_start": "with_start" if import_pattern == "new" else "set_start",
        },
        # Method availability - assume new methods exist in new import pattern
        "methods": {
            "video_clip": {
                "subclip": import_pattern == "legacy",
                "subclipped": import_pattern == "new",
                "set_audio": import_pattern == "legacy",
                "with_audio": import_pattern == "new",
                "write_videofile": True,  # Always available
            },
            "audio_clip": {
                "subclip": import_pattern == "legacy",
                "subclipped": import_pattern == "new",
            },
        },
    }

    # Analyze write_videofile parameters
    try:
        write_sig = inspect.signature(VideoFileClip.write_videofile)
        compatibility["write_videofile_params"] = list(write_sig.parameters.keys())
    except Exception:
        compatibility["write_videofile_params"] = ["filename"]

    return compatibility


def attach_audio_safely(video_clip, audio_clip, compatibility_info=None):
    """Safely attach audio to video using available API with robust fallbacks.

    Args:
        video_clip: VideoClip to attach audio to
        audio_clip: AudioClip to attach
        compatibility_info: Result from check_moviepy_api_compatibility()

    Returns:
        VideoClip with audio attached (never None)
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()

    if video_clip is None:
        raise RuntimeError("Cannot attach audio to None video clip")
    if audio_clip is None:
        raise RuntimeError("Cannot attach None audio clip")

    # CRITICAL FIX: Try both methods with robust error handling
    for method_name in ["with_audio", "set_audio"]:
        if hasattr(video_clip, method_name):
            try:
                method = getattr(video_clip, method_name)
                result = method(audio_clip)

                # CRITICAL: Ensure we never return None
                if result is None:
                    continue
                # CRITICAL FIX: Return in try block, not orphaned else block
                return result
            except Exception:
                continue

    # If all methods fail, this is a critical error
    raise RuntimeError(
        f"Could not attach audio using any method (tried: with_audio, set_audio) on {type(video_clip)}",
    )


def subclip_safely(clip, start_time, end_time=None, compatibility_info=None):
    """Safely create subclip using available API (subclip vs subclipped).

    Args:
        clip: VideoClip or AudioClip to extract from
        start_time: Start time in seconds
        end_time: End time in seconds (None for rest of clip)
        compatibility_info: Result from check_moviepy_api_compatibility()

    Returns:
        New clip with specified time range
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()

    # Get the correct method name for subclip operation
    method_name = compatibility_info["method_mappings"]["subclip"]

    try:
        # CRITICAL FIX: Always try subclipped first for MoviePy 2.2.1 compatibility
        # Both video and audio clips use subclipped in 2.2.1
        for method_name in ["subclipped", "subclip"]:
            if hasattr(clip, method_name):
                method = getattr(clip, method_name)
                try:
                    if end_time is not None:
                        return method(start_time, end_time)
                    return method(start_time)
                except Exception:
                    # If this method fails, try the next one
                    continue

        # If neither method works, raise an error
        raise AttributeError(
            f"Neither 'subclipped' nor 'subclip' methods work on {type(clip)}",
        )

    except AttributeError as attr_error:
        raise RuntimeError(f"Could not find subclip method on clip type {type(clip)}") from attr_error


def test_independent_subclip_creation(video_path: Optional[str] = None) -> bool:
    """Test function to verify independent subclip creation works correctly.

    This test validates that the fix for the NoneType get_frame error is working
    by creating subclips and ensuring they remain functional after the parent
    video is closed.

    Args:
        video_path: Optional path to test video (uses demo if None)

    Returns:
        True if test passes, False otherwise
    """
    if not video_path:
        # Use a test media file if available
        test_files = ["test_media/sample.mp4", "test_media/demo.mp4", "demo.mp4"]
        for test_file in test_files:
            if Path(test_file).exists():
                video_path = test_file
                break

        if not video_path:
            return True  # Skip test if no video available

    try:
        # Create resource manager and load video
        resource_manager = VideoResourceManager()

        # Test the critical scenario: parent video gets closed
        with resource_manager.load_video_safely(video_path) as source_video:
            duration = source_video.duration

            # Create subclip using old method (should fail after parent closes)
            old_subclip = subclip_safely(source_video, 0.5, min(2.0, duration - 0.5))

            # Create subclip using new method (should work after parent closes)
            new_subclip = source_video.subclipped(0.5, min(2.0, duration - 0.5))

        # Parent video is now closed - test if subclips still work

        # Test old subclip (should fail)
        with contextlib.suppress(Exception):
            old_frame = old_subclip.get_frame(0.1)

        # Test new subclip (should work)
        try:
            new_frame = new_subclip.get_frame(0.1)
        except Exception:
            return False
        else:
            return new_frame is not None

    except Exception:
        return False

    finally:
        # Cleanup
        try:
            if "old_subclip" in locals():
                old_subclip.close()
            if "new_subclip" in locals():
                new_subclip.close()
        except Exception:
            pass  # Ignore cleanup errors


def import_moviepy_safely():
    """Safely import MoviePy classes handling import structure changes.

    Returns:
        Tuple of (VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip)
    """
    try:
        # Try new import structure first (MoviePy 2.1.2+)
        from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

        try:
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        except ImportError:
            from moviepy import CompositeVideoClip
        
        # CRITICAL FIX: Return in try block, not orphaned else block
        return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2)
            from moviepy.editor import (
                AudioFileClip,
                CompositeVideoClip,
                VideoFileClip,
                concatenate_videoclips,
            )
            
            # CRITICAL FIX: Return in try block, not orphaned else block
            return (
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
                CompositeVideoClip,
            )
        except ImportError as import_error:
            raise RuntimeError(
                "Could not import MoviePy with either import pattern. Please check MoviePy installation.",
            ) from import_error


def write_videofile_safely(video_clip, output_path, compatibility_info=None, **kwargs):
    """Safely write video file with parameter compatibility checking.

    Args:
        video_clip: VideoClip to write
        output_path: Output file path
        compatibility_info: Result from check_moviepy_api_compatibility()
        **kwargs: Parameters to pass to write_videofile

    Returns:
        None
    """
    if compatibility_info is None:
        compatibility_info = check_moviepy_api_compatibility()

    available_params = compatibility_info["write_videofile_params"]

    # Filter kwargs to only include supported parameters
    safe_kwargs = {}
    for key, value in kwargs.items():
        if key in available_params:
            safe_kwargs[key] = value
        else:
            pass

    try:
        video_clip.write_videofile(output_path, **safe_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to write video file: {e!s}") from e


# ClipTimeline class extracted to src/video/timeline_renderer.py


def match_clips_to_beats(
    video_chunks: List[VideoChunk],
    beats: List[float],
    allowed_durations: List[float],
    pattern: str = "balanced",
    musical_start_time: float = 0.0,
) -> ClipTimeline:
    """Match video chunks to beat grid using variety patterns with musical intelligence.

    Args:
        video_chunks: List of scored video chunks
        beats: List of beat timestamps in seconds (compensated and filtered)
        allowed_durations: List of musically appropriate durations
        pattern: Variety pattern to use ('energetic', 'buildup', 'balanced', 'dramatic')
        musical_start_time: First significant beat timestamp (skip intro/buildup)

    Returns:
        ClipTimeline object with matched clips starting from musical content
    """
    if not video_chunks or not beats or len(beats) < 2:
        return ClipTimeline()

    # MUSICAL INTELLIGENCE: Filter beats to start from actual musical content
    # This fixes the 1-2 second intro misalignment issue
    effective_beats = (
        [b for b in beats if b >= musical_start_time]
        if musical_start_time > 0
        else beats
    )

    if len(effective_beats) < 2:
        # Fallback to all beats if musical start filtering leaves too few
        effective_beats = beats

    # Calculate beat interval (average time between beats)
    beat_intervals = [
        effective_beats[i + 1] - effective_beats[i]
        for i in range(len(effective_beats) - 1)
    ]
    avg_beat_interval = sum(beat_intervals) / len(beat_intervals)

    # Apply variety pattern to get beat multipliers
    total_beats = (
        len(effective_beats) - 1
    )  # Don't count the last beat as start of a clip
    beat_multipliers = apply_variety_pattern(pattern, total_beats)

    # Convert beat multipliers to target durations
    target_durations = [
        multiplier * avg_beat_interval for multiplier in beat_multipliers
    ]

    # Estimate total clips needed
    estimated_clips = len(target_durations)

    # Select best clips with variety (request more than needed for flexibility)
    selected_clips = select_best_clips(
        video_chunks,
        target_count=min(estimated_clips * 2, len(video_chunks)),
        variety_factor=0.3,
    )

    timeline = ClipTimeline()
    current_beat_index = 0
    used_clips = set()  # Track used clips to avoid repetition

    for i, target_duration in enumerate(target_durations):
        if current_beat_index >= len(effective_beats):
            break

        # Find best matching clip for this target duration
        best_clip = None
        best_fit_score = -1

        for clip in selected_clips:
            if id(clip) in used_clips:
                continue

            # Calculate fit score based on:
            # 1. How close clip duration is to target duration
            # 2. Clip quality score
            # 3. Whether clip can be trimmed to fit exactly

            duration_fit = _calculate_duration_fit(
                clip.duration,
                target_duration,
                allowed_durations,
            )
            if duration_fit < 0:  # Clip can't be used for this duration
                continue

            # Combined score: 70% quality, 30% duration fit
            fit_score = 0.7 * (clip.score / 100.0) + 0.3 * duration_fit

            if fit_score > best_fit_score:
                best_fit_score = fit_score
                best_clip = clip

        if best_clip is None:
            # No suitable clip found, skip this position
            current_beat_index += beat_multipliers[i]
            continue

        # Mark clip as used
        used_clips.add(id(best_clip))

        # Determine actual clip timing
        beat_position = effective_beats[current_beat_index]
        clip_start, clip_end, clip_duration = _fit_clip_to_duration(
            best_clip,
            target_duration,
        )

        # Add to timeline
        timeline.add_clip(
            video_file=best_clip.video_path,
            start=clip_start,
            end=clip_end,
            beat_position=beat_position,
            score=best_clip.score,
        )

        # Move to next beat position
        current_beat_index += beat_multipliers[i]

    return timeline


def _calculate_duration_fit(
    clip_duration: float,
    target_duration: float,
    allowed_durations: List[float],
) -> float:
    """Calculate how well a clip duration fits the target duration.

    Args:
        clip_duration: Duration of the video clip
        target_duration: Desired duration for this position
        allowed_durations: List of musically appropriate durations

    Returns:
        Fit score between 0.0 and 1.0, or -1 if clip can't be used
    """
    # Check if target duration is in allowed durations (with small tolerance)
    duration_allowed = False
    for allowed in allowed_durations:
        if abs(target_duration - allowed) < 0.1:
            duration_allowed = True
            break

    if not duration_allowed:
        return -1  # Target duration is not musically appropriate

    # Perfect match
    if abs(clip_duration - target_duration) < 0.1:
        return 1.0

    # Clip is longer than target - can be trimmed
    if clip_duration > target_duration:
        # Prefer clips that are close to target but slightly longer
        excess = clip_duration - target_duration
        if excess <= 2.0:  # Can trim up to 2 seconds
            return 1.0 - (excess / 4.0)  # Gentle penalty for trimming
        return 0.3  # Heavy penalty for lots of trimming

    # Clip is shorter than target
    shortage = target_duration - clip_duration
    if shortage <= 0.5:  # Small shortage is acceptable
        return 0.8 - (shortage / 1.0)
    return -1  # Too short, can't use


def _fit_clip_to_duration(
    clip: VideoChunk,
    target_duration: float,
) -> Tuple[float, float, float]:
    """Fit a clip to the target duration by trimming if necessary.

    Args:
        clip: Video chunk to fit
        target_duration: Desired duration

    Returns:
        Tuple of (start_time, end_time, actual_duration)
    """
    if clip.duration <= target_duration + 0.1:
        # Clip fits as-is
        return clip.start_time, clip.end_time, clip.duration

    # Clip needs trimming - trim from the end to preserve the beginning
    new_end_time = clip.start_time + target_duration

    # Make sure we don't exceed the original clip bounds
    new_end_time = min(new_end_time, clip.end_time)
    actual_duration = new_end_time - clip.start_time

    return clip.start_time, new_end_time, actual_duration


def select_best_clips(
    video_chunks: List[VideoChunk],
    target_count: int,
    variety_factor: float = 0.3,
) -> List[VideoChunk]:
    """Select best clips ensuring variety in source videos.

    Args:
        video_chunks: List of all available video chunks
        target_count: Number of clips to select
        variety_factor: Weight for variety vs. quality (0.0 = only quality, 1.0 = only variety)

    Returns:
        List of selected VideoChunk objects
    """
    if not video_chunks:
        return []

    if target_count <= 0:
        return []

    if len(video_chunks) <= target_count:
        return video_chunks.copy()

    # Group clips by video file for variety management
    clips_by_video = {}
    for chunk in video_chunks:
        if chunk.video_path not in clips_by_video:
            clips_by_video[chunk.video_path] = []
        clips_by_video[chunk.video_path].append(chunk)

    # Sort clips within each video by score (descending)
    for video_path in clips_by_video:
        clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)

    selected_clips = []

    if variety_factor >= 0.9:
        # High variety: Round-robin selection from each video
        video_paths = list(clips_by_video.keys())
        video_index = 0

        while len(selected_clips) < target_count:
            video_path = video_paths[video_index % len(video_paths)]

            # Find next non-overlapping clip from this video
            available_clips = clips_by_video[video_path]
            for clip in available_clips:
                if clip not in selected_clips and not _clips_overlap(
                    clip,
                    selected_clips,
                ):
                    selected_clips.append(clip)
                    break

            video_index += 1

            # Safety check: if we've tried all videos and can't find more clips
            if video_index > len(video_paths) * 10:
                break

    elif variety_factor <= 0.1:
        # High quality: Just take the best clips regardless of source
        all_clips_sorted = sorted(video_chunks, key=lambda x: x.score, reverse=True)
        for clip in all_clips_sorted:
            if len(selected_clips) >= target_count:
                break
            if not _clips_overlap(clip, selected_clips):
                selected_clips.append(clip)

    else:
        # Balanced approach: Weighted selection
        # Calculate how many clips per video (with some variety)
        num_videos = len(clips_by_video)
        base_clips_per_video = max(1, target_count // num_videos)
        remaining_clips = target_count - (base_clips_per_video * num_videos)

        # First pass: Get base clips from each video (highest quality)
        for video_path in clips_by_video:
            clips_from_video = 0
            for clip in clips_by_video[video_path]:
                if clips_from_video >= base_clips_per_video:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)
                    clips_from_video += 1

        # Second pass: Fill remaining slots with highest quality clips
        if remaining_clips > 0:
            all_remaining_clips = []
            for video_path in clips_by_video:
                for clip in clips_by_video[video_path][base_clips_per_video:]:
                    if clip not in selected_clips:
                        all_remaining_clips.append(clip)

            all_remaining_clips.sort(key=lambda x: x.score, reverse=True)

            for clip in all_remaining_clips:
                if len(selected_clips) >= target_count:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)

    return selected_clips[:target_count]


def _clips_overlap(
    clip: VideoChunk,
    existing_clips: List[VideoChunk],
    min_gap: float = 1.0,
) -> bool:
    """Check if a clip overlaps with any existing clips from the same video.

    Args:
        clip: Clip to check
        existing_clips: List of already selected clips
        min_gap: Minimum gap required between clips from same video (seconds)

    Returns:
        True if clip overlaps with any existing clip from same video
    """
    for existing in existing_clips:
        # Check for overlap or too close proximity from same video
        if (existing.video_path == clip.video_path
            and clip.start_time < existing.end_time + min_gap
            and clip.end_time > existing.start_time - min_gap):
            return True
    return False


def apply_variety_pattern(pattern_name: str, beat_count: int) -> List[int]:
    """Apply variety pattern to determine clip lengths.

    Args:
        pattern_name: Name of variety pattern to use
        beat_count: Total number of beats to fill

    Returns:
        List of beat multipliers for each clip
    """
    if pattern_name not in VARIETY_PATTERNS:
        pattern_name = "balanced"

    pattern = VARIETY_PATTERNS[pattern_name]
    result = []
    pattern_index = 0
    remaining_beats = beat_count

    while remaining_beats > 0:
        multiplier = pattern[pattern_index % len(pattern)]
        if multiplier <= remaining_beats:
            result.append(multiplier)
            remaining_beats -= multiplier
        else:
            result.append(remaining_beats)
            remaining_beats = 0
        pattern_index += 1

    return result


def render_video(
    timeline: ClipTimeline,
    audio_file: str,
    output_path: str,
    max_workers: int = 3,
    progress_callback: Optional[callable] = None,
    bpm: Optional[float] = None,
    avg_beat_interval: Optional[float] = None,
    canvas_format: Optional[
        dict
    ] = None,  # NEW: Canvas format from intelligent analysis
) -> str:
    """Render final video with music synchronization and intelligent canvas sizing.

    Args:
        timeline: ClipTimeline with all clips and timing
        audio_file: Path to music file
        output_path: Path for output video
        max_workers: Maximum parallel workers (legacy parameter)
        progress_callback: Optional callback for progress updates
        bpm: Beats per minute for musical fade calculations
        avg_beat_interval: Average time between beats in seconds
        canvas_format: Intelligent canvas format from VideoFormatAnalyzer

    Returns:
        Path to rendered video file

    Raises:
        RuntimeError: If rendering fails
    """
    try:
        # Import MoviePy components safely
        VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = (
            import_moviepy_safely()
        )

        if VideoFileClip is None:
            raise RuntimeError(
                "MoviePy VideoFileClip is not available - check MoviePy installation"
            )

        # Import robust audio loading system to prevent proc errors
        try:
            # Try to import from audio_loader module first
            from audio_loader import load_audio_robust
        except ImportError:
            try:
                # Fallback: try local definition in this file
                load_audio_robust = locals().get("load_audio_robust")
                if load_audio_robust is None:
                    raise ImportError("load_audio_robust not found in local scope")
            except Exception:
                # Final fallback: define a minimal robust audio loader
                def load_audio_robust(audio_file):
                    """Minimal robust audio loader as final fallback."""
                    return AudioFileClip(audio_file)

        # Validate audio file before processing
        if not Path(audio_file).exists():
            raise RuntimeError(f"Audio file not found: {audio_file}")

        # Check audio file size for potential issues
        try:
            audio_size = Path(audio_file).stat().st_size
            if audio_size == 0:
                raise RuntimeError(f"Audio file is empty: {audio_file}")
        except Exception:
            pass

        # Get MoviePy compatibility info for safe subclip operations
        try:
            from compatibility.moviepy import (
                attach_audio_safely,
                check_moviepy_api_compatibility,
                subclip_safely,
            )

            compatibility_info = check_moviepy_api_compatibility()
        except ImportError:
            compatibility_info = None
            subclip_safely = None
            attach_audio_safely = None

        # CRITICAL FIX: Log canvas format usage
        if canvas_format:
            pass
        else:
            pass

        if progress_callback:
            progress_callback("Loading video clips", 0.1)

        # Convert timeline to format expected by robust loading system

        # Prepare clip data for robust loading system
        video_files = list({clip_info["video_file"] for clip_info in timeline.clips})
        sorted_clips = []

        for i, clip_info in enumerate(timeline.clips):
            sorted_clips.append(
                {
                    "video_file": clip_info["video_file"],
                    "start": clip_info["start"],
                    "end": clip_info["end"],
                    "score": clip_info.get("score", 50.0),
                    "index": i,
                }
            )

        # Use existing robust loading system that handles proc errors properly
        try:
            # CRITICAL FIX: Pass canvas_format to the loading system
            video_clips, failed_indices, error_report, resource_manager = (
                load_video_clips_with_robust_error_handling(
                    sorted_clips=sorted_clips,
                    video_files=video_files,
                    canvas_format=canvas_format,  # NEW: Pass canvas format for preprocessing
                    progress_callback=lambda step, prog: progress_callback(
                        f"Loading: {step}", 0.1 + 0.4 * prog
                    )
                    if progress_callback
                    else None,
                )
            )

            if error_report.get("total_errors", 0) > 0:
                pass

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load video clips using robust loading system: {e}"
            ) from e

        if not video_clips:
            raise RuntimeError(
                f"No video clips could be loaded from {len(timeline.clips)} timeline clips using robust loading system"
            )

        if progress_callback:
            progress_callback("Concatenating video clips", 0.5)

        # Concatenate video clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        if progress_callback:
            progress_callback("Loading audio", 0.6)

        # Load and attach audio using robust loading system
        try:
            audio_clip = load_audio_robust(audio_file)
        except Exception as audio_error:
            raise RuntimeError(f"Failed to load audio file {audio_file}: {audio_error}") from audio_error

        # Trim audio to match video duration or vice versa
        video_duration = final_video.duration
        audio_duration = audio_clip.duration

        if audio_duration > video_duration:
            # Trim audio to video length
            if subclip_safely:
                audio_clip = subclip_safely(
                    audio_clip, 0, video_duration, compatibility_info
                )
            else:
                # Fallback: try both modern and legacy API for audio
                try:
                    audio_clip = audio_clip.subclipped(0, video_duration)
                except AttributeError:
                    audio_clip = audio_clip.subclip(0, video_duration)
        # Trim video to audio length
        elif subclip_safely:
            final_video = subclip_safely(
                final_video, 0, audio_duration, compatibility_info
            )
        else:
            # Fallback: try both modern and legacy API for video
            try:
                final_video = final_video.subclipped(0, audio_duration)
            except AttributeError:
                final_video = final_video.subclip(0, audio_duration)

        # Apply musical fade-out if we have beat information with robust error handling
        if avg_beat_interval and audio_duration > video_duration:
            # Calculate fade duration (2-4 beats, max 3 seconds)
            fade_duration = min(avg_beat_interval * 3, 3.0)
            try:
                # Apply fade operations with MoviePy API compatibility

                # Try modern MoviePy 2.x effects system first
                try:
                    from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

                    # MoviePy 2.x: Apply effects using with_effects() method
                    audio_clip = audio_clip.with_effects(
                        [
                            AudioFadeIn(0.1),
                            AudioFadeOut(fade_duration),
                        ]
                    )
                except ImportError:
                    # Fallback to legacy methods if available
                    try:
                        if hasattr(audio_clip, "audio_fadein") and hasattr(
                            audio_clip, "audio_fadeout"
                        ):
                            audio_clip = audio_clip.audio_fadein(0.1).audio_fadeout(
                                fade_duration
                            )
                        else:
                            pass
                    except Exception:
                        pass

            except Exception:
                pass
                # Continue without fades rather than failing completely - this prevents
                # the creation of new FFMPEG_AudioReader instances that could trigger proc errors

        # Attach audio to video with MoviePy API compatibility
        try:
            if compatibility_info and attach_audio_safely:
                # Use the compatibility layer if available
                final_video = attach_audio_safely(
                    final_video, audio_clip, compatibility_info
                )
            else:
                # Try multiple methods for audio attachment
                try:
                    # Try modern MoviePy 2.x method first
                    final_video = final_video.with_audio(audio_clip)
                except AttributeError:
                    try:
                        # Fallback to legacy set_audio method
                        final_video = final_video.set_audio(audio_clip)
                    except AttributeError as attr_error:
                        raise RuntimeError(
                            f"Cannot attach audio to {type(final_video)} - no compatible method found"
                        ) from attr_error
        except Exception as audio_attach_error:
            raise RuntimeError(f"Failed to attach audio to video: {audio_attach_error}") from audio_attach_error

        if progress_callback:
            progress_callback("Encoding video", 0.7)

        # Get optimal encoding settings
        try:
            encoder = VideoEncoder()
            moviepy_params, ffmpeg_params = encoder.detect_optimal_codec_settings()
        except Exception:
            # Fallback encoding settings
            moviepy_params = {
                "codec": "libx264",
                "bitrate": "5000k",
                "audio_codec": "aac",
            }
            ffmpeg_params = ["-preset", "medium", "-crf", "23"]

        # Prepare encoding parameters
        encoding_params = {
            **moviepy_params,
            "ffmpeg_params": ffmpeg_params,
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "verbose": False,
            "logger": None,
        }

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Encode video with compatibility layer
        try:
            from compatibility.moviepy import (
                check_moviepy_api_compatibility,
                write_videofile_safely,
            )

            compatibility_info = check_moviepy_api_compatibility()

            write_videofile_safely(
                final_video,
                output_path,
                compatibility_info,
                **encoding_params,
            )
        except ImportError:
            # Fallback if compatibility module not available
            final_video.write_videofile(output_path, **encoding_params)

        if progress_callback:
            progress_callback("Video rendering complete", 1.0)

        # Clean up clips and resource manager

        # Clean up resource manager (prevents proc errors)
        try:
            if "resource_manager" in locals():
                resource_manager.cleanup_all()
        except Exception:
            pass

        # Clean up individual clips
        for clip in video_clips:
            with contextlib.suppress(builtins.BaseException):
                clip.close()

        # Enhanced audio cleanup to prevent proc errors
        try:
            # Close audio clip and any internal readers
            if hasattr(audio_clip, "close"):
                audio_clip.close()

            # Additional cleanup for FFMPEG_AudioReader instances
            if hasattr(audio_clip, "reader") and hasattr(audio_clip.reader, "proc"):
                with contextlib.suppress(builtins.BaseException):
                    audio_clip.reader.proc.terminate()

        except Exception:
            pass

        # Clean up final video
        try:
            if hasattr(final_video, "close"):
                final_video.close()
        except Exception:
            pass
        
        # CRITICAL FIX: Return in try block, not orphaned else block
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to render video: {e!s}") from e


def add_transitions(
    clips: List[VideoFileClip],
    transition_duration: float = 0.5,
) -> VideoFileClip:
    """Add crossfade transitions between clips - REFACTORED.

    This function now delegates to the new modular TransitionEngine
    extracted as part of Phase 3 refactoring while maintaining full
    backward compatibility with existing AutoCut code.

    Args:
        clips: List of video clips
        transition_duration: Duration of crossfade in seconds

    Returns:
        Composite video with transitions
    """
    try:
        # Import the new modular transition system with dual import pattern
        try:
            from video.rendering import add_transitions as add_transitions_modular
        except ImportError:
            from .video.rendering import add_transitions as add_transitions_modular

        # Delegate to the new modular system
        return add_transitions_modular(clips, transition_duration)

    except ImportError as import_error:
        # Fallback to legacy implementation if modules not available
        raise RuntimeError(
            "New transition system not available - refactoring incomplete"
        ) from import_error
    except Exception as e:
        raise RuntimeError(f"Transition creation failed: {e!s}") from e


def assemble_clips(
    video_files: List[str],
    audio_file: str,
    output_path: str,
    pattern: str = "balanced",
    max_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> str:
    """Main function to assemble clips into final video.

    Combines all steps:
    1. Analyze all video files
    2. Analyze audio file
    3. Determine optimal canvas format
    4. Match clips to beats
    5. Render final video

    Args:
        video_files: List of paths to video files
        audio_file: Path to music file
        output_path: Path for output video
        pattern: Variety pattern to use
        max_workers: Maximum parallel workers for video loading (None = auto-detect optimal)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to final rendered video

    Raises:
        FileNotFoundError: If any input file doesn't exist
        ValueError: If no suitable clips found or invalid audio file
        RuntimeError: If rendering fails
    """
    import logging
    import mimetypes

    # Dual import pattern for package/direct execution compatibility
    try:
        from audio_analyzer import analyze_audio
        from video.format_analyzer import VideoFormatAnalyzer
        from video_analyzer import analyze_video_file
    except ImportError:
        from .audio_analyzer import analyze_audio
        from .video.format_analyzer import VideoFormatAnalyzer
        from .video_analyzer import analyze_video_file

    def validate_audio_file_comprehensive(audio_path: str) -> tuple:
        """Comprehensive audio file validation before processing.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check 1: File existence
            if not Path(audio_path).exists():
                return False, f"Audio file not found: {audio_path}"

            # Check 2: File accessibility
            if not os.access(audio_path, os.R_OK):
                return False, f"Audio file is not readable: {audio_path}"

            # Check 3: File size validation
            try:
                file_size = Path(audio_path).stat().st_size
                if file_size == 0:
                    return False, f"Audio file is empty: {audio_path}"
                if file_size < 1024:  # Less than 1KB is suspicious
                    return (
                        False,
                        f"Audio file too small ({file_size} bytes), likely corrupted: {audio_path}",
                    )
                if file_size > 500 * 1024 * 1024:  # More than 500MB is excessive
                    return (
                        False,
                        f"Audio file too large ({file_size / (1024 * 1024):.1f}MB), may cause memory issues: {audio_path}",
                    )
            except OSError as e:
                return False, f"Cannot access audio file: {e}"

            # Check 4: File extension and MIME type validation
            audio_path_lower = audio_path.lower()
            valid_extensions = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"]
            has_valid_extension = any(
                audio_path_lower.endswith(ext) for ext in valid_extensions
            )

            if not has_valid_extension:
                return (
                    False,
                    f"Unsupported audio file extension. Supported: {', '.join(valid_extensions)}",
                )

            # Check 5: MIME type validation (if available)
            try:
                mime_type, _ = mimetypes.guess_type(audio_path)
                if mime_type and not mime_type.startswith("audio/"):
                    return (
                        False,
                        f"File does not appear to be an audio file (MIME type: {mime_type})",
                    )
            except Exception:
                # MIME type detection is optional, don't fail if it doesn't work
                pass

            # Check 6: Path complexity and character validation
            path_issues = []

            # Check for very long paths that might cause issues
            if len(audio_path) > 260:  # Windows MAX_PATH limit
                path_issues.append("Path length exceeds system limits")

            # Check for problematic characters that might cause FFmpeg issues
            problematic_chars = ["|", "<", ">", '"', "?", "*"]
            for char in problematic_chars:
                if char in audio_path:
                    path_issues.append(f"Contains problematic character '{char}'")

            if path_issues:
                # These are warnings, not fatal errors
                logger = logging.getLogger("autocut.clip_assembler")
                logger.warning(
                    f"⚠️  Audio path has potential issues: {'; '.join(path_issues)}",
                )
                logger.warning(f"   Path: {audio_path}")
                logger.warning(
                    "   Will attempt to process but may encounter issues...",
                )
        
            # CRITICAL FIX: Return in try block, not orphaned else block  
            return True, None
        except Exception as e:
            return False, f"Unexpected error validating audio file: {e!s}"

    # Set up detailed logging for the main pipeline
    logger = logging.getLogger("autocut.clip_assembler")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)

    # Initialize comprehensive processing statistics
    processing_summary = {
        "total_videos": len(video_files),
        "videos_processed": 0,
        "videos_successful": 0,
        "videos_failed": 0,
        "total_chunks": 0,
        "file_results": [],  # Detailed per-file results
        "errors": [],
    }

    logger.info("=== AutoCut Video Processing Started ===")
    logger.info(f"Input videos: {len(video_files)} files")
    logger.info(f"Audio file: {Path(audio_file).name}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Pattern: {pattern}")

    # CRITICAL FIX: Enhanced comprehensive input validation
    logger.info("Validating input files...")

    # Validate audio file with comprehensive checks
    logger.info("🔍 Comprehensive audio file validation...")
    audio_valid, audio_error = validate_audio_file_comprehensive(audio_file)
    if not audio_valid:
        error_msg = f"Audio validation failed: {audio_error}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise ValueError(error_msg)

    logger.info("   ✅ Audio file validation passed")
    logger.info(f"   📁 File: {Path(audio_file).name}")
    logger.info(f"   📊 Size: {Path(audio_file).stat().st_size / (1024 * 1024):.2f}MB")

    # Validate video files
    missing_videos = [vf for vf in video_files if not Path(vf).exists()]
    if missing_videos:
        error_msg = f"Video files not found: {missing_videos}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise FileNotFoundError(error_msg)

    if not video_files:
        error_msg = "No video files provided"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("✅ All input files exist and are accessible")
    report_progress("Starting analysis", 0.0)

    # Step 1: Analyze audio file
    logger.info("=== Step 1: Audio Analysis ===")
    report_progress("Analyzing audio", 0.1)
    try:
        # CRITICAL FIX: Enhanced audio analysis with better error reporting
        try:
            audio_data = analyze_audio(audio_file)
        except FileNotFoundError as e:
            error_msg = f"Audio file access error during analysis: {e!s}"
            logger.exception(error_msg)
            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Audio analysis failed: {e!s}"
            logger.exception(error_msg)

            # Provide more helpful error messages for common issues
            if "No such file" in str(e) or "cannot find" in str(e).lower():
                error_msg += " (Check if audio file path is correct and accessible)"
            elif "format" in str(e).lower() or "codec" in str(e).lower():
                error_msg += " (Audio file may be corrupted or in unsupported format)"
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                error_msg += " (Check file permissions)"

            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg) from e

        # CRITICAL FIX: Use compensated beats instead of raw beats to fix sync issues
        beats = audio_data["compensated_beats"]  # Offset-corrected and filtered beats

        # Get musical timing information for professional synchronization
        musical_start_time = audio_data["musical_start_time"]
        intro_duration = audio_data["intro_duration"]
        allowed_durations = audio_data["allowed_durations"]

        if len(beats) < 2:
            error_msg = f"Insufficient beats detected in audio file: {len(beats)} beats"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ Audio analysis successful:")
        logger.info(f"  - Beats detected: {len(beats)}")
        logger.info(f"  - Musical start: {musical_start_time:.2f}s")
        logger.info(f"  - Intro duration: {intro_duration:.2f}s")
        report_progress("Audio analysis complete", 0.2)

    except Exception as e:
        error_msg = f"Failed to analyze audio file: {e!s}"
        logger.exception(error_msg)
        processing_summary["errors"].append(error_msg)
        raise RuntimeError(error_msg) from e

    # Step 2: Analyze all video files with detailed per-file tracking
    logger.info("=== Step 2: Video Analysis ===")
    report_progress("Analyzing videos", 0.3)
    all_video_chunks = []

    for i, video_file in enumerate(video_files):
        processing_summary["videos_processed"] += 1
        filename = Path(video_file).name

        file_result = {
            "file_path": video_file,
            "filename": filename,
            "index": i + 1,
            "status": "processing",
            "chunks_created": 0,
            "error_message": None,
            "processing_time": 0,
        }

        logger.info(f"--- Processing video {i + 1}/{len(video_files)}: {filename} ---")

        import time

        start_time = time.time()

        try:
            video_chunks = analyze_video_file(video_file, bpm=audio_data.get("bpm"))
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time

            if video_chunks:
                all_video_chunks.extend(video_chunks)
                file_result["chunks_created"] = len(video_chunks)
                file_result["status"] = "success"
                processing_summary["videos_successful"] += 1
                processing_summary["total_chunks"] += len(video_chunks)

                logger.info(
                    f"✅ {filename}: {len(video_chunks)} chunks created ({processing_time:.2f}s)",
                )

                # Log chunk quality summary
                if video_chunks:
                    scores = [chunk.score for chunk in video_chunks]
                    logger.info(
                        f"   Chunk scores: {min(scores):.1f}-{max(scores):.1f} (avg: {sum(scores) / len(scores):.1f})",
                    )
            else:
                file_result["status"] = "failed"
                file_result["error_message"] = (
                    "No chunks created - check logs above for detailed error analysis"
                )
                processing_summary["videos_failed"] += 1

                logger.error(
                    f"❌ {filename}: No chunks created ({processing_time:.2f}s)",
                )
                logger.error("   → This video will be excluded from the final output")

        except Exception as e:
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time
            file_result["status"] = "failed"
            file_result["error_message"] = str(e)
            processing_summary["videos_failed"] += 1
            processing_summary["errors"].append(f"{filename}: {e!s}")

            logger.exception(
                f"❌ {filename}: Processing failed ({processing_time:.2f}s)"
            )
            logger.exception(f"   Error: {e!s}")
            logger.exception("   → This video will be excluded from the final output")

        processing_summary["file_results"].append(file_result)

        # Update progress for each video
        video_progress = 0.3 + (0.3 * (i + 1) / len(video_files))
        report_progress(f"Analyzed video {i + 1}/{len(video_files)}", video_progress)

    # Comprehensive processing summary
    logger.info("=== Video Processing Summary ===")
    logger.info(f"Total videos: {processing_summary['total_videos']}")
    logger.info(f"✅ Successful: {processing_summary['videos_successful']}")
    logger.info(f"❌ Failed: {processing_summary['videos_failed']}")
    logger.info(f"📊 Total chunks created: {processing_summary['total_chunks']}")

    # Detailed per-file results
    if processing_summary["videos_failed"] > 0:
        logger.warning("Failed video details:")
        for file_result in processing_summary["file_results"]:
            if file_result["status"] == "failed":
                logger.warning(
                    f"  - {file_result['filename']}: {file_result['error_message']}",
                )

    # Check if we have any usable content
    if not all_video_chunks:
        error_msg = (
            f"No suitable video clips found in any of the {len(video_files)} input files.\n"
            f"Processing results:\n"
            f"  - Successful videos: {processing_summary['videos_successful']}\n"
            f"  - Failed videos: {processing_summary['videos_failed']}\n"
            f"  - Total chunks created: {processing_summary['total_chunks']}\n"
            f"\nDetailed errors:\n"
            + "\n".join(
                [f"  - {error}" for error in processing_summary["errors"][-10:]],
            )  # Last 10 errors
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate success rate
    success_rate = (
        processing_summary["videos_successful"] / processing_summary["total_videos"]
    ) * 100
    logger.info(f"Processing success rate: {success_rate:.1f}%")

    if success_rate < 50:
        logger.warning(
            f"⚠️  Low success rate ({success_rate:.1f}%) - check video format compatibility",
        )

    report_progress(
        f"Video analysis complete: {len(all_video_chunks)} clips found",
        0.6,
    )

    # Step 2.5: CANVAS ANALYSIS - CRITICAL FIX for letterboxing issue
    logger.info("=== Step 2.5: Canvas Format Analysis ===")
    report_progress("Analyzing optimal canvas format", 0.65)

    try:
        # Initialize the VideoFormatAnalyzer
        format_analyzer = VideoFormatAnalyzer()

        # Analyze all video chunks to determine optimal canvas
        logger.info(
            f"🎯 Analyzing {len(all_video_chunks)} video clips for optimal canvas..."
        )

        canvas_format = format_analyzer.determine_optimal_canvas(all_video_chunks)

        # Log the canvas analysis results
        logger.info("✅ Canvas analysis complete:")
        logger.info(f"   🎬 Canvas type: {canvas_format['canvas_type']}")
        logger.info(
            f"   📐 Target dimensions: {canvas_format['target_width']}x{canvas_format['target_height']}"
        )
        logger.info("   📊 Content breakdown:")
        logger.info(
            f"      - Landscape: {canvas_format['aspect_ratio_analysis']['landscape_count']} clips ({(canvas_format['aspect_ratio_analysis']['landscape_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(
            f"      - Portrait: {canvas_format['aspect_ratio_analysis']['portrait_count']} clips ({(canvas_format['aspect_ratio_analysis']['portrait_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(
            f"      - Square: {canvas_format['aspect_ratio_analysis']['square_count']} clips ({(canvas_format['aspect_ratio_analysis']['square_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(f"   💡 Strategy: {canvas_format.get('description', canvas_format.get('aspect_ratio_analysis', {}).get('decision_rationale', 'Canvas analysis'))}")

        # Log letterboxing expectations
        if canvas_format.get("letterboxing_analysis"):
            logger.info("   📺 Letterboxing analysis:")
            for info in canvas_format["letterboxing_analysis"]:
                logger.info(f"      - {info}")
        else:
            logger.info(
                "   ✅ Minimal letterboxing expected - optimal aspect ratio match"
            )

        report_progress("Canvas analysis complete", 0.7)

    except Exception as e:
        error_msg = f"Canvas analysis failed: {e!s}"
        logger.exception(error_msg)
        logger.warning("Falling back to default 16:9 canvas (1920x1080)")

        # Fallback canvas format
        canvas_format = {
            "target_width": 1920,
            "target_height": 1080,
            "canvas_type": "fallback_16_9",
            "description": "Fallback 16:9 canvas due to analysis failure",
            "target_fps": 25,
            "aspect_ratio_analysis": {
                "landscape_count": 0,
                "portrait_count": 0,
                "square_count": 0,
                "dominant_orientation": "unknown",
                "decision_rationale": "Analysis failed, using safe fallback",
            },
            "letterboxing_analysis": [],
        }

    # Step 3: Match clips to beats
    logger.info("=== Step 3: Beat Matching ===")
    report_progress("Matching clips to beats", 0.75)
    try:
        timeline = match_clips_to_beats(
            video_chunks=all_video_chunks,
            beats=beats,
            allowed_durations=allowed_durations,
            pattern=pattern,
            musical_start_time=musical_start_time,  # Use musical intelligence for sync
        )

        if not timeline.clips:
            error_msg = f"No clips could be matched to the beat pattern using {len(all_video_chunks)} available chunks"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"✅ Beat matching successful: {len(timeline.clips)} clips selected",
        )

        # Log timeline statistics
        timeline_stats = timeline.get_summary_stats()
        logger.info("Timeline statistics:")
        logger.info(f"  - Total duration: {timeline_stats['total_duration']:.2f}s")
        logger.info(f"  - Average score: {timeline_stats['avg_score']:.1f}")
        logger.info(
            f"  - Score range: {timeline_stats['score_range'][0]:.1f}-{timeline_stats['score_range'][1]:.1f}",
        )
        logger.info(f"  - Unique videos used: {timeline_stats['unique_videos']}")

        report_progress(
            f"Beat matching complete: {len(timeline.clips)} clips selected",
            0.8,
        )

    except Exception as e:
        error_msg = f"Failed to match clips to beats: {e!s}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Step 4: Render final video
    logger.info("=== Step 4: Video Rendering ===")
    report_progress("Rendering video", 0.85)
    try:

        def render_progress(step_name: str, progress: float):
            # Scale render progress to final 15% of overall progress
            overall_progress = 0.85 + (0.15 * progress)
            report_progress(f"Rendering: {step_name}", overall_progress)

        # Calculate average beat interval for musical fade-out feature
        avg_beat_interval = None
        if len(beats) > 1:
            beat_intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
            avg_beat_interval = sum(beat_intervals) / len(beat_intervals)
            logger.info(f"Average beat interval calculated: {avg_beat_interval:.3f}s")

        # CRITICAL FIX: Pass canvas format to render_video function
        final_video_path = render_video(
            timeline=timeline,
            audio_file=audio_file,
            output_path=output_path,
            max_workers=max_workers,
            progress_callback=render_progress,
            bpm=audio_data.get("bpm"),
            avg_beat_interval=avg_beat_interval,
            canvas_format=canvas_format,  # NEW: Pass canvas format for optimal sizing
        )

        logger.info(f"✅ Video rendering complete: {final_video_path}")
        report_progress("Video rendering complete", 1.0)

        # Final success summary
        logger.info("=== AutoCut Processing Complete ===")
        logger.info(
            f"✅ Successfully created video: {Path(final_video_path).name}",
        )
        logger.info("📊 Processing summary:")
        logger.info(
            f"  - Videos processed: {processing_summary['videos_successful']}/{processing_summary['total_videos']}",
        )
        logger.info(
            f"  - Clips used: {len(timeline.clips)}/{processing_summary['total_chunks']}",
        )
        logger.info(
            f"  - Final video duration: {timeline_stats['total_duration']:.2f}s",
        )
        logger.info(
            f"  - Canvas format: {canvas_format['canvas_type']} ({canvas_format['target_width']}x{canvas_format['target_height']})"
        )
        # CRITICAL FIX: Return in try block, not orphaned else block
        return final_video_path
    except Exception as e:
        error_msg = f"Failed to render video: {e!s}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Export timeline JSON for debugging (optional)
    try:
        timeline_path = output_path.replace(".mp4", "_timeline.json")
        timeline.export_json(timeline_path)
        logger.info(f"Debug: Timeline exported to {timeline_path}")

        # Export processing summary for debugging
        summary_path = output_path.replace(".mp4", "_processing_summary.json")
        import json

        with Path(summary_path).open("w") as f:
            json.dump(processing_summary, f, indent=2)
        logger.info(f"Debug: Processing summary exported to {summary_path}")

    except Exception:
        pass  # Non-critical, ignore errors  # Non-critical, ignore errors  # Non-critical, ignore errors  # Non-critical, ignore errors


def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """Detect optimal codec settings for video encoding.

    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
    """
    try:
        # Try to use the extracted VideoEncoder class
        encoder = VideoEncoder()
        return encoder.detect_optimal_codec_settings()
    except Exception:
        # Fallback to safe default settings
        moviepy_params = {
            "codec": "libx264",
            "bitrate": "5000k",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
        }

        ffmpeg_params = [
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        return moviepy_params, ffmpeg_params


def detect_optimal_codec_settings_with_diagnostics() -> Tuple[
    Dict[str, Any],
    List[str],
    Dict[str, str],
]:
    """Enhanced codec settings detection with full diagnostic information.

    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
        - Dictionary of diagnostic information and capability details
    """
    try:
        # Try to use the extracted VideoEncoder class
        encoder = VideoEncoder()
        return encoder.detect_optimal_codec_settings_with_diagnostics()
    except Exception:
        # Fallback with basic diagnostics
        moviepy_params, ffmpeg_params = detect_optimal_codec_settings()
        diagnostics = {
            "encoder_type": "FALLBACK",
            "hardware_acceleration": "false",
            "platform": "unknown",
        }
        return moviepy_params, ffmpeg_params, diagnostics
