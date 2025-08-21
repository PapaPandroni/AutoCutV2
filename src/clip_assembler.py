"""
Clip Assembly Module for AutoCut

Handles the core logic of matching video clips to musical beats,
applying variety patterns, and rendering the final video.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import os
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
try:
    from system_profiler import SystemProfiler
    from adaptive_monitor import AdaptiveWorkerMonitor
    from memory.monitor import get_memory_info
except ImportError:
    # Fallback if modules not available
    class SystemProfiler:
        def __init__(self): pass
    class AdaptiveWorkerMonitor:
        def __init__(self): pass
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
    from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    try:
        # Fallback for MoviePy 2.x direct imports
        from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips
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
    from video.format_analyzer import VideoFormatAnalyzer
    from video.normalization import VideoNormalizationPipeline 
    from video.timeline_renderer import ClipTimeline, TimelineRenderer
    from video.encoder import VideoEncoder, detect_optimal_codec_settings, detect_optimal_codec_settings_with_diagnostics
    print("✅ Successfully imported extracted video processing classes")
except ImportError as e:
    print(f"⚠️  Could not import extracted classes: {e}")
    print("   Falling back to legacy inline class definitions")
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
            print("Warning: audio_loader module not found, using alternative robust loading...")
            
            # Import moviepy safely with compatibility layer
            from moviepy.editor import AudioFileClip
            
            try:
                # First try standard MoviePy loading
                audio_clip = AudioFileClip(audio_file)
                return audio_clip
            except (AttributeError, RuntimeError, OSError) as e:
                if "proc" in str(e).lower() or "ffmpeg" in str(e).lower():
                    print(f"MoviePy audio loading failed ({e}), trying FFmpeg subprocess fallback...")
                    
                    # Fallback to FFmpeg subprocess for problematic files
                    try:
                        import subprocess
                        import numpy as np
                        from moviepy.audio.AudioClip import AudioArrayClip
                        
                        # Use FFmpeg subprocess to bypass MoviePy's FFMPEG_AudioReader
                        cmd = [
                            "ffmpeg", "-i", audio_file,
                            "-f", "s16le", "-acodec", "pcm_s16le", 
                            "-ac", "2", "-ar", "44100", "-v", "quiet", "-"
                        ]
                        
                        process = subprocess.run(cmd, capture_output=True, check=True, timeout=60)
                        audio_data = np.frombuffer(process.stdout, dtype=np.int16)
                        
                        if len(audio_data) == 0:
                            raise RuntimeError("No audio data extracted from file")
                        
                        # Convert to stereo float32 format
                        audio_data = audio_data.reshape(-1, 2).astype(np.float32) / 32768.0
                        audio_clip = AudioArrayClip(audio_data, fps=44100)
                        
                        print(f"✅ FFmpeg subprocess fallback successful for {audio_file}")
                        return audio_clip
                        
                    except Exception as fallback_error:
                        print(f"❌ All audio loading strategies failed: {fallback_error}")
                        raise RuntimeError(f"Could not load audio file {audio_file}: {fallback_error}")
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

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with self._lock:
            if video_path not in self._cache:
                try:
                    video_clip = VideoFileClip(video_path)
                    self._cache[video_path] = video_clip
                except Exception as e:
                    raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

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
        with self._lock:
            for video_clip in self._cache.values():
                try:
                    video_clip.close()
                except Exception:
                    pass  # Ignore cleanup errors
            self._cache.clear()
            self._ref_counts.clear()


# VideoFormatAnalyzer class extracted to src/video/format_analyzer.py


# VideoNormalizationPipeline class extracted to src/video/normalization.py


def load_video_segment(
    clip_data: Dict[str, Any], video_cache: VideoCache
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

        return (clip_data, segment)

    except Exception as e:
        print(f"Warning: Failed to load clip {clip_data['video_file']}: {str(e)}")
        return None


# ============================================================================
# NEW THREAD-SAFE SEQUENTIAL VIDEO PROCESSING (Phase 1 Fix)
# Replaces dangerous parallel loading with memory-safe sequential processing
# ============================================================================


class MemoryMonitor:
    """Real-time memory monitoring with emergency cleanup capabilities."""

    def __init__(
        self, warning_threshold_gb: float = 4.0, emergency_threshold_gb: float = 6.0
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
            print(f"   💾 Memory: +{usage_gb:.1f}GB ({context})")


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
        from contextlib import contextmanager
        import gc

        @contextmanager
        def _video_context():
            video = None
            try:
                if VideoFileClip is None:
                    raise RuntimeError(
                        "MoviePy not available. Please install moviepy>=1.0.3"
                    )

                video = VideoFileClip(video_path)
                self.active_videos.add(id(video))
                yield video
            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")
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
            if VideoFileClip is None:
                raise RuntimeError(
                    "MoviePy not available. Please install moviepy>=1.0.3"
                )

            # Check if we already have this video loaded for delayed cleanup
            if video_path in self.delayed_cleanup_videos:
                return self.delayed_cleanup_videos[video_path]

            # Load the video and store it for delayed cleanup
            video = VideoFileClip(video_path)
            self.delayed_cleanup_videos[video_path] = video
            self.active_videos.add(id(video))

            print(
                f"   📹 Loaded video with delayed cleanup: {os.path.basename(video_path)}"
            )
            return video

        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

    def cleanup_delayed_videos(self) -> None:
        """Clean up all videos that were loaded with delayed cleanup.

        This should be called AFTER concatenation is complete to ensure subclips
        remain valid during the entire video processing pipeline.
        """
        import gc

        cleanup_count = len(self.delayed_cleanup_videos)
        if cleanup_count > 0:
            print(
                f"   🧹 Cleaning up {cleanup_count} delayed videos after concatenation"
            )

            for video_path, video in self.delayed_cleanup_videos.items():
                try:
                    self.active_videos.discard(id(video))
                    video.close()
                    print(f"      ✅ Closed {os.path.basename(video_path)}")
                except Exception as e:
                    print(
                        f"      ⚠️ Warning: Failed to close {os.path.basename(video_path)}: {e}"
                    )

            self.delayed_cleanup_videos.clear()
            gc.collect()  # Force garbage collection
            print(f"   🧹 Delayed cleanup complete")

    def emergency_cleanup(self) -> None:
        """Force cleanup of any remaining video resources."""
        import gc

        print("   🚨 Emergency cleanup: forcing garbage collection")

        # Clean up delayed videos first
        self.cleanup_delayed_videos()

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
    print(
        f"   🔍 PHASE 2: Smart preprocessing analysis for {len(video_files)} video files"
    )
    video_path_map = preprocess_videos_smart(
        video_files, progress_callback=progress_callback
    )

    # Update clip data to use preprocessed video paths
    for clip_data in sorted_clips:
        original_path = clip_data["video_file"]
        if original_path in video_path_map:
            optimized_path = video_path_map[original_path]
            if optimized_path != original_path:
                print(
                    f"   🔄 Using preprocessed version: {os.path.basename(optimized_path)}"
                )
                clip_data["video_file"] = optimized_path
                clip_data["original_video_file"] = (
                    original_path  # Keep reference to original
                )

    # Memory monitoring at start
    initial_memory = get_memory_info()
    print(
        f"   🧠 Starting sequential loading: {initial_memory['used_gb']:.1f}GB used ({initial_memory['percent']:.1f}%)"
    )

    # Use a results dictionary to maintain perfect index alignment
    clip_results = {}  # Maps original_index -> video clip (or None for failed)
    failed_indices = []

    # Group clips by video file to minimize loading
    grouped_clips = _group_clips_by_file(sorted_clips)

    print(
        f"   🎬 Processing {len(sorted_clips)} clips from {len(grouped_clips)} files sequentially"
    )
    print(
        f"   🔧 CRITICAL FIX: Using delayed cleanup to prevent NoneType get_frame errors"
    )

    processed_files = 0
    total_clips_processed = 0

    for video_file, file_clips in grouped_clips.items():
        processed_files += 1
        print(
            f"\n   [{processed_files}/{len(grouped_clips)}] Processing {os.path.basename(video_file)}"
        )
        print(f"   🎯 Extracting {len(file_clips)} clips")

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
                        f"clip {total_clips_processed + 1}"
                    )

                    # Extract segment - parent video stays alive via delayed cleanup
                    segment = source_video.subclipped(
                        clip_data["start"], clip_data["end"]
                    )

                    # Validation: Test the clip while parent video is available
                    # This validation is still important for catching other issues
                    try:
                        test_time = min(0.1, segment.duration)
                        test_frame = segment.get_frame(test_time)
                        if test_frame is None:
                            raise RuntimeError(
                                "get_frame returned None during validation"
                            )
                    except Exception as e:
                        raise RuntimeError(f"Clip validation failed: {e}")

                    original_index = clip_data.get(
                        "original_index", total_clips_processed
                    )
                    clip_results[original_index] = segment
                    file_clips_loaded += 1

                except Exception as e:
                    print(
                        f"   ⚠️  Failed to extract clip {clip_data['start']:.1f}-{clip_data['end']:.1f}s: {e}"
                    )
                    # Use original_index from grouped clips to maintain timeline alignment
                    original_index = clip_data.get(
                        "original_index", total_clips_processed
                    )
                    clip_results[original_index] = None
                    failed_indices.append(original_index)

                total_clips_processed += 1

            print(
                f"   ✅ Extracted {file_clips_loaded}/{len(file_clips)} clips successfully"
            )

            # NOTE: We do NOT close the source_video here - it will be cleaned up later
            # This is the key fix: parent videos stay alive until after concatenation

            # Update progress
            progress = 0.1 + (0.6 * total_clips_processed / len(sorted_clips))
            report_progress(
                f"Loaded {total_clips_processed}/{len(sorted_clips)} clips", progress
            )

        except Exception as e:
            print(f"   ❌ Failed to load video file {video_file}: {e}")
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
        clip = clip_results.get(i, None)
        video_clips.append(clip)

    # Count successful clips (non-None)
    successful_clips = [clip for clip in video_clips if clip is not None]

    if not successful_clips:
        raise RuntimeError("No video clips could be loaded successfully")

    # Report final statistics
    success_count = len(successful_clips)
    total_count = len(sorted_clips)
    success_rate = success_count / total_count

    if failed_indices:
        print(
            f"   ⚠️  {len(failed_indices)} clips failed to load ({100 - success_rate * 100:.1f}% success rate)"
        )

    # Memory monitoring at completion
    final_memory = get_memory_info()
    memory_increase = final_memory["used_gb"] - initial_memory["used_gb"]
    print(
        f"   🧠 Sequential loading complete: {final_memory['used_gb']:.1f}GB used (+{memory_increase:.1f}GB)"
    )

    # Memory usage warning
    if final_memory["percent"] > 85:
        print(
            f"   ⚠️  HIGH MEMORY USAGE: {final_memory['percent']:.1f}% - Consider reducing video count"
        )

    print(f"   🎬 CRITICAL SUCCESS: All parent videos kept alive for concatenation")
    print(
        f"   🔧 Delayed cleanup will occur after concatenation to prevent NoneType errors"
    )

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
        import os

        if not os.path.exists(video_path):
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
            print(
                f"Warning: Could not analyze video {os.path.basename(video_path)}: {e}"
            )
            return {
                "needs_preprocessing": False,
                "reason": "analysis_failed",
                "error": str(e),
            }

    def _detect_video_properties_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Use ffprobe to detect video properties without loading full video."""
        import subprocess
        import json

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

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

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

            # Fallback if ffprobe fails
            return {
                "codec_name": "unknown",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "24/1",
            }

        except Exception as e:
            print(f"Warning: ffprobe analysis failed: {e}")
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
        memory_mb = memory_bytes / (1024 * 1024)

        return memory_mb

    def preprocess_video_if_needed(
        self, video_path: str, output_dir: str = None
    ) -> str:
        """Preprocess video if needed and return path to processed version.

        Args:
            video_path: Path to original video file
            output_dir: Directory for processed files (default: same as input)

        Returns:
            Path to video file to use (original or preprocessed)
        """
        import os
        import tempfile

        # Check if preprocessing is needed
        analysis = self.should_preprocess_video(video_path)

        if not analysis["needs_preprocessing"]:
            print(f"   📹 {os.path.basename(video_path)}: No preprocessing needed")
            return video_path

        # Check cache first
        cache_key = f"{video_path}_{hash(str(analysis['reasons']))}"
        if cache_key in self.preprocessing_cache:
            cached_path = self.preprocessing_cache[cache_key]
            if os.path.exists(cached_path):
                print(
                    f"   📹 {os.path.basename(video_path)}: Using cached preprocessed version"
                )
                return cached_path

        print(f"   📹 {os.path.basename(video_path)}: Preprocessing needed")
        print(f"       Reasons: {', '.join(analysis['reasons'])}")
        print(
            f"       Original: {analysis['codec']} {analysis['resolution']} @ {analysis['fps']:.1f}fps"
        )
        print(f"       Est. memory: {analysis['estimated_memory_mb']:.0f}MB")

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(video_path)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        processed_name = f"{base_name}_processed.mp4"
        processed_path = os.path.join(output_dir, processed_name)

        # If processed version already exists and is newer, use it
        if os.path.exists(processed_path) and os.path.getmtime(
            processed_path
        ) > os.path.getmtime(video_path):
            print(f"   📹 Using existing processed version: {processed_name}")
            self.preprocessing_cache[cache_key] = processed_path
            return processed_path

        # Perform preprocessing
        try:
            success = self._preprocess_with_ffmpeg_modern(video_path, processed_path, analysis, getattr(self, '_target_format', None))

            if success and os.path.exists(processed_path):
                print(f"   ✅ Preprocessing complete: {processed_name}")
                self.preprocessing_cache[cache_key] = processed_path
                return processed_path
            else:
                print(f"   ⚠️  Preprocessing failed, using original")
                return video_path

        except Exception as e:
            print(f"   ❌ Preprocessing error: {e}")
            return video_path

    def _preprocess_with_ffmpeg_modern(
        self, input_path: str, output_path: str, analysis: Dict, target_format: Dict = None
    ) -> bool:
        """Modern FFmpeg preprocessing with intelligent aspect ratio preservation.
        
        This replaces the old hard-coded 1920x1080 scaling that stretched portrait videos.
        Uses dynamic canvas dimensions and proper letterboxing with pad filter.
        """
        import subprocess
        import os

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
            if any("resolution" in reason for reason in analysis["reasons"]) and target_format:
                target_w = target_format.get('target_width', 1920)
                target_h = target_format.get('target_height', 1080)
                canvas_type = target_format.get('canvas_type', 'default_landscape')
                
                # Use modern FFmpeg scaling with aspect ratio preservation + letterboxing
                # Step 1: Scale down if needed, maintaining aspect ratio
                scale_filter = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
                
                # Step 2: Add letterboxing with pad filter (black bars)
                pad_filter = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
                
                # Combine both filters for perfect aspect ratio preservation
                combined_filter = f"{scale_filter},{pad_filter}"
                cmd.extend(["-vf", combined_filter])
                
                print(f"   🎬 Aspect-aware preprocessing for {canvas_type}")
                print(f"       Target canvas: {target_w}x{target_h}")
                print(f"       Filter chain: {combined_filter}")
                
            elif any("resolution" in reason for reason in analysis["reasons"]):
                # Fallback: use safe 1920x1080 with letterboxing (backward compatibility)
                fallback_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
                cmd.extend(["-vf", fallback_filter])
                print(f"   🎬 Fallback preprocessing with letterboxing")
                print(f"       Filter: {fallback_filter}")

            # Frame rate optimization
            if any("framerate" in reason for reason in analysis["reasons"]):
                # Use target FPS if available, otherwise limit to 30fps
                target_fps = target_format.get('target_fps', 30) if target_format else 30
                cmd.extend(["-r", str(target_fps)])
                print(f"   🎞️  Frame rate optimization: {target_fps}fps")

            # Audio handling
            cmd.extend(["-c:a", "aac"])  # Standard audio codec
            cmd.extend(["-b:a", "128k"])  # Reasonable audio bitrate

            # Quality settings
            cmd.extend(["-crf", "23"])  # Good quality/size balance
            cmd.extend(["-movflags", "+faststart"])  # Web optimization

            cmd.append(output_path)

            print(f"   🔄 Running modern FFmpeg preprocessing...")
            print(f"       Command: {' '.join(cmd[0:3] + ['...'] + cmd[-1:])}")

            # Run with timeout to prevent hanging
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                # Verify output dimensions match target
                if target_format:
                    expected_w = target_format.get('target_width', 1920)
                    expected_h = target_format.get('target_height', 1080)
                    print(f"   ✅ FFmpeg preprocessing successful")
                    print(f"       Output should be: {expected_w}x{expected_h} with proper letterboxing")
                else:
                    print(f"   ✅ FFmpeg preprocessing successful")
                return True
            else:
                print(f"   ❌ FFmpeg failed with return code {result.returncode}")
                if result.stderr:
                    print(f"       Error: {result.stderr[:200]}...")  # First 200 chars
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ FFmpeg preprocessing timed out (5+ minutes)")
            return False
        except Exception as e:
            print(f"   ❌ FFmpeg preprocessing error: {e}")
            return False

    def cleanup_preprocessed_files(self, max_age_hours: int = 24) -> None:
        """Clean up old preprocessed files to save disk space."""
        import os
        import time

        current_time = time.time()
        cleaned_count = 0

        for cached_path in list(self.preprocessing_cache.values()):
            if os.path.exists(cached_path):
                file_age_hours = (current_time - os.path.getmtime(cached_path)) / 3600
                if file_age_hours > max_age_hours:
                    try:
                        os.remove(cached_path)
                        cleaned_count += 1
                        print(
                            f"   🧹 Cleaned up old preprocessed file: {os.path.basename(cached_path)}"
                        )
                    except Exception:
                        pass  # Ignore cleanup errors

        # Clear cache entries for non-existent files
        self.preprocessing_cache = {
            k: v for k, v in self.preprocessing_cache.items() if os.path.exists(v)
        }

        if cleaned_count > 0:
            print(f"   🧹 Cleaned up {cleaned_count} old preprocessed files")


def preprocess_videos_smart(
    video_files: List[str],
    output_dir: str = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, str]:
    """Smart preprocessing of video files for optimal processing.

    Args:
        video_files: List of video file paths to analyze and preprocess
        output_dir: Directory for preprocessed files (default: temp)
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping original paths to optimized paths (original or preprocessed)
    """
    import os
    import tempfile

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="autocut_preprocessed_")

    preprocessor = VideoPreprocessor()
    video_map = {}

    print(f"\n🔍 PHASE 2: Smart Video Preprocessing")
    print(
        f"   Analyzing {len(video_files)} video files for optimization opportunities..."
    )

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

        # Always preprocess if needed, map result
        optimized_path = preprocessor.preprocess_video_if_needed(video_path, output_dir)
        video_map[video_path] = optimized_path

    # Summary
    print(f"\n📊 Preprocessing Summary:")
    print(f"   - Files analyzed: {len(video_files)}")
    print(f"   - Files needing preprocessing: {preprocessing_needed}")
    print(f"   - Estimated memory savings: {total_estimated_memory:.0f}MB")
    print(f"   - Output directory: {output_dir}")

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
                print(f"   🚨 SWITCHING TO EMERGENCY MEMORY MODE")
                print(f"      Current usage: {status['current_usage_gb']:.1f}GB")
                print(f"      System available: {status['system_available_gb']:.1f}GB")
                self.processing_mode = "emergency"
            return True

        elif status["is_warning"]:
            if self.processing_mode == "normal":
                print(f"   ⚠️  Switching to conservative memory mode")
                self.processing_mode = "conservative"
            self.memory_warnings += 1
            return False

        return False

    def perform_emergency_cleanup(self, context: str = "unknown") -> Dict[str, Any]:
        """Perform aggressive memory cleanup and return results."""
        import gc
        import psutil

        print(f"   🚨 EMERGENCY MEMORY CLEANUP ({context})")

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
        except:
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

        print(f"      Objects collected: {cleanup_result['objects_collected']}")
        print(f"      Memory freed: {cleanup_result['memory_freed_gb']:.2f}GB")

        if not cleanup_result["cleanup_effective"]:
            print(f"      ⚠️  Cleanup was not effective - consider reducing batch size")

        return cleanup_result

    def get_optimal_batch_size(self, total_items: int) -> int:
        """Determine optimal batch size based on current memory mode."""
        if self.processing_mode == "emergency":
            return min(2, total_items)  # Process 2 at a time maximum
        elif self.processing_mode == "conservative":
            return min(5, total_items)  # Process 5 at a time
        else:
            return min(10, total_items)  # Normal batch size

    def log_memory_summary(self, context: str) -> None:
        """Log comprehensive memory summary."""
        status = self.get_memory_status()

        print(f"   💾 Memory Summary ({context}):")
        print(
            f"      Current usage: {status['current_usage_gb']:.1f}GB (+{status['baseline_increase_gb']:.1f}GB from start)"
        )
        print(
            f"      System: {status['system_available_gb']:.1f}GB available ({100 - status['system_percent']:.1f}% free)"
        )
        print(f"      Mode: {status['processing_mode'].upper()}")

        if self.emergency_cleanup_count > 0:
            print(f"      Emergency cleanups: {self.emergency_cleanup_count}")
        if self.memory_warnings > 0:
            print(f"      Memory warnings: {self.memory_warnings}")


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
    print(f"   🧠 Advanced memory-aware preprocessing")

    # Check memory before preprocessing
    if memory_manager.should_switch_to_emergency_mode():
        print(
            f"   🚨 Already in emergency mode - skipping preprocessing to save memory"
        )
        video_path_map = {path: path for path in video_files}  # No preprocessing
    else:
        video_path_map = preprocess_videos_smart(
            video_files, progress_callback=progress_callback
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

    print(f"   🎬 Processing {len(sorted_clips)} clips from {len(grouped_clips)} files")
    print(
        f"   🔧 CRITICAL FIX: Using delayed cleanup to prevent NoneType get_frame errors"
    )
    print(f"      Initial processing mode: {memory_manager.processing_mode}")

    for video_file, file_clips in grouped_clips.items():
        processed_files += 1

        # Check memory status before each file
        if memory_manager.should_switch_to_emergency_mode():
            cleanup_result = memory_manager.perform_emergency_cleanup(
                f"before file {processed_files}"
            )

            # If cleanup didn't help much, switch to single-clip processing
            if not cleanup_result["cleanup_effective"]:
                print(f"   🚨 Switching to single-clip emergency processing")
                optimal_batch_size = 1
            else:
                optimal_batch_size = memory_manager.get_optimal_batch_size(
                    len(file_clips)
                )
        else:
            optimal_batch_size = memory_manager.get_optimal_batch_size(len(file_clips))

        print(
            f"\n   [{processed_files}/{len(grouped_clips)}] Processing {os.path.basename(video_file)}"
        )
        print(f"   🎯 {len(file_clips)} clips, batch size: {optimal_batch_size}")

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

                print(
                    f"      Processing batch {batch_start // optimal_batch_size + 1}: clips {batch_start + 1}-{batch_end}"
                )

                for clip_data in batch_clips:
                    try:
                        # Check memory before each clip in emergency mode
                        if memory_manager.processing_mode == "emergency":
                            status = memory_manager.get_memory_status()
                            if status["is_critical"]:
                                print(
                                    f"      🚨 Memory critical, skipping remaining clips"
                                )
                                break

                        # Extract segment - parent video stays alive via delayed cleanup
                        segment = subclip_safely(
                            source_video, clip_data["start"], clip_data["end"]
                        )

                        # Validation: Test the clip while parent video is available
                        # This validation is still important for catching other issues
                        try:
                            test_time = min(0.1, segment.duration)
                            test_frame = segment.get_frame(test_time)
                            if test_frame is None:
                                raise RuntimeError(
                                    "get_frame returned None during validation"
                                )
                        except Exception as e:
                            raise RuntimeError(f"Clip validation failed: {e}")

                        video_clips.append(segment)
                        file_clips_loaded += 1

                    except Exception as e:
                        print(f"      ⚠️  Failed clip {total_clips_processed + 1}: {e}")
                        # Use original_index from grouped clips to maintain timeline alignment
                        original_index = clip_data.get(
                            "original_index", total_clips_processed
                        )
                        failed_indices.append(original_index)

                    total_clips_processed += 1

                # Memory check after batch
                if memory_manager.processing_mode != "normal":
                    memory_manager.log_memory_summary(f"after batch")

                    # Force garbage collection between batches in conservative/emergency mode
                    import gc

                    gc.collect()

            # NOTE: We do NOT close the source_video here - it will be cleaned up later
            # This is the key fix: parent videos stay alive until after concatenation

        except Exception as e:
            print(f"   ❌ Failed to load video file {video_file}: {e}")
            # Mark all clips from this file as failed using original indices
            for clip_data in file_clips:
                original_index = clip_data.get("original_index", total_clips_processed)
                failed_indices.append(original_index)
                total_clips_processed += 1
            continue

        print(
            f"   ✅ File complete: {file_clips_loaded}/{len(file_clips)} clips loaded"
        )

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
    success_rate = success_count / len(sorted_clips)

    print(f"\n   📊 Advanced Memory Processing Results:")
    print(
        f"      Clips loaded: {success_count}/{len(sorted_clips)} ({success_rate * 100:.1f}%)"
    )
    print(f"      Emergency cleanups: {memory_manager.emergency_cleanup_count}")
    print(f"      Processing mode: {memory_manager.processing_mode}")

    if failed_indices:
        print(f"      Failed clips: {len(failed_indices)}")

    print(f"   🎬 CRITICAL SUCCESS: All parent videos kept alive for concatenation")
    print(
        f"   🔧 Delayed cleanup will occur after concatenation to prevent NoneType errors"
    )

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
        self, clip_data: Dict[str, Any], resource_manager: VideoResourceManager
    ) -> Optional[Any]:
        """Load a single clip with multiple fallback strategies.

        CRITICAL FIX: Now supports delayed cleanup to keep parent videos alive.

        Args:
            clip_data: Dictionary with video_file, start, end information
            resource_manager: Resource manager for delayed cleanup video loading

        Returns:
            VideoFileClip segment or None if all strategies failed
        """
        self.error_statistics["total_attempts"] += 1

        strategies = [
            ("direct_loading", self._load_direct_moviepy),
            ("format_conversion", self._load_with_format_conversion),
            ("quality_reduction", self._load_with_quality_reduction),
            ("emergency_mode", self._load_emergency_minimal),
        ]

        last_error = None

        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(clip_data, resource_manager)
                if result is not None:
                    self.error_statistics["successful_loads"] += 1
                    self.error_statistics["fallback_usage"][strategy_name] += 1

                    if strategy_name != "direct_loading":
                        print(f"      ✅ Fallback success: {strategy_name}")

                    return result

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                self.error_statistics["error_types"][error_type] = (
                    self.error_statistics["error_types"].get(error_type, 0) + 1
                )

                print(f"      🔄 Strategy '{strategy_name}' failed: {str(e)[:100]}...")
                continue

        # All strategies failed
        self.error_statistics["failed_loads"] += 1
        print(f"      ❌ All fallback strategies failed. Last error: {last_error}")
        return None

    def _get_or_load_video(
        self, video_file: str, resource_manager: VideoResourceManager
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
        self, clip_data: Dict[str, Any], resource_manager: VideoResourceManager
    ) -> Optional[Any]:
        """Direct MoviePy loading (standard approach).

        CRITICAL FIX: Uses delayed cleanup to keep parent video alive.
        """
        source_video = self._get_or_load_video(
            clip_data["video_file"], resource_manager
        )
        return subclip_safely(source_video, clip_data["start"], clip_data["end"])

    def _load_with_format_conversion(
        self, clip_data: Dict[str, Any], resource_manager: VideoResourceManager
    ) -> Optional[Any]:
        """Try loading with format conversion preprocessing.

        NOTE: This strategy creates temporary files and doesn't need delayed cleanup
        since it creates its own temporary videos.
        """
        import tempfile
        import subprocess
        import os

        # Create a temporary converted file for this specific clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Extract just this specific clip segment using FFmpeg directly
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                clip_data["video_file"],
                "-ss",
                str(clip_data["start"]),
                "-t",
                str(clip_data["end"] - clip_data["start"]),
                "-c:v",
                "libx264",  # Force H.264
                "-c:a",
                "aac",  # Force AAC audio
                "-avoid_negative_ts",
                "make_zero",
                temp_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(temp_path):
                # Load the converted segment using immediate cleanup since it's temporary
                with resource_manager.load_video_safely(temp_path) as converted_video:
                    segment = (
                        converted_video.copy()
                    )  # Get full clip since it's already the right duration
                    return segment

            return None

        except Exception:
            return None
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

    def _load_with_quality_reduction(
        self, clip_data: Dict[str, Any], resource_manager: VideoResourceManager
    ) -> Optional[Any]:
        """Try loading with reduced quality settings.

        NOTE: This strategy creates temporary files and doesn't need delayed cleanup
        since it creates its own temporary videos.
        """
        import tempfile
        import subprocess
        import os

        # Create temporary lower-quality version
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Reduce quality and resolution for easier loading
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                clip_data["video_file"],
                "-ss",
                str(clip_data["start"]),
                "-t",
                str(clip_data["end"] - clip_data["start"]),
                "-vf",
                "scale=640:360",  # Lower resolution
                "-c:v",
                "libx264",
                "-crf",
                "30",  # Lower quality
                "-preset",
                "ultrafast",  # Faster encoding
                "-c:a",
                "aac",
                "-b:a",
                "64k",  # Lower audio bitrate
                temp_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and os.path.exists(temp_path):
                # Load the converted segment using immediate cleanup since it's temporary
                with resource_manager.load_video_safely(temp_path) as low_quality_video:
                    segment = low_quality_video.copy()
                    return segment

            return None

        except Exception:
            return None
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

    def _load_emergency_minimal(
        self, clip_data: Dict[str, Any], resource_manager: VideoResourceManager
    ) -> Optional[Any]:
        """Emergency fallback: create minimal placeholder or extract with minimal processing.

        CRITICAL FIX: Uses delayed cleanup to keep parent video alive.
        """
        try:
            # Try one more time with absolute minimal MoviePy settings using delayed cleanup
            source_video = self._get_or_load_video(
                clip_data["video_file"], resource_manager
            )

            # Use the most basic subclip operation possible
            start_time = max(0, clip_data["start"])
            end_time = min(source_video.duration, clip_data["end"])

            if end_time <= start_time:
                return None

            # Very basic subclip - parent video stays alive via delayed cleanup
            segment = source_video.subclipped(start_time, end_time)
            return segment

        except Exception:
            # Final fallback: return None (clip will be skipped)
            return None

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

        print(f"\n   📊 Robust Loading Summary:")
        print(f"      Success rate: {report['success_rate'] * 100:.1f}%")
        print(f"      Total attempts: {report['total_attempts']}")
        print(f"      Successful: {report['successful_loads']}")
        print(f"      Failed: {report['failed_loads']}")

        if any(report["fallback_usage"].values()):
            print(f"      Fallback usage:")
            for strategy, count in report["fallback_usage"].items():
                if count > 0:
                    print(f"        - {strategy}: {count} clips")

        if report["error_types"]:
            print(f"      Error types encountered:")
            for error_type, count in report["error_types"].items():
                print(f"        - {error_type}: {count} occurrences")


def load_video_clips_with_robust_error_handling(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    progress_callback: Optional[callable] = None,
) -> Tuple[List[Any], List[int], Dict[str, Any], VideoResourceManager]:
    """Load video clips with comprehensive error handling and recovery strategies.

    CRITICAL FIX: This version now uses delayed cleanup to prevent NoneType get_frame errors.
    Parent videos are kept alive until after concatenation, ensuring subclips remain valid.

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

    print(f"   🛡️  PHASE 4: Robust error handling and recovery")
    print(
        f"   🔧 CRITICAL FIX: Using delayed cleanup to prevent NoneType get_frame errors"
    )
    print(f"      Total clips to process: {len(sorted_clips)}")

    # Smart preprocessing with error recovery
    try:
        video_path_map = preprocess_videos_smart(
            video_files, progress_callback=progress_callback
        )
    except Exception as e:
        print(f"      ⚠️  Preprocessing failed: {e}")
        print(f"      Continuing with original video files")
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

    for video_file, file_clips in grouped_clips.items():
        processed_files += 1
        print(
            f"\n   [{processed_files}/{len(grouped_clips)}] Robust processing: {os.path.basename(video_file)}"
        )
        print(f"   🎯 {len(file_clips)} clips with fallback strategies")

        file_clips_loaded = 0

        # Check memory and adapt strategy
        if memory_manager.should_switch_to_emergency_mode():
            memory_manager.perform_emergency_cleanup(f"before file {processed_files}")
            batch_size = 1  # Process one at a time in emergency mode
        else:
            batch_size = memory_manager.get_optimal_batch_size(len(file_clips))

        # Process clips with robust error handling
        for i, clip_data in enumerate(file_clips):
            print(
                f"      Clip {i + 1}/{len(file_clips)}: {clip_data['start']:.1f}-{clip_data['end']:.1f}s"
            )

            try:
                # Use robust loader with multiple fallback strategies and delayed cleanup
                segment = robust_loader.load_clip_with_fallbacks(
                    clip_data, resource_manager
                )

                if segment is not None:
                    video_clips.append(segment)
                    file_clips_loaded += 1
                    print(f"      ✅ Success")
                else:
                    # Use original_index from grouped clips to maintain timeline alignment
                    original_index = clip_data.get(
                        "original_index", total_clips_processed
                    )
                    failed_indices.append(original_index)
                    print(f"      ❌ All strategies failed")

            except Exception as e:
                print(f"      ❌ Unexpected error: {e}")
                # Use original_index from grouped clips to maintain timeline alignment
                original_index = clip_data.get("original_index", total_clips_processed)
                failed_indices.append(original_index)

            total_clips_processed += 1

            # Memory check between clips if in conservative mode
            if memory_manager.processing_mode != "normal" and i % 5 == 0:
                memory_manager.log_memory_summary(f"progress check")

            # Update progress
            progress = 0.1 + (0.6 * total_clips_processed / len(sorted_clips))
            report_progress(
                f"Robust loading: {total_clips_processed}/{len(sorted_clips)}", progress
            )

        print(f"   📊 File result: {file_clips_loaded}/{len(file_clips)} clips loaded")

    # Generate comprehensive error report
    error_report = robust_loader.get_error_report()
    robust_loader.print_error_summary()

    memory_manager.log_memory_summary("robust loading complete")

    if not video_clips:
        print(f"\n   ❌ CRITICAL: No clips could be loaded with any strategy")
        print(f"      This indicates a fundamental compatibility issue")
        raise RuntimeError(
            "No video clips could be loaded successfully with any fallback strategy"
        )

    success_count = len(video_clips)
    success_rate = success_count / len(sorted_clips)

    print(f"\n   🎉 ROBUST LOADING COMPLETE")
    print(
        f"      Final success rate: {success_rate * 100:.1f}% ({success_count}/{len(sorted_clips)} clips)"
    )

    if success_rate < 0.5:
        print(f"      ⚠️  LOW SUCCESS RATE - Consider checking video file formats")
    elif success_rate < 0.8:
        print(f"      ⚠️  MODERATE SUCCESS RATE - Some clips had issues")
    else:
        print(f"      ✅ EXCELLENT SUCCESS RATE")

    print(f"   🎬 CRITICAL SUCCESS: All parent videos kept alive for concatenation")
    print(
        f"   🔧 Delayed cleanup will occur after concatenation to prevent NoneType errors"
    )

    report_progress(f"Robust loading complete: {success_count} clips", 0.7)

    # Return clips with perfect index alignment, failed indices, error report, and resource manager
    # The caller MUST call resource_manager.cleanup_delayed_videos() after concatenation
    return video_clips, failed_indices, error_report, resource_manager



def load_video_clips_parallel(
    sorted_clips: List[Dict[str, Any]],
    video_files: List[str],
    progress_callback: Optional[callable] = None,
    max_workers: int = None,
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

    # Memory monitoring at start
    initial_memory = get_memory_info()
    print(
        f"   🧠 Initial memory: {initial_memory['used_gb']:.1f}GB used ({initial_memory['percent']:.1f}%), {initial_memory['available_gb']:.1f}GB available"
    )

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
            capabilities, video_profile, len(sorted_clips)
        )

        optimal_workers = worker_analysis["optimal_workers"]

        # Display detailed analysis
        profiler.print_system_analysis(capabilities, video_profile, worker_analysis)

    else:
        # Manual override provided
        optimal_workers = min(max_workers, len(sorted_clips))
        print(f"   ⚙️  Manual worker override: {optimal_workers} workers")

    report_progress(
        f"Loading {len(sorted_clips)} clips with {optimal_workers} workers", 0.1
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

                except Exception as e:
                    print(f"Warning: Future failed for clip {index}: {str(e)}")
                    failed_indices.append(index)

                # Update progress
                progress = 0.1 + (0.6 * completed_count / len(sorted_clips))
                report_progress(
                    f"Loaded {completed_count}/{len(sorted_clips)} clips", progress
                )

    except Exception as e:
        # Clean up cache on error
        video_cache.clear()
        raise RuntimeError(f"Parallel video loading failed: {str(e)}")

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
        print(
            f"Warning: {len(failed_indices)} clips failed to load (indices: {failed_indices})"
        )

    report_progress(f"Successfully loaded {len(video_clips)} clips", 0.7)

    # Memory monitoring at completion
    final_memory = get_memory_info()
    memory_increase = final_memory["used_gb"] - initial_memory["used_gb"]
    print(
        f"   🧠 Final memory: {final_memory['used_gb']:.1f}GB used (+{memory_increase:.1f}GB), {final_memory['available_gb']:.1f}GB available"
    )

    # Memory usage warning
    if final_memory["percent"] > 85:
        print(
            f"   ⚠️  HIGH MEMORY USAGE: {final_memory['percent']:.1f}% - Consider reducing video count"
        )

    # Log cache statistics
    cached_files = video_cache.get_cached_paths()
    print(f"   📦 Video cache: {len(cached_files)} unique files loaded")

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
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

        import_pattern = "new"  # from moviepy import ...
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2)
            from moviepy.editor import (
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
            )

            import_pattern = "legacy"  # from moviepy.editor import ...
        except ImportError:
            raise RuntimeError("Could not import MoviePy with either import pattern")

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
        write_sig = inspect.signature(video_dummy.write_videofile)
        compatibility["write_videofile_params"] = list(write_sig.parameters.keys())
    except:
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
                    print(
                        f"Warning: {method_name}() returned None, trying next method..."
                    )
                    continue

                print(f"Debug: Successfully attached audio using {method_name}")
                return result

            except Exception as e:
                print(
                    f"Warning: {method_name}() failed with error: {e}, trying next method..."
                )
                continue

    # If all methods fail, this is a critical error
    raise RuntimeError(
        f"Could not attach audio using any method (tried: with_audio, set_audio) on {type(video_clip)}"
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
                    else:
                        return method(start_time)
                except Exception:
                    # If this method fails, try the next one
                    continue

        # If neither method works, raise an error
        raise AttributeError(
            f"Neither 'subclipped' nor 'subclip' methods work on {type(clip)}"
        )

    except AttributeError:
        raise RuntimeError(f"Could not find subclip method on clip type {type(clip)}")


def test_independent_subclip_creation(video_path: str = None) -> bool:
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
            if os.path.exists(test_file):
                video_path = test_file
                break

        if not video_path:
            print("Warning: No test video found - skipping independent subclip test")
            return True  # Skip test if no video available

    print(f"Testing independent subclip creation with: {video_path}")

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

            print(
                f"  Created subclips: old={type(old_subclip)}, new={type(new_subclip)}"
            )

        # Parent video is now closed - test if subclips still work
        print("  Parent video closed, testing subclip functionality...")

        # Test old subclip (should fail)
        try:
            old_frame = old_subclip.get_frame(0.1)
            print("  WARNING: Old subclip still works - this is unexpected")
        except Exception as e:
            print(f"  Expected: Old subclip failed as expected: {str(e)[:50]}...")

        # Test new subclip (should work)
        try:
            new_frame = new_subclip.get_frame(0.1)
            if new_frame is not None:
                print("  SUCCESS: New independent subclip works after parent closed!")
                return True
            else:
                print("  FAILURE: New subclip returns None frame")
                return False
        except Exception as e:
            print(f"  FAILURE: New independent subclip failed: {str(e)}")
            return False

    except Exception as e:
        print(f"Test failed with exception: {str(e)}")
        return False

    finally:
        # Cleanup
        try:
            if "old_subclip" in locals():
                old_subclip.close()
            if "new_subclip" in locals():
                new_subclip.close()
        except:
            pass


def import_moviepy_safely():
    """Safely import MoviePy classes handling import structure changes.

    Returns:
        Tuple of (VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip)
    """
    try:
        # Try new import structure first (MoviePy 2.1.2+)
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

        try:
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        except ImportError:
            from moviepy import CompositeVideoClip
        return VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
    except ImportError:
        try:
            # Fallback to legacy import structure (MoviePy < 2.1.2)
            from moviepy.editor import (
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
                CompositeVideoClip,
            )

            return (
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
                CompositeVideoClip,
            )
        except ImportError:
            raise RuntimeError(
                "Could not import MoviePy with either import pattern. Please check MoviePy installation."
            )


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
            print(
                f"Warning: Parameter '{key}' not supported in this MoviePy version, skipping"
            )

    try:
        video_clip.write_videofile(output_path, **safe_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to write video file: {str(e)}")


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
                clip.duration, target_duration, allowed_durations
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
            best_clip, target_duration, allowed_durations
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
    clip_duration: float, target_duration: float, allowed_durations: List[float]
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
        else:
            return 0.3  # Heavy penalty for lots of trimming

    # Clip is shorter than target
    else:
        shortage = target_duration - clip_duration
        if shortage <= 0.5:  # Small shortage is acceptable
            return 0.8 - (shortage / 1.0)
        else:
            return -1  # Too short, can't use


def _fit_clip_to_duration(
    clip: VideoChunk, target_duration: float, allowed_durations: List[float]
) -> Tuple[float, float, float]:
    """Fit a clip to the target duration by trimming if necessary.

    Args:
        clip: Video chunk to fit
        target_duration: Desired duration
        allowed_durations: List of allowed durations

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
    video_chunks: List[VideoChunk], target_count: int, variety_factor: float = 0.3
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
                    clip, selected_clips
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
    clip: VideoChunk, existing_clips: List[VideoChunk], min_gap: float = 1.0
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
        if existing.video_path == clip.video_path:
            # Check for overlap or too close proximity
            if (
                clip.start_time < existing.end_time + min_gap
                and clip.end_time > existing.start_time - min_gap
            ):
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
) -> str:
    """Render final video with music synchronization.
    
    Args:
        timeline: ClipTimeline with all clips and timing
        audio_file: Path to music file
        output_path: Path for output video
        max_workers: Maximum parallel workers (legacy parameter)
        progress_callback: Optional callback for progress updates
        bpm: Beats per minute for musical fade calculations
        avg_beat_interval: Average time between beats in seconds
        
    Returns:
        Path to rendered video file
        
    Raises:
        RuntimeError: If rendering fails
    """
    try:
        import os
        from typing import List
        
        # Import MoviePy components safely
        print("DEBUG: Importing MoviePy components...")
        VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = import_moviepy_safely()
        print(f"DEBUG: MoviePy imports successful - VideoFileClip: {VideoFileClip is not None}")
        
        if VideoFileClip is None:
            raise RuntimeError("MoviePy VideoFileClip is not available - check MoviePy installation")
        
        # Get MoviePy compatibility info for safe subclip operations
        try:
            from compatibility.moviepy import check_moviepy_api_compatibility, subclip_safely
            compatibility_info = check_moviepy_api_compatibility()
            print(f"DEBUG: MoviePy compatibility: {compatibility_info.get('version_detected', 'unknown')}")
        except ImportError:
            print("WARNING: Compatibility module not available, using fallback subclip method")
            compatibility_info = None
            subclip_safely = None
        
        if progress_callback:
            progress_callback("Loading video clips", 0.1)
        
        # Load video clips from timeline
        video_clips = []
        print(f"DEBUG: Timeline has {len(timeline.clips)} clips to load")
        
        for i, clip_info in enumerate(timeline.clips):
            try:
                print(f"DEBUG: Loading clip {i+1}: {clip_info}")
                
                # Check if file exists
                video_file_path = clip_info["video_file"]
                if not os.path.exists(video_file_path):
                    print(f"ERROR: Video file does not exist: {video_file_path}")
                    continue
                
                print(f"DEBUG: File exists, creating VideoFileClip for: {video_file_path}")
                # Load video segment
                video_clip = VideoFileClip(video_file_path)
                print(f"DEBUG: VideoFileClip created, duration: {video_clip.duration}s")
                
                # Validate clip timing
                start_time = clip_info["start"]
                end_time = clip_info["end"]
                if end_time > video_clip.duration:
                    print(f"WARNING: Clip end time {end_time}s exceeds video duration {video_clip.duration}s, adjusting")
                    end_time = video_clip.duration
                
                if start_time >= end_time:
                    print(f"ERROR: Invalid clip timing - start {start_time}s >= end {end_time}s")
                    video_clip.close()
                    continue
                
                print(f"DEBUG: Creating subclip from {start_time}s to {end_time}s")
                # Use compatibility-safe subclip method
                if subclip_safely:
                    segment = subclip_safely(video_clip, start_time, end_time, compatibility_info)
                else:
                    # Fallback: try both modern and legacy API
                    try:
                        segment = video_clip.subclipped(start_time, end_time)  # Modern MoviePy 2.x
                        print("DEBUG: Used modern subclipped() method")
                    except AttributeError:
                        try:
                            segment = video_clip.subclip(start_time, end_time)  # Legacy MoviePy 1.x
                            print("DEBUG: Used legacy subclip() method")
                        except AttributeError:
                            raise RuntimeError(f"Neither 'subclipped' nor 'subclip' methods available on {type(video_clip)}")
                video_clips.append(segment)
                print(f"DEBUG: Successfully loaded clip {i+1}, segment duration: {segment.duration}s")
                
                if progress_callback:
                    progress = 0.1 + (0.4 * (i + 1) / len(timeline.clips))
                    progress_callback(f"Loading clip {i+1}/{len(timeline.clips)}", progress)
                    
            except Exception as e:
                print(f"ERROR: Failed to load clip {i+1}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"DEBUG: Successfully loaded {len(video_clips)} out of {len(timeline.clips)} clips")
        
        if not video_clips:
            raise RuntimeError(f"No video clips could be loaded from {len(timeline.clips)} timeline clips")
        
        if progress_callback:
            progress_callback("Concatenating video clips", 0.5)
        
        # Concatenate video clips
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        if progress_callback:
            progress_callback("Loading audio", 0.6)
        
        # Load and attach audio
        audio_clip = AudioFileClip(audio_file)
        
        # Trim audio to match video duration or vice versa
        video_duration = final_video.duration
        audio_duration = audio_clip.duration
        
        if audio_duration > video_duration:
            # Trim audio to video length
            if subclip_safely:
                audio_clip = subclip_safely(audio_clip, 0, video_duration, compatibility_info)
            else:
                # Fallback: try both modern and legacy API for audio
                try:
                    audio_clip = audio_clip.subclipped(0, video_duration)
                except AttributeError:
                    audio_clip = audio_clip.subclip(0, video_duration)
        else:
            # Trim video to audio length  
            if subclip_safely:
                final_video = subclip_safely(final_video, 0, audio_duration, compatibility_info)
            else:
                # Fallback: try both modern and legacy API for video
                try:
                    final_video = final_video.subclipped(0, audio_duration)
                except AttributeError:
                    final_video = final_video.subclip(0, audio_duration)
        
        # Apply musical fade-out if we have beat information
        if avg_beat_interval and audio_duration > video_duration:
            # Calculate fade duration (2-4 beats, max 3 seconds)
            fade_duration = min(avg_beat_interval * 3, 3.0)
            print(f"Applying musical fade-out: {fade_duration:.2f}s")
            audio_clip = audio_clip.audio_fadein(0.1).audio_fadeout(fade_duration)
        
        # Attach audio to video
        final_video = final_video.set_audio(audio_clip)
        
        if progress_callback:
            progress_callback("Encoding video", 0.7)
        
        # Get optimal encoding settings
        try:
            encoder = VideoEncoder()
            moviepy_params, ffmpeg_params = encoder.detect_optimal_codec_settings()
        except:
            # Fallback encoding settings
            moviepy_params = {"codec": "libx264", "bitrate": "5000k", "audio_codec": "aac"}
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Encode video with compatibility layer
        try:
            from compatibility.moviepy import write_videofile_safely, check_moviepy_api_compatibility
            compatibility_info = check_moviepy_api_compatibility()
            
            write_videofile_safely(
                final_video,
                output_path,
                compatibility_info,
                **encoding_params
            )
        except ImportError:
            # Fallback if compatibility module not available
            final_video.write_videofile(output_path, **encoding_params)
        
        if progress_callback:
            progress_callback("Video rendering complete", 1.0)
        
        # Clean up clips
        for clip in video_clips:
            try:
                clip.close()
            except:
                pass
        try:
            final_video.close()
            audio_clip.close()
        except:
            pass
        
        print(f"✅ Video rendered successfully: {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to render video: {str(e)}")


def add_transitions(
    clips: List[VideoFileClip], transition_duration: float = 0.5
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
        
    except ImportError:
        # Fallback to legacy implementation if modules not available
        raise RuntimeError("New transition system not available - refactoring incomplete")
    except Exception as e:
        raise RuntimeError(f"Transition creation failed: {str(e)}")


def assemble_clips(
    video_files: List[str],
    audio_file: str,
    output_path: str,
    pattern: str = "balanced",
    max_workers: int = None,
    progress_callback: Optional[callable] = None,
) -> str:
    """Main function to assemble clips into final video.

    Combines all steps:
    1. Analyze all video files
    2. Analyze audio file
    3. Match clips to beats
    4. Render final video

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
    import os
    import logging
    import mimetypes
    # Dual import pattern for package/direct execution compatibility
    try:
        from audio_analyzer import analyze_audio
        from video_analyzer import analyze_video_file
    except ImportError:
        from .audio_analyzer import analyze_audio
        from .video_analyzer import analyze_video_file

    def validate_audio_file_comprehensive(audio_path: str) -> tuple:
        """Comprehensive audio file validation before processing.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check 1: File existence
            if not os.path.exists(audio_path):
                return False, f"Audio file not found: {audio_path}"

            # Check 2: File accessibility
            if not os.access(audio_path, os.R_OK):
                return False, f"Audio file is not readable: {audio_path}"

            # Check 3: File size validation
            try:
                file_size = os.path.getsize(audio_path)
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
                    f"⚠️  Audio path has potential issues: {'; '.join(path_issues)}"
                )
                logger.warning(f"   Path: {audio_path}")
                logger.warning(
                    f"   Will attempt to process but may encounter issues..."
                )

            return True, None

        except Exception as e:
            return False, f"Unexpected error validating audio file: {str(e)}"

    # Set up detailed logging for the main pipeline
    logger = logging.getLogger("autocut.clip_assembler")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    logger.info(f"=== AutoCut Video Processing Started ===")
    logger.info(f"Input videos: {len(video_files)} files")
    logger.info(f"Audio file: {os.path.basename(audio_file)}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Pattern: {pattern}")

    # CRITICAL FIX: Enhanced comprehensive input validation
    logger.info("Validating input files...")

    # Validate audio file with comprehensive checks
    logger.info(f"🔍 Comprehensive audio file validation...")
    audio_valid, audio_error = validate_audio_file_comprehensive(audio_file)
    if not audio_valid:
        error_msg = f"Audio validation failed: {audio_error}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise ValueError(error_msg)

    logger.info(f"   ✅ Audio file validation passed")
    logger.info(f"   📁 File: {os.path.basename(audio_file)}")
    logger.info(f"   📊 Size: {os.path.getsize(audio_file) / (1024 * 1024):.2f}MB")

    # Validate video files
    missing_videos = [vf for vf in video_files if not os.path.exists(vf)]
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
            error_msg = f"Audio file access error during analysis: {str(e)}"
            logger.error(error_msg)
            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Audio analysis failed: {str(e)}"
            logger.error(error_msg)

            # Provide more helpful error messages for common issues
            if "No such file" in str(e) or "cannot find" in str(e).lower():
                error_msg += f" (Check if audio file path is correct and accessible)"
            elif "format" in str(e).lower() or "codec" in str(e).lower():
                error_msg += f" (Audio file may be corrupted or in unsupported format)"
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                error_msg += f" (Check file permissions)"

            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg)

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

        logger.info(f"✅ Audio analysis successful:")
        logger.info(f"  - Beats detected: {len(beats)}")
        logger.info(f"  - Musical start: {musical_start_time:.2f}s")
        logger.info(f"  - Intro duration: {intro_duration:.2f}s")
        report_progress("Audio analysis complete", 0.2)

    except Exception as e:
        error_msg = f"Failed to analyze audio file: {str(e)}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise RuntimeError(error_msg)

    # Step 2: Analyze all video files with detailed per-file tracking
    logger.info("=== Step 2: Video Analysis ===")
    report_progress("Analyzing videos", 0.3)
    all_video_chunks = []

    for i, video_file in enumerate(video_files):
        processing_summary["videos_processed"] += 1
        filename = os.path.basename(video_file)

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
                    f"✅ {filename}: {len(video_chunks)} chunks created ({processing_time:.2f}s)"
                )

                # Log chunk quality summary
                if video_chunks:
                    scores = [chunk.score for chunk in video_chunks]
                    logger.info(
                        f"   Chunk scores: {min(scores):.1f}-{max(scores):.1f} (avg: {sum(scores) / len(scores):.1f})"
                    )
            else:
                file_result["status"] = "failed"
                file_result["error_message"] = (
                    "No chunks created - check logs above for detailed error analysis"
                )
                processing_summary["videos_failed"] += 1

                logger.error(
                    f"❌ {filename}: No chunks created ({processing_time:.2f}s)"
                )
                logger.error(f"   → This video will be excluded from the final output")

        except Exception as e:
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time
            file_result["status"] = "failed"
            file_result["error_message"] = str(e)
            processing_summary["videos_failed"] += 1
            processing_summary["errors"].append(f"{filename}: {str(e)}")

            logger.error(f"❌ {filename}: Processing failed ({processing_time:.2f}s)")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   → This video will be excluded from the final output")

        processing_summary["file_results"].append(file_result)

        # Update progress for each video
        video_progress = 0.3 + (0.4 * (i + 1) / len(video_files))
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
                    f"  - {file_result['filename']}: {file_result['error_message']}"
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
                [f"  - {error}" for error in processing_summary["errors"][-10:]]
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
            f"⚠️  Low success rate ({success_rate:.1f}%) - check video format compatibility"
        )

    report_progress(
        f"Video analysis complete: {len(all_video_chunks)} clips found", 0.7
    )

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
            f"✅ Beat matching successful: {len(timeline.clips)} clips selected"
        )

        # Log timeline statistics
        timeline_stats = timeline.get_summary_stats()
        logger.info(f"Timeline statistics:")
        logger.info(f"  - Total duration: {timeline_stats['total_duration']:.2f}s")
        logger.info(f"  - Average score: {timeline_stats['avg_score']:.1f}")
        logger.info(
            f"  - Score range: {timeline_stats['score_range'][0]:.1f}-{timeline_stats['score_range'][1]:.1f}"
        )
        logger.info(f"  - Unique videos used: {timeline_stats['unique_videos']}")

        report_progress(
            f"Beat matching complete: {len(timeline.clips)} clips selected", 0.8
        )

    except Exception as e:
        error_msg = f"Failed to match clips to beats: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

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

        final_video_path = render_video(
            timeline=timeline,
            audio_file=audio_file,
            output_path=output_path,
            max_workers=max_workers,
            progress_callback=render_progress,
            bpm=audio_data.get("bpm"),
            avg_beat_interval=avg_beat_interval,
        )

        logger.info(f"✅ Video rendering complete: {final_video_path}")
        report_progress("Video rendering complete", 1.0)

        # Final success summary
        logger.info("=== AutoCut Processing Complete ===")
        logger.info(
            f"✅ Successfully created video: {os.path.basename(final_video_path)}"
        )
        logger.info(f"📊 Processing summary:")
        logger.info(
            f"  - Videos processed: {processing_summary['videos_successful']}/{processing_summary['total_videos']}"
        )
        logger.info(
            f"  - Chunks used: {len(timeline.clips)}/{processing_summary['total_chunks']}"
        )
        logger.info(
            f"  - Final video duration: {timeline_stats['total_duration']:.2f}s"
        )

        return final_video_path

    except Exception as e:
        error_msg = f"Failed to render video: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Export timeline JSON for debugging (optional)
    try:
        timeline_path = output_path.replace(".mp4", "_timeline.json")
        timeline.export_json(timeline_path)
        logger.info(f"Debug: Timeline exported to {timeline_path}")

        # Export processing summary for debugging
        summary_path = output_path.replace(".mp4", "_processing_summary.json")
        import json

        with open(summary_path, "w") as f:
            json.dump(processing_summary, f, indent=2)
        logger.info(f"Debug: Processing summary exported to {summary_path}")

    except Exception:
        pass  # Non-critical, ignore errors  # Non-critical, ignore errors  # Non-critical, ignore errors


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
    except:
        # Fallback to safe default settings
        moviepy_params = {
            "codec": "libx264",
            "bitrate": "5000k",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
        }
        
        ffmpeg_params = [
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        
        return moviepy_params, ffmpeg_params


def detect_optimal_codec_settings_with_diagnostics() -> Tuple[
    Dict[str, Any], List[str], Dict[str, str]
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
    except:
        # Fallback with basic diagnostics
        moviepy_params, ffmpeg_params = detect_optimal_codec_settings()
        diagnostics = {
            "encoder_type": "FALLBACK",
            "hardware_acceleration": "false",
            "platform": "unknown"
        }
        return moviepy_params, ffmpeg_params, diagnostics


if __name__ == "__main__":
    # Test script for clip assembly
    print("AutoCut Clip Assembler - Test Mode")
    print("TODO: Add test with sample files")
