"""Robust video loading and preprocessing for AutoCut.

Extracted from clip_assembler.py during the codebase streamline. Handles
format-aware preprocessing (H.265/transcoding, canvas scaling), memory-aware
batching, and multi-strategy clip loading with delayed cleanup so subclips stay
valid until after concatenation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from compatibility.moviepy import import_moviepy_safely
except ImportError:
    from .compatibility.moviepy import import_moviepy_safely


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
        logger = logging.getLogger("autocut.clip_assembler")
        logger.info(f"📁 Loading video with delayed cleanup: {video_path}")

        # Check if file exists first
        from pathlib import Path

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        try:
            # Use safe import pattern instead of global variable
            VideoFileClip, _, _, _ = import_moviepy_safely()
            logger.info("📦 MoviePy classes imported successfully")

            # Check if we already have this video loaded for delayed cleanup
            if video_path in self.delayed_cleanup_videos:
                logger.info(f"♻️ Using cached video: {video_path}")
                return self.delayed_cleanup_videos[video_path]

            logger.info(f"🎬 Creating VideoFileClip for: {video_path}")
            # Load the video and store it for delayed cleanup
            video = VideoFileClip(video_path)
            logger.info(f"✅ VideoFileClip created successfully: {type(video)}")
            logger.info(
                f"   Duration: {video.duration:.2f}s, FPS: {video.fps}, Size: {video.size}"
            )

            self.delayed_cleanup_videos[video_path] = video
            self.active_videos.add(id(video))

            # CRITICAL FIX: Return video in try block, not orphaned else block
            return video

        except Exception as e:
            logger.exception(f"❌ Failed to load video {video_path}: {e}")
            import traceback

            traceback.print_exc()
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
        if not hasattr(self, "_temp_files"):
            self._temp_files = set()
        self._temp_files.add(str(temp_file_path))

    def cleanup_all(self) -> None:
        """Clean up all resources including delayed videos and temporary files."""
        import gc

        # Clean up delayed videos
        self.cleanup_delayed_videos()

        # Clean up temporary files
        if hasattr(self, "_temp_files"):
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
        if (
            processed_path.exists()
            and processed_path.stat().st_mtime > Path(video_path).stat().st_mtime
        ):
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
                file_age_hours = (
                    current_time - Path(cached_path).stat().st_mtime
                ) / 3600
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

        def _try_loading_strategy(
            strategy_name: str,
            strategy_func,
            clip_data,
            resource_manager,
            canvas_format,
        ):
            """Try a single loading strategy and return result or None."""
            logger = logging.getLogger("autocut.clip_assembler")
            video_file = clip_data.get("video_file", "UNKNOWN")
            logger.info(f"🔄 Trying {strategy_name} for {video_file}")

            try:
                # CRITICAL FIX: Pass canvas_format to all fallback strategies
                result = strategy_func(
                    clip_data, resource_manager, canvas_format=canvas_format
                )
                if result is not None:
                    logger.info(f"✅ {strategy_name} SUCCESS for {video_file}")
                    self.error_statistics["successful_loads"] += 1
                    self.error_statistics["fallback_usage"][strategy_name] += 1

                    if strategy_name != "direct_loading":
                        pass
                    else:
                        pass

                    return result, None
                logger.warning(f"⚠️ {strategy_name} returned None for {video_file}")

                # CRITICAL FIX: Return None for unsuccessful result, not in else block
                return None, None

            except Exception as e:
                logger.exception(f"❌ {strategy_name} FAILED for {video_file}: {e}")
                error_type = type(e).__name__
                self.error_statistics["error_types"][error_type] = (
                    self.error_statistics["error_types"].get(error_type, 0) + 1
                )
                return None, e

        for strategy_name, strategy_func in strategies:
            result, error = _try_loading_strategy(
                strategy_name, strategy_func, clip_data, resource_manager, canvas_format
            )
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
        canvas_format: Optional[
            dict
        ] = None,  # Kept for backward compatibility, no longer used
    ) -> Optional[Any]:
        """Direct loading with MoviePy.

        NOTE: Canvas scaling is now handled centrally in uniformize_dimensions()
        before concatenation for better performance and consistency.
        """
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

            # NOTE: Canvas scaling is now handled centrally in uniformize_dimensions()

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

            # Load converted clip using safe import pattern
            VideoFileClip, _, _, _ = import_moviepy_safely()

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

            # NOTE: Canvas scaling is now handled centrally in uniformize_dimensions()
            # Apply standard quality reduction
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

            # Load reduced quality clip using safe import pattern
            VideoFileClip, _, _, _ = import_moviepy_safely()

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

            # NOTE: Canvas scaling is now handled centrally in uniformize_dimensions()
            # Apply standard emergency settings for minimal resource usage
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

            # Load minimal clip using safe import pattern
            VideoFileClip, _, _, _ = import_moviepy_safely()

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
        Tuple of (video_clips, failed_indices, error_report, resource_manager).
        video_clips is in timeline order and length == len(sorted_clips), with
        None at any position whose clip failed to load. The caller must replace
        those None gaps (e.g. with placeholders) before concatenation.
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

    # DIAGNOSTIC: Log what we're about to process
    logger = logging.getLogger("autocut.clip_assembler")
    logger.info("🎯 Starting robust video loading:")
    logger.info(f"   Total clips to process: {len(sorted_clips)}")
    logger.info(f"   Grouped into {len(grouped_clips)} video files:")
    for video_file, file_clips in grouped_clips.items():
        logger.info(f"     📄 {video_file}: {len(file_clips)} clips")

    clip_results = {}  # Maps original_index -> segment (None for failed clips)
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
            # DIAGNOSTIC: Log every clip attempt
            logger = logging.getLogger("autocut.clip_assembler")
            video_file = clip_data.get("video_file", "UNKNOWN")
            start_time = clip_data.get("start", 0)
            end_time = clip_data.get("end", 0)
            logger.info(
                f"🎬 Attempting to load clip {i+1}: {video_file} ({start_time:.2f}-{end_time:.2f}s)"
            )

            try:
                # Use robust loader with multiple fallback strategies and delayed cleanup
                # NEW: Pass canvas_format to the loader for intelligent scaling
                segment = robust_loader.load_clip_with_fallbacks(
                    clip_data,
                    resource_manager,
                    canvas_format=canvas_format,
                )

                # Use original_index from grouped clips to maintain timeline alignment
                original_index = clip_data.get(
                    "original_index",
                    total_clips_processed,
                )
                if segment is not None:
                    logger.info(f"✅ Clip {i+1} loaded successfully: {type(segment)}")
                    clip_results[original_index] = segment
                    file_clips_loaded += 1
                else:
                    logger.error(
                        f"❌ Clip {i+1} returned None - all fallback strategies failed"
                    )
                    clip_results[original_index] = None
                    failed_indices.append(original_index)

            except Exception:
                # Use original_index from grouped clips to maintain timeline alignment
                original_index = clip_data.get("original_index", total_clips_processed)
                clip_results[original_index] = None
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

    # Reconstruct the clip list in timeline (beat) order, with None where a clip
    # failed to load. This preserves the beat-matched sequencing the timeline
    # assigned (clips are loaded grouped-by-file for memory efficiency, so their
    # load order does NOT match timeline order); the caller substitutes
    # placeholders for the None gaps.
    video_clips = [clip_results.get(i) for i in range(len(sorted_clips))]

    successful_clips = [clip for clip in video_clips if clip is not None]
    if not successful_clips:
        raise RuntimeError(
            "No video clips could be loaded successfully with any fallback strategy",
        )

    success_count = len(successful_clips)
    success_rate = success_count / len(sorted_clips)

    if success_rate < 1.0:
        logger.warning(
            f"⚠️ Loaded {success_count}/{len(sorted_clips)} clips "
            f"({success_rate:.0%}); {len(failed_indices)} failed and will be "
            f"replaced with placeholder gaps to preserve beat alignment.",
        )

    report_progress(f"Robust loading complete: {success_count} clips", 0.7)

    # Return clips in timeline order (None for failed positions), failed indices,
    # error report, and resource manager. The caller MUST call
    # resource_manager.cleanup_delayed_videos() after concatenation.
    return video_clips, failed_indices, error_report, resource_manager
