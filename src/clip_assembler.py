"""
Clip Assembly Module for AutoCut

Handles the core logic of matching video clips to musical beats,
applying variety patterns, and rendering the final video.
"""

import builtins
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# MoviePy 1.x/2.x compatibility shims all live in src/compatibility/moviepy.
# import_moviepy_safely is called at module scope here; the other shims
# (subclip_safely / attach_audio_safely / check_moviepy_api_compatibility /
# write_videofile_safely / resize_clip_safely) are imported locally where used.
try:
    from compatibility.moviepy import import_moviepy_safely
except ImportError:
    from .compatibility.moviepy import import_moviepy_safely

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


# Beat matching / clip selection extracted to src/beat_matching.py
try:
    from beat_matching import (
        VARIETY_PATTERNS,
        apply_variety_pattern,
        match_clips_to_beats,
        select_best_clips,
        _calculate_duration_fit,
        _clips_overlap,
        _fit_clip_to_duration,
    )
except ImportError:
    from .beat_matching import (
        VARIETY_PATTERNS,
        apply_variety_pattern,
        match_clips_to_beats,
        select_best_clips,
        _calculate_duration_fit,
        _clips_overlap,
        _fit_clip_to_duration,
    )


# VideoFormatAnalyzer class extracted to src/video/format_analyzer.py


# VideoNormalizationPipeline class extracted to src/video/normalization.py


# Video loading / preprocessing extracted to src/video_loading.py
try:
    from video_loading import (
        AdvancedMemoryManager,
        RobustVideoLoader,
        VideoPreprocessor,
        VideoResourceManager,
        _group_clips_by_file,
        load_video_clips_with_robust_error_handling,
        preprocess_videos_smart,
    )
except ImportError:
    from .video_loading import (
        AdvancedMemoryManager,
        RobustVideoLoader,
        VideoPreprocessor,
        VideoResourceManager,
        _group_clips_by_file,
        load_video_clips_with_robust_error_handling,
        preprocess_videos_smart,
    )


# ClipTimeline class extracted to src/video/timeline_renderer.py


def uniformize_dimensions(clips, target_width, target_height):
    """Force all clips to exact canvas dimensions with maximum scaling and minimal letterboxing.

    This function ensures all clips have identical final dimensions before concatenation,
    preventing black bars caused by dimension mismatches. Each clip is scaled to use
    the maximum possible screen space within the target canvas while preserving aspect ratio.

    Args:
        clips: List of MoviePy VideoClip objects
        target_width: Canvas width in pixels
        target_height: Canvas height in pixels

    Returns:
        List of VideoClip objects, all exactly target_width x target_height
    """
    import logging

    logger = logging.getLogger("autocut.clip_assembler")
    logger.info(f"🎯 Uniformizing {len(clips)} clips to {target_width}x{target_height}")

    # Import MoviePy classes safely using the same pattern as render_video
    try:
        VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = (
            import_moviepy_safely()
        )

        # Import ColorClip with fallback pattern like in compatibility module
        ColorClip = None
        try:
            from moviepy.editor import ColorClip
        except ImportError:
            try:
                from moviepy import ColorClip
            except ImportError:
                logger.exception("❌ Cannot import ColorClip - letterboxing not available")
                # Return clips unchanged if we can't do letterboxing
                return clips

        if ColorClip is None or CompositeVideoClip is None:
            logger.error("❌ Required MoviePy classes not available for uniformization")
            return clips

        # Import safe resize function from compatibility module (dual-import
        # pattern — matches the rest of this file so it works both when src/ is
        # on sys.path via autocut.py and when imported as the src package).
        try:
            from compatibility.moviepy import resize_clip_safely
        except ImportError:
            from .compatibility.moviepy import resize_clip_safely

    except RuntimeError as e:
        logger.exception(f"❌ MoviePy import failed: {e}")
        return clips
    except ImportError as e:
        logger.exception(f"❌ Failed to import compatibility resize function: {e}")
        return clips

    uniform_clips = []
    failed_count = 0
    target_aspect = target_width / target_height

    def _force_target_size(source_clip):
        """Last-resort: force exact canvas dims via composite (no rescale).

        Used when the aspect-preserving resize fails, so concatenation still
        receives a uniformly-sized clip instead of a wrong-sized one that would
        create dimension-mismatch artifacts.
        """
        return CompositeVideoClip([source_clip], size=(target_width, target_height))

    for i, clip in enumerate(clips):
        try:
            # Get original clip dimensions
            original_width, original_height = clip.size
            original_aspect = original_width / original_height

            # Calculate maximum scale factor that preserves aspect ratio
            if original_aspect > target_aspect:
                # Clip is wider than target - fit to width (letterbox top/bottom)
                scale_factor = target_width / original_width
                scaled_width = target_width
                scaled_height = int(original_height * scale_factor)
            else:
                # Clip is taller than target - fit to height (pillarbox left/right)
                scale_factor = target_height / original_height
                scaled_width = int(original_width * scale_factor)
                scaled_height = target_height

            # Use safe resize function from compatibility module instead of direct resize
            resized_clip = resize_clip_safely(
                clip,
                newsize=(scaled_width, scaled_height),
                scaling_mode="fit"  # Preserve aspect ratio, no cropping
            )

            # CRITICAL: Check if resize_clip_safely returned None or failed
            if resized_clip is None:
                logger.error(
                    f"❌ resize_clip_safely returned None for clip {i+1}; "
                    f"forcing exact canvas size as fallback"
                )
                uniform_clips.append(_force_target_size(clip))
                failed_count += 1
                continue

            # Validate resized clip has required attributes
            if not hasattr(resized_clip, "duration"):
                logger.error(
                    f"❌ Resized clip {i+1} missing duration attribute; "
                    f"forcing exact canvas size as fallback"
                )
                uniform_clips.append(_force_target_size(clip))
                failed_count += 1
                continue

            # Calculate position to center the resized clip
            x_offset = (target_width - scaled_width) // 2
            y_offset = (target_height - scaled_height) // 2

            # Create black background canvas
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),  # Black background
                duration=resized_clip.duration
            )

            # FIXED: Use with_position instead of set_position for MoviePy v2.0 compatibility
            positioned_clip = resized_clip.with_position((x_offset, y_offset))

            # Composite to create exact target dimensions
            uniform_clip = CompositeVideoClip(
                [background, positioned_clip],
                size=(target_width, target_height)
            )

            uniform_clips.append(uniform_clip)

            # Enhanced diagnostic logging
            letterbox_type = "top/bottom" if original_aspect > target_aspect else "left/right"
            logger.info(f"🔧 Clip {i+1}: {original_width}x{original_height} → {target_width}x{target_height}")
            logger.info(f"   📏 Scale factor: {scale_factor:.3f}")
            logger.info(f"   📐 Content size: {scaled_width}x{scaled_height} (centered)")
            logger.info(f"   ⬛ Letterbox: {letterbox_type} bars")
            logger.info(f"   📍 Position: ({x_offset}, {y_offset})")

        except Exception as e:
            logger.exception(f"❌ Failed to uniformize clip {i+1}: {e}")
            # Last resort: force exact canvas dims so concatenation stays uniform
            try:
                uniform_clips.append(_force_target_size(clip))
            except Exception:
                logger.exception(
                    f"❌ Could not force canvas size for clip {i+1}; appending "
                    f"original (may cause dimension mismatch)"
                )
                uniform_clips.append(clip)
            failed_count += 1

    if failed_count:
        logger.warning(
            f"⚠️ Uniformization finished with {failed_count}/{len(clips)} clip(s) "
            f"using the forced-size fallback; output is {target_width}x{target_height} "
            f"but those clips may be cropped/letterboxed unexpectedly."
        )
    else:
        logger.info(
            f"✅ Uniformization complete: all clips are {target_width}x{target_height}"
        )
    return uniform_clips

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
    # Initialize logger for this function
    logger = logging.getLogger("autocut.clip_assembler")

    # Cleanup handles, initialized up front so the finally block is always safe
    # even if a failure occurs before these are assigned.
    resource_manager = None
    video_clips: List[Any] = []
    audio_clip = None
    final_video = None

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

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load video clips using robust loading system: {e}"
            ) from e

        # video_clips is now in timeline order with None where a clip failed to
        # load. Abort if too many failed, otherwise fill the gaps with black
        # placeholders of the intended duration so surviving clips keep their
        # beat-aligned positions (prevents silent audio desync — see C1).
        total_positions = len(sorted_clips)
        loaded_count = sum(1 for c in video_clips if c is not None)
        if loaded_count == 0:
            raise RuntimeError(
                f"No video clips could be loaded from {total_positions} timeline "
                f"clips using robust loading system"
            )

        def _failed_sources() -> list:
            return sorted(
                {
                    sorted_clips[i]["video_file"]
                    for i in failed_indices
                    if 0 <= i < total_positions
                }
            )

        success_rate = loaded_count / total_positions
        if success_rate < 0.5:
            raise RuntimeError(
                f"Only {loaded_count}/{total_positions} clips loaded "
                f"({success_rate:.0%}); too many failures to produce a usable "
                f"video. Problem sources: {_failed_sources()}"
            )

        if failed_indices:
            logger.warning(
                f"⚠️ {len(failed_indices)} clip(s) failed to load and will be "
                f"replaced with black placeholders to preserve beat sync. "
                f"Sources: {_failed_sources()}"
            )

            # Import ColorClip using the same fallback pattern as uniformize_dimensions
            ColorClip = None
            try:
                from moviepy.editor import ColorClip
            except ImportError:
                try:
                    from moviepy import ColorClip
                except ImportError:
                    ColorClip = None

            if ColorClip is None:
                raise RuntimeError(
                    "Cannot create placeholder clips (ColorClip unavailable); "
                    f"{len(failed_indices)} clips failed to load."
                )

            placeholder_w = canvas_format["target_width"] if canvas_format else 1920
            placeholder_h = canvas_format["target_height"] if canvas_format else 1080
            for i in range(total_positions):
                if video_clips[i] is None:
                    clip_meta = sorted_clips[i]
                    gap_duration = max(
                        float(clip_meta["end"]) - float(clip_meta["start"]), 0.1
                    )
                    video_clips[i] = ColorClip(
                        size=(placeholder_w, placeholder_h),
                        color=(0, 0, 0),
                        duration=gap_duration,
                    )

        if progress_callback:
            progress_callback("Uniformizing video dimensions", 0.45)

        # CRITICAL FIX: Uniformize all clip dimensions before concatenation
        # This ensures all clips are exactly target_width x target_height,
        # preventing black bars caused by dimension mismatches
        if canvas_format:
            video_clips = uniformize_dimensions(
                video_clips,
                canvas_format["target_width"],
                canvas_format["target_height"]
            )
        else:
            logger.warning("⚠️ No canvas format provided - skipping dimension uniformization")

        if progress_callback:
            progress_callback("Concatenating video clips", 0.5)

        # Concatenate video clips (now all have identical dimensions)
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

        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to render video: {e!s}") from e
    finally:
        # Always release clips, audio readers, and FFmpeg subprocesses — on both
        # success and failure — so a mid-render error can't orphan proc handles.

        # Clean up resource manager (prevents proc errors)
        try:
            if resource_manager is not None:
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
            path_issues.extend(
                f"Contains problematic character '{char}'"
                for char in problematic_chars
                if char in audio_path
            )

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
