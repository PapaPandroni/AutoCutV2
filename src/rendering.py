"""Final-video rendering for AutoCut.

Extracted from clip_assembler.py during the codebase streamline. Loads the
beat-matched clips, uniformizes their dimensions to the target canvas,
concatenates them, attaches the (untouched) music track with an optional musical
fade, and encodes the result with hardware-accelerated settings when available.
"""

import builtins
import contextlib
import logging
from pathlib import Path
from typing import Any, List, Optional

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        VideoFileClip = None

try:
    from compatibility.moviepy import import_moviepy_safely
except ImportError:
    from .compatibility.moviepy import import_moviepy_safely

try:
    from video.encoder import VideoEncoder
except ImportError:
    from .video.encoder import VideoEncoder

try:
    from video.timeline_renderer import ClipTimeline
except ImportError:
    from .video.timeline_renderer import ClipTimeline

try:
    from video_loading import load_video_clips_with_robust_error_handling
except ImportError:
    from .video_loading import load_video_clips_with_robust_error_handling


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
                logger.exception(
                    "❌ Cannot import ColorClip - letterboxing not available"
                )
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
                scaling_mode="fit",  # Preserve aspect ratio, no cropping
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
                duration=resized_clip.duration,
            )

            # FIXED: Use with_position instead of set_position for MoviePy v2.0 compatibility
            positioned_clip = resized_clip.with_position((x_offset, y_offset))

            # Composite to create exact target dimensions
            uniform_clip = CompositeVideoClip(
                [background, positioned_clip], size=(target_width, target_height)
            )

            uniform_clips.append(uniform_clip)

            # Enhanced diagnostic logging
            letterbox_type = (
                "top/bottom" if original_aspect > target_aspect else "left/right"
            )
            logger.debug(
                f"🔧 Clip {i+1}: {original_width}x{original_height} → {target_width}x{target_height}"
            )
            logger.debug(f"   📏 Scale factor: {scale_factor:.3f}")
            logger.debug(
                f"   📐 Content size: {scaled_width}x{scaled_height} (centered)"
            )
            logger.debug(f"   ⬛ Letterbox: {letterbox_type} bars")
            logger.debug(f"   📍 Position: ({x_offset}, {y_offset})")

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
    progress_callback: Optional[callable] = None,
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
        progress_callback: Optional callback for progress updates
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
            from audio_loader import load_audio_robust
        except ImportError:

            def load_audio_robust(audio_file):
                """Minimal robust audio loader as fallback."""
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
                canvas_format["target_height"],
            )
        else:
            logger.warning(
                "⚠️ No canvas format provided - skipping dimension uniformization"
            )

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
            raise RuntimeError(
                f"Failed to load audio file {audio_file}: {audio_error}"
            ) from audio_error

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
            raise RuntimeError(
                f"Failed to attach audio to video: {audio_attach_error}"
            ) from audio_attach_error

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
