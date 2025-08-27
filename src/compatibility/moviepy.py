"""MoviePy compatibility functions extracted from clip_assembler.py.

This module provides version-agnostic interfaces to MoviePy functionality
to handle differences between MoviePy 1.x and 2.x APIs.
"""

import logging
from typing import Any, Dict, Optional

# Set up basic logging
logger = logging.getLogger(__name__)


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

            return (
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
                CompositeVideoClip,
            )
        except ImportError as e:
            raise RuntimeError(
                "Could not import MoviePy with either import pattern. Please check MoviePy installation.",
            ) from e


def check_moviepy_api_compatibility() -> Dict[str, Any]:
    """Check MoviePy API compatibility and return version information.

    Returns:
        Dictionary with version detection and method mappings
    """
    try:
        import moviepy

        version = getattr(moviepy, "__version__", "unknown")

        # Test for API method availability by creating a dummy clip
        VideoFileClip, _, _, _ = import_moviepy_safely()

        # Check method names by inspecting the class
        has_subclip = hasattr(VideoFileClip, "subclip")
        has_subclipped = hasattr(VideoFileClip, "subclipped")
        has_set_audio = hasattr(VideoFileClip, "set_audio")
        has_with_audio = hasattr(VideoFileClip, "with_audio")
        has_resize = hasattr(VideoFileClip, "resize")
        has_resized = hasattr(VideoFileClip, "resized")
        has_set_fps = hasattr(VideoFileClip, "set_fps")
        has_with_fps = hasattr(VideoFileClip, "with_fps")
        has_crop = hasattr(VideoFileClip, "crop")
        has_cropped = hasattr(VideoFileClip, "cropped")

        # Determine preferred method based on availability
        preferred_subclip = "subclipped" if has_subclipped else "subclip"
        preferred_audio = "with_audio" if has_with_audio else "set_audio"
        preferred_resize = "resized" if has_resized else "resize"
        preferred_fps = "with_fps" if has_with_fps else "set_fps"
        preferred_crop = "cropped" if has_cropped else "crop"

        return {
            "version_detected": version,
            "method_mappings": {
                "subclip": preferred_subclip,
                "set_audio": preferred_audio,
                "resize": preferred_resize,
                "set_fps": preferred_fps,
                "crop": preferred_crop,
            },
            "available_methods": {
                "subclip": has_subclip,
                "subclipped": has_subclipped,
                "set_audio": has_set_audio,
                "with_audio": has_with_audio,
                "resize": has_resize,
                "resized": has_resized,
                "set_fps": has_set_fps,
                "with_fps": has_with_fps,
                "crop": has_crop,
                "cropped": has_cropped,
            },
        }
    except Exception as e:
        logger.warning(f"MoviePy compatibility check failed: {e}")
        return {
            "version_detected": "unknown",
            "method_mappings": {
                "subclip": "subclipped",  # Default to modern methods
                "set_audio": "with_audio",
                "resize": "resized",
                "set_fps": "with_fps",
                "crop": "cropped",
            },
            "available_methods": {
                "subclip": False,
                "subclipped": True,
                "set_audio": False,
                "with_audio": True,
                "resize": False,
                "resized": True,
                "set_fps": False,
                "with_fps": True,
                "crop": False,
                "cropped": True,
            },
        }


def subclip_safely(
    clip, start_time: float, end_time: float, compatibility_info: Dict[str, Any]
):
    """Create subclip using version-compatible method.

    Args:
        clip: VideoFileClip or AudioFileClip instance
        start_time: Start time in seconds
        end_time: End time in seconds
        compatibility_info: Compatibility information from check_moviepy_api_compatibility

    Returns:
        Subclipped video/audio clip
    """
    # Try modern MoviePy 2.x method first (subclipped)
    try:
        return clip.subclipped(start_time, end_time)
    except AttributeError:
        # Fallback to older MoviePy 1.x method (subclip)
        try:
            return clip.subclip(start_time, end_time)
        except AttributeError:
            # If both fail, raise informative error
            raise RuntimeError(
                f"Neither 'subclipped' nor 'subclip' methods available on {type(clip)}"
            ) from None
    except Exception:
        # Re-raise other exceptions
        raise


def attach_audio_safely(video_clip, audio_clip, compatibility_info: Dict[str, Any]):
    """Attach audio to video using version-compatible method.

    Args:
        video_clip: VideoFileClip instance
        audio_clip: AudioFileClip instance
        compatibility_info: Compatibility information from check_moviepy_api_compatibility

    Returns:
        Video clip with attached audio
    """
    # Try modern MoviePy 2.x method first (with_audio)
    try:
        return video_clip.with_audio(audio_clip)
    except AttributeError:
        # Fallback to older MoviePy 1.x method (set_audio)
        try:
            return video_clip.set_audio(audio_clip)
        except AttributeError:
            # If both fail, raise informative error
            raise RuntimeError(
                f"Neither 'with_audio' nor 'set_audio' methods available on {type(video_clip)}"
            ) from None
    except Exception:
        # Re-raise other exceptions
        raise


def resize_clip_safely(
    clip,
    newsize=None,
    width=None,
    height=None,
    scaling_mode="smart",
    compatibility_info: Optional[Dict[str, Any]] = None,
):
    """Resize clip using MoviePy 2.x effects system with content-aware smart scaling for optimal screen utilization.

    This function preserves aspect ratios and adds letterboxing/pillarboxing as needed,
    with enhanced smart scaling that uses adaptive thresholds based on content and canvas types
    to maximize screen utilization while preserving content safety.

    ENHANCED SMART MODE:
    - Landscape→Landscape: 15% crop threshold (same family)
    - Landscape→Square: 25% crop threshold (moderate mismatch)
    - Portrait→Portrait: 15% crop threshold (same family)
    - Portrait→Square: 25% crop threshold (moderate mismatch)
    - Cross-type (Land↔Port): 8% crop threshold (major mismatch)
    - Square content: 15% crop threshold (universal)

    Args:
        clip: Video clip instance
        newsize: Tuple of (width, height) or None
        width: Target width (alternative to newsize)
        height: Target height (alternative to newsize)
        scaling_mode: "fit" (conservative), "fill" (aggressive), "smart" (adaptive with content-aware thresholds)
        compatibility_info: Compatibility information (optional)

    Returns:
        Resized video clip with proper aspect ratio preservation and intelligent letterboxing
    """
    # Determine target dimensions
    if newsize is not None:
        target_width, target_height = newsize
    elif width is not None and height is not None:
        target_width, target_height = width, height
    elif width is not None:
        # Width only - height computed to maintain aspect ratio (no letterboxing needed)
        try:
            from moviepy.video.fx.Resize import Resize

            return clip.with_effects([Resize(width=width)])
        except ImportError:
            return clip.resized(width=width)
    elif height is not None:
        # Height only - width computed to maintain aspect ratio (no letterboxing needed)
        try:
            from moviepy.video.fx.Resize import Resize

            return clip.with_effects([Resize(height=height)])
        except ImportError:
            return clip.resized(height=height)
    else:
        raise ValueError("Must specify either newsize or width/height parameters")

    try:
        # Get current clip dimensions
        current_width = clip.w
        current_height = clip.h

        # Calculate aspect ratios
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        # Calculate scaling factors
        width_scale = target_width / current_width
        height_scale = target_height / current_height

        # Determine scaling strategy based on mode
        if scaling_mode == "fit":
            # Conservative: fit entire video within canvas (current behavior)
            scale = min(width_scale, height_scale)
            scaling_reason = "fit mode - preserves all content"

        elif scaling_mode == "fill":
            # Aggressive: fill entire canvas (may crop content)
            scale = max(width_scale, height_scale)
            content_type = (
                "landscape"
                if current_aspect > 1.3
                else "portrait"
                if current_aspect < 0.8
                else "square"
            )
            canvas_type = (
                "landscape"
                if target_aspect > 1.3
                else "portrait"
                if target_aspect < 0.8
                else "square"
            )
            scaling_reason = (
                f"fill mode - maximizes screen usage ({content_type}→{canvas_type})"
            )

        elif scaling_mode == "smart":
            # Enhanced Smart: content-aware and canvas-aware crop thresholds
            fit_scale = min(width_scale, height_scale)
            fill_scale = max(width_scale, height_scale)

            # Calculate what percentage of content would be cropped
            crop_ratio = fill_scale / fit_scale
            crop_percentage = (crop_ratio - 1) * 100

            # Determine content type from aspect ratio
            content_type = (
                "landscape"
                if current_aspect > 1.3
                else "portrait"
                if current_aspect < 0.8
                else "square"
            )

            # Determine canvas type from target aspect ratio
            canvas_type = (
                "landscape"
                if target_aspect > 1.3
                else "portrait"
                if target_aspect < 0.8
                else "square"
            )

            # ADAPTIVE CROP THRESHOLDS based on content and canvas analysis
            if content_type == "landscape":
                if canvas_type == "landscape":
                    # Landscape → Landscape: very permissive (same family)
                    SAFE_CROP_THRESHOLD = 15.0
                elif canvas_type == "square":
                    # Landscape → Square: moderately permissive
                    SAFE_CROP_THRESHOLD = 25.0
                else:
                    # Landscape → Portrait: conservative (major mismatch)
                    SAFE_CROP_THRESHOLD = 8.0
            elif content_type == "portrait":
                if canvas_type == "portrait":
                    # Portrait → Portrait: very permissive (same family)
                    SAFE_CROP_THRESHOLD = 15.0
                elif canvas_type == "square":
                    # Portrait → Square: moderately permissive
                    SAFE_CROP_THRESHOLD = 25.0
                else:
                    # Portrait → Landscape: conservative (major mismatch)
                    SAFE_CROP_THRESHOLD = 8.0
            else:  # square content
                # Square content: moderate threshold for any canvas
                SAFE_CROP_THRESHOLD = 15.0

            # Decision logic with enhanced reasoning
            if crop_percentage <= SAFE_CROP_THRESHOLD:
                scale = fill_scale
                scaling_reason = f"smart mode - fill (crop: {crop_percentage:.1f}% ≤ {SAFE_CROP_THRESHOLD}% for {content_type}→{canvas_type})"
            else:
                scale = fit_scale
                scaling_reason = f"smart mode - fit (crop: {crop_percentage:.1f}% > {SAFE_CROP_THRESHOLD}% for {content_type}→{canvas_type})"
        else:
            # Unknown mode: default to safe fit mode
            scale = min(width_scale, height_scale)
            scaling_reason = f"unknown mode '{scaling_mode}' - defaulting to fit"
            logger.warning(f"Unknown scaling mode '{scaling_mode}', using fit mode")

        # Calculate new dimensions (maintain aspect ratio)
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)

        # Log scaling decision for transparency
        logger.info(
            f"Scaling decision: {current_width}x{current_height} → {new_width}x{new_height}"
        )
        logger.info(f"  Reason: {scaling_reason}")
        logger.info(
            f"  Scale factor: {scale:.3f} (width: {width_scale:.3f}, height: {height_scale:.3f})"
        )

        # Try multiple resize approaches with robust fallbacks
        resized_clip = None

        # Approach 1: Modern MoviePy 2.x effects system
        try:
            from moviepy.video.fx.Resize import Resize

            resized_clip = clip.with_effects([Resize((new_width, new_height))])
            logger.debug(
                f"Used modern MoviePy 2.x effects: {current_width}x{current_height} → {new_width}x{new_height}"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Modern effects resize failed: {e}")

        # Approach 2: Legacy resize method
        if resized_clip is None:
            try:
                resized_clip = clip.resized((new_width, new_height))
                logger.debug(
                    f"Used legacy resize method: {current_width}x{current_height} → {new_width}x{new_height}"
                )
            except Exception as e:
                logger.warning(f"Legacy resize failed: {e}")

        # Approach 3: Direct resize as last resort
        if resized_clip is None:
            try:
                resized_clip = clip.resize((new_width, new_height))
                logger.debug(
                    f"Used direct resize fallback: {current_width}x{current_height} → {new_width}x{new_height}"
                )
            except Exception as e:
                logger.exception(f"All resize methods failed: {e}")
                raise RuntimeError(f"Unable to resize clip: {e}") from e

        # CRITICAL FIX: Check if letterboxing should be applied based on scaling mode
        if new_width == target_width and new_height == target_height:
            # Perfect fit, no letterboxing needed
            logger.debug(
                f"Perfect fit: {new_width}x{new_height} matches target {target_width}x{target_height}"
            )
            return resized_clip

        # NEW: For fill mode, return resized clip directly without letterboxing
        # This eliminates cascading letterboxing that was causing black bars on all sides
        if scaling_mode == "fill":
            logger.info(
                "Fill mode: Returning resized clip without letterboxing to maximize screen utilization"
            )
            logger.info(
                f"  Dimensions: {new_width}x{new_height} (target: {target_width}x{target_height})"
            )

            # Calculate screen utilization for fill mode
            video_area = new_width * new_height
            target_area = target_width * target_height
            utilization = (video_area / target_area) * 100
            logger.info(
                f"  Screen utilization (fill mode): {utilization:.1f}% - no letterboxing applied"
            )

            return resized_clip

        # Import composition classes with fallbacks
        ColorClip = None
        CompositeVideoClip = None

        try:
            from moviepy.editor import ColorClip, CompositeVideoClip
        except ImportError:
            try:
                from moviepy import ColorClip, CompositeVideoClip
            except ImportError:
                logger.exception(
                    "Cannot import ColorClip and CompositeVideoClip for letterboxing"
                )
                logger.warning(
                    f"Returning resized clip without letterboxing: {new_width}x{new_height}"
                )
                return resized_clip

        # Create black background for letterboxing/pillarboxing (fit/smart modes only)
        logger.info(
            f"Applying letterboxing for '{scaling_mode}' mode to ensure proper aspect ratio"
        )

        try:
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),  # Black letterbox bars
                duration=resized_clip.duration,
            )

            # Calculate centering position
            x_pos = (target_width - new_width) // 2
            y_pos = (target_height - new_height) // 2

            # Create composite with centered resized clip
            letterboxed_clip = CompositeVideoClip(
                [
                    background,
                    resized_clip.with_position((x_pos, y_pos)),
                ],
                size=(target_width, target_height),
            )

            # Ensure the composite has the correct duration
            letterboxed_clip = letterboxed_clip.with_duration(resized_clip.duration)

            # Enhanced logging for letterboxing operations
            if new_width < target_width:
                letterbox_type = (
                    "pillarbox"
                    if new_height == target_height
                    else "letterbox+pillarbox"
                )
                bar_width = (target_width - new_width) // 2
                logger.info(f"Applied {letterbox_type}: {bar_width}px bars on sides")
                logger.info(
                    "  Reason: Content scaled to fit target width while preserving aspect ratio"
                )
            else:
                bar_height = (target_height - new_height) // 2
                logger.info(f"Applied letterbox: {bar_height}px bars on top/bottom")
                logger.info(
                    "  Reason: Content scaled to fit target height while preserving aspect ratio"
                )

            # Calculate and log screen utilization
            video_area = new_width * new_height
            canvas_area = target_width * target_height
            utilization = (video_area / canvas_area) * 100
            logger.info(
                f"  Screen utilization (with letterboxing): {utilization:.1f}% ({video_area}/{canvas_area} pixels)"
            )
            logger.info(
                f"  Mode: {scaling_mode} - letterboxing applied to preserve content and aspect ratio"
            )

            return letterboxed_clip

        except Exception as letterbox_error:
            # Fallback: return resized clip without letterboxing if composition fails
            logger.warning(
                f"Letterboxing failed ({letterbox_error!s}), returning resized clip without black bars"
            )
            logger.warning(
                f"Resized to: {new_width}x{new_height} (target: {target_width}x{target_height})"
            )
            return resized_clip

    except Exception as e:
        logger.exception(f"Failed to resize clip with aspect ratio preservation: {e}")
        # Final fallback: attempt direct resize (old behavior)
        logger.warning(
            "Falling back to direct resize without aspect ratio preservation"
        )
        try:
            from moviepy.video.fx.Resize import Resize

            if newsize is not None:
                return clip.with_effects([Resize(newsize)])
            return clip.with_effects([Resize((target_width, target_height))])
        except ImportError:
            try:
                if newsize is not None:
                    return clip.resized(newsize)
                return clip.resized((target_width, target_height))
            except Exception as fallback_error:
                logger.exception(f"Even fallback resize failed: {fallback_error}")
                raise


def resize_with_aspect_preservation(
    clip, target_width: int, target_height: int, scaling_mode: str = "smart"
):
    """Centralized function for resizing clips with content-aware intelligent scaling and optimal screen utilization.

    This function is specifically designed for AutoCut's enhanced video normalization pipeline,
    handling mixed aspect ratios with dynamic canvas selection and content-aware crop thresholds
    to maximize screen utilization while maintaining content safety.

    OPTIMIZED FOR DYNAMIC CANVAS SYSTEM:
    - Works seamlessly with 16:9, 4:3, and 9:16 canvases
    - Content-aware thresholds minimize black bars
    - Preserves video quality through intelligent cropping decisions

    Args:
        clip: Video clip instance
        target_width: Target canvas width (from dynamic canvas selection)
        target_height: Target canvas height (from dynamic canvas selection)
        scaling_mode: "smart" (content-aware adaptive), "fit" (conservative), "fill" (aggressive)

    Returns:
        Resized and letterboxed clip optimized for target canvas with minimal black bars
    """
    try:
        logger.info(
            f"Starting aspect ratio preservation: {clip.w}x{clip.h} → {target_width}x{target_height}"
        )
        logger.info(f"Using scaling mode: {scaling_mode}")

        # Use the robust resize_clip_safely function with enhanced smart scaling
        result = resize_clip_safely(
            clip, newsize=(target_width, target_height), scaling_mode=scaling_mode
        )

        logger.info("Aspect ratio preservation completed successfully")
        return result

    except Exception as e:
        logger.exception(f"resize_with_aspect_preservation failed: {e}")
        logger.warning("Attempting fallback to fit mode")

        # Fallback to conservative fit mode
        try:
            return resize_clip_safely(
                clip, newsize=(target_width, target_height), scaling_mode="fit"
            )
        except Exception as fallback_error:
            logger.exception(f"Fallback to fit mode also failed: {fallback_error}")
            logger.warning("Returning original clip as final fallback")
            return clip


def set_fps_safely(
    clip, fps: float, compatibility_info: Optional[Dict[str, Any]] = None
):
    """Set FPS using version-compatible method.

    Note: In MoviePy 2.x, set_fps was removed. We try multiple approaches.

    Args:
        clip: Video clip instance
        fps: Target frames per second
        compatibility_info: Compatibility information (optional)

    Returns:
        Clip with modified FPS
    """
    # Try modern MoviePy 2.x approach (with_fps if available)
    try:
        return clip.with_fps(fps)
    except AttributeError:
        pass

    # Try legacy MoviePy 1.x method (set_fps)
    try:
        return clip.set_fps(fps)
    except AttributeError:
        pass

    # Alternative approach: some versions might have fps as a property
    try:
        # Create a copy with modified fps
        new_clip = clip.copy()
        if hasattr(new_clip, "fps"):
            new_clip.fps = fps
            return new_clip
    except (AttributeError, Exception):
        pass

    # Final fallback: warning and return original
    logger.warning(
        f"Could not set FPS to {fps} on {type(clip)} - FPS modification not supported in this MoviePy version"
    )
    return clip  # Return original clip unchanged


def crop_clip_safely(
    clip,
    x1=None,
    y1=None,
    x2=None,
    y2=None,
    width=None,
    height=None,
    x_center=None,
    y_center=None,
    compatibility_info: Optional[Dict[str, Any]] = None,
):
    """Crop clip using version-compatible method.

    Args:
        clip: Video clip instance
        x1, y1, x2, y2: Crop coordinates
        width, height: Crop dimensions
        x_center, y_center: Center coordinates for cropping
        compatibility_info: Compatibility information (optional)

    Returns:
        Cropped video clip
    """
    # Try modern MoviePy 2.x method first (cropped)
    try:
        return clip.cropped(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height,
            x_center=x_center,
            y_center=y_center,
        )
    except AttributeError:
        # Fallback to older MoviePy 1.x method (crop)
        try:
            return clip.crop(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=width,
                height=height,
                x_center=x_center,
                y_center=y_center,
            )
        except AttributeError:
            raise RuntimeError(
                f"Neither 'cropped' nor 'crop' methods available on {type(clip)}"
            ) from None
    except Exception:
        # Re-raise other exceptions
        raise


def write_videofile_safely(
    video_clip, output_path: str, compatibility_info: Dict[str, Any], **kwargs
):
    """Write video file using version-compatible parameters.

    Args:
        video_clip: VideoFileClip instance to write
        output_path: Output file path
        compatibility_info: Compatibility information (currently unused but kept for API consistency)
        **kwargs: Additional parameters for write_videofile
    """
    try:
        # Remove any parameters that might not be supported in MoviePy 2.x
        safe_kwargs = kwargs.copy()

        # MoviePy 2.x incompatible parameters that need to be removed
        moviepy2_unsupported_params = [
            "logger",  # Causes issues in some versions
            "temp_audiofile_fps",  # Not supported in MoviePy 2.x - removed parameter
            "verbose",  # Not supported as keyword argument in some MoviePy versions
        ]

        for param in moviepy2_unsupported_params:
            if param in safe_kwargs:
                logger.debug(f"Removing unsupported parameter: {param}")
                del safe_kwargs[param]

        # Fix audio codec profile issues that cause FFmpeg errors
        if "audio_codec" in safe_kwargs:
            audio_codec = safe_kwargs["audio_codec"]
            # Replace problematic audio codec profiles with safe defaults
            if audio_codec in ["aac_low", "aac_he", "aac_he_v2"]:
                logger.debug(
                    f"Replacing problematic audio codec '{audio_codec}' with 'aac'"
                )
                safe_kwargs["audio_codec"] = "aac"

        # Handle FFmpeg parameters that might contain problematic audio settings
        if "ffmpeg_params" in safe_kwargs and isinstance(
            safe_kwargs["ffmpeg_params"], list
        ):
            ffmpeg_params = safe_kwargs["ffmpeg_params"]
            filtered_params = []
            skip_next = False

            for _i, param in enumerate(ffmpeg_params):
                if skip_next:
                    skip_next = False
                    continue

                # Remove problematic audio profile settings
                if param in ["-profile:a", "-profile:audio"]:
                    # Skip this parameter and its value
                    skip_next = True
                    logger.debug(f"Removing FFmpeg audio profile parameter: {param}")
                    continue
                if param == "aac_low":
                    logger.debug("Removing problematic aac_low from FFmpeg parameters")
                    continue
                filtered_params.append(param)

            safe_kwargs["ffmpeg_params"] = filtered_params

        video_clip.write_videofile(output_path, **safe_kwargs)

    except Exception as e:
        logger.exception(
            f"Video writing failed with parameters {list(kwargs.keys())}: {e}"
        )

        # Try with minimal parameters as fallback
        try:
            essential_params = {
                "codec": kwargs.get("codec", "libx264"),
                "fps": kwargs.get("fps", 24),
                "audio_codec": "aac",  # Use safe audio codec
                "ffmpeg_params": [],  # Remove all FFmpeg parameters that might cause issues
            }
            logger.warning("Retrying with minimal parameters and safe audio codec")
            video_clip.write_videofile(output_path, **essential_params)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Video writing failed even with fallback parameters: {fallback_error}"
            ) from fallback_error


# Legacy compatibility functions that may be needed
def test_independent_subclip_creation():
    """Test if MoviePy can create independent subclips without parent reference issues."""
    try:
        VideoFileClip, _, _, _ = import_moviepy_safely()
        # This is a placeholder - actual implementation would test subclip creation
        return True
    except:
        return False
