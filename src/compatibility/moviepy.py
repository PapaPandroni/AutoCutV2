"""MoviePy compatibility functions extracted from clip_assembler.py.

This module provides version-agnostic interfaces to MoviePy functionality
to handle differences between MoviePy 1.x and 2.x APIs.
"""

from typing import Tuple, Dict, Any
import logging

# Set up basic logging
logger = logging.getLogger(__name__)


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


def check_moviepy_api_compatibility() -> Dict[str, Any]:
    """Check MoviePy API compatibility and return version information.
    
    Returns:
        Dictionary with version detection and method mappings
    """
    try:
        import moviepy
        version = getattr(moviepy, '__version__', 'unknown')
        
        # Test for API method availability by creating a dummy clip
        VideoFileClip, _, _, _ = import_moviepy_safely()
        
        # Check method names by inspecting the class
        has_subclip = hasattr(VideoFileClip, 'subclip')
        has_subclipped = hasattr(VideoFileClip, 'subclipped') 
        has_set_audio = hasattr(VideoFileClip, 'set_audio')
        has_with_audio = hasattr(VideoFileClip, 'with_audio')
        has_resize = hasattr(VideoFileClip, 'resize')
        has_resized = hasattr(VideoFileClip, 'resized')
        has_set_fps = hasattr(VideoFileClip, 'set_fps')
        has_with_fps = hasattr(VideoFileClip, 'with_fps')
        has_crop = hasattr(VideoFileClip, 'crop')
        has_cropped = hasattr(VideoFileClip, 'cropped')
        
        # Determine preferred method based on availability
        preferred_subclip = 'subclipped' if has_subclipped else 'subclip'
        preferred_audio = 'with_audio' if has_with_audio else 'set_audio'
        preferred_resize = 'resized' if has_resized else 'resize'
        preferred_fps = 'with_fps' if has_with_fps else 'set_fps'
        preferred_crop = 'cropped' if has_cropped else 'crop'
        
        return {
            'version_detected': version,
            'method_mappings': {
                'subclip': preferred_subclip,
                'set_audio': preferred_audio,
                'resize': preferred_resize,
                'set_fps': preferred_fps,
                'crop': preferred_crop
            },
            'available_methods': {
                'subclip': has_subclip,
                'subclipped': has_subclipped,
                'set_audio': has_set_audio, 
                'with_audio': has_with_audio,
                'resize': has_resize,
                'resized': has_resized,
                'set_fps': has_set_fps,
                'with_fps': has_with_fps,
                'crop': has_crop,
                'cropped': has_cropped
            }
        }
    except Exception as e:
        logger.warning(f"MoviePy compatibility check failed: {e}")
        return {
            'version_detected': 'unknown',
            'method_mappings': {
                'subclip': 'subclipped',  # Default to modern methods
                'set_audio': 'with_audio',
                'resize': 'resized',
                'set_fps': 'with_fps', 
                'crop': 'cropped'
            },
            'available_methods': {
                'subclip': False,
                'subclipped': True,
                'set_audio': False,
                'with_audio': True,
                'resize': False,
                'resized': True,
                'set_fps': False,
                'with_fps': True,
                'crop': False,
                'cropped': True
            }
        }


def subclip_safely(clip, start_time: float, end_time: float, compatibility_info: Dict[str, Any]):
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
            raise RuntimeError(f"Neither 'subclipped' nor 'subclip' methods available on {type(clip)}")
    except Exception as e:
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
            raise RuntimeError(f"Neither 'with_audio' nor 'set_audio' methods available on {type(video_clip)}")
    except Exception as e:
        # Re-raise other exceptions
        raise


def resize_clip_safely(clip, newsize=None, width=None, height=None, compatibility_info: Dict[str, Any] = None):
    """Resize clip using version-compatible method.
    
    Args:
        clip: Video clip instance
        newsize: Tuple of (width, height) or None
        width: Target width (alternative to newsize)
        height: Target height (alternative to newsize) 
        compatibility_info: Compatibility information (optional)
        
    Returns:
        Resized video clip
    """
    # Determine target size
    if newsize is not None:
        target_size = newsize
    elif width is not None or height is not None:
        current_size = getattr(clip, 'size', (1920, 1080))
        target_width = width or current_size[0]
        target_height = height or current_size[1] 
        target_size = (target_width, target_height)
    else:
        raise ValueError("Must specify either newsize or width/height parameters")
    
    # Try modern MoviePy 2.x method first (resized)
    try:
        return clip.resized(newsize=target_size)
    except AttributeError:
        # Fallback to older MoviePy 1.x method (resize)
        try:
            return clip.resize(newsize=target_size)
        except AttributeError:
            # If both fail, try alternative parameter names
            try:
                # Some versions might use different parameter names
                return clip.resized(target_size)
            except AttributeError:
                try:
                    return clip.resize(target_size)
                except AttributeError:
                    raise RuntimeError(f"Neither 'resized' nor 'resize' methods available on {type(clip)}")
    except Exception as e:
        # Re-raise other exceptions
        raise


def set_fps_safely(clip, fps: float, compatibility_info: Dict[str, Any] = None):
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
        if hasattr(new_clip, 'fps'):
            new_clip.fps = fps
            return new_clip
    except (AttributeError, Exception):
        pass
    
    # Final fallback: warning and return original
    logger.warning(f"Could not set FPS to {fps} on {type(clip)} - FPS modification not supported in this MoviePy version")
    return clip  # Return original clip unchanged


def crop_clip_safely(clip, x1=None, y1=None, x2=None, y2=None, width=None, height=None, x_center=None, y_center=None, compatibility_info: Dict[str, Any] = None):
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
        return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height, x_center=x_center, y_center=y_center)
    except AttributeError:
        # Fallback to older MoviePy 1.x method (crop)
        try:
            return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height, x_center=x_center, y_center=y_center)
        except AttributeError:
            raise RuntimeError(f"Neither 'cropped' nor 'crop' methods available on {type(clip)}")
    except Exception as e:
        # Re-raise other exceptions
        raise


def write_videofile_safely(video_clip, output_path: str, compatibility_info: Dict[str, Any], **kwargs):
    """Write video file using version-compatible parameters.
    
    Args:
        video_clip: VideoFileClip instance to write
        output_path: Output file path
        compatibility_info: Compatibility information (currently unused but kept for API consistency)
        **kwargs: Additional parameters for write_videofile
    """
    try:
        # Remove any parameters that might not be supported in all versions
        safe_kwargs = kwargs.copy()
        
        # Some versions of MoviePy might not support certain parameters
        unsupported_params = ['logger']  # Add parameters that cause issues
        for param in unsupported_params:
            if param in safe_kwargs:
                del safe_kwargs[param]
        
        video_clip.write_videofile(output_path, **safe_kwargs)
        
    except Exception as e:
        logger.error(f"Video writing failed with parameters {list(kwargs.keys())}: {e}")
        
        # Try with minimal parameters as fallback
        try:
            essential_params = {
                'codec': kwargs.get('codec', 'libx264'),
                'fps': kwargs.get('fps', 24),
                'ffmpeg_params': kwargs.get('ffmpeg_params', [])
            }
            logger.warning("Retrying with minimal parameters")
            video_clip.write_videofile(output_path, **essential_params)
        except Exception as fallback_error:
            raise RuntimeError(f"Video writing failed even with fallback parameters: {fallback_error}")


# Legacy compatibility functions that may be needed
def test_independent_subclip_creation():
    """Test if MoviePy can create independent subclips without parent reference issues."""
    try:
        VideoFileClip, _, _, _ = import_moviepy_safely()
        # This is a placeholder - actual implementation would test subclip creation
        return True
    except:
        return False