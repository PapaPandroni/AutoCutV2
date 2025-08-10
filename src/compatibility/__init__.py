"""MoviePy compatibility layer for AutoCut.

This module provides version-agnostic interfaces to MoviePy functionality
to handle differences between MoviePy 1.x and 2.x APIs.
"""

from .moviepy import (
    import_moviepy_safely,
    check_moviepy_api_compatibility, 
    write_videofile_safely,
    attach_audio_safely,
    subclip_safely
)

__all__ = [
    'import_moviepy_safely',
    'check_moviepy_api_compatibility',
    'write_videofile_safely', 
    'attach_audio_safely',
    'subclip_safely'
]