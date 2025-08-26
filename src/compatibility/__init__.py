"""MoviePy compatibility layer for AutoCut.

This module provides version-agnostic interfaces to MoviePy functionality
to handle differences between MoviePy 1.x and 2.x APIs.
"""

from .moviepy import (
    attach_audio_safely,
    check_moviepy_api_compatibility,
    import_moviepy_safely,
    subclip_safely,
    write_videofile_safely,
)

__all__ = [
    "attach_audio_safely",
    "check_moviepy_api_compatibility",
    "import_moviepy_safely",
    "subclip_safely",
    "write_videofile_safely",
]
