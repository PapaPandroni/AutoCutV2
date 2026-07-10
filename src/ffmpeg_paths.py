"""
FFmpeg/FFprobe binary resolution.

Resolves the ffmpeg and ffprobe executables so subprocess calls work in
three contexts:

1. Frozen app (PyInstaller): static binaries bundled next to the executable
2. Dev environment: imageio-ffmpeg's ffmpeg (the same binary MoviePy uses)
3. Fallback: whatever is on PATH (previous behavior)
"""

import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _bundled(name: str) -> Optional[str]:
    """Return the path to a binary bundled in the frozen app, if present."""
    if not getattr(sys, "frozen", False):
        return None
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    candidate = base / name
    if candidate.exists():
        return str(candidate)
    return None


@lru_cache(maxsize=1)
def get_ffmpeg_exe() -> str:
    """Path to the ffmpeg executable to use for subprocess calls."""
    bundled = _bundled("ffmpeg")
    if bundled:
        return bundled
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


@lru_cache(maxsize=1)
def get_ffprobe_exe() -> str:
    """Path to the ffprobe executable to use for subprocess calls.

    Note: imageio-ffmpeg does NOT ship ffprobe, so outside the frozen app
    this requires ffprobe on PATH (e.g. brew install ffmpeg).
    """
    bundled = _bundled("ffprobe")
    if bundled:
        return bundled
    return shutil.which("ffprobe") or "ffprobe"
