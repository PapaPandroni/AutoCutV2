"""Clip-selection helpers for AutoCut V2.

Only ``VideoChunk`` (and the clip-selection types alongside it) remain in live
use. The former BeatMatcher / AssemblyEngine / ClipTimeline reimplementation in
this package was never wired into the live pipeline and was removed during the
codebase streamline — the live assembly/render path lives in
``src.clip_assembler``.
"""

from .clip_selector import (
    ClipSelector,
    SelectionCriteria,
    SelectionStrategy,
    VideoChunk,
)

__all__ = [
    "ClipSelector",
    "SelectionCriteria",
    "SelectionStrategy",
    "VideoChunk",
]
