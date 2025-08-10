"""Video Assembly System for AutoCut V2.

This module provides the clip assembly engine that handles beat-synced video
composition. Extracted from the original god module's timeline and beat
matching functionality (lines 2249-2693).

The system provides:
- Timeline management and clip organization
- Beat-to-clip synchronization algorithms
- Quality-based clip selection with variety patterns
- Duration optimization and overlap detection

Key Components:
- ClipTimeline: Timeline management and organization
- BeatMatcher: Beat synchronization algorithms
- ClipSelector: Quality-based selection with variety patterns
- AssemblyEngine: Main orchestration interface

Usage:
    >>> from src.video.assembly import AssemblyEngine
    >>> engine = AssemblyEngine()
    >>> assembled_clips = engine.assemble_clips(clips, beats, pattern="balanced")
"""

from .timeline import (
    ClipTimeline,
    TimelineEntry,
    TimelinePosition,
)
from .beat_matcher import (
    BeatMatcher,
    BeatMatchResult,
    BeatSyncSettings,
)
from .clip_selector import (
    ClipSelector,
    SelectionCriteria,
    VarietyPattern,
)
from .engine import (
    AssemblyEngine,
    AssemblyResult,
    AssemblySettings,
)

# Main interface
ClipAssembler = AssemblyEngine

__all__ = [
    # Main interface
    "AssemblyEngine",
    "ClipAssembler",
    "AssemblyResult", 
    "AssemblySettings",
    
    # Timeline management
    "ClipTimeline",
    "TimelineEntry",
    "TimelinePosition",
    
    # Beat matching
    "BeatMatcher",
    "BeatMatchResult",
    "BeatSyncSettings",
    
    # Clip selection
    "ClipSelector",
    "SelectionCriteria", 
    "VarietyPattern",
]