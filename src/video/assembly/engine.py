"""Main assembly engine orchestrating video clip assembly.

Provides the primary interface for the refactored clip assembly system,
replacing the original god module's functionality with clean architecture.
This engine coordinates timeline management, beat matching, and clip selection.

Features:
- Unified interface for clip assembly operations
- Integration of timeline, beat matching, and selection systems
- Backward compatibility with legacy interfaces
- Comprehensive result analysis and statistics

Key Components:
- AssemblyEngine: Main orchestration engine
- AssemblySettings: Configuration for assembly behavior  
- AssemblyResult: Comprehensive assembly results
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from ...core.exceptions import ValidationError, raise_validation_error
from ...core.logging_config import get_logger, log_performance, LoggingContext
from .timeline import ClipTimeline, TimelineEntry, TimelinePosition
from .beat_matcher import BeatMatcher, BeatMatchResult, BeatSyncSettings, VarietyPattern
from .clip_selector import ClipSelector, SelectionCriteria, SelectionStrategy


# Import VideoChunk from canonical location (already available in clip_selector)
from .clip_selector import VideoChunk


@dataclass
class AssemblySettings:
    """Configuration for assembly engine behavior."""
    
    # Selection strategy
    selection_strategy: SelectionStrategy = SelectionStrategy.BALANCED
    variety_factor: float = 0.3
    min_quality_score: float = 30.0
    
    # Beat synchronization
    variety_pattern: VarietyPattern = VarietyPattern.BALANCED
    use_musical_start: bool = True
    beat_alignment_tolerance: float = 0.1
    
    # Duration preferences
    min_clip_duration: float = 0.5
    max_clip_duration: float = 8.0
    preferred_durations: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    
    # Quality settings
    quality_weight: float = 0.7
    enable_quality_boost: bool = True
    
    # Timeline preferences
    timeline_name: str = "AutoCut Assembly"
    enable_overlap_detection: bool = True
    min_gap_between_clips: float = 1.0


@dataclass
class AssemblyResult:
    """Comprehensive result of assembly operation."""
    
    # Core results
    timeline: ClipTimeline
    selected_clips: List[VideoChunk]
    
    # Assembly statistics
    clips_processed: int
    clips_selected: int
    clips_rejected: int
    
    # Quality metrics
    average_quality: float
    quality_range: Tuple[float, float]
    variety_score: float
    
    # Beat matching results
    beat_match_result: BeatMatchResult
    musical_coverage: float
    
    # Processing metadata
    processing_time: float
    settings_used: AssemblySettings
    
    def get_success_rate(self) -> float:
        """Get assembly success rate."""
        if self.clips_processed == 0:
            return 0.0
        return self.clips_selected / self.clips_processed
    
    def get_timeline_duration(self) -> float:
        """Get total timeline duration."""
        return self.timeline.get_total_duration()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for analysis."""
        return {
            "success_rate": self.get_success_rate(),
            "clips_selected": self.clips_selected,
            "clips_rejected": self.clips_rejected,
            "timeline_duration": self.get_timeline_duration(),
            "average_quality": self.average_quality,
            "variety_score": self.variety_score,
            "musical_coverage": self.musical_coverage,
            "processing_time": self.processing_time,
            "timeline_stats": self.timeline.get_summary_stats(),
            "beat_match_stats": self.beat_match_result.to_dict(),
        }


class AssemblyEngine:
    """Main assembly engine for beat-synced video clip assembly.
    
    Orchestrates the complete assembly process by coordinating timeline
    management, beat matching, and clip selection systems. Provides a
    clean interface that replaces the original god module functionality.
    """
    
    def __init__(self, settings: Optional[AssemblySettings] = None):
        """Initialize assembly engine.
        
        Args:
            settings: Assembly configuration settings
        """
        self.settings = settings or AssemblySettings()
        self.logger = get_logger("autocut.video.assembly.AssemblyEngine")
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Statistics tracking
        self._stats = {
            "assemblies_performed": 0,
            "total_clips_processed": 0,
            "total_processing_time": 0.0,
        }
    
    def _init_subsystems(self) -> None:
        """Initialize assembly subsystems."""
        # Beat matcher configuration
        beat_settings = BeatSyncSettings(
            use_musical_start=self.settings.use_musical_start,
            beat_alignment_tolerance=self.settings.beat_alignment_tolerance,
            min_clip_duration=self.settings.min_clip_duration,
            max_clip_duration=self.settings.max_clip_duration,
            preferred_durations=self.settings.preferred_durations,
            quality_weight=self.settings.quality_weight,
            variety_factor=self.settings.variety_factor,
            variety_pattern=self.settings.variety_pattern,
        )
        self.beat_matcher = BeatMatcher(beat_settings)
        
        # Clip selector configuration
        selection_criteria = SelectionCriteria(
            strategy=self.settings.selection_strategy,
            min_quality_score=self.settings.min_quality_score,
            quality_weight=self.settings.quality_weight,
            variety_weight=self.settings.variety_factor,
            enable_quality_boost=self.settings.enable_quality_boost,
            min_gap_between_clips=self.settings.min_gap_between_clips,
        )
        self.clip_selector = ClipSelector(selection_criteria)
    
    @log_performance("clip_assembly")
    def assemble_clips(self,
                      video_chunks: List[VideoChunk],
                      beats: List[float],
                      timeline_name: Optional[str] = None) -> AssemblyResult:
        """Assemble video clips into beat-synced timeline.
        
        This is the main assembly method that coordinates all subsystems
        to create a beat-synchronized timeline from input video chunks.
        
        Args:
            video_chunks: Available video chunks for assembly
            beats: Beat timestamps for synchronization
            timeline_name: Optional name for the timeline
            
        Returns:
            AssemblyResult with complete assembly information
            
        Raises:
            ValidationError: If inputs are invalid
        """
        start_time = time.time()
        
        # Validate inputs
        self._validate_assembly_inputs(video_chunks, beats)
        
        timeline_name = timeline_name or self.settings.timeline_name
        
        with LoggingContext("clip_assembly", self.logger) as ctx:
            ctx.log(f"Starting assembly: {len(video_chunks)} clips, {len(beats)} beats")
            
            # Phase 1: Pre-select clips using clip selector
            estimated_clips_needed = min(len(beats) // 2, len(video_chunks))
            ctx.log(f"Pre-selecting approximately {estimated_clips_needed} clips")
            
            selection_result = self.clip_selector.select_clips(
                video_chunks, estimated_clips_needed
            )
            
            # Phase 2: Beat matching with selected clips
            ctx.log(f"Beat matching {len(selection_result.selected_clips)} selected clips")
            
            beat_match_result = self.beat_matcher.match_clips_to_beats(
                selection_result.selected_clips, beats, timeline_name
            )
            
            # Phase 3: Create comprehensive result
            processing_time = time.time() - start_time
            result = self._create_assembly_result(
                video_chunks, selection_result, beat_match_result, processing_time
            )
            
            ctx.log(
                f"Assembly completed: {result.clips_selected} clips in timeline",
                extra={
                    "success_rate": f"{result.get_success_rate() * 100:.1f}%",
                    "timeline_duration": f"{result.get_timeline_duration():.1f}s",
                    "processing_time": f"{processing_time:.2f}s",
                }
            )
            
            # Update statistics
            self._stats["assemblies_performed"] += 1
            self._stats["total_clips_processed"] += len(video_chunks)
            self._stats["total_processing_time"] += processing_time
            
            return result
    
    def assemble_clips_legacy(self,
                             video_chunks: List[VideoChunk],
                             beats: List[float],
                             variety_factor: float = 0.3,
                             pattern: str = "balanced") -> List[VideoChunk]:
        """Legacy interface for backward compatibility.
        
        Args:
            video_chunks: Available video chunks
            beats: Beat timestamps
            variety_factor: Balance between quality and variety (0-1)
            pattern: Variety pattern name
            
        Returns:
            List of assembled video chunks in timeline order
        """
        # Convert legacy parameters
        try:
            variety_pattern = VarietyPattern(pattern)
        except ValueError:
            variety_pattern = VarietyPattern.BALANCED
        
        # Create temporary settings
        legacy_settings = AssemblySettings(
            variety_factor=variety_factor,
            variety_pattern=variety_pattern,
            timeline_name="Legacy Assembly"
        )
        
        # Store original settings and use legacy settings
        original_settings = self.settings
        self.settings = legacy_settings
        self._init_subsystems()
        
        try:
            result = self.assemble_clips(video_chunks, beats)
            # Return clips in timeline order
            sorted_entries = result.timeline.get_clips_sorted_by_beat()
            return [self._timeline_entry_to_chunk(entry) for entry in sorted_entries]
        finally:
            # Restore original settings
            self.settings = original_settings
            self._init_subsystems()
    
    def create_clip_timeline(self,
                            video_chunks: List[VideoChunk],
                            beats: List[float],
                            timeline_name: str = "AutoCut Timeline") -> ClipTimeline:
        """Create timeline from video chunks and beats (legacy interface).
        
        Args:
            video_chunks: Available video chunks
            beats: Beat timestamps
            timeline_name: Name for the timeline
            
        Returns:
            ClipTimeline with assembled clips
        """
        result = self.assemble_clips(video_chunks, beats, timeline_name)
        return result.timeline
    
    def _validate_assembly_inputs(self, video_chunks: List[VideoChunk], beats: List[float]) -> None:
        """Validate assembly inputs."""
        if not video_chunks:
            raise_validation_error(
                "No video chunks provided for assembly",
                validation_type="input_validation"
            )
        
        if not beats:
            raise_validation_error(
                "No beats provided for assembly",
                validation_type="input_validation"
            )
        
        if len(beats) < 2:
            raise_validation_error(
                f"Need at least 2 beats for assembly, got {len(beats)}",
                validation_type="input_validation"
            )
        
        # Validate video chunks
        for i, chunk in enumerate(video_chunks):
            if chunk.duration <= 0:
                raise_validation_error(
                    f"Video chunk {i} has invalid duration: {chunk.duration}",
                    validation_type="chunk_validation",
                    file_path=chunk.video_path
                )
            
            if not (0 <= chunk.score <= 100):
                self.logger.warning(
                    f"Video chunk {i} has unusual score: {chunk.score}",
                    extra={"video_path": chunk.video_path}
                )
    
    def _create_assembly_result(self,
                              original_chunks: List[VideoChunk],
                              selection_result,
                              beat_match_result: BeatMatchResult,
                              processing_time: float) -> AssemblyResult:
        """Create comprehensive assembly result."""
        # Calculate quality metrics
        timeline_entries = beat_match_result.timeline.entries
        if timeline_entries:
            quality_scores = [entry.quality_score for entry in timeline_entries]
            average_quality = sum(quality_scores) / len(quality_scores)
            quality_range = (min(quality_scores), max(quality_scores))
        else:
            average_quality = 0.0
            quality_range = (0.0, 0.0)
        
        # Calculate musical coverage
        timeline_stats = beat_match_result.timeline.get_summary_stats()
        musical_coverage = beat_match_result.coverage_ratio
        
        # Get selected clips from timeline
        selected_clips = [
            self._timeline_entry_to_chunk(entry) 
            for entry in timeline_entries
        ]
        
        return AssemblyResult(
            timeline=beat_match_result.timeline,
            selected_clips=selected_clips,
            clips_processed=len(original_chunks),
            clips_selected=len(selected_clips),
            clips_rejected=len(original_chunks) - len(selected_clips),
            average_quality=average_quality,
            quality_range=quality_range,
            variety_score=beat_match_result.variety_score,
            beat_match_result=beat_match_result,
            musical_coverage=musical_coverage,
            processing_time=processing_time,
            settings_used=self.settings,
        )
    
    def _timeline_entry_to_chunk(self, entry: TimelineEntry) -> VideoChunk:
        """Convert timeline entry back to video chunk for compatibility."""
        return VideoChunk(
            video_path=entry.video_file,
            start_time=entry.start_time,
            end_time=entry.end_time,
            score=entry.quality_score,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assembly engine statistics."""
        stats = self._stats.copy()
        
        # Add subsystem statistics
        stats["beat_matcher_stats"] = self.beat_matcher.get_statistics()
        stats["clip_selector_stats"] = self.clip_selector.get_statistics()
        
        # Calculate derived statistics
        if stats["assemblies_performed"] > 0:
            stats["avg_clips_per_assembly"] = stats["total_clips_processed"] / stats["assemblies_performed"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["assemblies_performed"]
        else:
            stats["avg_clips_per_assembly"] = 0.0
            stats["avg_processing_time"] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        for key in self._stats:
            self._stats[key] = 0
        
        self.beat_matcher.reset_statistics()
        self.clip_selector.reset_statistics()
    
    def update_settings(self, new_settings: AssemblySettings) -> None:
        """Update assembly settings and reinitialize subsystems.
        
        Args:
            new_settings: New settings to apply
        """
        self.settings = new_settings
        self._init_subsystems()
        self.logger.info("Assembly settings updated and subsystems reinitialized")


# Convenience functions for common assembly operations
def assemble_clips_simple(clips: List[VideoChunk], beats: List[float]) -> ClipTimeline:
    """Simple clip assembly with default settings."""
    engine = AssemblyEngine()
    result = engine.assemble_clips(clips, beats)
    return result.timeline


def assemble_clips_with_pattern(clips: List[VideoChunk], 
                               beats: List[float],
                               pattern: str = "balanced") -> List[VideoChunk]:
    """Assemble clips with specified variety pattern."""
    try:
        variety_pattern = VarietyPattern(pattern)
    except ValueError:
        variety_pattern = VarietyPattern.BALANCED
    
    settings = AssemblySettings(variety_pattern=variety_pattern)
    engine = AssemblyEngine(settings)
    result = engine.assemble_clips(clips, beats)
    
    return result.selected_clips


def create_balanced_timeline(clips: List[VideoChunk], beats: List[float]) -> ClipTimeline:
    """Create timeline with balanced quality/variety selection."""
    settings = AssemblySettings(
        selection_strategy=SelectionStrategy.BALANCED,
        variety_pattern=VarietyPattern.BALANCED
    )
    engine = AssemblyEngine(settings)
    result = engine.assemble_clips(clips, beats)
    return result.timeline