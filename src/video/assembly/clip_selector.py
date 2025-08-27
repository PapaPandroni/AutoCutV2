"""Clip selection system for video assembly.

Provides intelligent clip selection algorithms that balance quality,
variety, and musical suitability. Extracted and enhanced from the
original god module's clip selection functionality.

Features:
- Quality-based clip ranking and selection
- Video variety management to prevent repetition
- Overlap detection and conflict resolution
- Configurable selection criteria and strategies

Key Components:
- ClipSelector: Main clip selection engine
- SelectionCriteria: Configuration for selection behavior
- SelectionResult: Results with detailed analysis
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Dual import pattern for package/direct execution compatibility
try:
    from ...core.exceptions import ValidationError, raise_validation_error
    from ...core.logging_config import LoggingContext, get_logger, log_performance
except ImportError:
    # Fallback for direct execution
    from core.exceptions import raise_validation_error
    from core.logging_config import LoggingContext, get_logger, log_performance


class SelectionStrategy(Enum):
    """Available clip selection strategies."""

    QUALITY_FOCUSED = "quality_focused"  # Prioritize highest quality clips
    VARIETY_FOCUSED = "variety_focused"  # Maximize video source diversity
    BALANCED = "balanced"  # Balance quality and variety
    DURATION_OPTIMIZED = "duration_optimized"  # Optimize for target durations


class OverlapResolution(Enum):
    """Strategies for resolving clip overlaps."""

    SKIP_OVERLAPPING = "skip_overlapping"  # Skip clips that overlap
    PREFER_HIGHER_QUALITY = "prefer_higher_quality"  # Keep higher quality clip
    PREFER_EARLIER = "prefer_earlier"  # Keep clip with earlier timestamp
    ALLOW_OVERLAPS = "allow_overlaps"  # Allow overlapping clips


@dataclass
class SelectionCriteria:
    """Configuration for clip selection behavior."""

    # Selection strategy
    strategy: SelectionStrategy = SelectionStrategy.BALANCED

    # Quality constraints
    min_quality_score: float = 30.0
    quality_weight: float = 0.7

    # Variety settings
    variety_weight: float = 0.3
    max_clips_per_video: Optional[int] = None
    min_gap_between_clips: float = 1.0

    # Duration preferences
    preferred_duration_range: Tuple[float, float] = (0.5, 8.0)
    duration_tolerance: float = 0.2

    # Overlap handling
    overlap_resolution: OverlapResolution = OverlapResolution.SKIP_OVERLAPPING
    overlap_threshold: float = 1.0

    # Advanced options
    enable_quality_boost: bool = True
    quality_boost_factor: float = 1.2
    enable_diversity_penalty: bool = True
    diversity_penalty_factor: float = 0.8


@dataclass
class VideoChunk:
    """Represents a video chunk for selection (enhanced compatibility)."""

    video_path: str
    start_time: float
    end_time: float
    score: float

    # Optional metadata
    motion_score: Optional[float] = None
    face_score: Optional[float] = None
    brightness_score: Optional[float] = None
    sharpness_score: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def get_quality_breakdown(self) -> Dict[str, float]:
        """Get breakdown of quality components."""
        return {
            "overall": self.score,
            "motion": self.motion_score or 0.0,
            "face": self.face_score or 0.0,
            "brightness": self.brightness_score or 0.0,
            "sharpness": self.sharpness_score or 0.0,
        }

    def __str__(self) -> str:
        from pathlib import Path

        return f"VideoChunk({Path(self.video_path).name}, {self.duration:.1f}s, score={self.score:.1f})"


@dataclass
class SelectionResult:
    """Result of clip selection operation."""

    # Selected clips
    selected_clips: List[VideoChunk]

    # Selection statistics
    total_candidates: int
    clips_selected: int
    clips_rejected: int

    # Quality analysis
    average_quality: float
    quality_range: Tuple[float, float]

    # Variety analysis
    unique_videos: int
    clips_per_video: Dict[str, int]
    variety_score: float

    # Duration analysis
    total_duration: float
    average_duration: float
    duration_range: Tuple[float, float]

    # Processing metadata
    processing_time: float
    strategy_used: str
    criteria_used: SelectionCriteria

    def get_success_rate(self) -> float:
        """Get selection success rate."""
        if self.total_candidates == 0:
            return 0.0
        return self.clips_selected / self.total_candidates

    def get_variety_score(self) -> float:
        """Calculate variety score (0-1)."""
        if self.clips_selected == 0:
            return 0.0
        return self.unique_videos / self.clips_selected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/analysis."""
        return {
            "clips_selected": self.clips_selected,
            "success_rate": self.get_success_rate(),
            "average_quality": self.average_quality,
            "variety_score": self.variety_score,
            "total_duration": self.total_duration,
            "unique_videos": self.unique_videos,
            "processing_time": self.processing_time,
            "strategy_used": self.strategy_used,
            "quality_analysis": {
                "range": self.quality_range,
                "average": self.average_quality,
            },
            "variety_analysis": {
                "unique_videos": self.unique_videos,
                "clips_per_video": self.clips_per_video,
                "variety_score": self.variety_score,
            },
            "duration_analysis": {
                "total": self.total_duration,
                "average": self.average_duration,
                "range": self.duration_range,
            },
        }


class ClipSelector:
    """Intelligent clip selection engine for video assembly.

    Provides sophisticated clip selection algorithms that balance
    quality, variety, and suitability for beat-synced video creation.
    """

    def __init__(self, criteria: Optional[SelectionCriteria] = None):
        """Initialize clip selector.

        Args:
            criteria: Selection criteria configuration
        """
        self.criteria = criteria or SelectionCriteria()
        self.logger = get_logger("autocut.video.assembly.ClipSelector")

        # Statistics tracking
        self._stats = {
            "selections_performed": 0,
            "clips_processed": 0,
            "quality_rejections": 0,
            "overlap_rejections": 0,
            "variety_adjustments": 0,
        }

    @log_performance("clip_selection")
    def select_clips(
        self,
        candidates: List[VideoChunk],
        target_count: int,
        target_durations: Optional[List[float]] = None,
    ) -> SelectionResult:
        """Select best clips from candidates using configured strategy.

        Args:
            candidates: Available video chunks
            target_count: Number of clips to select
            target_durations: Optional list of target durations for optimization

        Returns:
            SelectionResult with selected clips and analysis

        Raises:
            ValidationError: If inputs are invalid
        """
        start_time = time.time()

        # Validate inputs
        self._validate_selection_inputs(candidates, target_count)

        with LoggingContext("clip_selection", self.logger) as ctx:
            ctx.log(f"Selecting {target_count} clips from {len(candidates)} candidates")

            # Filter candidates by quality threshold
            qualified_clips = self._filter_by_quality(candidates)
            ctx.log(
                f"Quality filtering: {len(qualified_clips)}/{len(candidates)} clips passed"
            )

            if not qualified_clips:
                # Return empty result if no clips meet quality threshold
                return self._create_empty_result(candidates, start_time)

            # Apply selection strategy
            selected_clips = self._apply_selection_strategy(
                qualified_clips,
                target_count,
                target_durations,
            )

            # Create comprehensive result
            processing_time = time.time() - start_time
            result = self._create_selection_result(
                candidates,
                selected_clips,
                processing_time,
            )

            ctx.log(
                f"Selection completed: {result.clips_selected} clips selected",
                extra={
                    "success_rate": f"{result.get_success_rate() * 100:.1f}%",
                    "average_quality": f"{result.average_quality:.1f}",
                    "variety_score": f"{result.variety_score:.2f}",
                },
            )

            self._stats["selections_performed"] += 1
            self._stats["clips_processed"] += len(candidates)

            return result

    def select_best_clips(
        self,
        video_chunks: List[VideoChunk],
        target_count: int,
        variety_factor: float = 0.3,
    ) -> List[VideoChunk]:
        """Legacy interface for backward compatibility.

        Args:
            video_chunks: Available video chunks
            target_count: Number of clips to select
            variety_factor: Variety vs quality balance (0-1)

        Returns:
            List of selected video chunks
        """
        # Convert legacy variety_factor to modern criteria
        if variety_factor >= 0.9:
            strategy = SelectionStrategy.VARIETY_FOCUSED
        elif variety_factor <= 0.1:
            strategy = SelectionStrategy.QUALITY_FOCUSED
        else:
            strategy = SelectionStrategy.BALANCED

        # Create temporary criteria
        criteria = SelectionCriteria(
            strategy=strategy,
            variety_weight=variety_factor,
            quality_weight=1.0 - variety_factor,
        )

        # Use modern selection system
        original_criteria = self.criteria
        self.criteria = criteria

        try:
            result = self.select_clips(video_chunks, target_count)
            return result.selected_clips
        finally:
            self.criteria = original_criteria

    def _validate_selection_inputs(
        self, candidates: List[VideoChunk], target_count: int
    ) -> None:
        """Validate selection inputs."""
        if not candidates:
            raise_validation_error(
                "No candidate clips provided for selection",
                validation_type="input_validation",
            )

        if target_count <= 0:
            raise_validation_error(
                f"Target count must be positive, got {target_count}",
                validation_type="input_validation",
            )

        # Validate video chunks
        for i, chunk in enumerate(candidates):
            if chunk.duration <= 0:
                raise_validation_error(
                    f"Candidate {i} has invalid duration: {chunk.duration}",
                    validation_type="chunk_validation",
                    file_path=chunk.video_path,
                )

            if not (0 <= chunk.score <= 100):
                self.logger.warning(
                    f"Candidate {i} has unusual score: {chunk.score}",
                    extra={"video_path": chunk.video_path},
                )

    def _filter_by_quality(self, candidates: List[VideoChunk]) -> List[VideoChunk]:
        """Filter candidates by minimum quality threshold."""
        qualified = []

        for chunk in candidates:
            if chunk.score >= self.criteria.min_quality_score:
                # Apply quality boost if enabled
                if self.criteria.enable_quality_boost and chunk.score >= 80:
                    # Boost high-quality clips for better selection
                    boosted_chunk = VideoChunk(
                        video_path=chunk.video_path,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        score=min(
                            100, chunk.score * self.criteria.quality_boost_factor
                        ),
                        motion_score=chunk.motion_score,
                        face_score=chunk.face_score,
                        brightness_score=chunk.brightness_score,
                        sharpness_score=chunk.sharpness_score,
                    )
                    qualified.append(boosted_chunk)
                else:
                    qualified.append(chunk)
            else:
                self._stats["quality_rejections"] += 1

        return qualified

    def _apply_selection_strategy(
        self,
        candidates: List[VideoChunk],
        target_count: int,
        target_durations: Optional[List[float]],
    ) -> List[VideoChunk]:
        """Apply the configured selection strategy."""
        strategy = self.criteria.strategy

        if strategy == SelectionStrategy.QUALITY_FOCUSED:
            return self._select_by_quality(candidates, target_count)

        if strategy == SelectionStrategy.VARIETY_FOCUSED:
            return self._select_by_variety(candidates, target_count)

        if strategy == SelectionStrategy.BALANCED:
            return self._select_balanced(candidates, target_count)

        if strategy == SelectionStrategy.DURATION_OPTIMIZED:
            return self._select_by_duration(candidates, target_count, target_durations)

        self.logger.warning(f"Unknown strategy {strategy}, using balanced")
        return self._select_balanced(candidates, target_count)

    def _select_by_quality(
        self, candidates: List[VideoChunk], target_count: int
    ) -> List[VideoChunk]:
        """Select clips prioritizing quality above all else."""
        # Sort by quality score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        selected = []

        for candidate in sorted_candidates:
            if len(selected) >= target_count:
                break

            # Check for overlaps if required
            if not self._has_unacceptable_overlap(candidate, selected):
                selected.append(candidate)
            else:
                self._stats["overlap_rejections"] += 1

        return selected

    def _select_by_variety(
        self, candidates: List[VideoChunk], target_count: int
    ) -> List[VideoChunk]:
        """Select clips maximizing video source variety."""
        # Group by video file
        clips_by_video = {}
        for chunk in candidates:
            if chunk.video_path not in clips_by_video:
                clips_by_video[chunk.video_path] = []
            clips_by_video[chunk.video_path].append(chunk)

        # Sort clips within each video by quality
        for video_path in clips_by_video:
            clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)

        # Round-robin selection from each video
        selected = []
        video_paths = list(clips_by_video.keys())
        video_index = 0
        attempts = 0
        max_attempts = target_count * len(video_paths)

        while len(selected) < target_count and attempts < max_attempts:
            video_path = video_paths[video_index % len(video_paths)]
            available_clips = clips_by_video[video_path]

            # Find next non-overlapping clip from this video
            for clip in available_clips:
                if clip not in selected and not self._has_unacceptable_overlap(
                    clip, selected
                ):
                    selected.append(clip)
                    self._stats["variety_adjustments"] += 1
                    break

            video_index += 1
            attempts += 1

        return selected

    def _select_balanced(
        self, candidates: List[VideoChunk], target_count: int
    ) -> List[VideoChunk]:
        """Select clips balancing quality and variety."""
        # Group by video file
        clips_by_video = {}
        for chunk in candidates:
            if chunk.video_path not in clips_by_video:
                clips_by_video[chunk.video_path] = []
            clips_by_video[chunk.video_path].append(chunk)

        # Sort clips within each video by quality
        for video_path in clips_by_video:
            clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)

        # Calculate clips per video with variety consideration
        num_videos = len(clips_by_video)
        if self.criteria.max_clips_per_video:
            max_per_video = self.criteria.max_clips_per_video
        else:
            max_per_video = max(1, target_count // num_videos + 1)

        selected = []

        # First pass: Get base clips from each video
        for video_path in clips_by_video:
            clips_from_video = 0
            for clip in clips_by_video[video_path]:
                if clips_from_video >= max_per_video:
                    break
                if not self._has_unacceptable_overlap(clip, selected):
                    selected.append(clip)
                    clips_from_video += 1

        # Second pass: Fill remaining slots with highest quality
        if len(selected) < target_count:
            remaining_clips = []
            for video_path in clips_by_video:
                # Get clips beyond the base allocation
                remaining_clips.extend(
                    [
                        clip
                        for clip in clips_by_video[video_path]
                        if clip not in selected
                    ]
                )

            # Sort by quality and add remaining clips
            remaining_clips.sort(key=lambda x: x.score, reverse=True)

            for clip in remaining_clips:
                if len(selected) >= target_count:
                    break
                if not self._has_unacceptable_overlap(clip, selected):
                    selected.append(clip)

        return selected[:target_count]

    def _select_by_duration(
        self,
        candidates: List[VideoChunk],
        target_count: int,
        target_durations: Optional[List[float]],
    ) -> List[VideoChunk]:
        """Select clips optimizing for target durations."""
        if not target_durations:
            # Fall back to balanced selection if no target durations
            return self._select_balanced(candidates, target_count)

        selected = []
        used_candidates = set()

        # Match each target duration with best candidate
        for target_duration in target_durations[:target_count]:
            best_candidate = None
            best_score = -1

            for candidate in candidates:
                if id(candidate) in used_candidates:
                    continue

                # Calculate duration fit score
                duration_fit = self._calculate_duration_fit(
                    candidate.duration, target_duration
                )
                if duration_fit < 0:
                    continue

                # Check for overlaps
                if self._has_unacceptable_overlap(candidate, selected):
                    continue

                # Combined score: quality + duration fit
                quality_score = candidate.score / 100.0
                combined_score = (
                    self.criteria.quality_weight * quality_score
                    + (1 - self.criteria.quality_weight) * duration_fit
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                used_candidates.add(id(best_candidate))

        return selected

    def _calculate_duration_fit(
        self, clip_duration: float, target_duration: float
    ) -> float:
        """Calculate how well clip duration fits target duration."""
        min_dur, max_dur = self.criteria.preferred_duration_range

        # Check if target is within preferred range
        if not (min_dur <= target_duration <= max_dur):
            return -1

        # Perfect match
        if abs(clip_duration - target_duration) <= self.criteria.duration_tolerance:
            return 1.0

        # Calculate fit score based on difference
        difference = abs(clip_duration - target_duration)
        max_acceptable_diff = max_dur - min_dur

        if difference > max_acceptable_diff:
            return -1

        # Linear scoring based on difference
        fit_score = 1.0 - (difference / max_acceptable_diff)
        return max(0.0, fit_score)

    def _has_unacceptable_overlap(
        self, candidate: VideoChunk, selected: List[VideoChunk]
    ) -> bool:
        """Check if candidate has unacceptable overlap with selected clips."""
        resolution = self.criteria.overlap_resolution

        if resolution == OverlapResolution.ALLOW_OVERLAPS:
            return False

        for selected_clip in selected:
            if selected_clip.video_path != candidate.video_path:
                continue  # Only check overlaps within same video

            # Calculate overlap
            overlap_start = max(candidate.start_time, selected_clip.start_time)
            overlap_end = min(candidate.end_time, selected_clip.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Check gap between clips
            gap = min(
                abs(candidate.start_time - selected_clip.end_time),
                abs(selected_clip.start_time - candidate.end_time),
            )

            if overlap_duration > 0 or gap < self.criteria.min_gap_between_clips:
                if resolution == OverlapResolution.SKIP_OVERLAPPING:
                    return True
                if resolution == OverlapResolution.PREFER_HIGHER_QUALITY:
                    return candidate.score <= selected_clip.score
                if resolution == OverlapResolution.PREFER_EARLIER:
                    return candidate.start_time >= selected_clip.start_time

        return False

    def _create_empty_result(
        self, candidates: List[VideoChunk], start_time: float
    ) -> SelectionResult:
        """Create empty selection result."""
        processing_time = time.time() - start_time

        return SelectionResult(
            selected_clips=[],
            total_candidates=len(candidates),
            clips_selected=0,
            clips_rejected=len(candidates),
            average_quality=0.0,
            quality_range=(0.0, 0.0),
            unique_videos=0,
            clips_per_video={},
            variety_score=0.0,
            total_duration=0.0,
            average_duration=0.0,
            duration_range=(0.0, 0.0),
            processing_time=processing_time,
            strategy_used=self.criteria.strategy.value,
            criteria_used=self.criteria,
        )

    def _create_selection_result(
        self,
        candidates: List[VideoChunk],
        selected: List[VideoChunk],
        processing_time: float,
    ) -> SelectionResult:
        """Create comprehensive selection result."""
        if not selected:
            return self._create_empty_result(candidates, time.time() - processing_time)

        # Quality analysis
        quality_scores = [clip.score for clip in selected]
        average_quality = sum(quality_scores) / len(quality_scores)
        quality_range = (min(quality_scores), max(quality_scores))

        # Variety analysis
        clips_per_video = {}
        for clip in selected:
            clips_per_video[clip.video_path] = (
                clips_per_video.get(clip.video_path, 0) + 1
            )

        unique_videos = len(clips_per_video)
        variety_score = unique_videos / len(selected) if selected else 0.0

        # Duration analysis
        durations = [clip.duration for clip in selected]
        total_duration = sum(durations)
        average_duration = total_duration / len(durations)
        duration_range = (min(durations), max(durations))

        return SelectionResult(
            selected_clips=selected,
            total_candidates=len(candidates),
            clips_selected=len(selected),
            clips_rejected=len(candidates) - len(selected),
            average_quality=average_quality,
            quality_range=quality_range,
            unique_videos=unique_videos,
            clips_per_video=clips_per_video,
            variety_score=variety_score,
            total_duration=total_duration,
            average_duration=average_duration,
            duration_range=duration_range,
            processing_time=processing_time,
            strategy_used=self.criteria.strategy.value,
            criteria_used=self.criteria,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get clip selector statistics."""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


# Convenience functions
def select_clips_by_quality(clips: List[VideoChunk], count: int) -> List[VideoChunk]:
    """Select clips prioritizing quality."""
    criteria = SelectionCriteria(strategy=SelectionStrategy.QUALITY_FOCUSED)
    selector = ClipSelector(criteria)
    result = selector.select_clips(clips, count)
    return result.selected_clips


def select_clips_by_variety(clips: List[VideoChunk], count: int) -> List[VideoChunk]:
    """Select clips prioritizing variety."""
    criteria = SelectionCriteria(strategy=SelectionStrategy.VARIETY_FOCUSED)
    selector = ClipSelector(criteria)
    result = selector.select_clips(clips, count)
    return result.selected_clips


def select_clips_balanced(clips: List[VideoChunk], count: int) -> List[VideoChunk]:
    """Select clips with balanced quality and variety."""
    criteria = SelectionCriteria(strategy=SelectionStrategy.BALANCED)
    selector = ClipSelector(criteria)
    result = selector.select_clips(clips, count)
    return result.selected_clips
