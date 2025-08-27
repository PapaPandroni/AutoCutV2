"""Beat matching engine for video clip assembly.

Provides beat-to-clip synchronization algorithms that align video clips
with musical beats. Extracted and enhanced from the original god module's
beat matching functionality (lines 2682-2876).

Features:
- Musical intelligence with beat filtering
- Duration optimization for beat alignment
- Clip fitting algorithms with trimming support
- Variety pattern application for dynamic pacing

Key Components:
- BeatMatcher: Main beat synchronization engine
- BeatSyncSettings: Configuration for beat matching behavior
- BeatMatchResult: Results of beat matching operation
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Dual import pattern for package/direct execution compatibility
try:
    from ...core.exceptions import ValidationError, raise_validation_error
    from ...core.logging_config import get_logger, log_performance
except ImportError:
    # Fallback for direct execution
    from core.exceptions import raise_validation_error
    from core.logging_config import LoggingContext, get_logger, log_performance

# Dual import pattern for timeline
try:
    from .timeline import ClipTimeline, TimelineEntry, TimelinePosition
except ImportError:
    from video.assembly.timeline import ClipTimeline, TimelineEntry, TimelinePosition


# Variety patterns for dynamic clip pacing
VARIETY_PATTERNS = {
    "energetic": [1, 1, 2, 1, 1, 4],  # Mostly fast with occasional pause
    "buildup": [4, 2, 2, 1, 1, 1],  # Start slow, increase pace
    "balanced": [2, 1, 2, 4, 2, 1],  # Mixed pacing
    "dramatic": [1, 1, 1, 1, 8],  # Fast cuts then long hold
}


class VarietyPattern(Enum):
    """Available variety patterns for clip assembly."""

    ENERGETIC = "energetic"
    BUILDUP = "buildup"
    BALANCED = "balanced"
    DRAMATIC = "dramatic"


# Import VideoChunk from canonical location
from .clip_selector import VideoChunk


@dataclass
class BeatSyncSettings:
    """Configuration for beat synchronization behavior."""

    # Musical intelligence settings
    use_musical_start: bool = True
    musical_start_threshold: float = 0.1
    beat_alignment_tolerance: float = 0.1

    # Duration constraints
    min_clip_duration: float = 0.5
    max_clip_duration: float = 8.0
    preferred_durations: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0]
    )

    # Quality vs variety balance
    quality_weight: float = 0.7
    duration_fit_weight: float = 0.3
    variety_factor: float = 0.3

    # Trimming behavior
    max_trim_seconds: float = 2.0
    prefer_trim_end: bool = True

    # Pattern settings
    variety_pattern: VarietyPattern = VarietyPattern.BALANCED
    allow_pattern_adaptation: bool = True


@dataclass
class BeatMatchResult:
    """Result of beat matching operation."""

    # Timeline result
    timeline: ClipTimeline

    # Matching statistics
    total_beats_processed: int
    clips_matched: int
    clips_skipped: int
    average_fit_score: float

    # Musical analysis
    musical_start_time: float
    effective_beat_count: int
    average_beat_interval: float

    # Quality metrics
    average_clip_quality: float
    variety_score: float
    coverage_ratio: float

    # Processing metadata
    processing_time: float
    pattern_used: str
    settings_used: BeatSyncSettings

    def get_success_rate(self) -> float:
        """Get the success rate of beat matching."""
        if self.total_beats_processed == 0:
            return 0.0
        return self.clips_matched / self.total_beats_processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging/analysis."""
        return {
            "success_rate": self.get_success_rate(),
            "clips_matched": self.clips_matched,
            "clips_skipped": self.clips_skipped,
            "average_fit_score": self.average_fit_score,
            "average_clip_quality": self.average_clip_quality,
            "variety_score": self.variety_score,
            "coverage_ratio": self.coverage_ratio,
            "processing_time": self.processing_time,
            "pattern_used": self.pattern_used,
            "musical_analysis": {
                "musical_start_time": self.musical_start_time,
                "effective_beat_count": self.effective_beat_count,
                "average_beat_interval": self.average_beat_interval,
            },
        }


class BeatMatcher:
    """Beat matching engine for video clip assembly.

    Provides intelligent beat-to-clip synchronization with musical awareness,
    duration optimization, and variety pattern support.
    """

    def __init__(self, settings: Optional[BeatSyncSettings] = None):
        """Initialize beat matcher.

        Args:
            settings: Beat synchronization settings
        """
        self.settings = settings or BeatSyncSettings()
        self.logger = get_logger("autocut.video.assembly.BeatMatcher")

        # Statistics tracking
        self._stats = {
            "matches_attempted": 0,
            "matches_successful": 0,
            "clips_trimmed": 0,
            "beats_filtered": 0,
        }

    @log_performance("beat_matching")
    def match_clips_to_beats(
        self,
        video_chunks: List[VideoChunk],
        beats: List[float],
        timeline_name: str = "Beat-Matched Timeline",
    ) -> BeatMatchResult:
        """Match video chunks to beat grid with musical intelligence.

        Args:
            video_chunks: List of scored video chunks
            beats: List of beat timestamps in seconds
            timeline_name: Name for the resulting timeline

        Returns:
            BeatMatchResult with timeline and matching statistics

        Raises:
            ValidationError: If inputs are invalid
        """
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(video_chunks, beats)

        with LoggingContext("beat_matching", self.logger) as ctx:
            ctx.log(f"Matching {len(video_chunks)} clips to {len(beats)} beats")

            # Apply musical intelligence to filter beats
            effective_beats, musical_start_time = (
                self._filter_beats_for_musical_content(beats)
            )

            if len(effective_beats) < 2:
                ctx.log("Insufficient effective beats after filtering", level="WARNING")
                effective_beats = beats[:10] if len(beats) >= 10 else beats
                musical_start_time = beats[0] if beats else 0.0

            # Calculate beat timing characteristics
            beat_analysis = self._analyze_beat_timing(effective_beats)

            # Apply variety pattern to get beat multipliers
            beat_multipliers = self._apply_variety_pattern(
                self.settings.variety_pattern,
                len(effective_beats) - 1,
            )

            # Convert beat multipliers to target durations
            target_durations = [
                multiplier * beat_analysis["avg_interval"]
                for multiplier in beat_multipliers
            ]

            # Create timeline and match clips
            timeline = ClipTimeline(name=timeline_name)
            match_stats = self._perform_beat_matching(
                timeline,
                video_chunks,
                effective_beats,
                target_durations,
                beat_multipliers,
            )

            # Calculate final statistics
            processing_time = time.time() - start_time
            result = self._create_match_result(
                timeline,
                video_chunks,
                beats,
                effective_beats,
                musical_start_time,
                beat_analysis,
                match_stats,
                processing_time,
            )

            ctx.log(
                f"Beat matching completed: {result.clips_matched}/{result.total_beats_processed} beats matched",
                extra={"success_rate": f"{result.get_success_rate() * 100:.1f}%"},
            )

            return result

    def _validate_inputs(
        self, video_chunks: List[VideoChunk], beats: List[float]
    ) -> None:
        """Validate beat matching inputs."""
        if not video_chunks:
            raise_validation_error(
                "No video chunks provided for beat matching",
                validation_type="input_validation",
            )

        if not beats:
            raise_validation_error(
                "No beats provided for beat matching",
                validation_type="input_validation",
            )

        if len(beats) < 2:
            raise_validation_error(
                f"Need at least 2 beats for matching, got {len(beats)}",
                validation_type="input_validation",
            )

        # Validate beats are in ascending order
        for i in range(1, len(beats)):
            if beats[i] <= beats[i - 1]:
                raise_validation_error(
                    f"Beats must be in ascending order: {beats[i - 1]} >= {beats[i]} at index {i}",
                    validation_type="beat_timing",
                )

        # Validate video chunks
        for i, chunk in enumerate(video_chunks):
            if chunk.duration <= 0:
                raise_validation_error(
                    f"Video chunk {i} has invalid duration: {chunk.duration}",
                    validation_type="chunk_validation",
                    file_path=chunk.video_path,
                )

    def _filter_beats_for_musical_content(
        self, beats: List[float]
    ) -> Tuple[List[float], float]:
        """Apply musical intelligence to filter beats.

        Args:
            beats: Raw beat timestamps

        Returns:
            Tuple of (filtered_beats, musical_start_time)
        """
        if not self.settings.use_musical_start or len(beats) < 4:
            return beats, beats[0] if beats else 0.0

        # Find the first significant musical beat (skip intro/buildup)
        # Look for consistent beat intervals after initial variation
        beat_intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]

        if len(beat_intervals) < 3:
            return beats, beats[0]

        # Calculate median interval for consistency check
        sorted_intervals = sorted(beat_intervals)
        median_interval = sorted_intervals[len(sorted_intervals) // 2]

        # Find first position where beat intervals stabilize
        musical_start_index = 0
        consistency_window = 3

        for i in range(len(beat_intervals) - consistency_window):
            window_intervals = beat_intervals[i : i + consistency_window]
            consistent_count = sum(
                1
                for interval in window_intervals
                if abs(interval - median_interval) / median_interval
                < self.settings.musical_start_threshold
            )

            if consistent_count >= consistency_window - 1:  # Allow one outlier
                musical_start_index = i
                break

        musical_start_time = beats[musical_start_index]
        effective_beats = beats[musical_start_index:]

        self._stats["beats_filtered"] = musical_start_index

        self.logger.debug(
            f"Musical intelligence: filtered {musical_start_index} intro beats",
            extra={
                "original_beats": len(beats),
                "effective_beats": len(effective_beats),
                "musical_start_time": musical_start_time,
            },
        )

        return effective_beats, musical_start_time

    def _analyze_beat_timing(self, beats: List[float]) -> Dict[str, float]:
        """Analyze beat timing characteristics.

        Args:
            beats: Beat timestamps to analyze

        Returns:
            Dictionary with timing analysis results
        """
        if len(beats) < 2:
            return {"avg_interval": 1.0, "interval_variance": 0.0, "tempo_bpm": 60.0}

        intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        # Calculate variance
        variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(
            intervals
        )

        # Estimate BPM
        tempo_bpm = 60.0 / avg_interval if avg_interval > 0 else 60.0

        return {
            "avg_interval": avg_interval,
            "interval_variance": variance,
            "tempo_bpm": tempo_bpm,
            "interval_range": (min(intervals), max(intervals)),
        }

    def _apply_variety_pattern(
        self, pattern: VarietyPattern, beat_count: int
    ) -> List[int]:
        """Apply variety pattern to determine clip lengths.

        Args:
            pattern: Variety pattern to apply
            beat_count: Total number of beats to fill

        Returns:
            List of beat multipliers for each clip
        """
        if pattern.value not in VARIETY_PATTERNS:
            self.logger.warning(f"Unknown pattern {pattern.value}, using balanced")
            pattern = VarietyPattern.BALANCED

        pattern_multipliers = VARIETY_PATTERNS[pattern.value]
        result = []
        pattern_index = 0
        remaining_beats = beat_count

        while remaining_beats > 0:
            multiplier = pattern_multipliers[pattern_index % len(pattern_multipliers)]

            if multiplier <= remaining_beats:
                result.append(multiplier)
                remaining_beats -= multiplier
            # Adapt pattern if allowed
            elif self.settings.allow_pattern_adaptation and remaining_beats > 0:
                result.append(remaining_beats)
                remaining_beats = 0
            else:
                break

            pattern_index += 1

        self.logger.debug(
            f"Applied {pattern.value} pattern: {len(result)} clips from {beat_count} beats",
        )

        return result

    def _perform_beat_matching(
        self,
        timeline: ClipTimeline,
        video_chunks: List[VideoChunk],
        effective_beats: List[float],
        target_durations: List[float],
        beat_multipliers: List[int],
    ) -> Dict[str, Any]:
        """Perform the actual beat matching process.

        Args:
            timeline: Timeline to populate
            video_chunks: Available video chunks
            effective_beats: Filtered beat timestamps
            target_durations: Target duration for each position
            beat_multipliers: Beat multipliers for each position

        Returns:
            Dictionary with matching statistics
        """
        # Pre-select best clips for variety
        estimated_clips_needed = len(target_durations)
        selected_clips = self._select_clips_for_variety(
            video_chunks,
            min(estimated_clips_needed * 2, len(video_chunks)),
        )

        used_clips = set()
        fit_scores = []
        clips_matched = 0
        clips_skipped = 0

        current_beat_index = 0

        for i, target_duration in enumerate(target_durations):
            if current_beat_index >= len(effective_beats):
                clips_skipped += 1
                continue

            self._stats["matches_attempted"] += 1

            # Find best matching clip for this target duration
            best_clip, fit_score = self._find_best_clip_for_duration(
                selected_clips,
                target_duration,
                used_clips,
            )

            if best_clip is None:
                self.logger.debug(
                    f"No suitable clip found for duration {target_duration:.2f}s"
                )
                current_beat_index += beat_multipliers[i]
                clips_skipped += 1
                continue

            # Mark clip as used
            used_clips.add(id(best_clip))
            fit_scores.append(fit_score)

            # Determine actual clip timing
            beat_position = effective_beats[current_beat_index]
            clip_start, clip_end, actual_duration = self._fit_clip_to_duration(
                best_clip,
                target_duration,
            )

            # Create timeline position
            position = TimelinePosition(
                beat_position=beat_position,
                beat_index=current_beat_index,
                beat_multiplier=beat_multipliers[i],
                musical_time=beat_position,
            )

            # Create timeline entry
            entry = TimelineEntry(
                video_file=best_clip.video_path,
                start_time=clip_start,
                end_time=clip_end,
                duration=actual_duration,
                position=position,
                quality_score=best_clip.score,
                selection_reason="beat_match",
            )

            timeline.add_entry(entry)
            clips_matched += 1
            self._stats["matches_successful"] += 1

            # Move to next beat position
            current_beat_index += beat_multipliers[i]

        return {
            "clips_matched": clips_matched,
            "clips_skipped": clips_skipped,
            "average_fit_score": sum(fit_scores) / len(fit_scores)
            if fit_scores
            else 0.0,
            "clips_used": len(used_clips),
        }

    def _select_clips_for_variety(
        self, video_chunks: List[VideoChunk], count: int
    ) -> List[VideoChunk]:
        """Select clips ensuring variety in source videos.

        Args:
            video_chunks: Available video chunks
            count: Number of clips to select

        Returns:
            List of selected video chunks
        """
        if len(video_chunks) <= count:
            return video_chunks.copy()

        # Group by video file
        clips_by_video = {}
        for chunk in video_chunks:
            if chunk.video_path not in clips_by_video:
                clips_by_video[chunk.video_path] = []
            clips_by_video[chunk.video_path].append(chunk)

        # Sort clips within each video by score
        for video_path in clips_by_video:
            clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)

        selected = []
        variety_factor = self.settings.variety_factor

        if variety_factor >= 0.9:
            # High variety: Round-robin from each video
            video_paths = list(clips_by_video.keys())
            video_index = 0

            while len(selected) < count and video_index < len(video_paths) * count:
                video_path = video_paths[video_index % len(video_paths)]
                available = clips_by_video[video_path]

                for clip in available:
                    if clip not in selected and not self._clips_overlap(clip, selected):
                        selected.append(clip)
                        break

                video_index += 1

        elif variety_factor <= 0.1:
            # High quality: Best clips regardless of source
            all_sorted = sorted(video_chunks, key=lambda x: x.score, reverse=True)
            for clip in all_sorted:
                if len(selected) >= count:
                    break
                if not self._clips_overlap(clip, selected):
                    selected.append(clip)

        else:
            # Balanced approach
            num_videos = len(clips_by_video)
            base_per_video = max(1, count // num_videos)

            # First pass: Base clips from each video
            for video_path in clips_by_video:
                clips_from_video = 0
                for clip in clips_by_video[video_path]:
                    if clips_from_video >= base_per_video:
                        break
                    if not self._clips_overlap(clip, selected):
                        selected.append(clip)
                        clips_from_video += 1

            # Second pass: Fill remaining with best quality
            remaining_clips = []
            for video_path in clips_by_video:
                remaining_clips.extend(clips_by_video[video_path][base_per_video:])

            remaining_clips.sort(key=lambda x: x.score, reverse=True)

            for clip in remaining_clips:
                if len(selected) >= count:
                    break
                if clip not in selected and not self._clips_overlap(clip, selected):
                    selected.append(clip)

        return selected[:count]

    def _find_best_clip_for_duration(
        self, clips: List[VideoChunk], target_duration: float, used_clips: set
    ) -> Tuple[Optional[VideoChunk], float]:
        """Find the best clip for a target duration.

        Args:
            clips: Available clips
            target_duration: Target duration in seconds
            used_clips: Set of already used clip IDs

        Returns:
            Tuple of (best_clip, fit_score) or (None, 0.0)
        """
        best_clip = None
        best_score = -1.0

        for clip in clips:
            if id(clip) in used_clips:
                continue

            # Calculate duration fit score
            duration_fit = self._calculate_duration_fit(clip.duration, target_duration)
            if duration_fit < 0:  # Clip can't be used
                continue

            # Combined score: quality + duration fit
            quality_score = clip.score / 100.0  # Normalize to 0-1
            fit_score = (
                self.settings.quality_weight * quality_score
                + self.settings.duration_fit_weight * duration_fit
            )

            if fit_score > best_score:
                best_score = fit_score
                best_clip = clip

        return best_clip, best_score

    def _calculate_duration_fit(
        self, clip_duration: float, target_duration: float
    ) -> float:
        """Calculate how well a clip duration fits the target.

        Args:
            clip_duration: Duration of video clip
            target_duration: Target duration

        Returns:
            Fit score 0.0-1.0, or -1 if unusable
        """
        # Check if target is in preferred durations
        duration_allowed = any(
            abs(target_duration - preferred) < self.settings.beat_alignment_tolerance
            for preferred in self.settings.preferred_durations
        )

        if not duration_allowed:
            return -1  # Target not musically appropriate

        # Perfect match
        if (
            abs(clip_duration - target_duration)
            < self.settings.beat_alignment_tolerance
        ):
            return 1.0

        # Clip longer than target - can trim
        if clip_duration > target_duration:
            excess = clip_duration - target_duration
            if excess <= self.settings.max_trim_seconds:
                return 1.0 - (
                    excess / (self.settings.max_trim_seconds * 2)
                )  # Gentle penalty
            return 0.3  # Heavy penalty for excessive trimming

        # Clip shorter than target
        shortage = target_duration - clip_duration
        if shortage <= 0.5:  # Small shortage acceptable
            return 0.8 - (shortage / 1.0)
        return -1  # Too short

    def _fit_clip_to_duration(
        self, clip: VideoChunk, target_duration: float
    ) -> Tuple[float, float, float]:
        """Fit clip to target duration by trimming if necessary.

        Args:
            clip: Video chunk to fit
            target_duration: Target duration

        Returns:
            Tuple of (start_time, end_time, actual_duration)
        """
        if clip.duration <= target_duration + self.settings.beat_alignment_tolerance:
            # Clip fits as-is
            return clip.start_time, clip.end_time, clip.duration

        # Clip needs trimming
        if self.settings.prefer_trim_end:
            # Trim from end (preserve beginning)
            new_end_time = clip.start_time + target_duration
            new_end_time = min(new_end_time, clip.end_time)
            actual_duration = new_end_time - clip.start_time
            self._stats["clips_trimmed"] += 1
            return clip.start_time, new_end_time, actual_duration
        # Trim from beginning (preserve end)
        new_start_time = clip.end_time - target_duration
        new_start_time = max(new_start_time, clip.start_time)
        actual_duration = clip.end_time - new_start_time
        self._stats["clips_trimmed"] += 1
        return new_start_time, clip.end_time, actual_duration

    def _clips_overlap(
        self, clip: VideoChunk, existing_clips: List[VideoChunk], min_gap: float = 1.0
    ) -> bool:
        """Check if clip overlaps with existing clips from same video.

        Args:
            clip: Clip to check
            existing_clips: Already selected clips
            min_gap: Minimum gap between clips from same video

        Returns:
            True if clip overlaps with any existing clip
        """
        for existing in existing_clips:
            if existing.video_path == clip.video_path:
                if (
                    clip.start_time < existing.end_time + min_gap
                    and clip.end_time > existing.start_time - min_gap
                ):
                    return True
        return False

    def _create_match_result(
        self,
        timeline: ClipTimeline,
        original_chunks: List[VideoChunk],
        original_beats: List[float],
        effective_beats: List[float],
        musical_start_time: float,
        beat_analysis: Dict[str, Any],
        match_stats: Dict[str, Any],
        processing_time: float,
    ) -> BeatMatchResult:
        """Create comprehensive beat match result.

        Returns:
            BeatMatchResult with all statistics and analysis
        """
        timeline_stats = timeline.get_summary_stats()

        # Calculate variety score (uniqueness of video sources)
        unique_videos = len(timeline.get_unique_video_files())
        total_clips = timeline_stats["total_clips"]
        variety_score = unique_videos / total_clips if total_clips > 0 else 0.0

        # Calculate coverage ratio
        if original_beats:
            timeline_span = timeline_stats["coverage_duration"]
            total_song_span = original_beats[-1] - original_beats[0]
            coverage_ratio = (
                timeline_span / total_song_span if total_song_span > 0 else 0.0
            )
        else:
            coverage_ratio = 0.0

        return BeatMatchResult(
            timeline=timeline,
            total_beats_processed=len(effective_beats) - 1,
            clips_matched=match_stats["clips_matched"],
            clips_skipped=match_stats["clips_skipped"],
            average_fit_score=match_stats["average_fit_score"],
            musical_start_time=musical_start_time,
            effective_beat_count=len(effective_beats),
            average_beat_interval=beat_analysis["avg_interval"],
            average_clip_quality=timeline_stats["avg_quality"],
            variety_score=variety_score,
            coverage_ratio=coverage_ratio,
            processing_time=processing_time,
            pattern_used=self.settings.variety_pattern.value,
            settings_used=self.settings,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get beat matcher statistics."""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0
