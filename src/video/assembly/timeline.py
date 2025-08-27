"""Timeline management system for video clip assembly.

Provides timeline organization, beat synchronization, and clip management
for the AutoCut video assembly engine. Extracted and enhanced from the
original god module's ClipTimeline class.

Features:
- Timeline entry management with beat positions
- Duration and coverage validation
- Timeline statistics and analytics
- JSON export for debugging and analysis
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Dual import pattern for package/direct execution compatibility
try:
    from ...core.exceptions import ValidationError, raise_validation_error
    from ...core.logging_config import get_logger, log_performance
except ImportError:
    # Fallback for direct execution
    from core.exceptions import ValidationError, raise_validation_error
    from core.logging_config import get_logger, log_performance


@dataclass
class TimelinePosition:
    """Position information for a clip in the timeline."""

    beat_position: float
    beat_index: int
    beat_multiplier: int
    musical_time: float


@dataclass
class TimelineEntry:
    """Individual entry in the video timeline.

    Represents a single video clip with its positioning,
    timing, and quality information.
    """

    # Video information
    video_file: str
    start_time: float
    end_time: float
    duration: float

    # Timeline positioning
    position: TimelinePosition

    # Quality and scoring
    quality_score: float
    selection_reason: str = "quality"

    # Metadata
    created_at: float = None
    clip_id: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

        if self.clip_id is None:
            # Generate unique clip ID
            self.clip_id = f"{Path(self.video_file).stem}_{self.start_time:.2f}_{self.end_time:.2f}"

        # Validate timing
        if self.end_time <= self.start_time:
            raise_validation_error(
                f"Invalid clip timing: end_time ({self.end_time}) <= start_time ({self.start_time})",
                validation_type="clip_timing",
                file_path=self.video_file,
            )

        if abs(self.duration - (self.end_time - self.start_time)) > 0.01:
            raise_validation_error(
                f"Duration mismatch: calculated {self.end_time - self.start_time:.3f}s != stored {self.duration:.3f}s",
                validation_type="duration_consistency",
                file_path=self.video_file,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline entry to dictionary for JSON export."""
        return {
            "video_file": self.video_file,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "beat_position": self.position.beat_position,
            "beat_index": self.position.beat_index,
            "beat_multiplier": self.position.beat_multiplier,
            "musical_time": self.position.musical_time,
            "quality_score": self.quality_score,
            "selection_reason": self.selection_reason,
            "clip_id": self.clip_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineEntry":
        """Create timeline entry from dictionary."""
        position = TimelinePosition(
            beat_position=data["beat_position"],
            beat_index=data["beat_index"],
            beat_multiplier=data["beat_multiplier"],
            musical_time=data["musical_time"],
        )

        return cls(
            video_file=data["video_file"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration=data["duration"],
            position=position,
            quality_score=data["quality_score"],
            selection_reason=data.get("selection_reason", "quality"),
            created_at=data.get("created_at", time.time()),
            clip_id=data.get("clip_id"),
        )

    def __str__(self) -> str:
        return f"TimelineEntry({Path(self.video_file).name}, {self.duration:.1f}s @ {self.position.beat_position:.1f}s)"


class ClipTimeline:
    """Timeline manager for beat-synced video clips.

    Manages the assembly and organization of video clips according to
    musical beat timing. Provides validation, statistics, and export
    capabilities for timeline analysis.
    """

    def __init__(self, name: str = "AutoCut Timeline"):
        """Initialize timeline.

        Args:
            name: Human-readable name for this timeline
        """
        self.name = name
        self.entries: List[TimelineEntry] = []
        self.created_at = time.time()
        self.logger = get_logger("autocut.video.assembly.ClipTimeline")

        # Timeline metadata
        self._metadata = {
            "version": "2.0",
            "created_by": "AutoCut Assembly Engine",
            "created_at": self.created_at,
        }

    def add_entry(self, entry: TimelineEntry) -> None:
        """Add a timeline entry.

        Args:
            entry: Timeline entry to add

        Raises:
            ValidationError: If entry is invalid
        """
        # Validate entry doesn't conflict with existing entries
        for existing in self.entries:
            if existing.video_file == entry.video_file and self._entries_overlap(
                existing, entry
            ):
                raise_validation_error(
                    "Timeline entry conflicts with existing entry",
                    validation_type="timeline_conflict",
                    file_path=entry.video_file,
                )

        self.entries.append(entry)
        self.logger.debug(f"Added timeline entry: {entry}")

    def add_clip(
        self,
        video_file: str,
        start_time: float,
        end_time: float,
        beat_position: float,
        beat_index: int,
        beat_multiplier: int,
        quality_score: float,
        selection_reason: str = "quality",
    ) -> None:
        """Add a clip to the timeline (legacy interface).

        Args:
            video_file: Path to video file
            start_time: Start time in video
            end_time: End time in video
            beat_position: Position in musical timeline
            beat_index: Index of beat in beat array
            beat_multiplier: Number of beats this clip spans
            quality_score: Quality score of clip
            selection_reason: Reason for selection
        """
        position = TimelinePosition(
            beat_position=beat_position,
            beat_index=beat_index,
            beat_multiplier=beat_multiplier,
            musical_time=beat_position,
        )

        entry = TimelineEntry(
            video_file=video_file,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            position=position,
            quality_score=quality_score,
            selection_reason=selection_reason,
        )

        self.add_entry(entry)

    def remove_entry(self, clip_id: str) -> bool:
        """Remove timeline entry by clip ID.

        Args:
            clip_id: ID of clip to remove

        Returns:
            True if entry was removed, False if not found
        """
        for i, entry in enumerate(self.entries):
            if entry.clip_id == clip_id:
                removed_entry = self.entries.pop(i)
                self.logger.debug(f"Removed timeline entry: {removed_entry}")
                return True
        return False

    def clear(self) -> int:
        """Clear all timeline entries.

        Returns:
            Number of entries removed
        """
        count = len(self.entries)
        self.entries.clear()
        self.logger.info(f"Cleared timeline: removed {count} entries")
        return count

    def get_clips_sorted_by_beat(self) -> List[TimelineEntry]:
        """Get timeline entries sorted by beat position."""
        return sorted(self.entries, key=lambda x: x.position.beat_position)

    def get_clips_sorted_by_time(self) -> List[TimelineEntry]:
        """Get timeline entries sorted by musical time."""
        return sorted(self.entries, key=lambda x: x.position.musical_time)

    def get_unique_video_files(self) -> Set[str]:
        """Get set of unique video files used in timeline."""
        return {entry.video_file for entry in self.entries}

    def get_total_duration(self) -> float:
        """Get total duration of all clips in timeline."""
        return sum(entry.duration for entry in self.entries)

    def get_coverage_span(self) -> Tuple[float, float]:
        """Get the time span covered by timeline.

        Returns:
            Tuple of (start_time, end_time) in musical timeline
        """
        if not self.entries:
            return (0.0, 0.0)

        sorted_entries = self.get_clips_sorted_by_beat()
        start_time = sorted_entries[0].position.beat_position
        end_time = (
            sorted_entries[-1].position.beat_position + sorted_entries[-1].duration
        )

        return (start_time, end_time)

    @log_performance("timeline_statistics")
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive timeline statistics."""
        if not self.entries:
            return {
                "total_clips": 0,
                "total_duration": 0.0,
                "unique_videos": 0,
                "coverage_span": (0.0, 0.0),
                "avg_quality": 0.0,
                "quality_range": (0.0, 0.0),
                "duration_range": (0.0, 0.0),
                "beat_coverage": 0.0,
            }

        # Basic statistics
        durations = [entry.duration for entry in self.entries]
        quality_scores = [entry.quality_score for entry in self.entries]

        # Coverage analysis
        coverage_span = self.get_coverage_span()
        coverage_duration = coverage_span[1] - coverage_span[0]

        # Beat analysis
        beat_positions = [entry.position.beat_position for entry in self.entries]
        beat_multipliers = [entry.position.beat_multiplier for entry in self.entries]

        # Selection reason analysis
        selection_reasons = {}
        for entry in self.entries:
            reason = entry.selection_reason
            selection_reasons[reason] = selection_reasons.get(reason, 0) + 1

        return {
            "total_clips": len(self.entries),
            "total_duration": self.get_total_duration(),
            "unique_videos": len(self.get_unique_video_files()),
            "coverage_span": coverage_span,
            "coverage_duration": coverage_duration,
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "quality_range": (min(quality_scores), max(quality_scores)),
            "duration_range": (min(durations), max(durations)),
            "beat_coverage": {
                "total_beats": sum(beat_multipliers),
                "avg_beat_multiplier": sum(beat_multipliers) / len(beat_multipliers),
                "beat_range": (min(beat_positions), max(beat_positions)),
            },
            "selection_breakdown": selection_reasons,
            "timeline_metadata": {
                "name": self.name,
                "created_at": self.created_at,
                "last_modified": max(entry.created_at for entry in self.entries),
            },
        }

    @log_performance("timeline_validation")
    def validate_timeline(
        self,
        song_duration: Optional[float] = None,
        min_coverage: float = 0.8,
        min_quality: float = 50.0,
    ) -> Dict[str, Any]:
        """Validate timeline for common issues.

        Args:
            song_duration: Total song duration for coverage check
            min_coverage: Minimum coverage ratio required
            min_quality: Minimum average quality score required

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        if not self.entries:
            issues.append("Timeline is empty")
            return {
                "valid": False,
                "issues": issues,
                "warnings": warnings,
                "stats": self.get_summary_stats(),
            }

        # Get statistics for analysis
        stats = self.get_summary_stats()

        # Check for very short clips
        short_clips = [e for e in self.entries if e.duration < 0.5]
        if short_clips:
            warnings.append(f"{len(short_clips)} clips are very short (<0.5s)")

        # Check for very long clips
        long_clips = [e for e in self.entries if e.duration > 8.0]
        if long_clips:
            warnings.append(f"{len(long_clips)} clips are very long (>8s)")

        # Check quality distribution
        if stats["avg_quality"] < min_quality:
            warnings.append(f"Average clip quality is low: {stats['avg_quality']:.1f}")

        # Check video variety
        if len(self.entries) > 5 and stats["unique_videos"] == 1:
            warnings.append("All clips are from the same video - low variety")

        # Check timeline coverage if song duration provided
        if song_duration and song_duration > 0:
            coverage_ratio = stats["coverage_duration"] / song_duration
            if coverage_ratio < min_coverage:
                warnings.append(
                    f"Timeline covers only {coverage_ratio * 100:.1f}% of song"
                )

        # Check for overlapping clips from same video
        overlaps = self._find_overlapping_clips()
        if overlaps:
            issues.append(f"Found {len(overlaps)} overlapping clips from same video")

        # Check beat consistency
        sorted_entries = self.get_clips_sorted_by_beat()
        beat_gaps = []
        for i in range(1, len(sorted_entries)):
            current_end = (
                sorted_entries[i - 1].position.beat_position
                + sorted_entries[i - 1].duration
            )
            next_start = sorted_entries[i].position.beat_position
            gap = next_start - current_end
            beat_gaps.append(gap)

        if beat_gaps:
            large_gaps = [g for g in beat_gaps if g > 2.0]
            if large_gaps:
                warnings.append(
                    f"Found {len(large_gaps)} large gaps between clips (>2s)"
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": stats,
            "validation_details": {
                "overlapping_clips": len(overlaps),
                "large_gaps": len(large_gaps) if beat_gaps else 0,
                "coverage_ratio": stats["coverage_duration"] / song_duration
                if song_duration
                else None,
            },
        }

    def export_json(self, file_path: str, include_metadata: bool = True) -> None:
        """Export timeline as JSON for debugging and analysis.

        Args:
            file_path: Output file path
            include_metadata: Whether to include timeline metadata
        """
        export_data = {
            "timeline": {
                "name": self.name,
                "entries": [entry.to_dict() for entry in self.entries],
                "stats": self.get_summary_stats(),
            },
        }

        if include_metadata:
            export_data["metadata"] = self._metadata
            export_data["metadata"]["exported_at"] = time.time()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported timeline to {file_path}")

    def import_json(self, file_path: str) -> None:
        """Import timeline from JSON file.

        Args:
            file_path: Input file path
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        timeline_data = data.get("timeline", {})
        self.name = timeline_data.get("name", "Imported Timeline")

        # Clear existing entries
        self.clear()

        # Import entries
        for entry_data in timeline_data.get("entries", []):
            entry = TimelineEntry.from_dict(entry_data)
            self.add_entry(entry)

        self.logger.info(
            f"Imported timeline from {file_path}: {len(self.entries)} entries"
        )

    def _entries_overlap(
        self, entry1: TimelineEntry, entry2: TimelineEntry, min_gap: float = 1.0
    ) -> bool:
        """Check if two timeline entries overlap.

        Args:
            entry1: First timeline entry
            entry2: Second timeline entry
            min_gap: Minimum gap required between clips

        Returns:
            True if entries overlap or are too close
        """
        return (
            entry1.start_time < entry2.end_time + min_gap
            and entry1.end_time > entry2.start_time - min_gap
        )

    def _find_overlapping_clips(self) -> List[Tuple[TimelineEntry, TimelineEntry]]:
        """Find all pairs of overlapping clips from the same video.

        Returns:
            List of tuples containing overlapping entry pairs
        """
        overlaps = []

        for i, entry1 in enumerate(self.entries):
            for entry2 in self.entries[i + 1 :]:
                if entry1.video_file == entry2.video_file and self._entries_overlap(
                    entry1, entry2
                ):
                    overlaps.append((entry1, entry2))

        return overlaps

    @property
    def clips(self):
        """Compatibility property for video.rendering system.

        The rendering system expects timeline.clips but ClipTimeline uses entries.
        This property provides interface compatibility without breaking existing code.
        """
        return self.entries

    @clips.setter
    def clips(self, value):
        """Allow rendering system to set clips (maps to entries)."""
        self.entries = value

    def __len__(self) -> int:
        """Get number of entries in timeline."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over timeline entries."""
        return iter(self.entries)

    def __str__(self) -> str:
        stats = self.get_summary_stats()
        return (
            f"ClipTimeline('{self.name}', {stats['total_clips']} clips, "
            f"{stats['total_duration']:.1f}s, {stats['unique_videos']} videos)"
        )


# Convenience functions for timeline creation
def create_empty_timeline(name: str = "AutoCut Timeline") -> ClipTimeline:
    """Create an empty timeline with specified name."""
    return ClipTimeline(name=name)


def merge_timelines(
    timelines: List[ClipTimeline], name: str = "Merged Timeline"
) -> ClipTimeline:
    """Merge multiple timelines into one.

    Args:
        timelines: List of timelines to merge
        name: Name for merged timeline

    Returns:
        New timeline containing all entries
    """
    merged = ClipTimeline(name=name)

    def _safe_add_entry(merged_timeline, entry):
        """Safely add entry to merged timeline, logging conflicts."""
        try:
            merged_timeline.add_entry(entry)
            return True
        except ValidationError:
            # Skip conflicting entries
            merged_timeline.logger.warning(f"Skipped conflicting entry: {entry}")
            return False

    for timeline in timelines:
        for entry in timeline.entries:
            _safe_add_entry(merged, entry)

    return merged
