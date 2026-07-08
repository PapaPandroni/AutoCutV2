"""Beat-to-clip matching and quality-based clip selection for AutoCut.

Extracted from clip_assembler.py during the codebase streamline. This is pure
timeline-planning logic: it maps scored :class:`VideoChunk` objects onto a beat
grid using variety patterns, with no video loading or rendering.
"""

from typing import List, Tuple

try:
    from video import VideoChunk
except ImportError:
    from .video import VideoChunk

try:
    from video.timeline_renderer import ClipTimeline
except ImportError:
    from .video.timeline_renderer import ClipTimeline


VARIETY_PATTERNS = {
    "energetic": [2, 2, 4, 2, 2, 8],  # Fast 2-beat cuts with occasional longer pause
    "buildup": [8, 4, 4, 4, 4, 4],  # Start slow, maintain deliberate 4-beat pace
    "balanced": [4, 4, 4, 8, 4, 4],  # Consistent 4-beat pacing with variety
    "dramatic": [4, 4, 4, 4, 16],  # Build tension with 4-beat base, long dramatic hold
}


def match_clips_to_beats(
    video_chunks: List[VideoChunk],
    beats: List[float],
    allowed_durations: List[float],
    pattern: str = "balanced",
    musical_start_time: float = 0.0,
) -> ClipTimeline:
    """Match video chunks to beat grid using variety patterns with musical intelligence.

    Args:
        video_chunks: List of scored video chunks
        beats: List of beat timestamps in seconds (compensated and filtered)
        allowed_durations: List of musically appropriate durations
        pattern: Variety pattern to use ('energetic', 'buildup', 'balanced', 'dramatic')
        musical_start_time: First significant beat timestamp (skip intro/buildup)

    Returns:
        ClipTimeline object with matched clips starting from musical content
    """
    if not video_chunks or not beats or len(beats) < 2:
        return ClipTimeline()

    # MUSICAL INTELLIGENCE: Filter beats to start from actual musical content
    # This fixes the 1-2 second intro misalignment issue
    effective_beats = (
        [b for b in beats if b >= musical_start_time]
        if musical_start_time > 0
        else beats
    )

    if len(effective_beats) < 2:
        # Fallback to all beats if musical start filtering leaves too few
        effective_beats = beats

    # Calculate beat interval (average time between beats)
    beat_intervals = [
        effective_beats[i + 1] - effective_beats[i]
        for i in range(len(effective_beats) - 1)
    ]
    avg_beat_interval = sum(beat_intervals) / len(beat_intervals)

    # Apply variety pattern to get beat multipliers
    total_beats = (
        len(effective_beats) - 1
    )  # Don't count the last beat as start of a clip
    beat_multipliers = apply_variety_pattern(pattern, total_beats)

    # Convert beat multipliers to target durations
    target_durations = [
        multiplier * avg_beat_interval for multiplier in beat_multipliers
    ]

    # Estimate total clips needed
    estimated_clips = len(target_durations)

    # Select best clips with variety (request more than needed for flexibility)
    selected_clips = select_best_clips(
        video_chunks,
        target_count=min(estimated_clips * 2, len(video_chunks)),
        variety_factor=0.3,
    )

    timeline = ClipTimeline()
    current_beat_index = 0
    used_clips = set()  # Track used clips to avoid repetition

    for i, target_duration in enumerate(target_durations):
        if current_beat_index >= len(effective_beats):
            break

        # Find best matching clip for this target duration
        best_clip = None
        best_fit_score = -1

        for clip in selected_clips:
            if id(clip) in used_clips:
                continue

            # Calculate fit score based on:
            # 1. How close clip duration is to target duration
            # 2. Clip quality score
            # 3. Whether clip can be trimmed to fit exactly

            duration_fit = _calculate_duration_fit(
                clip.duration,
                target_duration,
                allowed_durations,
            )
            if duration_fit < 0:  # Clip can't be used for this duration
                continue

            # Combined score: 70% quality, 30% duration fit
            fit_score = 0.7 * (clip.score / 100.0) + 0.3 * duration_fit

            if fit_score > best_fit_score:
                best_fit_score = fit_score
                best_clip = clip

        if best_clip is None:
            # No suitable clip found, skip this position
            current_beat_index += beat_multipliers[i]
            continue

        # Mark clip as used
        used_clips.add(id(best_clip))

        # Determine actual clip timing
        beat_position = effective_beats[current_beat_index]
        clip_start, clip_end, clip_duration = _fit_clip_to_duration(
            best_clip,
            target_duration,
        )

        # Add to timeline
        timeline.add_clip(
            video_file=best_clip.video_path,
            start=clip_start,
            end=clip_end,
            beat_position=beat_position,
            score=best_clip.score,
        )

        # Move to next beat position
        current_beat_index += beat_multipliers[i]

    return timeline


def _calculate_duration_fit(
    clip_duration: float,
    target_duration: float,
    allowed_durations: List[float],
) -> float:
    """Calculate how well a clip duration fits the target duration.

    Args:
        clip_duration: Duration of the video clip
        target_duration: Desired duration for this position
        allowed_durations: List of musically appropriate durations

    Returns:
        Fit score between 0.0 and 1.0, or -1 if clip can't be used
    """
    # Check if target duration is in allowed durations. Target durations come from
    # multiplier * avg_beat_interval while allowed come from multiplier * (60/bpm),
    # so acceptable drift grows with the multiplier. A fixed 0.1s tolerance wrongly
    # rejected long holds (e.g. the 16-beat "dramatic" cut) on tiny beat drift.
    duration_allowed = False
    for allowed in allowed_durations:
        if abs(target_duration - allowed) <= max(0.1, 0.04 * allowed):
            duration_allowed = True
            break

    if not duration_allowed:
        return -1  # Target duration is not musically appropriate

    # Perfect match
    if abs(clip_duration - target_duration) < 0.1:
        return 1.0

    # Clip is longer than target - can be trimmed
    if clip_duration > target_duration:
        # Prefer clips that are close to target but slightly longer
        excess = clip_duration - target_duration
        if excess <= 2.0:  # Can trim up to 2 seconds
            return 1.0 - (excess / 4.0)  # Gentle penalty for trimming
        return 0.3  # Heavy penalty for lots of trimming

    # Clip is shorter than target
    shortage = target_duration - clip_duration
    if shortage <= 0.5:  # Small shortage is acceptable
        return 0.8 - (shortage / 1.0)
    return -1  # Too short, can't use


def _fit_clip_to_duration(
    clip: VideoChunk,
    target_duration: float,
) -> Tuple[float, float, float]:
    """Fit a clip to the target duration by trimming if necessary.

    Args:
        clip: Video chunk to fit
        target_duration: Desired duration

    Returns:
        Tuple of (start_time, end_time, actual_duration)
    """
    if clip.duration <= target_duration + 0.1:
        # Clip fits as-is
        return clip.start_time, clip.end_time, clip.duration

    # Clip needs trimming - trim from the end to preserve the beginning
    new_end_time = clip.start_time + target_duration

    # Make sure we don't exceed the original clip bounds
    new_end_time = min(new_end_time, clip.end_time)
    actual_duration = new_end_time - clip.start_time

    return clip.start_time, new_end_time, actual_duration


def select_best_clips(
    video_chunks: List[VideoChunk],
    target_count: int,
    variety_factor: float = 0.3,
) -> List[VideoChunk]:
    """Select best clips ensuring variety in source videos.

    Args:
        video_chunks: List of all available video chunks
        target_count: Number of clips to select
        variety_factor: Weight for variety vs. quality (0.0 = only quality, 1.0 = only variety)

    Returns:
        List of selected VideoChunk objects
    """
    if not video_chunks:
        return []

    if target_count <= 0:
        return []

    if len(video_chunks) <= target_count:
        return video_chunks.copy()

    # Group clips by video file for variety management
    clips_by_video = {}
    for chunk in video_chunks:
        if chunk.video_path not in clips_by_video:
            clips_by_video[chunk.video_path] = []
        clips_by_video[chunk.video_path].append(chunk)

    # Sort clips within each video by score (descending)
    for video_path in clips_by_video:
        clips_by_video[video_path].sort(key=lambda x: x.score, reverse=True)

    selected_clips = []

    if variety_factor >= 0.9:
        # High variety: Round-robin selection from each video
        video_paths = list(clips_by_video.keys())
        video_index = 0

        while len(selected_clips) < target_count:
            video_path = video_paths[video_index % len(video_paths)]

            # Find next non-overlapping clip from this video
            available_clips = clips_by_video[video_path]
            for clip in available_clips:
                if clip not in selected_clips and not _clips_overlap(
                    clip,
                    selected_clips,
                ):
                    selected_clips.append(clip)
                    break

            video_index += 1

            # Safety check: if we've tried all videos and can't find more clips
            if video_index > len(video_paths) * 10:
                break

    elif variety_factor <= 0.1:
        # High quality: Just take the best clips regardless of source
        all_clips_sorted = sorted(video_chunks, key=lambda x: x.score, reverse=True)
        for clip in all_clips_sorted:
            if len(selected_clips) >= target_count:
                break
            if not _clips_overlap(clip, selected_clips):
                selected_clips.append(clip)

    else:
        # Balanced approach: Weighted selection
        # Calculate how many clips per video (with some variety)
        num_videos = len(clips_by_video)
        base_clips_per_video = max(1, target_count // num_videos)
        remaining_clips = target_count - (base_clips_per_video * num_videos)

        # First pass: Get base clips from each video (highest quality)
        for video_path in clips_by_video:
            clips_from_video = 0
            for clip in clips_by_video[video_path]:
                if clips_from_video >= base_clips_per_video:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)
                    clips_from_video += 1

        # Second pass: Fill remaining slots with highest quality clips
        if remaining_clips > 0:
            all_remaining_clips = [
                clip
                for video_path in clips_by_video
                for clip in clips_by_video[video_path][base_clips_per_video:]
                if clip not in selected_clips
            ]

            all_remaining_clips.sort(key=lambda x: x.score, reverse=True)

            for clip in all_remaining_clips:
                if len(selected_clips) >= target_count:
                    break
                if not _clips_overlap(clip, selected_clips):
                    selected_clips.append(clip)

    return selected_clips[:target_count]


def _clips_overlap(
    clip: VideoChunk,
    existing_clips: List[VideoChunk],
    min_gap: float = 1.0,
) -> bool:
    """Check if a clip overlaps with any existing clips from the same video.

    Args:
        clip: Clip to check
        existing_clips: List of already selected clips
        min_gap: Minimum gap required between clips from same video (seconds)

    Returns:
        True if clip overlaps with any existing clip from same video
    """
    for existing in existing_clips:
        # Check for overlap or too close proximity from same video
        if (
            existing.video_path == clip.video_path
            and clip.start_time < existing.end_time + min_gap
            and clip.end_time > existing.start_time - min_gap
        ):
            return True
    return False


def apply_variety_pattern(pattern_name: str, beat_count: int) -> List[int]:
    """Apply variety pattern to determine clip lengths.

    Args:
        pattern_name: Name of variety pattern to use
        beat_count: Total number of beats to fill

    Returns:
        List of beat multipliers for each clip
    """
    if pattern_name not in VARIETY_PATTERNS:
        pattern_name = "balanced"

    pattern = VARIETY_PATTERNS[pattern_name]
    result = []
    pattern_index = 0
    remaining_beats = beat_count

    while remaining_beats > 0:
        multiplier = pattern[pattern_index % len(pattern)]
        if multiplier <= remaining_beats:
            result.append(multiplier)
            remaining_beats -= multiplier
        else:
            result.append(remaining_beats)
            remaining_beats = 0
        pattern_index += 1

    return result
