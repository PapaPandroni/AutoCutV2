"""Beat-to-clip matching and quality-based clip selection for AutoCut.

Extracted from clip_assembler.py during the codebase streamline. This is pure
timeline-planning logic: it maps scored :class:`VideoChunk` objects onto a beat
grid using variety patterns, with no video loading or rendering.
"""

import logging
from typing import List, Optional, Tuple

try:
    from video import VideoChunk
except ImportError:
    from .video import VideoChunk

try:
    from video.timeline_renderer import ClipTimeline
except ImportError:
    from .video.timeline_renderer import ClipTimeline

logger = logging.getLogger("autocut.beat_matching")


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
    downbeat_times: Optional[List[float]] = None,
) -> ClipTimeline:
    """Match video chunks to beat grid using variety patterns with musical intelligence.

    Args:
        video_chunks: List of scored video chunks
        beats: List of beat timestamps in seconds (compensated and filtered)
        allowed_durations: List of musically appropriate durations
        pattern: Variety pattern to use ('energetic', 'buildup', 'balanced', 'dramatic')
        musical_start_time: First significant beat timestamp (skip intro/buildup)
        downbeat_times: Timestamps of estimated downbeats (D6). Starting the
            pattern on the first downbeat found in the (filtered) beat grid
            means every 4/8/16-beat cut lands on a downbeat (all multiples of
            a 4/4 bar); 2-beat cuts may still land mid-bar by design.
            Timestamps rather than a phase index, because ``beats`` has
            usually been trimmed (weak-intro filter, musical_start) since
            analysis and an index into the original grid would point at the
            wrong beat here.

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

    # Calculate beat interval (average time between beats). Only used as a
    # fallback for the tail past the last detected beat (D3) and for the
    # audio fade calculation in rendering.py -- per-segment targets below use
    # the actual beat gap instead.
    beat_intervals = [
        effective_beats[i + 1] - effective_beats[i]
        for i in range(len(effective_beats) - 1)
    ]
    avg_beat_interval = sum(beat_intervals) / len(beat_intervals)

    # D6: start the pattern on the estimated downbeat so 4/8/16-beat cuts
    # (all multiples of a 4/4 bar) land on downbeats, not an arbitrary parity.
    # Resolve the downbeat timestamps against the (possibly trimmed) grid:
    # the first effective beat lying on a downbeat becomes the start index.
    downbeat_offset = _resolve_downbeat_start(
        effective_beats,
        downbeat_times,
        avg_beat_interval,
    )

    # Apply variety pattern to get beat multipliers
    total_beats = (
        len(effective_beats) - 1 - downbeat_offset
    )  # Don't count the last beat as start of a clip
    beat_multipliers = apply_variety_pattern(pattern, total_beats)

    # Estimate total clips needed
    estimated_clips = len(beat_multipliers)

    # Select best clips with variety (request more than needed for flexibility)
    selected_clips = select_best_clips(
        video_chunks,
        target_count=min(estimated_clips * 2, len(video_chunks)),
        variety_factor=0.3,
    )

    timeline = ClipTimeline()
    current_beat_index = downbeat_offset
    used_clips = set()  # Track used clips to avoid repetition
    # Total beat-grid span of slots dropped for lack of footage (D4 step 3).
    # Clips are concatenated with no holes, so after a drop the whole rest of
    # the output plays that much earlier against the music; later targets and
    # beat_position labels must subtract it or they'd chase (and report
    # against) grid positions the output can no longer reach.
    dropped_span = 0.0
    # Bookkeeping for D4's "extend the previous clip" fallback: the source
    # chunk, actual output-time start, and intended grid label of the most
    # recently committed clip (None until the first clip is actually added).
    last_clip: Optional[VideoChunk] = None
    last_anchor = 0.0
    last_intended = 0.0

    def _segment_target(anchor: float, multiplier: int) -> Tuple[float, int]:
        """Target duration for a slot starting at ``anchor`` (an absolute
        output-time position) and spanning ``multiplier`` beats from
        ``current_beat_index`` (D3: the actual beat gap, not an average).
        Falls back to the song-wide average only past the last detected beat.
        """
        end_index = current_beat_index + multiplier
        if end_index < len(effective_beats):
            return effective_beats[end_index] - dropped_span - anchor, end_index
        return multiplier * avg_beat_interval, end_index

    def _find_best_clip(pool, target_duration: float, ignore_fit_penalty=False):
        best_clip, best_score = None, -1.0
        for clip in pool:
            if id(clip) in used_clips:
                continue
            if ignore_fit_penalty:
                # D4 last-resort: any long-enough, unused clip is acceptable
                # (a visible longer clip beats a permanently shifted grid).
                if clip.duration < target_duration:
                    continue
                score = clip.score / 100.0
            else:
                duration_fit = _calculate_duration_fit(
                    clip.duration,
                    target_duration,
                    allowed_durations,
                )
                if duration_fit < 0:  # Clip can't be used for this duration
                    continue
                # Combined score: 70% quality, 30% duration fit
                score = 0.7 * (clip.score / 100.0) + 0.3 * duration_fit
            if score > best_score:
                best_score = score
                best_clip = clip
        return best_clip

    def _extend_last_clip(new_target_duration: float) -> None:
        """Pop the most recently committed clip and re-add it trimmed to
        ``new_target_duration`` from the same source chunk/anchor (D4: swallow
        an unfillable slot into the previous one instead of leaving a hole)."""
        popped = timeline.clips.pop()
        timeline._cumulative_start -= popped["duration"]
        start, end, _duration = _fit_clip_to_duration(last_clip, new_target_duration)
        timeline.add_clip(
            video_file=last_clip.video_path,
            start=start,
            end=end,
            beat_position=last_intended,
            score=last_clip.score,
        )

    for slot_index, multiplier in enumerate(beat_multipliers):
        if current_beat_index >= len(effective_beats):
            break

        # D1: the very first slot spans from absolute output-time 0 (the video
        # absorbs the intro); the widened-pool retry below is a one-time
        # attempt tied to loop position, not "nothing committed yet" --
        # retrying it on every slot until one finally commits would make the
        # target grow without bound if the first slot(s) get dropped for lack
        # of long-enough footage.
        #
        # Every slot's target is measured from the *actual* end of the
        # assembled timeline (which is where its cut really lands in the
        # output), not from the theoretical grid position: if an earlier clip
        # came up short despite the fit checks (e.g. the D4 extend fallback
        # clamps to the source chunk's end), measuring from the grid would
        # carry that shortfall into every later cut, while measuring from the
        # timeline end lets the very next clip absorb it and re-sync.
        is_first_slot = slot_index == 0
        anchor = timeline.get_total_duration()
        intended_position = (
            0.0 if is_first_slot else effective_beats[current_beat_index] - dropped_span
        )
        target_duration, next_index = _segment_target(anchor, multiplier)

        best_clip = _find_best_clip(selected_clips, target_duration)
        if best_clip is None and is_first_slot:
            # The anchored first slot needs extra footage to absorb the
            # intro -- the pre-filtered pool may lack a long-enough
            # candidate even though the full chunk list has one.
            best_clip = _find_best_clip(video_chunks, target_duration)

        if best_clip is None:
            # D4, step 1: extend the previous clip to swallow this slot too,
            # if its source chunk has the footage for the combined span.
            if last_clip is not None:
                combined_duration, combined_next_index = _segment_target(
                    last_anchor,
                    multiplier,
                )
                if last_clip.duration >= combined_duration:
                    _extend_last_clip(combined_duration)
                    current_beat_index = combined_next_index
                    continue

            # D4, step 2: fall back to any long-enough clip, ignoring the
            # fit-score rejection (a visible longer clip beats a hole).
            best_clip = _find_best_clip(
                video_chunks,
                target_duration,
                ignore_fit_penalty=True,
            )

        if best_clip is None:
            # D4, step 3: nothing long enough exists -- drop the slot and
            # record the output-time span it was going to fill (its target,
            # which for the first slot includes the intro), so later targets
            # aim for the beats as the output will actually hear them
            # (shifted earlier by the drop) rather than chasing unreachable
            # absolute grid positions.
            dropped_span += target_duration
            current_beat_index = next_index
            continue

        # Mark clip as used
        used_clips.add(id(best_clip))

        clip_start, clip_end, _clip_duration = _fit_clip_to_duration(
            best_clip,
            target_duration,
        )

        # Add to timeline
        timeline.add_clip(
            video_file=best_clip.video_path,
            start=clip_start,
            end=clip_end,
            beat_position=intended_position,
            score=best_clip.score,
        )
        last_clip, last_anchor, last_intended = best_clip, anchor, intended_position

        # Move to next beat position
        current_beat_index = next_index

    report = timeline.get_alignment_report()
    # >50ms is at the edge of audibility for an off-beat cut -- surface it.
    log = logger.warning if report["max_abs_delta"] > 0.05 else logger.info
    log(
        f"Beat alignment: max |delta|={report['max_abs_delta']:.3f}s, "
        f"mean |delta|={report['mean_abs_delta']:.3f}s over {len(timeline.clips)} clips"
    )

    return timeline


def _resolve_downbeat_start(
    effective_beats: List[float],
    downbeat_times: Optional[List[float]],
    avg_beat_interval: float,
) -> int:
    """Index of the first beat in ``effective_beats`` sitting on an estimated
    downbeat (within a quarter beat), or 0 if none matches.

    ``downbeat_times`` come from the full analysis-time grid while
    ``effective_beats`` has typically lost beats to the weak-intro filter and
    the musical_start trim, so matching must be by timestamp -- an index
    carried over from analysis would be rotated by every dropped beat.
    """
    if not downbeat_times:
        return 0
    tolerance = avg_beat_interval / 4
    for index, beat in enumerate(effective_beats):
        if any(abs(beat - downbeat) <= tolerance for downbeat in downbeat_times):
            return index
    return 0


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
    # allowed_durations is kept in the signature for call-site compatibility,
    # but no longer used to reject target_duration: earlier versions rejected
    # anything outside tolerance of it because targets came from
    # multiplier * avg_beat_interval, a global estimate that could drift away
    # from anything musically sensible. Targets are now the actual
    # beat-to-beat gap for a real pattern multiplier (D3) and so are musically
    # appropriate by construction.

    # Clip is shorter than target: unusable. A clip that comes up short can't
    # be trimmed to fit -- accepting it leaves that shortfall as permanent
    # drift in every later cut (D2). This must be checked before any
    # "close enough" tolerance: scene detection samples on whole seconds, so
    # chunk durations are integers, and at e.g. 117 BPM a 4-beat target is
    # 2.04s -- a 2.00s chunk is 40ms short *every time*, which used to pass
    # the old symmetric <0.1 "perfect match" branch and accumulate into
    # audible off-beat cuts.
    if clip_duration < target_duration:
        return -1

    # Essentially exact -- nothing (or almost nothing) to trim away.
    excess = clip_duration - target_duration
    if excess < 0.1:
        return 1.0

    # Clip is longer than target - can be trimmed exactly (D2: trimming to
    # the exact target never loses precision, so only the trim's *size* -- not
    # whether it happens -- affects the score).
    if excess <= 2.0:  # Can trim up to 2 seconds
        return 1.0 - (excess / 4.0)  # Gentle penalty for trimming
    return 0.3  # Heavy penalty for lots of trimming


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
    if clip.duration <= target_duration:
        # Clip is already exactly (or, in the D4 extend-previous fallback,
        # as close as possible to) the target -- no +/- slack here, since any
        # slack becomes permanent drift once clips are concatenated (D2).
        return clip.start_time, clip.end_time, clip.duration

    # Clip needs trimming - trim from the end to preserve the beginning,
    # landing on exactly target_duration.
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
