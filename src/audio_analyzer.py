"""
Audio Analysis Module for AutoCut

Handles music analysis including BPM detection, beat tracking, and
calculation of musically appropriate clip durations.
"""

import contextlib
import os
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np


def detect_musical_start(
    y: np.ndarray, sr: int, tempo: float, beats: np.ndarray,
) -> Tuple[float, float]:
    """Detect the start of significant musical content using onset detection and energy analysis.

    Args:
        y: Audio time series
        sr: Sample rate
        tempo: Detected BPM
        beats: Beat frame indices

    Returns:
        Tuple of (musical_start_time, intro_duration)
    """
    # Calculate energy-based onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        pre_max=20,  # Look ahead 20 frames
        post_max=20,  # Look back 20 frames
        pre_avg=100,  # Average over 100 frames before
        post_avg=100,  # Average over 100 frames after
        delta=0.07,  # Minimum threshold for onset
        wait=15,  # Minimum frames between onsets
    )

    if len(onset_frames) == 0:
        return 0.0, 0.0

    # Convert to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate RMS energy over time (hop_length frames)
    hop_length = 512
    frame_length = 2048
    rms_energy = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]

    # Convert RMS frame indices to time
    rms_times = librosa.frames_to_time(
        np.arange(len(rms_energy)), sr=sr, hop_length=hop_length,
    )

    # Find significant energy increase (musical content start)
    # Use 70th percentile of energy as "significant" threshold
    energy_threshold = np.percentile(rms_energy, 70)

    # Find first sustained period of high energy
    sustained_duration = 60.0 / tempo * 2  # 2 beats worth of sustain

    musical_start_time = 0.0
    for i, energy in enumerate(rms_energy):
        if energy >= energy_threshold:
            current_time = rms_times[i]

            # Check if energy stays high for sustained_duration
            end_idx = min(
                i + int(sustained_duration * sr / hop_length), len(rms_energy),
            )

            if np.mean(rms_energy[i:end_idx]) >= energy_threshold * 0.8:
                musical_start_time = current_time
                break

    # Combine onset and energy analysis
    # Use the first significant onset that's close to energy start
    for onset_time in onset_times:
        if abs(onset_time - musical_start_time) <= 1.0:  # Within 1 second
            musical_start_time = onset_time
            break

    intro_duration = musical_start_time

    return musical_start_time, intro_duration


def detect_intro_duration(
    y: np.ndarray,
    sr: int,
    tempo: float,
    energy_threshold: float = 0.3,
    min_intro: float = 0.5,
    max_intro: float = 8.0,
) -> float:
    """Detect intro/buildup duration using configurable energy thresholds.

    Args:
        y: Audio time series
        sr: Sample rate
        tempo: Detected BPM
        energy_threshold: Relative energy threshold (0.0-1.0)
        min_intro: Minimum intro duration in seconds
        max_intro: Maximum intro duration in seconds

    Returns:
        Intro duration in seconds
    """
    # Calculate spectral centroid (brightness) over time
    hop_length = 512
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Calculate RMS energy
    rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Calculate chroma (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma_energy = np.sum(chroma, axis=0)

    # Time axis for features
    times = librosa.frames_to_time(
        np.arange(len(rms_energy)), sr=sr, hop_length=hop_length,
    )

    # Normalize features
    centroid_norm = (centroid - np.min(centroid)) / (
        np.max(centroid) - np.min(centroid) + 1e-8
    )
    rms_norm = (rms_energy - np.min(rms_energy)) / (
        np.max(rms_energy) - np.min(rms_energy) + 1e-8
    )
    chroma_norm = (chroma_energy - np.min(chroma_energy)) / (
        np.max(chroma_energy) - np.min(chroma_energy) + 1e-8
    )

    # Combined musical complexity score
    complexity_score = rms_norm * 0.4 + centroid_norm * 0.3 + chroma_norm * 0.3

    # Find where complexity exceeds threshold consistently
    threshold = energy_threshold
    beat_duration = 60.0 / tempo
    min_sustain_frames = int((beat_duration * 4) * sr / hop_length)  # 4 beats

    intro_end_time = min_intro

    for i in range(len(complexity_score)):
        if complexity_score[i] >= threshold:
            # Check for sustained complexity
            end_idx = min(i + min_sustain_frames, len(complexity_score))
            if np.mean(complexity_score[i:end_idx]) >= threshold * 0.8:
                intro_end_time = times[i]
                break

    # Clamp to reasonable bounds
    return max(min_intro, min(intro_end_time, max_intro))



def create_beat_hierarchy(
    beats: np.ndarray, tempo: float, sr: int,
) -> Dict[str, List[float]]:
    """Create hierarchical beat structure with downbeats, half-beats, and measures.

    Args:
        beats: Beat frame indices
        tempo: BPM
        sr: Sample rate

    Returns:
        Dictionary with beat hierarchy
    """
    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_duration = 60.0 / tempo

    # Generate half-beats (between main beats)
    half_beats = []
    for i in range(len(beat_times) - 1):
        half_beat_time = beat_times[i] + (beat_times[i + 1] - beat_times[i]) / 2
        half_beats.append(half_beat_time)

    # Estimate time signature (assume 4/4 for most popular music)
    # Downbeats occur every 4 beats
    downbeats = [beat_times[i] for i in range(0, len(beat_times), 4)]

    # Measures (bars) - same as downbeats for 4/4 time
    measures = downbeats.copy()

    # Quarter note subdivisions (double-time)
    quarter_notes = []
    for i in range(len(beat_times) - 1):
        beat_gap = beat_times[i + 1] - beat_times[i]
        quarter_notes.append(beat_times[i] + beat_gap * 0.25)
        quarter_notes.append(beat_times[i] + beat_gap * 0.5)
        quarter_notes.append(beat_times[i] + beat_gap * 0.75)

    return {
        "main_beats": beat_times.tolist(),
        "half_beats": half_beats,
        "downbeats": downbeats,
        "measures": measures,
        "quarter_notes": quarter_notes,
    }


def apply_offset_compensation(beats: List[float], offset: float = -0.04) -> List[float]:
    """Apply systematic offset compensation for librosa timing latency.

    Args:
        beats: List of beat timestamps
        offset: Offset in seconds (negative to shift earlier)

    Returns:
        Compensated beat timestamps
    """
    return [max(0.0, beat + offset) for beat in beats]


def filter_weak_beats_in_intro(
    beats: List[float],
    y: np.ndarray,
    sr: int,
    intro_duration: float,
    strength_threshold: float = 0.3,
) -> List[float]:
    """Filter out weak beats during intro sections using energy analysis.

    Args:
        beats: Beat timestamps
        y: Audio time series
        sr: Sample rate
        intro_duration: Duration of intro section
        strength_threshold: Minimum beat strength (0.0-1.0)

    Returns:
        Filtered beat list with weak intro beats removed
    """
    if intro_duration <= 0:
        return beats

    # Calculate beat strength using onset strength
    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_envelope)), sr=sr, hop_length=hop_length,
    )

    filtered_beats = []

    for beat_time in beats:
        if beat_time > intro_duration:
            # After intro, keep all beats
            filtered_beats.append(beat_time)
        else:
            # During intro, check beat strength
            # Find closest onset envelope value
            closest_idx = np.argmin(np.abs(onset_times - beat_time))
            beat_strength = onset_envelope[closest_idx]

            # Normalize strength (0-1 scale based on max in song)
            max_strength = np.max(onset_envelope)
            normalized_strength = (
                beat_strength / max_strength if max_strength > 0 else 0
            )

            if normalized_strength >= strength_threshold:
                filtered_beats.append(beat_time)

    return filtered_beats


def analyze_audio(file_path: str) -> Dict[str, Union[float, List[float]]]:
    """Analyze audio file and extract comprehensive tempo and beat information.

    This enhanced version provides musical intelligence including intro detection,
    beat hierarchy, and offset compensation for professional synchronization.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary containing:
        - 'bpm': Detected beats per minute (float)
        - 'beats': Original beat timestamps in seconds (List[float])
        - 'compensated_beats': Offset-corrected beat timestamps (List[float])
        - 'musical_start_time': First significant beat timestamp (float)
        - 'intro_duration': Length of intro section in seconds (float)
        - 'beat_hierarchy': Hierarchical beat structure (Dict)
        - 'duration': Total audio duration in seconds (float)
        - 'allowed_durations': Musically appropriate clip durations (List[float])
        - 'min_duration': Minimum clip duration (float)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If BPM is outside valid range (30-300)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Load audio file with librosa
        y, sr = librosa.load(file_path)

        # Get audio duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Separate harmonic and percussive components for better beat detection
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Detect tempo and beats using the percussive component
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Convert frame indices to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # === ENHANCED MUSICAL INTELLIGENCE ===

        # 1. Detect musical start and intro duration
        musical_start_time, intro_duration = detect_musical_start(
            y, sr, tempo, beat_frames,
        )

        # 2. Apply systematic offset compensation for librosa latency
        compensated_beats = apply_offset_compensation(beat_times, offset=-0.04)

        # 3. Filter weak beats during intro sections
        filtered_beats = filter_weak_beats_in_intro(
            compensated_beats, y, sr, intro_duration, strength_threshold=0.3,
        )

        # 4. Create beat hierarchy structure
        beat_hierarchy = create_beat_hierarchy(beat_frames, tempo, sr)

        # 5. Enhanced intro detection with configurable thresholds
        refined_intro_duration = detect_intro_duration(
            y, sr, tempo, energy_threshold=0.3, min_intro=0.5, max_intro=8.0,
        )

        # Validate BPM range
        if not 30 <= tempo <= 300:
            # If tempo is outside range, try alternative methods
            if tempo < 30:
                tempo = tempo * 2  # Double tempo for very slow songs
            elif tempo > 300:
                tempo = tempo / 2  # Half tempo for very fast songs

        # Calculate allowed clip durations based on BPM
        min_duration, allowed_durations = calculate_clip_constraints(tempo)

        return {
            # === BACKWARD COMPATIBLE FIELDS ===
            "bpm": float(tempo),
            "beats": beat_times,  # Original beats for backward compatibility
            "duration": float(duration),
            "allowed_durations": allowed_durations,
            "min_duration": min_duration,
            # === ENHANCED MUSICAL INTELLIGENCE FIELDS ===
            "compensated_beats": filtered_beats,  # Offset-corrected and filtered beats
            "musical_start_time": float(musical_start_time),
            "intro_duration": float(refined_intro_duration),
            "beat_hierarchy": beat_hierarchy,
            # === METADATA ===
            "analysis_version": "2.0",
            "librosa_offset_compensation": -0.04,
            "intro_detection_method": "onset_energy_analysis",
        }

    except Exception as e:
        raise ValueError(f"Failed to analyze audio file {file_path}: {e!s}")


def calculate_clip_constraints(bpm: float) -> Tuple[float, List[float]]:
    """Calculate allowed clip durations based on BPM.

    For a given BPM, calculate musically appropriate clip durations.

    Examples:
    - 60 BPM = 1 beat/second → clips: 4s, 8s, 16s
    - 120 BPM = 2 beats/second → clips: 2s, 4s, 8s
    - 90 BPM = 1.5 beats/second → clips: 2.67s, 5.33s, 10.67s

    Args:
        bpm: Beats per minute of the music track

    Returns:
        Tuple containing minimum duration and list of allowed durations

    Raises:
        ValueError: If BPM is not within valid range (30-300)
    """
    if not 30 <= bpm <= 300:
        raise ValueError(f"BPM {bpm} is outside valid range (30-300)")

    beat_duration = 60.0 / bpm

    # Minimum clip is 4 beats (but at least 1.0 seconds for very slow songs)
    min_duration = max(float(beat_duration * 4), 1.0)

    # Allowed durations are musical multiples starting from 4 beats
    multipliers = [4, 8, 16]
    allowed_durations = [float(beat_duration * m) for m in multipliers]

    # Filter out clips longer than 16 seconds
    allowed_durations = [d for d in allowed_durations if d <= 16.0]

    return min_duration, allowed_durations


def get_cut_points(beats: List[float], song_duration: float) -> List[float]:
    """Convert beat timestamps to potential video cut points.

    Args:
        beats: List of beat timestamps in seconds
        song_duration: Total duration of the song in seconds

    Returns:
        List of timestamps suitable for video cuts
    """
    if not beats:
        return []

    # Filter beats to avoid cuts too close together (minimum 0.5 seconds apart)
    min_gap = 0.5
    filtered_beats = [beats[0]]  # Always include first beat

    for beat in beats[1:]:
        if beat - filtered_beats[-1] >= min_gap:
            filtered_beats.append(beat)

    # Ensure we have cuts throughout the song duration
    cut_points = []

    # Add beginning if first beat is not at the start
    if filtered_beats[0] > 1.0:
        cut_points.append(0.0)

    # Add all filtered beats as cut points
    cut_points.extend(filtered_beats)

    # Add end point if needed
    if cut_points[-1] < song_duration - 1.0:
        cut_points.append(song_duration)

    # Remove any cut points beyond song duration
    return [cp for cp in cut_points if cp <= song_duration]



def test_audio_analyzer():
    """Test the audio analyzer with sample data."""

    # Test calculate_clip_constraints function
    test_bpms = [60, 120, 90, 140]

    for bpm in test_bpms:
        with contextlib.suppress(ValueError):
            min_dur, allowed_durs = calculate_clip_constraints(bpm)

    # Test edge cases
    edge_cases = [25, 350]  # Should be rejected
    for bpm in edge_cases:
        with contextlib.suppress(ValueError):
            min_dur, allowed_durs = calculate_clip_constraints(bpm)

    # Test get_cut_points function
    test_beats = [0.5, 1.0, 1.3, 2.0, 2.8, 3.5, 4.0, 4.2, 5.0]
    song_duration = 6.0

    cut_points = get_cut_points(test_beats, song_duration)

    # Test with empty beats
    empty_cuts = get_cut_points([], 5.0)


