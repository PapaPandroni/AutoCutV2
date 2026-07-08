"""
Audio Analysis Module for AutoCut

Handles music analysis including BPM detection, beat tracking, and
calculation of musically appropriate clip durations.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np

# librosa.beat.beat_track's detected beat consistently lags the true beat by
# a small, fairly constant amount. Measured against tests/synthetic_media.py's
# WAV click track (exact known click times) across several BPMs (90/120/140):
# mean offset ~+0.034s, std ~0.007s. Compensation shifts beats earlier by that
# amount to cancel it out.
LIBROSA_BEAT_LATENCY_OFFSET_SECONDS = -0.034


def detect_musical_start(
    y: np.ndarray,
    sr: int,
    tempo: float,
) -> float:
    """Detect the start of significant musical content using onset detection and energy analysis.

    Args:
        y: Audio time series
        sr: Sample rate
        tempo: Detected BPM

    Returns:
        musical_start_time in seconds
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
        return 0.0

    # Convert to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate RMS energy over time (hop_length frames)
    hop_length = 512
    frame_length = 2048
    rms_energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # Convert RMS frame indices to time
    rms_times = librosa.frames_to_time(
        np.arange(len(rms_energy)),
        sr=sr,
        hop_length=hop_length,
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
                i + int(sustained_duration * sr / hop_length),
                len(rms_energy),
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

    return musical_start_time


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
        np.arange(len(rms_energy)),
        sr=sr,
        hop_length=hop_length,
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


def _onset_envelope_and_times(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Onset-strength envelope and its frame timestamps, shared by the
    beat-phase (D5) and downbeat (D6) estimators."""
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_envelope)),
        sr=sr,
        hop_length=hop_length,
    )
    return onset_envelope, onset_times


def _mean_onset_strength_at(
    times: List[float],
    onset_envelope: np.ndarray,
    onset_times: np.ndarray,
) -> float:
    """Mean onset-strength value at the nearest frame to each of ``times``."""
    if not times:
        return 0.0
    strengths = [onset_envelope[int(np.argmin(np.abs(onset_times - t)))] for t in times]
    return float(np.mean(strengths))


def verify_beat_phase(
    beat_times: List[float],
    y: np.ndarray,
    sr: int,
) -> List[float]:
    """Check whether librosa locked onto the off-beat phase and correct it.

    ``librosa.beat.beat_track`` sometimes tracks the up-beats instead of the
    downbeats. Compare mean onset-envelope strength at the detected beat
    times against the same grid shifted by half a beat interval; if the
    shifted grid is meaningfully (>15%) stronger, the detector locked onto
    the wrong phase, so shift all beats to match (D5).

    Args:
        beat_times: Detected beat timestamps in seconds
        y: Audio time series used for beat detection (e.g. the percussive
            component), so the phase check reflects the same signal
        sr: Sample rate

    Returns:
        Beat timestamps, shifted by half a beat interval if the shifted grid
        is the stronger phase, otherwise unchanged
    """
    if len(beat_times) < 2:
        return beat_times

    onset_envelope, onset_times = _onset_envelope_and_times(y, sr)

    avg_interval = sum(
        beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)
    ) / (len(beat_times) - 1)
    half_interval = avg_interval / 2

    detected_strength = _mean_onset_strength_at(beat_times, onset_envelope, onset_times)
    shifted_times = [t + half_interval for t in beat_times]
    shifted_strength = _mean_onset_strength_at(
        shifted_times,
        onset_envelope,
        onset_times,
    )

    # max(..., 1e-6) so a completely silent detected phase (0.0) still
    # counts as "meaningfully weaker" rather than failing the check.
    if shifted_strength > max(detected_strength * 1.15, 1e-6):
        return shifted_times
    return beat_times


def estimate_downbeat_offset(beat_times: List[float], y: np.ndarray, sr: int) -> int:
    """Estimate which beat index (0-3) is the downbeat, assuming 4/4 time.

    The old ``create_beat_hierarchy`` hardcoded "downbeat = beat index 0".
    Instead, evaluate mean onset strength at each of the 4 candidate phases'
    downbeat positions (``beat_times[phase::4]``) and pick the strongest (D6).

    Args:
        beat_times: Detected (and phase-corrected) beat timestamps in seconds
        y: Audio time series used for beat detection
        sr: Sample rate

    Returns:
        The strongest candidate phase, 0-3
    """
    if len(beat_times) < 4:
        return 0

    onset_envelope, onset_times = _onset_envelope_and_times(y, sr)

    best_phase, best_strength = 0, -1.0
    for phase in range(4):
        strength = _mean_onset_strength_at(
            beat_times[phase::4],
            onset_envelope,
            onset_times,
        )
        if strength > best_strength:
            best_strength = strength
            best_phase = phase
    return best_phase


def apply_offset_compensation(
    beats: List[float],
    offset: float = LIBROSA_BEAT_LATENCY_OFFSET_SECONDS,
) -> List[float]:
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
        np.arange(len(onset_envelope)),
        sr=sr,
        hop_length=hop_length,
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
    downbeat estimation, and offset compensation for professional synchronization.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary containing:
        - 'bpm': Detected beats per minute (float)
        - 'beats': Original beat timestamps in seconds (List[float])
        - 'compensated_beats': Offset-corrected beat timestamps (List[float])
        - 'musical_start_time': First significant beat timestamp (float)
        - 'intro_duration': Length of intro section in seconds (float)
        - 'downbeat_offset': Beat index (0-3) estimated to be the downbeat (int)
        - 'duration': Total audio duration in seconds (float)
        - 'allowed_durations': Musically appropriate clip durations (List[float])
        - 'min_duration': Minimum clip duration (float)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If BPM is outside valid range (30-300)
    """
    if not Path(file_path).exists():
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

        # librosa >=0.10 may return tempo as a 0-d/1-d ndarray. Coerce to a plain
        # float so all downstream scalar math (comparisons, int(), etc.) is safe.
        tempo = float(np.atleast_1d(tempo)[0])

        # Validate BPM range UP FRONT. detect_musical_start()/detect_intro_duration()
        # below divide by tempo, so it must be valid before they run. Octave errors
        # are common in beat tracking, so fold the tempo into 30-300 by repeatedly
        # doubling/halving, then hard-clamp (this also turns a degenerate tempo of
        # 0 into 30, preventing a division-by-zero further down).
        for _ in range(8):
            if 30 <= tempo <= 300:
                break
            if tempo < 30:
                tempo *= 2  # Double tempo for very slow songs
            else:
                tempo /= 2  # Halve tempo for very fast songs
        tempo = min(max(tempo, 30.0), 300.0)

        # Convert frame indices to timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Verify librosa didn't lock onto the off-beat phase (D5), before any
        # further processing that assumes beat_times are on the downbeat.
        beat_times = verify_beat_phase(beat_times, y_percussive, sr)

        # === ENHANCED MUSICAL INTELLIGENCE ===

        # 1. Detect the start of significant musical content (for beat-grid
        # filtering downstream, e.g. match_clips_to_beats' musical_start_time).
        musical_start_time = detect_musical_start(y, sr, tempo)

        # 2. Detect intro/buildup duration -- the single canonical value, used
        # both for weak-beat filtering below and reported in the result dict
        # (previously a second, cruder duplicate of musical_start_time was
        # used for filtering while this one was only reported, never applied).
        intro_duration = detect_intro_duration(
            y,
            sr,
            tempo,
            energy_threshold=0.3,
            min_intro=0.5,
            max_intro=8.0,
        )

        # 3. Apply systematic offset compensation for librosa latency
        compensated_beats = apply_offset_compensation(
            beat_times,
            offset=LIBROSA_BEAT_LATENCY_OFFSET_SECONDS,
        )

        # 4. Filter weak beats during intro sections
        filtered_beats = filter_weak_beats_in_intro(
            compensated_beats,
            y,
            sr,
            intro_duration,
            strength_threshold=0.3,
        )

        # 5. Estimate which beat index is the downbeat (D6)
        downbeat_offset = estimate_downbeat_offset(beat_times, y_percussive, sr)

        # Calculate allowed clip durations based on BPM (already validated above)
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
            "intro_duration": float(intro_duration),
            "downbeat_offset": downbeat_offset,
            # === METADATA ===
            "analysis_version": "2.0",
            "librosa_offset_compensation": LIBROSA_BEAT_LATENCY_OFFSET_SECONDS,
            "intro_detection_method": "onset_energy_analysis",
        }

    except Exception as e:
        raise ValueError(f"Failed to analyze audio file {file_path}: {e!s}") from e


def calculate_clip_constraints(bpm: float) -> Tuple[float, List[float]]:
    """Calculate allowed clip durations based on BPM.

    For a given BPM, calculate musically appropriate clip durations.

    Examples:
    - 60 BPM = 1 beat/second → clips: 2s, 4s, 8s, 16s
    - 120 BPM = 2 beats/second → clips: 1s, 2s, 4s, 8s
    - 90 BPM = 1.5 beats/second → clips: 1.33s, 2.67s, 5.33s, 10.67s

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

    # Allowed durations are the musical multiples used by the variety patterns
    # (2, 4, 8 and 16 beats). Without the 2-beat entry the "energetic" pattern's
    # fast cuts get rejected by _calculate_duration_fit and silently dropped.
    multipliers = [2, 4, 8, 16]
    allowed_durations = [float(beat_duration * m) for m in multipliers]

    # Filter out clips longer than 16 seconds
    allowed_durations = [d for d in allowed_durations if d <= 16.0]

    return min_duration, allowed_durations
