"""
Audio Analysis Module for AutoCut

Handles music analysis including BPM detection, beat tracking, and
calculation of musically appropriate clip durations.
"""

from typing import Dict, List, Tuple, Union, Optional
import os
import librosa
import numpy as np


def analyze_audio(file_path: str) -> Dict[str, Union[float, List[float]]]:
    """Analyze audio file and extract tempo and beat information.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary containing:
        - 'bpm': Detected beats per minute
        - 'beats': List of beat timestamps in seconds
        - 'duration': Total audio duration in seconds
        - 'allowed_durations': List of musically appropriate clip durations
        
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
            'bpm': float(tempo),
            'beats': beat_times,
            'duration': float(duration),
            'allowed_durations': allowed_durations,
            'min_duration': min_duration
        }
        
    except Exception as e:
        raise ValueError(f"Failed to analyze audio file {file_path}: {str(e)}")


def calculate_clip_constraints(bpm: float) -> Tuple[float, List[float]]:
    """Calculate allowed clip durations based on BPM.
    
    For a given BPM, calculate musically appropriate clip durations.
    
    Examples:
    - 60 BPM = 1 beat/second → clips: 1s, 2s, 4s, 8s
    - 120 BPM = 2 beats/second → clips: 0.5s, 1s, 2s, 4s
    - 90 BPM = 1.5 beats/second → clips: 0.67s, 1.33s, 2.67s, 5.33s
    
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
    
    # Minimum clip is 1 beat (but at least 0.5 seconds)
    min_duration = max(beat_duration, 0.5)
    
    # Allowed durations are musical multiples
    multipliers = [1, 2, 4, 8, 16]
    allowed_durations = [beat_duration * m for m in multipliers]
    
    # Filter out clips longer than 8 seconds
    allowed_durations = [d for d in allowed_durations if d <= 8.0]
    
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
    cut_points = [cp for cp in cut_points if cp <= song_duration]
    
    return cut_points


def test_audio_analyzer():
    """Test the audio analyzer with sample data."""
    print("🎵 AutoCut Audio Analyzer - Test Mode")
    print("=" * 50)
    
    # Test calculate_clip_constraints function
    print("\n1. Testing calculate_clip_constraints():")
    test_bpms = [60, 120, 90, 140]
    
    for bpm in test_bpms:
        try:
            min_dur, allowed_durs = calculate_clip_constraints(bpm)
            print(f"   BPM {bpm:3d}: min={min_dur:.2f}s, allowed={[f'{d:.2f}' for d in allowed_durs]}")
        except ValueError as e:
            print(f"   BPM {bpm:3d}: ERROR - {e}")
    
    # Test edge cases
    print("\n2. Testing edge cases:")
    edge_cases = [25, 350]  # Should be rejected
    for bpm in edge_cases:
        try:
            min_dur, allowed_durs = calculate_clip_constraints(bpm)
            print(f"   BPM {bpm:3d}: ERROR - should have failed")
        except ValueError as e:
            print(f"   BPM {bpm:3d}: ✓ Correctly rejected - {e}")
    
    # Test get_cut_points function
    print("\n3. Testing get_cut_points():")
    test_beats = [0.5, 1.0, 1.3, 2.0, 2.8, 3.5, 4.0, 4.2, 5.0]
    song_duration = 6.0
    
    cut_points = get_cut_points(test_beats, song_duration)
    print(f"   Input beats: {test_beats}")
    print(f"   Song duration: {song_duration}s")
    print(f"   Cut points: {cut_points}")
    
    # Test with empty beats
    empty_cuts = get_cut_points([], 5.0)
    print(f"   Empty beats: {empty_cuts} (should be [])")
    
    print("\n4. Audio file analysis test:")
    print("   To test with real audio files, place them in test_media/")
    print("   Supported formats: MP3, WAV, M4A, FLAC")
    
    print("\n✅ Audio analyzer tests completed!")