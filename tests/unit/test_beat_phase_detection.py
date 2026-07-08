"""Beat-detection phase-quality tests (D5/D6, IMPROVEMENTS.md Stage 2).

These build a synthetic percussive signal with a known accent pattern and
check that ``verify_beat_phase``/``estimate_downbeat_offset`` correctly find
the strongest phase -- without depending on librosa's own beat_track locking
onto a particular phase (which isn't reliably controllable from a synthetic
signal), so these test the phase-correction mechanism directly and
deterministically.
"""

import librosa
import numpy as np

from audio_analyzer import (
    apply_offset_compensation,
    estimate_downbeat_offset,
    verify_beat_phase,
)

SR = 22050
BPM = 120
INTERVAL = 60.0 / BPM  # 0.5s
DURATION = 12.0


def _kick(sr: int, freq: float = 150.0, length_s: float = 0.15) -> np.ndarray:
    n = int(length_s * sr)
    env = np.exp(-np.linspace(0, 8, n))
    return env * np.sin(2 * np.pi * freq * np.linspace(0, length_s, n))


def _place_kicks(y: np.ndarray, sr: int, times: list, kick: np.ndarray) -> None:
    for t in times:
        i = int(t * sr)
        n = min(len(kick), len(y) - i)
        if n > 0:
            y[i : i + n] += kick[:n]


def _normalized(y: np.ndarray) -> np.ndarray:
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


# --------------------------------- D5 ---------------------------------------
def test_verify_beat_phase_shifts_onto_the_stronger_accent():
    """A beat grid sitting on silence, with real accents a half-interval
    away, gets shifted onto the accents."""
    weak_beats = list(np.arange(0, DURATION, INTERVAL))
    strong_times = [t + INTERVAL / 2 for t in weak_beats]

    y = np.zeros(int(SR * DURATION), dtype=np.float64)
    _place_kicks(y, SR, strong_times, _kick(SR))
    y = _normalized(y)

    corrected = verify_beat_phase(weak_beats, y, SR)
    assert abs(corrected[0] - strong_times[0]) < 0.01


def test_verify_beat_phase_leaves_a_correct_grid_unchanged():
    """A beat grid already sitting on the accents is left alone."""
    beats = list(np.arange(0, DURATION, INTERVAL))

    y = np.zeros(int(SR * DURATION), dtype=np.float64)
    _place_kicks(y, SR, beats, _kick(SR))
    y = _normalized(y)

    unchanged = verify_beat_phase(beats, y, SR)
    assert abs(unchanged[0] - beats[0]) < 0.01


# --------------------------------- D6 ---------------------------------------
def test_estimate_downbeat_offset_finds_the_accented_phase():
    """Beat index 2 (of every 4) is accented -- the estimator should pick it,
    not assume index 0 like the old create_beat_hierarchy did."""
    beats = list(np.arange(0, DURATION, INTERVAL))
    accent_phase = 2
    accented_beats = beats[accent_phase::4]

    y = np.zeros(int(SR * DURATION), dtype=np.float64)
    _place_kicks(y, SR, beats, _kick(SR, freq=150.0) * 0.3)  # weak on every beat
    _place_kicks(y, SR, accented_beats, _kick(SR, freq=150.0))  # strong on the accent
    y = _normalized(y)

    assert estimate_downbeat_offset(beats, y, SR) == accent_phase


# ------------------------- offset compensation --------------------------------
def test_offset_compensation_matches_measured_click_latency(tmp_path):
    """apply_offset_compensation's default should bring librosa's detected
    beats within ~1 frame of the true click times on the synthetic click
    track -- locks in the empirical measurement it was set from."""
    from tests.synthetic_media import generate_audio

    bpm = 120
    duration = 16.0
    path = generate_audio(tmp_path, duration=duration, bpm=bpm)

    y, sr = librosa.load(path)
    _y_harmonic, y_percussive = librosa.effects.hpss(y)
    _tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    compensated = apply_offset_compensation(beat_times)

    true_clicks = list(np.arange(0, duration, 60.0 / bpm))
    deltas = [min(abs(bt - c) for c in true_clicks) for bt in compensated]
    assert max(deltas) < 1 / 24, deltas
