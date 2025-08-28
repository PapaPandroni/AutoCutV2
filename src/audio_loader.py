"""
Robust Audio Loading System for AutoCut
======================================

This module provides reliable audio loading that bypasses MoviePy's problematic
FFMPEG_AudioReader, which becomes corrupted after complex video processing.

The system uses multiple strategies in fallback order:
1. FFmpeg subprocess (primary) - Complete process isolation
2. Librosa backend (secondary) - Alternative audio processing library
3. Additional fallbacks as needed

Key Benefits:
- Prevents "At least one output file must be specified" FFmpeg errors
- Avoids "object has no attribute 'proc'" AudioReader corruption
- Maintains support for all audio formats (WAV, MP3, M4A, FLAC)
- Production-ready error handling and diagnostics
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# Import for type annotations
try:
    from moviepy.audio.AudioClip import AudioArrayClip
except ImportError:
    # Fallback for type checking when moviepy not available
    AudioArrayClip = Any

# Setup logging
logger = logging.getLogger(__name__)


def load_audio_with_ffmpeg_subprocess(audio_file: str) -> "AudioArrayClip":
    """
    Load audio using direct FFmpeg subprocess for complete process isolation.

    This method bypasses MoviePy's FFMPEG_AudioReader entirely, preventing
    state corruption issues that occur after complex video processing.
    Enhanced with WAV-specific optimizations and error handling.

    Args:
        audio_file: Path to audio file (WAV, MP3, M4A, FLAC supported)

    Returns:
        AudioArrayClip: MoviePy-compatible audio clip

    Raises:
        RuntimeError: If FFmpeg subprocess fails
    """
    from moviepy.audio.AudioClip import AudioArrayClip

    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logger.info(
        f"🔧 Loading audio with FFmpeg subprocess: {Path(audio_file).name}",
    )

    # Detect file format for optimized processing
    file_ext = Path(audio_file).suffix.lower()
    is_wav_file = file_ext in [".wav", ".wave"]

    if is_wav_file:
        logger.info("📻 WAV file detected - using optimized WAV processing parameters")

    # Base FFmpeg command for reliable audio extraction
    # -f s16le: 16-bit signed integer, little endian output format
    # -acodec pcm_s16le: PCM codec for raw audio data
    # -ac 2: Force stereo output (easier to handle)
    # -ar 44100: Standard sample rate
    # -: Output to stdout for capture
    base_cmd = [
        "ffmpeg",
        "-i",
        audio_file,  # Input file
        "-f",
        "s16le",  # Output format: signed 16-bit little endian
        "-acodec",
        "pcm_s16le",  # Audio codec: PCM 16-bit
        "-ac",
        "2",  # Audio channels: stereo
        "-ar",
        "44100",  # Audio sample rate: 44.1kHz
        "-v",
        "quiet",  # Suppress FFmpeg output
        "-",  # Output to stdout
    ]

    # WAV-specific optimizations
    if is_wav_file:
        # Add WAV-optimized parameters to prevent common WAV processing issues
        wav_cmd = [
            "ffmpeg",
            "-i",
            audio_file,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-sample_fmt",
            "s16",  # Explicit sample format for WAV
            "-channel_layout",
            "stereo",  # Explicit channel layout
            "-avoid_negative_ts",
            "make_zero",  # Fix timestamp issues
            "-v",
            "quiet",
            "-",
        ]
        cmd = wav_cmd
        logger.info("🎯 Using WAV-optimized FFmpeg parameters")
    else:
        cmd = base_cmd

    try:
        # Run FFmpeg subprocess with output capture
        process = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=60,  # Timeout after 60 seconds
        )

        # Convert raw audio bytes to numpy array
        audio_data = np.frombuffer(process.stdout, dtype=np.int16)

        if len(audio_data) == 0:
            raise RuntimeError("No audio data extracted from file")

        # Reshape to stereo format (N, 2) and normalize to float32 [-1, 1]
        audio_data = audio_data.reshape(-1, 2).astype(np.float32) / 32768.0

        # Create AudioArrayClip (bypasses FFMPEG_AudioReader completely)
        audio_clip = AudioArrayClip(audio_data, fps=44100)

        duration = len(audio_data) / 44100.0
        logger.info(f"✅ FFmpeg subprocess success: {duration:.2f}s duration")

    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg subprocess failed (exit code {e.returncode})"
        if e.stderr:
            stderr_text = e.stderr.decode("utf-8", errors="ignore")
            error_msg += f": {stderr_text}"

            # WAV-specific error handling and fallback strategies
            if is_wav_file and (
                "At least one output file must be specified" in stderr_text
                or "Invalid data found" in stderr_text
            ):
                logger.warning(
                    "⚠️ WAV-specific error detected, trying fallback approach..."
                )
                return _fallback_wav_processing(audio_file)

        logger.exception(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

    except subprocess.TimeoutExpired:
        error_msg = "FFmpeg subprocess timed out after 60 seconds"
        logger.exception(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from None  # TimeoutExpired doesn't need chaining

    except Exception as e:
        error_msg = f"FFmpeg subprocess unexpected error: {e!s}"

        # If it's a WAV file and we get unexpected errors, try fallback
        if is_wav_file and ("reshape" in str(e) or "frombuffer" in str(e)):
            logger.warning("⚠️ WAV data processing error, trying fallback approach...")
            try:
                return _fallback_wav_processing(audio_file)
            except Exception as fallback_error:
                error_msg += f" (fallback also failed: {fallback_error})"

        logger.exception(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e
    else:
        return audio_clip


def _fallback_wav_processing(audio_file: str) -> "AudioArrayClip":
    """
    Specialized fallback for WAV files that fail standard FFmpeg processing.

    This function handles problematic WAV files that trigger the specific errors:
    - "At least one output file must be specified"
    - "Invalid data found when processing input"
    - FFMPEG_AudioReader corruption issues

    Args:
        audio_file: Path to problematic WAV file

    Returns:
        AudioArrayClip: Successfully processed audio clip

    Raises:
        RuntimeError: If all fallback strategies fail
    """
    from moviepy.audio.AudioClip import AudioArrayClip

    logger.info(
        f"🔄 Attempting WAV fallback processing for {Path(audio_file).name}"
    )

    # Strategy 1: Use simplified FFmpeg command for WAV files
    try:
        logger.info("📋 Fallback Strategy 1: Simplified WAV FFmpeg command")

        # Ultra-simple FFmpeg command that avoids complex parameter parsing
        simple_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            audio_file,
            "-f",
            "wav",  # Explicit WAV output format
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "pipe:1",  # Output to stdout as pipe
        ]

        process = subprocess.run(
            simple_cmd, capture_output=True, check=True, timeout=30
        )

        # Parse WAV data from stdout
        import io
        import wave

        wav_buffer = io.BytesIO(process.stdout)
        with wave.open(wav_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()

            # Convert to numpy array
            if wav_file.getsampwidth() == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:  # Other formats
                raise RuntimeError("Unsupported sample width")

            # Handle mono/stereo
            if n_channels == 1:
                # Convert mono to stereo
                audio_data = np.column_stack((audio_data, audio_data))
            else:
                audio_data = audio_data.reshape(-1, n_channels)

            # Ensure stereo output
            if audio_data.shape[1] != 2:
                audio_data = audio_data[:, :2]  # Take first two channels

            # Normalize to float32 [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Create clip with original sample rate
            audio_clip = AudioArrayClip(audio_data, fps=sample_rate)
            logger.info(
                f"✅ WAV Fallback Strategy 1 successful: {len(audio_data) / sample_rate:.2f}s"
            )

    except Exception as e1:
        logger.warning(f"⚠️ Fallback Strategy 1 failed: {e1}")
    else:
        return audio_clip

    # Strategy 2: Use librosa as ultimate fallback
    try:
        logger.info("📋 Fallback Strategy 2: Using librosa for WAV processing")

        import librosa

        # Load with librosa (handles most WAV format issues)
        audio_data, sample_rate = librosa.load(audio_file, sr=44100, mono=False)

        # Ensure stereo format
        if audio_data.ndim == 1:
            # Mono to stereo
            audio_data = np.column_stack((audio_data, audio_data))
        else:
            # Multi-channel to stereo
            audio_data = audio_data.T  # Transpose to (samples, channels)
            if audio_data.shape[1] == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            elif audio_data.shape[1] > 2:
                audio_data = audio_data[:, :2]  # Take first two channels

        # Create AudioArrayClip
        audio_clip = AudioArrayClip(audio_data, fps=sample_rate)
        logger.info(
            f"✅ WAV Fallback Strategy 2 (librosa) successful: {len(audio_data) / sample_rate:.2f}s"
        )

    except Exception as e2:
        logger.warning(f"⚠️ Fallback Strategy 2 failed: {e2}")
    else:
        return audio_clip

    # All strategies failed
    error_msg = (
        f"All WAV fallback strategies failed for {audio_file}. "
        f"This WAV file may be corrupted or use an unsupported format."
    )
    logger.error(f"❌ {error_msg}")
    raise RuntimeError(error_msg)


def load_audio_with_librosa(audio_file: str) -> "AudioArrayClip":
    """
    Load audio using librosa backend as fallback strategy.

    This method completely bypasses MoviePy's audio system, using librosa
    for audio loading and converting to AudioArrayClip for compatibility.

    Args:
        audio_file: Path to audio file

    Returns:
        AudioArrayClip: MoviePy-compatible audio clip

    Raises:
        RuntimeError: If librosa loading fails
    """
    from moviepy.audio.AudioClip import AudioArrayClip

    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("Librosa not available - install with: pip install librosa") from e

    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logger.info(f"🎵 Loading audio with librosa: {Path(audio_file).name}")

    try:
        # Load audio with librosa (sr=44100 for consistency, mono=False for stereo)
        audio_data, sample_rate = librosa.load(
            audio_file,
            sr=44100,  # Target sample rate
            mono=False,  # Preserve stereo if available
        )

        # Handle mono vs stereo format conversion
        if audio_data.ndim == 1:
            # Mono: reshape to (N, 1) format
            audio_data = audio_data.reshape(-1, 1)
            logger.info("   📻 Detected mono audio")
        else:
            # Stereo: transpose from (2, N) to (N, 2) for MoviePy format
            audio_data = audio_data.T
            logger.info("   🔊 Detected stereo audio")

        # Ensure float32 format and proper range [-1, 1]
        audio_data = audio_data.astype(np.float32)

        # Create AudioArrayClip
        audio_clip = AudioArrayClip(audio_data, fps=sample_rate)

        duration = len(audio_data) / sample_rate
        logger.info(f"✅ Librosa success: {duration:.2f}s duration, {sample_rate}Hz")

    except Exception as e:
        error_msg = f"Librosa loading failed: {e!s}"
        logger.exception(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e
    else:
        return audio_clip


def get_audio_info(audio_file: str) -> dict:
    """
    Get audio file information for diagnostics.

    Args:
        audio_file: Path to audio file

    Returns:
        dict: Audio file information
    """
    if not Path(audio_file).exists():
        return {"error": "File not found"}

    info = {
        "file": Path(audio_file).name,
        "size_mb": round(Path(audio_file).stat().st_size / (1024 * 1024), 2),
        "extension": Path(audio_file).suffix.lower(),
    }

    # Try to get additional info with FFmpeg
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            audio_file,
        ]
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            if "format" in data:
                info["duration"] = float(data["format"].get("duration", 0))
                info["bitrate"] = int(data["format"].get("bit_rate", 0))
            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                info["sample_rate"] = int(stream.get("sample_rate", 0))
                info["channels"] = int(stream.get("channels", 0))
                info["codec"] = stream.get("codec_name", "unknown")
    except Exception:
        pass  # Ignore errors in info gathering

    return info


def load_audio_robust(audio_file: str) -> "AudioArrayClip":
    """
    Load audio with multiple fallback strategies for maximum reliability.

    This is the main entry point that should replace AudioFileClip() calls.
    It tries multiple strategies in order of preference and provides detailed
    error reporting and diagnostics for common FFMPEG_AudioReader issues.

    Strategy Order:
    1. FFmpeg subprocess (most reliable, complete process isolation)
    2. Librosa backend (alternative processing, still bypasses MoviePy)
    3. Future: Additional fallbacks can be added here

    Args:
        audio_file: Path to audio file

    Returns:
        AudioArrayClip: MoviePy-compatible audio clip

    Raises:
        RuntimeError: If all loading strategies fail
    """
    if not audio_file:
        raise ValueError("Audio file path cannot be empty")

    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Log audio file information for diagnostics
    audio_info = get_audio_info(audio_file)
    logger.info(f"🎯 Loading audio: {audio_info}")

    # Detect common problematic scenarios upfront
    file_ext = Path(audio_file).suffix.lower()
    is_wav_file = file_ext in [".wav", ".wave"]

    if is_wav_file:
        logger.info("📻 WAV file detected - will use enhanced WAV processing")

    # Define loading strategies in priority order
    strategies = [
        ("FFmpeg Subprocess", load_audio_with_ffmpeg_subprocess),
        ("Librosa Backend", load_audio_with_librosa),
    ]

    last_error = None
    error_diagnostics = []

    # Helper function to try a single loading strategy
    def _try_loading_strategy(strategy_name: str, loader_func) -> Tuple[Optional[Any], Optional[Exception]]:
        try:
            logger.info(f"🔄 Attempting: {strategy_name}")
            result = loader_func(audio_file)
            logger.info(f"✅ Audio loading successful with {strategy_name}")
        except Exception as e:
            error_str = str(e)

            # Enhanced error diagnostics for common FFMPEG_AudioReader issues
            if _is_ffmpeg_audioreader_error(error_str):
                diagnostic = _diagnose_ffmpeg_audioreader_error(
                    error_str, audio_file, is_wav_file
                )
                error_diagnostics.append(f"{strategy_name}: {diagnostic}")
                logger.warning(
                    f"⚠️ FFMPEG_AudioReader issue detected in {strategy_name}: {diagnostic}"
                )
            else:
                error_diagnostics.append(f"{strategy_name}: {error_str}")
                logger.warning(f"⚠️ {strategy_name} failed: {error_str}")

            return None, e
        else:
            return result, None

    # Try each strategy in order
    for strategy_name, loader_func in strategies:
        result, error = _try_loading_strategy(strategy_name, loader_func)
        if result is not None:
            return result
        last_error = error

    # All strategies failed - provide comprehensive error report
    error_msg = (
        f"All audio loading strategies failed for {Path(audio_file).name}"
    )

    if error_diagnostics:
        error_msg += "\n\nDiagnostic Details:"
        for i, diagnostic in enumerate(error_diagnostics, 1):
            error_msg += f"\n  {i}. {diagnostic}"

    if last_error:
        error_msg += f"\n\nLast error: {last_error!s}"

    error_msg += f"\n\nFile info: {audio_info}"

    # Add format-specific troubleshooting suggestions
    if is_wav_file:
        error_msg += "\n\nWAV Troubleshooting Suggestions:"
        error_msg += "\n• Try converting with: ffmpeg -i input.wav -acodec pcm_s16le -ar 44100 -ac 2 output.wav"
        error_msg += "\n• Check if WAV file has unusual encoding or metadata"
        error_msg += "\n• Verify file is not corrupted with: ffmpeg -v error -i input.wav -f null -"

    logger.error(f"❌ {error_msg}")
    raise RuntimeError(error_msg)


# Convenience function for backwards compatibility
def load_audio_safe(audio_file: str) -> "AudioArrayClip":
    """Alias for load_audio_robust() for backwards compatibility."""
    return load_audio_robust(audio_file)


def _is_ffmpeg_audioreader_error(error_str: str) -> bool:
    """
    Detect if an error is related to FFMPEG_AudioReader issues.

    Args:
        error_str: Error message string

    Returns:
        bool: True if error is likely from FFMPEG_AudioReader
    """
    ffmpeg_audioreader_indicators = [
        "ffmpeg_audioreader",
        "ffmpeg_audioread",
        "object has no attribute 'proc'",
        "at least one output file must be specified",
        "invalid data found when processing input",
        "ffmpeg.*pipe",
        "could not find codec parameters",
        "moov atom not found",
        "readers.py",
        "__del__",
    ]

    error_lower = error_str.lower()
    return any(indicator in error_lower for indicator in ffmpeg_audioreader_indicators)


def _diagnose_ffmpeg_audioreader_error(
    error_str: str, audio_file: str, is_wav_file: bool
) -> str:
    """
    Provide specific diagnostic information for FFMPEG_AudioReader errors.

    Args:
        error_str: Error message string
        audio_file: Path to the problematic audio file
        is_wav_file: Whether the file is a WAV file

    Returns:
        str: Diagnostic message with specific guidance
    """
    error_lower = error_str.lower()
    filename = Path(audio_file).name

    # Specific error pattern matching and diagnostics
    if "at least one output file must be specified" in error_lower:
        return (
            f"FFMPEG command construction failed for {filename}. "
            f"MoviePy's FFMPEG_AudioReader built an incomplete command. "
            f"{'WAV files commonly trigger this due to parameter parsing issues.' if is_wav_file else 'This indicates MoviePy version compatibility issues.'}"
        )

    if "object has no attribute 'proc'" in error_lower:
        return (
            f"FFMPEG_AudioReader cleanup race condition in {filename}. "
            f"The subprocess handle was not properly initialized before cleanup. "
            f"This is a known MoviePy 2.2.1 resource management bug."
        )

    if "invalid data found" in error_lower:
        return (
            f"FFmpeg could not parse audio data in {filename}. "
            f"{'WAV file may have unusual encoding, metadata, or corruption.' if is_wav_file else 'Audio format may not be supported or file is corrupted.'}"
        )

    if "could not find codec parameters" in error_lower:
        return (
            f"FFmpeg could not detect audio codec in {filename}. "
            f"{'WAV file may have non-standard header or missing metadata.' if is_wav_file else 'Audio codec may be unsupported or file is corrupted.'}"
        )

    if "moov atom not found" in error_lower:
        return (
            f"Missing metadata container in {filename}. "
            f"This typically indicates a truncated or corrupted audio file."
        )

    if "pipe" in error_lower and "ffmpeg" in error_lower:
        return (
            f"FFmpeg subprocess communication failure with {filename}. "
            f"Data transfer between MoviePy and FFmpeg was interrupted."
        )

    if "readers.py" in error_lower or "__del__" in error_lower:
        return (
            f"MoviePy audio reader destructor error for {filename}. "
            f"This is a known cleanup timing issue in MoviePy 2.2.1's FFMPEG_AudioReader."
        )

    # Generic FFMPEG_AudioReader error
    return (
        f"FFMPEG_AudioReader processing failed for {filename}. "
        f"{'WAV files are particularly susceptible to these MoviePy issues.' if is_wav_file else 'This indicates a MoviePy internal error.'} "
        f"Error details: {error_str[:100]}{'...' if len(error_str) > 100 else ''}"
    )
