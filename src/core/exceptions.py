"""
Custom exception classes for AutoCut V2.

This module provides a structured exception hierarchy to replace
inconsistent error handling patterns throughout the codebase.
All AutoCut-specific errors inherit from AutoCutError for consistent
error handling and logging.
"""

from typing import Any, Dict, List, Optional


class AutoCutError(Exception):
    """
    Base exception for all AutoCut operations.

    Provides consistent error handling with structured information
    for logging, debugging, and user feedback.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AutoCut error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            details: Additional error context and debugging information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class VideoProcessingError(AutoCutError):
    """
    Video processing specific errors.

    Raised when video loading, analysis, or processing operations fail.
    Common causes include codec issues, corrupted files, or unsupported formats.
    """



class iPhoneCompatibilityError(VideoProcessingError):
    """
    iPhone H.265 compatibility issues.

    Specifically handles errors related to iPhone H.265/HEVC video processing,
    transcoding failures, and platform-specific compatibility issues.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        platform: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize iPhone compatibility error.

        Args:
            message: Error description
            file_path: Path to problematic video file
            platform: Platform where error occurred (Linux/Mac)
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if platform:
            details["platform"] = platform

        super().__init__(
            message, error_code="IPHONE_COMPATIBILITY_ERROR", details=details,
        )


class HardwareAccelerationError(AutoCutError):
    """
    Hardware acceleration failures.

    Raised when hardware-accelerated encoding/decoding fails,
    GPU detection fails, or hardware-specific operations encounter errors.
    """

    def __init__(
        self,
        message: str,
        encoder_type: Optional[str] = None,
        hardware_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize hardware acceleration error.

        Args:
            message: Error description
            encoder_type: Type of encoder that failed (NVENC, QSV, CPU)
            hardware_info: Hardware detection information
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if encoder_type:
            details["encoder_type"] = encoder_type
        if hardware_info:
            details["hardware_info"] = hardware_info

        super().__init__(
            message, error_code="HARDWARE_ACCELERATION_ERROR", details=details,
        )


class ValidationError(AutoCutError):
    """
    Input validation failures.

    Raised when video files, audio files, or other inputs fail
    validation checks for format, codec, or content requirements.
    """

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        failed_checks: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize validation error.

        Args:
            message: Error description
            validation_type: Type of validation that failed
            failed_checks: List of specific validation checks that failed
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if validation_type:
            details["validation_type"] = validation_type
        if failed_checks:
            details["failed_checks"] = failed_checks

        super().__init__(message, error_code="VALIDATION_ERROR", details=details)


class TranscodingError(VideoProcessingError):
    """
    Video transcoding operation failures.

    Raised when H.265 to H.264 transcoding fails, FFmpeg operations fail,
    or transcoding parameters are invalid.
    """

    def __init__(
        self,
        message: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        ffmpeg_command: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize transcoding error.

        Args:
            message: Error description
            input_format: Input video format/codec
            output_format: Target output format/codec
            ffmpeg_command: FFmpeg command that failed
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if input_format:
            details["input_format"] = input_format
        if output_format:
            details["output_format"] = output_format
        if ffmpeg_command:
            details["ffmpeg_command"] = ffmpeg_command

        super().__init__(message, error_code="TRANSCODING_ERROR", details=details)


class ConfigurationError(AutoCutError):
    """
    Configuration and settings errors.

    Raised when configuration files are invalid, required settings are missing,
    or configuration values are out of valid ranges.
    """



class AudioProcessingError(AutoCutError):
    """
    Audio analysis and processing errors.

    Raised when audio file loading, BPM detection, or beat analysis fails.
    """

    def __init__(
        self,
        message: str,
        audio_file: Optional[str] = None,
        analysis_stage: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize audio processing error.

        Args:
            message: Error description
            audio_file: Path to problematic audio file
            analysis_stage: Stage of analysis that failed (BPM, beats, etc.)
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if audio_file:
            details["audio_file"] = audio_file
        if analysis_stage:
            details["analysis_stage"] = analysis_stage

        super().__init__(message, error_code="AUDIO_PROCESSING_ERROR", details=details)


class RenderingError(AutoCutError):
    """
    Video rendering and composition errors.

    Raised when final video rendering fails, clip composition fails,
    or MoviePy operations encounter errors.
    """

    def __init__(
        self,
        message: str,
        rendering_stage: Optional[str] = None,
        clip_count: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize rendering error.

        Args:
            message: Error description
            rendering_stage: Stage of rendering that failed
            clip_count: Number of clips being processed
            **kwargs: Additional error context
        """
        details = kwargs.get("details", {})
        if rendering_stage:
            details["rendering_stage"] = rendering_stage
        if clip_count:
            details["clip_count"] = clip_count

        super().__init__(message, error_code="RENDERING_ERROR", details=details)


# Convenience functions for common error patterns


def raise_validation_error(
    message: str,
    validation_type: str,
    file_path: Optional[str] = None,
    failed_checks: Optional[List[str]] = None,
) -> None:
    """Raise a validation error with standard formatting."""
    details = {}
    if file_path:
        details["file_path"] = file_path

    raise ValidationError(
        message=message,
        validation_type=validation_type,
        failed_checks=failed_checks,
        details=details,
    )


def raise_iphone_error(
    message: str,
    file_path: str,
    platform: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise an iPhone compatibility error with standard formatting."""
    details = additional_context or {}

    raise iPhoneCompatibilityError(
        message=message, file_path=file_path, platform=platform, details=details,
    )


def raise_transcoding_error(
    message: str,
    input_file: str,
    ffmpeg_command: Optional[str] = None,
    error_output: Optional[str] = None,
) -> None:
    """Raise a transcoding error with standard formatting."""
    details = {"input_file": input_file}
    if error_output:
        details["error_output"] = error_output

    raise TranscodingError(
        message=message, ffmpeg_command=ffmpeg_command, details=details,
    )


# Export all exception classes
__all__ = [
    "AudioProcessingError",
    "AutoCutError",
    "ConfigurationError",
    "HardwareAccelerationError",
    "RenderingError",
    "TranscodingError",
    "ValidationError",
    "VideoProcessingError",
    "iPhoneCompatibilityError",
    "raise_iphone_error",
    "raise_transcoding_error",
    "raise_validation_error",
]
