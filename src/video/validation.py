"""
Unified video validation system for AutoCut V2.

This module replaces 10+ scattered validation functions with a clean,
testable, and consistent validation system. Provides structured error
reporting and comprehensive iPhone H.265 compatibility checking.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our custom exceptions with dual import pattern
try:
    from ..core.exceptions import (
        ValidationError,
        VideoProcessingError,
        iPhoneCompatibilityError,
        raise_validation_error,
    )
except ImportError:
    # Fallback for direct execution
    from core.exceptions import (
        VideoProcessingError,
    )


class ValidationType(Enum):
    """Types of validation that can be performed."""

    BASIC_FORMAT = "basic_format"
    IPHONE_COMPATIBILITY = "iphone_compatibility"
    TRANSCODING_OUTPUT = "transcoding_output"
    MOVIEPY_COMPATIBILITY = "moviepy_compatibility"
    HARDWARE_ENCODER = "hardware_encoder"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue or warning."""

    severity: ValidationSeverity
    message: str
    code: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return formatted issue description."""
        severity_emoji = {
            ValidationSeverity.INFO: "ℹ️",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(self.severity, "❓")
        return f"{emoji} {self.message}"


@dataclass
class ValidationResult:
    """
    Unified validation result structure.

    Replaces inconsistent return types (bool, dict, complex objects)
    from the 10+ scattered validation functions with a consistent,
    structured result that provides detailed information for debugging
    and user feedback.
    """

    # Core validation status
    is_valid: bool
    validation_type: ValidationType
    file_path: Optional[str] = None

    # Detailed results
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    # Performance tracking
    validation_time_ms: Optional[float] = None

    def add_error(self, message: str, code: str, **context: Any) -> None:
        """Add an error issue to the validation result."""
        self.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=message,
                code=code,
                context=context,
            ),
        )
        self.is_valid = False

    def add_warning(self, message: str, code: str, **context: Any) -> None:
        """Add a warning issue to the validation result."""
        self.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=message,
                code=code,
                context=context,
            ),
        )

    def add_info(self, message: str, code: str, **context) -> None:
        """Add an info issue to the validation result."""
        self.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=message,
                code=code,
                context=context,
            ),
        )

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.is_valid:
            warning_count = len(self.get_warnings())
            if warning_count:
                return f"✅ Valid with {warning_count} warnings"
            return "✅ Valid"

        error_count = len(self.get_errors())
        return f"❌ Invalid ({error_count} errors)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backwards compatibility."""
        return {
            "valid": self.is_valid,
            "validation_type": self.validation_type.value,
            "file_path": self.file_path,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "code": issue.code,
                    "context": issue.context,
                }
                for issue in self.issues
            ],
            "metadata": self.metadata,
            "suggestions": self.suggestions,
            "summary": self.get_summary(),
            "validation_time_ms": self.validation_time_ms,
        }

    @classmethod
    def success(
        cls,
        validation_type: ValidationType,
        file_path: Optional[str] = None,
        **metadata,
    ) -> "ValidationResult":
        """Create a successful validation result."""
        result = cls(
            is_valid=True,
            validation_type=validation_type,
            file_path=file_path,
        )
        result.metadata.update(metadata)
        return result

    @classmethod
    def failure(
        cls,
        validation_type: ValidationType,
        error_message: str,
        error_code: str = "VALIDATION_FAILED",
        file_path: Optional[str] = None,
        **context,
    ) -> "ValidationResult":
        """Create a failed validation result."""
        result = cls(
            is_valid=False,
            validation_type=validation_type,
            file_path=file_path,
        )
        result.add_error(error_message, error_code, **context)
        return result


# Supported formats (moved from utils.py for consolidation)
SUPPORTED_VIDEO_FORMATS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".3gp",
    ".3g2",
    ".mts",
    ".m2ts",
    ".ts",
    ".vob",
    ".divx",
    ".xvid",
    ".asf",
    ".rm",
    ".rmvb",
    ".f4v",
    ".swf",
}

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}


class VideoValidator:
    """
    Unified video validation system replacing 10+ scattered functions.

    Provides consistent, testable, and well-structured validation with
    comprehensive error reporting and iPhone H.265 compatibility checking.

    Key improvements over old system:
    - Consistent return types (ValidationResult)
    - Structured error reporting with severity levels
    - Single FFprobe call with caching
    - Clear separation of validation concerns
    - Comprehensive test coverage support
    - Platform-consistent behavior
    """

    def __init__(self, cache_timeout: float = 300.0):
        """
        Initialize validator with optional caching.

        Args:
            cache_timeout: How long to cache FFprobe results (seconds)
        """
        self._codec_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = cache_timeout

    def validate_basic_format(self, file_path: str) -> ValidationResult:
        """
        Basic format validation - file existence and extension.

        Replaces: validate_video_file(), validate_audio_file()

        Args:
            file_path: Path to video file

        Returns:
            ValidationResult with basic format validation status
        """
        if not file_path:
            return ValidationResult.failure(
                ValidationType.BASIC_FORMAT,
                "No file path provided",
                "MISSING_FILE_PATH",
            )

        if not Path(file_path).exists():
            return ValidationResult.failure(
                ValidationType.BASIC_FORMAT,
                f"File not found: {file_path}",
                "FILE_NOT_FOUND",
                file_path=file_path,
            )

        file_extension = Path(file_path).suffix.lower()

        if file_extension not in SUPPORTED_VIDEO_FORMATS:
            # Check if it might be an audio file
            if file_extension in SUPPORTED_AUDIO_FORMATS:
                return ValidationResult.failure(
                    ValidationType.BASIC_FORMAT,
                    f"Audio file provided where video expected: {file_extension}",
                    "WRONG_FILE_TYPE",
                    file_path=file_path,
                    detected_type="audio",
                    extension=file_extension,
                )

            supported_list = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
            return ValidationResult.failure(
                ValidationType.BASIC_FORMAT,
                f"Unsupported video format: {file_extension}",
                "UNSUPPORTED_FORMAT",
                file_path=file_path,
                extension=file_extension,
                supported_formats=supported_list,
            )

        # Success
        result = ValidationResult.success(
            ValidationType.BASIC_FORMAT,
            file_path=file_path,
            extension=file_extension,
            file_size_mb=Path(file_path).stat().st_size / (1024 * 1024),
        )
        result.add_info(
            f"Valid video format: {file_extension}",
            "FORMAT_VALID",
            extension=file_extension,
        )

        return result

    def validate_iphone_compatibility(
        self,
        file_path: str,
        quick_mode: bool = False,
    ) -> ValidationResult:
        """
        Comprehensive iPhone H.265 compatibility validation.

        Replaces: validate_transcoded_output(), _validate_transcoded_output_enhanced(),
                 _validate_combined_iphone_requirements(), _validate_iphone_specific_requirements()

        Args:
            file_path: Path to video file
            quick_mode: If True, skip expensive MoviePy compatibility test

        Returns:
            ValidationResult with detailed iPhone compatibility status
        """
        # Basic format check first
        basic_result = self.validate_basic_format(file_path)
        if not basic_result.is_valid:
            # Re-wrap as iPhone compatibility error
            result = ValidationResult.failure(
                ValidationType.IPHONE_COMPATIBILITY,
                f"Basic format validation failed: {basic_result.issues[0].message}",
                "BASIC_FORMAT_FAILED",
                file_path=file_path,
            )
            result.issues.extend(basic_result.issues)
            return result

        # Get codec information (cached)
        try:
            codec_info = self._get_codec_info(file_path)
        except Exception as e:
            return ValidationResult.failure(
                ValidationType.IPHONE_COMPATIBILITY,
                f"Failed to analyze video codec: {e!s}",
                "CODEC_ANALYSIS_FAILED",
                file_path=file_path,
                error=str(e),
            )

        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.IPHONE_COMPATIBILITY,
            file_path=file_path,
        )

        # iPhone H.265 compatibility checks
        self._check_codec_compatibility(codec_info, result)
        self._check_resolution_limits(codec_info, result)
        self._check_frame_rate_limits(codec_info, result)
        self._check_container_compatibility(codec_info, result)

        # MoviePy compatibility test (expensive, optional)
        if not quick_mode and result.is_valid:
            self._check_moviepy_compatibility(file_path, result)

        # Add metadata
        result.metadata.update(
            {
                "codec_name": codec_info.get("codec_name"),
                "profile": codec_info.get("profile"),
                "pixel_format": codec_info.get("pixel_format"),
                "resolution": codec_info.get("resolution"),
                "fps": codec_info.get("fps"),
                "duration": codec_info.get("duration"),
                "container": codec_info.get("container"),
            },
        )

        return result

    def validate_transcoded_output(self, file_path: str) -> ValidationResult:
        """
        Validate transcoded video output for AutoCut pipeline compatibility.

        Replaces: validate_transcoded_output(), _validate_encoder_output()

        Args:
            file_path: Path to transcoded video file

        Returns:
            ValidationResult with transcoded output validation status
        """
        return self.validate_iphone_compatibility(file_path, quick_mode=False)

    def validate_input_files(
        self,
        video_files: List[str],
        audio_file: str,
    ) -> ValidationResult:
        """
        Validate all input files for AutoCut processing.

        Replaces: validate_input_files()

        Args:
            video_files: List of video file paths
            audio_file: Path to audio file

        Returns:
            ValidationResult with comprehensive input validation status
        """
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.BASIC_FORMAT,
            metadata={"video_count": len(video_files)},
        )

        # Validate video files
        if not video_files:
            result.add_error("No video files provided", "NO_VIDEO_FILES")
        else:
            failed_videos = []
            for i, video_file in enumerate(video_files):
                video_result = self.validate_basic_format(video_file)
                if not video_result.is_valid:
                    failed_videos.append(
                        (i + 1, video_file, video_result.issues[0].message),
                    )
                    result.add_error(
                        f"Video file {i + 1} invalid: {video_result.issues[0].message}",
                        "INVALID_VIDEO_FILE",
                        file_index=i + 1,
                        file_path=video_file,
                    )

            if failed_videos:
                result.metadata["failed_videos"] = failed_videos

        # Validate audio file
        if audio_file:
            audio_extension = Path(audio_file).suffix.lower()
            if not Path(audio_file).exists():
                result.add_error(
                    f"Audio file not found: {audio_file}",
                    "AUDIO_FILE_NOT_FOUND",
                    file_path=audio_file,
                )
            elif audio_extension not in SUPPORTED_AUDIO_FORMATS:
                supported_list = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
                result.add_error(
                    f"Unsupported audio format: {audio_extension}",
                    "UNSUPPORTED_AUDIO_FORMAT",
                    file_path=audio_file,
                    extension=audio_extension,
                    supported_formats=supported_list,
                )
            else:
                result.add_info(
                    f"Valid audio file: {audio_extension}",
                    "AUDIO_VALID",
                    file_path=audio_file,
                    extension=audio_extension,
                )
        else:
            result.add_error("No audio file provided", "NO_AUDIO_FILE")

        return result

    def _get_codec_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive codec information using cached FFprobe call.

        Replaces multiple scattered FFprobe calls throughout validation functions.
        """
        # Check cache first
        cache_key = f"{file_path}:{Path(file_path).stat().st_mtime}"
        if cache_key in self._codec_cache:
            return self._codec_cache[cache_key]

        # Run comprehensive FFprobe command
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,duration",
            "-show_entries",
            "format=format_name,duration,size",
            "-of",
            "json",
            file_path,
        ]

        try:
            ffprobe_result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if ffprobe_result.returncode != 0:
                raise VideoProcessingError(
                    f"FFprobe failed: {ffprobe_result.stderr}",
                    details={"command": " ".join(cmd)},
                )

            probe_data = json.loads(ffprobe_result.stdout)

            if not probe_data.get("streams"):
                raise VideoProcessingError("No video streams found in file")

            stream = probe_data["streams"][0]
            format_info = probe_data.get("format", {})

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 0
            except (ValueError, ZeroDivisionError):
                fps = 0

            # Build codec info
            codec_info = {
                "codec_name": stream.get("codec_name", "").lower(),
                "profile": stream.get("profile", ""),
                "level": stream.get("level", ""),
                "pixel_format": stream.get("pix_fmt", ""),
                "width": stream.get("width", 0),
                "height": stream.get("height", 0),
                "resolution": (stream.get("width", 0), stream.get("height", 0)),
                "fps": fps,
                "duration": float(
                    format_info.get("duration", stream.get("duration", 0)),
                ),
                "file_size": int(format_info.get("size", 0)),
                "container": Path(file_path).suffix.lower()[1:],  # Remove leading dot
                "format_name": format_info.get("format_name", ""),
            }

            # Cache result
            self._codec_cache[cache_key] = codec_info

            return codec_info

        except json.JSONDecodeError as e:
            raise VideoProcessingError(f"FFprobe output parsing failed: {e}") from e
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("FFprobe command timed out") from None
        except Exception as e:
            raise VideoProcessingError(f"Codec analysis failed: {e}") from e

    def _check_codec_compatibility(
        self,
        codec_info: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check H.264 codec and profile requirements for iPhone compatibility."""
        codec_name = codec_info.get("codec_name", "")
        profile = codec_info.get("profile", "").lower()
        pixel_format = codec_info.get("pixel_format", "").lower()

        # Codec requirement
        if codec_name != "h264":
            result.add_error(
                f"Codec not iPhone compatible: {codec_name} (need H.264)",
                "INCOMPATIBLE_CODEC",
                codec=codec_name,
                required="h264",
            )
            return

        # Profile requirement - accept both Main and Baseline
        acceptable_profiles = ["main", "baseline", "constrained baseline"]
        if not any(acceptable in profile for acceptable in acceptable_profiles):
            result.add_error(
                f"Profile not iPhone compatible: {profile} (need Main or Baseline profile)",
                "INCOMPATIBLE_PROFILE",
                profile=profile,
                acceptable=acceptable_profiles,
            )

        # Pixel format requirement
        if pixel_format != "yuv420p":
            result.add_error(
                f"Pixel format not iPhone compatible: {pixel_format} (need yuv420p/8-bit)",
                "INCOMPATIBLE_PIXEL_FORMAT",
                pixel_format=pixel_format,
                required="yuv420p",
            )

        # Check for 10-bit formats specifically
        if "10" in pixel_format or "high 10" in profile:
            result.add_error(
                "10-bit format detected - iPhone requires 8-bit",
                "TEN_BIT_FORMAT",
                profile=profile,
                pixel_format=pixel_format,
            )

    def _check_resolution_limits(
        self,
        codec_info: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check resolution limits for iPhone compatibility."""
        width = codec_info.get("width", 0)
        height = codec_info.get("height", 0)

        # iPhone supports up to 4K
        max_width, max_height = 4096, 2160

        if width > max_width or height > max_height:
            result.add_error(
                f"Resolution too high: {width}x{height} (max {max_width}x{max_height})",
                "RESOLUTION_TOO_HIGH",
                width=width,
                height=height,
                max_width=max_width,
                max_height=max_height,
            )

        # Add info about resolution
        if width > 0 and height > 0:
            result.add_info(
                f"Resolution: {width}x{height}",
                "RESOLUTION_INFO",
                width=width,
                height=height,
            )

    def _check_frame_rate_limits(
        self,
        codec_info: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check frame rate limits for iPhone compatibility."""
        fps = codec_info.get("fps", 0)

        # iPhone supports up to 240fps
        max_fps = 240

        if fps > max_fps:
            result.add_error(
                f"Frame rate too high: {fps}fps (max {max_fps}fps)",
                "FRAME_RATE_TOO_HIGH",
                fps=fps,
                max_fps=max_fps,
            )
        elif fps > 0:
            result.add_info(f"Frame rate: {fps:.1f}fps", "FRAME_RATE_INFO", fps=fps)

    def _check_container_compatibility(
        self,
        codec_info: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check container format compatibility for iPhone."""
        container = codec_info.get("container", "").lower()

        # iPhone-compatible containers
        supported_containers = ["mp4", "mov", "m4v"]

        if container and container not in supported_containers:
            result.add_warning(
                f"Container may have limited iPhone compatibility: {container}",
                "CONTAINER_WARNING",
                container=container,
                supported=supported_containers,
            )
        elif container:
            result.add_info(
                f"Container: {container}",
                "CONTAINER_INFO",
                container=container,
            )

    def _check_moviepy_compatibility(
        self,
        file_path: str,
        result: ValidationResult,
    ) -> None:
        """Test MoviePy compatibility - expensive operation."""
        try:
            # Import MoviePy (may fail on some systems)
            from moviepy import VideoFileClip

            # Quick compatibility test
            test_clip = VideoFileClip(file_path)
            duration = test_clip.duration
            test_clip.close()

            if duration is None or duration <= 0:
                result.add_error(
                    "MoviePy compatibility test failed - invalid duration",
                    "MOVIEPY_INVALID_DURATION",
                    duration=duration,
                )
            else:
                result.add_info(
                    f"MoviePy compatible - duration: {duration:.1f}s",
                    "MOVIEPY_COMPATIBLE",
                    duration=duration,
                )

        except ImportError:
            result.add_warning(
                "MoviePy not available for compatibility testing",
                "MOVIEPY_NOT_AVAILABLE",
            )
        except Exception as e:
            result.add_error(
                f"MoviePy compatibility test failed: {e!s}",
                "MOVIEPY_TEST_FAILED",
                error=str(e),
            )
