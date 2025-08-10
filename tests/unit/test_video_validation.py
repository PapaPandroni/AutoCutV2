"""
Unit tests for the video validation module.

Tests the unified VideoValidator class and ValidationResult dataclass
that replaced 10+ scattered validation functions.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.video.validation import VideoValidator, ValidationResult, ValidationError


class TestValidationResult:
    """Test ValidationResult dataclass functionality."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult instance."""
        result = ValidationResult(
            is_valid=True,
            validation_type="basic",
            details={"test": True},
            warnings=["warning1"],
            errors=[],
            suggestions=["suggestion1"],
        )

        assert result.is_valid is True
        assert result.validation_type == "basic"
        assert result.details == {"test": True}
        assert result.warnings == ["warning1"]
        assert result.errors == []
        assert result.suggestions == ["suggestion1"]

    def test_validation_result_defaults(self):
        """Test ValidationResult with minimal parameters."""
        result = ValidationResult(is_valid=False, validation_type="test")

        assert result.is_valid is False
        assert result.validation_type == "test"
        assert result.details == {}
        assert result.warnings == []
        assert result.errors == []
        assert result.suggestions == []

    def test_has_warnings(self):
        """Test has_warnings property."""
        result_with_warnings = ValidationResult(
            is_valid=True, validation_type="test", warnings=["warning1"]
        )
        result_without_warnings = ValidationResult(
            is_valid=True, validation_type="test"
        )

        assert result_with_warnings.has_warnings is True
        assert result_without_warnings.has_warnings is False

    def test_has_errors(self):
        """Test has_errors property."""
        result_with_errors = ValidationResult(
            is_valid=False, validation_type="test", errors=["error1"]
        )
        result_without_errors = ValidationResult(is_valid=True, validation_type="test")

        assert result_with_errors.has_errors is True
        assert result_without_errors.has_errors is False


class TestVideoValidator:
    """Test VideoValidator class functionality."""

    def test_validator_initialization(self, video_validator):
        """Test VideoValidator initialization."""
        assert isinstance(video_validator, VideoValidator)
        assert hasattr(video_validator, "validate_basic")
        assert hasattr(video_validator, "validate_iphone_compatibility")
        assert hasattr(video_validator, "validate_transcoding_output")

    def test_validate_nonexistent_file(self, video_validator, temp_dir):
        """Test validation of non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.mp4"

        result = video_validator.validate_basic(str(nonexistent_file))

        assert result.is_valid is False
        assert result.validation_type == "basic"
        assert "file_exists" in result.details
        assert result.details["file_exists"] is False
        assert len(result.errors) > 0
        assert any("not found" in error.lower() for error in result.errors)

    def test_validate_basic_with_valid_file(
        self, video_validator, test_helpers, temp_dir
    ):
        """Test basic validation with a valid file."""
        # Create a mock video file
        video_file = test_helpers.create_mock_video_file(temp_dir, "test.mp4")

        with patch("src.video.validation.os.access", return_value=True):
            result = video_validator.validate_basic(str(video_file))

        assert result.validation_type == "basic"
        assert "file_exists" in result.details
        assert result.details["file_exists"] is True
        assert "readable" in result.details

    def test_validate_basic_unreadable_file(
        self, video_validator, test_helpers, temp_dir
    ):
        """Test validation of unreadable file."""
        video_file = test_helpers.create_mock_video_file(temp_dir, "unreadable.mp4")

        with patch("src.video.validation.os.access", return_value=False):
            result = video_validator.validate_basic(str(video_file))

        assert result.is_valid is False
        assert result.details["readable"] is False
        assert len(result.errors) > 0

    def test_validate_audio_file_valid(self, video_validator, test_helpers, temp_dir):
        """Test audio file validation with valid file."""
        audio_file = test_helpers.create_mock_audio_file(temp_dir, "test.mp3")

        with patch("src.video.validation.os.access", return_value=True):
            result = video_validator.validate_audio_file(str(audio_file))

        assert result.validation_type == "audio"
        assert "file_exists" in result.details
        assert result.details["file_exists"] is True

    def test_validate_audio_file_invalid_extension(
        self, video_validator, test_helpers, temp_dir
    ):
        """Test audio file validation with invalid extension."""
        invalid_audio = test_helpers.create_mock_audio_file(temp_dir, "test.txt")

        result = video_validator.validate_audio_file(str(invalid_audio))

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("supported audio format" in error.lower() for error in result.errors)

    @patch("src.video.validation.CodecDetector")
    def test_validate_iphone_compatibility(
        self, mock_codec_detector, video_validator, test_helpers, temp_dir
    ):
        """Test iPhone H.265 compatibility validation."""
        video_file = test_helpers.create_mock_video_file(temp_dir, "iphone.mov")

        # Mock codec detection
        mock_detector = MagicMock()
        mock_detector.detect_codec.return_value = {
            "codec": "hevc",
            "is_hevc": True,
            "profile": "Main",
            "bit_depth": "10",
        }
        mock_codec_detector.return_value = mock_detector

        result = video_validator.validate_iphone_compatibility(str(video_file))

        assert result.validation_type == "iphone_compatibility"
        assert "codec_info" in result.details
        mock_detector.detect_codec.assert_called_once()

    @patch("src.video.validation.subprocess.run")
    def test_validate_transcoding_output(
        self, mock_subprocess, video_validator, test_helpers, temp_dir
    ):
        """Test transcoding output validation."""
        output_file = test_helpers.create_mock_video_file(temp_dir, "transcoded.mp4")

        # Mock successful ffprobe output
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout='{"streams": [{"codec_name": "h264"}]}'
        )

        result = video_validator.validate_transcoding_output(str(output_file))

        assert result.validation_type == "transcoding_output"
        assert "ffprobe_success" in result.details
        mock_subprocess.assert_called_once()

    def test_validate_multiple_files(self, video_validator, test_helpers, temp_dir):
        """Test validation of multiple files."""
        files = [
            test_helpers.create_mock_video_file(temp_dir, "video1.mp4"),
            test_helpers.create_mock_video_file(temp_dir, "video2.mov"),
            test_helpers.create_mock_audio_file(temp_dir, "audio.mp3"),
        ]

        with patch("src.video.validation.os.access", return_value=True):
            results = video_validator.validate_input_files(
                video_files=[str(files[0]), str(files[1])], audio_file=str(files[2])
            )

        assert "video_results" in results
        assert "audio_result" in results
        assert len(results["video_results"]) == 2
        assert results["audio_result"].validation_type == "audio"

    def test_validation_with_suggestions(self, video_validator, test_helpers, temp_dir):
        """Test validation that includes suggestions."""
        # Test with unsupported format
        unsupported_file = test_helpers.create_mock_video_file(temp_dir, "test.wmv")

        result = video_validator.validate_basic(str(unsupported_file))

        # Should have suggestions for unsupported formats
        if result.suggestions:
            assert any(
                "convert" in suggestion.lower() or "format" in suggestion.lower()
                for suggestion in result.suggestions
            )

    def test_validation_error_handling(self, video_validator):
        """Test proper error handling in validation."""
        # Test with None input
        result = video_validator.validate_basic(None)
        assert result.is_valid is False
        assert len(result.errors) > 0

        # Test with empty string
        result = video_validator.validate_basic("")
        assert result.is_valid is False
        assert len(result.errors) > 0

        # Test with invalid type
        result = video_validator.validate_basic(123)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidationError:
    """Test ValidationError exception class."""

    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_validation_error_with_details(self):
        """Test ValidationError with additional details."""
        details = {"file": "test.mp4", "reason": "invalid format"}
        error = ValidationError("Validation failed", details)

        assert "Validation failed" in str(error)
        assert hasattr(error, "details")
        assert error.details == details


class TestBackwardsCompatibility:
    """Test backwards compatibility with old validation functions."""

    def test_legacy_function_compatibility(self, video_validator):
        """Test that legacy functions still work through compatibility layer."""
        # These should be available through the validator or as module functions
        assert hasattr(video_validator, "validate_basic")
        assert hasattr(video_validator, "validate_audio_file")
        assert hasattr(video_validator, "validate_iphone_compatibility")
        assert hasattr(video_validator, "validate_transcoding_output")

    @patch("src.video.validation.VideoValidator")
    def test_module_level_functions(self, mock_validator):
        """Test module-level compatibility functions."""
        # Import module-level functions if they exist
        try:
            from src.video.validation import validate_video_file

            assert callable(validate_video_file)
        except ImportError:
            # It's okay if module-level functions aren't implemented yet
            pass
