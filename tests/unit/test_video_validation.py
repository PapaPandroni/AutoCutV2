"""
Simplified unit tests for the video validation module.

Tests the unified VideoValidator class and ValidationResult dataclass
that replaced 10+ scattered validation functions.

This is a simplified version that focuses on testing the current API
without complex mocking or outdated method calls.
"""

import pytest

from src.video.validation import (
    ValidationError,
    ValidationResult,
    ValidationType,
    ValidationSeverity,
    ValidationIssue,
    VideoValidator,
)


class TestValidationResult:
    """Test ValidationResult dataclass functionality."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult instance."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.BASIC_FORMAT,
            file_path="test.mp4",
            metadata={"test": True},
            suggestions=["suggestion1"],
        )
        
        # Add a warning to test the issues system
        result.add_warning("warning1", "WARN_001")

        assert result.is_valid is True
        assert result.validation_type == ValidationType.BASIC_FORMAT
        assert result.metadata == {"test": True}
        assert len(result.get_warnings()) == 1
        assert result.get_warnings()[0].message == "warning1"
        assert len(result.get_errors()) == 0
        assert result.suggestions == ["suggestion1"]

    def test_validation_result_defaults(self):
        """Test ValidationResult with minimal parameters."""
        result = ValidationResult(
            is_valid=False, 
            validation_type=ValidationType.BASIC_FORMAT
        )

        assert result.is_valid is False
        assert result.validation_type == ValidationType.BASIC_FORMAT
        assert result.metadata == {}
        assert result.get_warnings() == []
        assert result.get_errors() == []
        assert result.suggestions == []

    def test_add_warning(self):
        """Test warnings functionality."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.BASIC_FORMAT,
        )
        result.add_warning("warning1", "WARN_001")
        
        assert len(result.get_warnings()) > 0
        assert result.get_warnings()[0].message == "warning1"

    def test_add_error(self):
        """Test error functionality."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.BASIC_FORMAT,
        )
        result.add_error("error1", "ERR_001")
        
        # Adding an error should make the result invalid
        assert result.is_valid is False
        assert len(result.get_errors()) > 0
        assert result.get_errors()[0].message == "error1"

    def test_get_summary(self):
        """Test summary generation."""
        result = ValidationResult(
            is_valid=True,
            validation_type=ValidationType.BASIC_FORMAT,
        )
        
        # Valid result
        assert "Valid" in result.get_summary()
        
        # Add warning and test
        result.add_warning("warning", "WARN_001")
        assert "warning" in result.get_summary()
        
        # Make invalid and test
        result.add_error("error", "ERR_001")
        assert "Invalid" in result.get_summary()


class TestVideoValidator:
    """Test VideoValidator class functionality."""

    def test_validator_initialization(self):
        """Test VideoValidator initialization."""
        validator = VideoValidator()
        assert isinstance(validator, VideoValidator)
        assert hasattr(validator, "validate_basic_format")
        assert hasattr(validator, "validate_iphone_compatibility")
        assert hasattr(validator, "validate_transcoded_output")
        assert hasattr(validator, "validate_input_files")

    def test_validate_basic_format_nonexistent_file(self):
        """Test basic format validation with nonexistent file."""
        validator = VideoValidator()
        result = validator.validate_basic_format("nonexistent.mp4")
        
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.BASIC_FORMAT
        # File doesn't exist, so should be invalid
        assert result.is_valid is False
        assert len(result.get_errors()) > 0

    def test_validate_iphone_compatibility_nonexistent_file(self):
        """Test iPhone compatibility validation with nonexistent file."""
        validator = VideoValidator()
        result = validator.validate_iphone_compatibility("nonexistent.mp4")
        
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.IPHONE_COMPATIBILITY
        # File doesn't exist, so should be invalid
        assert result.is_valid is False

    def test_validate_transcoded_output_nonexistent_file(self):
        """Test transcoded output validation with nonexistent file.""" 
        validator = VideoValidator()
        result = validator.validate_transcoded_output("nonexistent.mp4")
        
        assert isinstance(result, ValidationResult)
        # validate_transcoded_output delegates to validate_iphone_compatibility
        assert result.validation_type == ValidationType.IPHONE_COMPATIBILITY
        # File doesn't exist, so should be invalid
        assert result.is_valid is False

    def test_validate_input_files_empty_lists(self):
        """Test input files validation with empty lists."""
        validator = VideoValidator()
        result = validator.validate_input_files(
            video_files=[], 
            audio_file="nonexistent.mp3"
        )
        
        assert isinstance(result, ValidationResult)
        # Should be invalid due to empty video files list
        assert result.is_valid is False


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_issue_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            code="TEST_001",
            context={"file": "test.mp4"}
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error"
        assert issue.code == "TEST_001"
        assert issue.context == {"file": "test.mp4"}

    def test_issue_string_representation(self):
        """Test ValidationIssue string representation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            code="WARN_001"
        )
        
        str_repr = str(issue)
        assert "Test warning" in str_repr
        assert "⚠️" in str_repr  # Warning emoji should be present


class TestValidationType:
    """Test ValidationType enum."""

    def test_validation_types_exist(self):
        """Test that all expected validation types exist."""
        assert ValidationType.BASIC_FORMAT
        assert ValidationType.IPHONE_COMPATIBILITY
        assert ValidationType.TRANSCODING_OUTPUT
        assert ValidationType.MOVIEPY_COMPATIBILITY
        assert ValidationType.HARDWARE_ENCODER

    def test_validation_type_values(self):
        """Test validation type string values."""
        assert ValidationType.BASIC_FORMAT.value == "basic_format"
        assert ValidationType.IPHONE_COMPATIBILITY.value == "iphone_compatibility"


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_severity_levels_exist(self):
        """Test that all expected severity levels exist."""
        assert ValidationSeverity.INFO
        assert ValidationSeverity.WARNING
        assert ValidationSeverity.ERROR
        assert ValidationSeverity.CRITICAL

    def test_severity_values(self):
        """Test severity string values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"