"""
Pytest configuration and shared fixtures for AutoCut V2 testing.

This file provides common fixtures, test utilities, and configuration
for the AutoCut V2 test suite.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import core modules
from src.hardware.detection import HardwareDetector
from src.video.codec_detection import CodecDetector
from src.video.transcoding import TranscodingService
from src.video.validation import ValidationResult, VideoValidator


@pytest.fixture(scope="session")
def test_media_dir() -> Path:
    """Path to test_media directory."""
    return Path(__file__).parent.parent / "test_media"


@pytest.fixture(scope="session")
def sample_video_files(test_media_dir: Path) -> List[Path]:
    """List of available sample video files."""
    video_extensions = [".mov", ".mp4", ".avi", ".mkv"]
    video_files = []

    if test_media_dir.exists():
        for ext in video_extensions:
            video_files.extend(list(test_media_dir.glob(f"*{ext}")))
            video_files.extend(list(test_media_dir.glob(f"*{ext.upper()}")))

    return sorted(video_files)


@pytest.fixture(scope="session")
def sample_audio_files(test_media_dir: Path) -> List[Path]:
    """List of available sample audio files."""
    audio_extensions = [".mp3", ".wav", ".m4a", ".flac"]
    audio_files = []

    if test_media_dir.exists():
        for ext in audio_extensions:
            audio_files.extend(list(test_media_dir.glob(f"*{ext}")))
            audio_files.extend(list(test_media_dir.glob(f"*{ext.upper()}")))

    return sorted(audio_files)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def video_validator() -> VideoValidator:
    """VideoValidator instance for testing."""
    return VideoValidator()


@pytest.fixture
def codec_detector() -> CodecDetector:
    """CodecDetector instance for testing."""
    return CodecDetector()


@pytest.fixture
def hardware_detector() -> HardwareDetector:
    """HardwareDetector instance for testing."""
    return HardwareDetector()


@pytest.fixture
def transcoding_service() -> TranscodingService:
    """TranscodingService instance for testing."""
    return TranscodingService()


@pytest.fixture
def mock_ffprobe_output() -> Dict[str, Any]:
    """Mock FFprobe output for testing."""
    return {
        "streams": [
            {
                "codec_name": "hevc",
                "codec_long_name": "H.265 / HEVC (High Efficiency Video Coding)",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "24/1",
                "duration": "10.5",
            },
        ],
        "format": {
            "filename": "test_video.mov",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "10.5",
            "size": "5242880",
        },
    }


@pytest.fixture
def mock_validation_result() -> ValidationResult:
    """Mock validation result for testing."""
    return ValidationResult(
        is_valid=True,
        validation_type="basic",
        details={"file_exists": True, "readable": True, "codec_supported": True},
        warnings=[],
        errors=[],
        suggestions=[],
    )


@pytest.fixture
def iphone_h265_file_path(test_media_dir: Path) -> Path:
    """Path to iPhone H.265 test file (if available)."""
    # Look for typical iPhone naming patterns
    iphone_patterns = ["IMG_*.mov", "IMG_*.MOV", "*iphone*.mov", "*iPhone*.mov"]

    for pattern in iphone_patterns:
        files = list(test_media_dir.glob(pattern))
        if files:
            return files[0]

    # Return first .mov file as fallback
    mov_files = list(test_media_dir.glob("*.mov")) + list(test_media_dir.glob("*.MOV"))
    if mov_files:
        return mov_files[0]

    # Create a dummy path if no test files available
    return test_media_dir / "test_iphone.mov"


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset all caches before each test to ensure clean state."""
    # Clear any module-level caches
    if hasattr(CodecDetector, "_cache"):
        CodecDetector._cache.clear()
    if hasattr(HardwareDetector, "_cache"):
        HardwareDetector._cache.clear()
    if hasattr(TranscodingService, "_cache"):
        TranscodingService._cache.clear()


# Test utilities
class TestHelpers:
    """Helper functions for testing."""

    @staticmethod
    def create_mock_video_file(
        temp_dir: Path, filename: str = "test.mp4", size_mb: float = 1.0,
    ) -> Path:
        """Create a mock video file for testing."""
        file_path = temp_dir / filename
        # Create a dummy file with specified size
        with open(file_path, "wb") as f:
            f.write(b"0" * int(size_mb * 1024 * 1024))
        return file_path

    @staticmethod
    def create_mock_audio_file(
        temp_dir: Path, filename: str = "test.mp3", size_mb: float = 0.5,
    ) -> Path:
        """Create a mock audio file for testing."""
        file_path = temp_dir / filename
        # Create a dummy file with specified size
        with open(file_path, "wb") as f:
            f.write(b"0" * int(size_mb * 1024 * 1024))
        return file_path


@pytest.fixture
def test_helpers() -> TestHelpers:
    """Test helper utilities."""
    return TestHelpers()


# Skip conditions
def skip_if_no_gpu():
    """Skip test if no GPU hardware available."""
    detector = HardwareDetector()
    # Use actual method from the implemented HardwareDetector
    optimal_settings = detector.detect_optimal_settings("fast")
    if optimal_settings.get("encoder_type", "").startswith("CPU"):
        pytest.skip("GPU hardware not available")


def skip_if_no_test_media():
    """Skip test if no test media files available."""
    test_media_dir = Path(__file__).parent.parent / "test_media"
    if not test_media_dir.exists() or not any(test_media_dir.iterdir()):
        pytest.skip("No test media files available")


# Custom markers
def _has_gpu():
    """Check if GPU hardware is available."""
    try:
        detector = HardwareDetector()
        optimal_settings = detector.detect_optimal_settings("fast")
        return not optimal_settings.get("encoder_type", "").startswith("CPU")
    except:
        return False


pytest.mark.gpu = pytest.mark.skipif(
    not _has_gpu(), reason="GPU hardware not available",
)

pytest.mark.media_required = pytest.mark.skipif(
    not (Path(__file__).parent.parent / "test_media").exists(),
    reason="Test media directory not available",
)
