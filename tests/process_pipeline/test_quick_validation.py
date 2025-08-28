"""
Quick validation tests for core pipeline functionality.

Fast tests to validate the system is working without long processing times.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api import AutoCutAPI


class TestQuickValidation:
    """Quick validation tests for core functionality."""

    @pytest.fixture
    def api_client(self) -> AutoCutAPI:
        """AutoCut API client for testing."""
        return AutoCutAPI()

    def test_api_initialization(self, api_client: AutoCutAPI):
        """Test that the API can be initialized successfully."""
        assert api_client is not None
        assert hasattr(api_client, "process_videos")
        assert hasattr(api_client, "validate_video")
        assert hasattr(api_client, "get_system_info")
        print("✅ API initialized successfully")

    def test_system_info_retrieval(self, api_client: AutoCutAPI):
        """Test system information retrieval."""
        system_info = api_client.get_system_info()

        assert system_info is not None
        # Should have some basic system information
        assert hasattr(system_info, "cpu_cores") or "cpu_cores" in str(system_info)
        print(f"✅ System info retrieved: {system_info}")

    @pytest.mark.media_required
    def test_media_files_detected(self, sample_video_files, sample_audio_files):
        """Test that real media files are detected (not just metadata)."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        assert len(real_video_files) > 0, "Should have real video files available"
        assert len(real_audio_files) > 0, "Should have real audio files available"

        print(
            f"✅ Found {len(real_video_files)} video files and {len(real_audio_files)} audio files"
        )
        for video in real_video_files[:3]:
            print(f"  📹 {video.name} ({video.stat().st_size / (1024 * 1024):.1f} MB)")
        for audio in real_audio_files[:1]:
            print(f"  🎵 {audio.name} ({audio.stat().st_size / (1024 * 1024):.1f} MB)")

    def test_input_validation_errors(self, api_client: AutoCutAPI, temp_dir):
        """Test that input validation properly rejects invalid inputs."""
        output_path = str(temp_dir / "validation_test.mp4")

        # Test with non-existent files
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            api_client.process_videos(
                video_files=["nonexistent_video.mp4"],
                audio_file="nonexistent_audio.mp3",
                output_path=output_path,
                pattern="balanced",
            )
        print("✅ Input validation correctly rejects invalid files")

        # Test with empty video list
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            api_client.process_videos(
                video_files=[],
                audio_file="nonexistent_audio.mp3",
                output_path=output_path,
                pattern="balanced",
            )
        print("✅ Input validation correctly rejects empty file list")

    def test_valid_patterns_accepted(self, api_client: AutoCutAPI):
        """Test that all valid patterns are accepted by the API."""
        patterns = ["energetic", "balanced", "dramatic", "buildup"]

        # We can't actually process without real files, but we can test the pattern validation
        # by checking that these patterns don't immediately cause parameter errors
        for pattern in patterns:
            try:
                # This will fail on missing files, but should not fail on invalid pattern
                api_client.process_videos(
                    video_files=["fake.mp4"],
                    audio_file="fake.mp3",
                    output_path="fake.mp4",
                    pattern=pattern,
                )
            except (ValueError, RuntimeError) as e:
                # Should fail on file validation, not pattern validation
                error_msg = str(e).lower()
                assert "pattern" not in error_msg, f"Pattern {pattern} should be valid"
                assert any(
                    word in error_msg for word in ["file", "not", "found", "exist"]
                ), f"Should fail on file issues, not pattern for {pattern}"

        print("✅ All valid patterns accepted: " + ", ".join(patterns))

    @pytest.mark.media_required
    def test_audio_analysis_basic(self, sample_audio_files):
        """Test basic audio analysis functionality."""
        # Filter out metadata files
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if not real_audio_files:
            pytest.skip("No real audio files available")

        audio_file = str(real_audio_files[0])

        # Import the audio analyzer directly for a quick test
        from audio_analyzer import analyze_audio

        try:
            result = analyze_audio(audio_file)
            assert isinstance(result, dict), "Audio analysis should return a dictionary"
            assert "bpm" in result, "Should detect BPM"
            assert "beats" in result, "Should detect beats"
            assert "duration" in result, "Should detect duration"
            assert len(result["beats"]) > 0, "Should find some beats"

            print("✅ Audio analysis working:")
            print(f"   🎵 File: {Path(audio_file).name}")
            print(f"   🥁 BPM: {result['bpm']:.1f}")
            print(f"   ⏱️ Duration: {result['duration']:.1f}s")
            print(f"   🎯 Beats: {len(result['beats'])}")

        except Exception as e:
            pytest.fail(f"Audio analysis failed: {e}")
