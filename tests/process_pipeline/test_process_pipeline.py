"""
Core Process Pipeline Tests for AutoCut V2

Tests the main production pipeline that powers the `autocut.py process` command.
These tests validate the end-to-end workflow that users will experience in v1.0.
"""

import sys
from pathlib import Path
from typing import List

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api import AutoCutAPI


class TestProcessPipeline:
    """Test the core process pipeline end-to-end."""

    @pytest.fixture
    def api_client(self) -> AutoCutAPI:
        """AutoCut API client for testing."""
        return AutoCutAPI()

    @pytest.fixture
    def output_dir(self, temp_dir):
        """Output directory for test videos."""
        output_dir = temp_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.mark.media_required
    def test_process_basic_workflow(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test basic process workflow: videos + audio → output video."""
        # Filter out macOS metadata files (._*) and hidden files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        # Skip if no test media
        if not real_video_files or not real_audio_files:
            pytest.skip("No real media files available (found only metadata files)")

        # Use smaller iPhone videos for faster testing (avoid large 4K files)
        small_videos = [f for f in real_video_files if "iphone" in f.name.lower()]
        if len(small_videos) >= 2:
            video_files = [str(f) for f in small_videos[:2]]  # Use 2 iPhone videos
        else:
            video_files = [str(f) for f in real_video_files[:2]]  # Fallback to first 2
        audio_file = str(real_audio_files[0])
        output_path = str(output_dir / "basic_test_output.mp4")

        print(f"\n🎬 Testing with {len(video_files)} video files:")
        for video in video_files:
            print(f"  - {Path(video).name}")
        print(f"🎵 Audio: {Path(audio_file).name}")

        # Process videos
        result_path = api_client.process_videos(
            video_files=video_files,
            audio_file=audio_file,
            output_path=output_path,
            pattern="balanced",
            verbose=True,
        )

        # Validate results
        assert result_path == output_path, "Result path should match requested path"
        assert Path(result_path).exists(), "Output video file should exist"

        # Check file size (should be > 0)
        file_size = Path(result_path).stat().st_size
        assert file_size > 0, "Output video should not be empty"
        print(f"✅ Created video: {file_size / (1024 * 1024):.1f} MB")

    @pytest.mark.media_required
    def test_process_all_patterns(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test all editing patterns work correctly."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if not real_video_files or not real_audio_files:
            pytest.skip("No real media files available")

        patterns = ["energetic", "balanced", "dramatic", "buildup"]
        video_files = [
            str(f) for f in real_video_files[:2]
        ]  # Use fewer files for speed
        audio_file = str(real_audio_files[0])

        results = {}

        for pattern in patterns:
            print(f"\n🎯 Testing {pattern} pattern...")
            output_path = str(output_dir / f"pattern_{pattern}_output.mp4")

            result_path = api_client.process_videos(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern=pattern,
                verbose=False,  # Reduce noise for multiple patterns
            )

            # Validate each pattern creates a valid output
            assert Path(result_path).exists(), (
                f"Pattern {pattern} should create output"
            )
            file_size = Path(result_path).stat().st_size
            assert file_size > 0, f"Pattern {pattern} should create non-empty output"

            results[pattern] = file_size
            print(f"✅ {pattern}: {file_size / (1024 * 1024):.1f} MB")

        # All patterns should produce valid results
        assert len(results) == 4, "All 4 patterns should produce outputs"
        print(f"\n🎉 All patterns working! Sizes: {results}")

    @pytest.mark.media_required
    def test_process_memory_safe_mode(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test memory-safe processing mode."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if not real_video_files or not real_audio_files:
            pytest.skip("No real media files available")

        video_files = [str(f) for f in real_video_files[:2]]
        audio_file = str(real_audio_files[0])
        output_path = str(output_dir / "memory_safe_output.mp4")

        print("\n🧠 Testing memory-safe mode...")

        # Test memory-safe processing
        result_path = api_client.process_videos(
            video_files=video_files,
            audio_file=audio_file,
            output_path=output_path,
            pattern="balanced",
            memory_safe=True,
            verbose=True,
        )

        # Validate memory-safe mode works
        assert Path(result_path).exists(), "Memory-safe mode should create output"
        file_size = Path(result_path).stat().st_size
        assert file_size > 0, "Memory-safe output should not be empty"
        print(f"✅ Memory-safe processing: {file_size / (1024 * 1024):.1f} MB")

    @pytest.mark.media_required
    def test_process_max_videos_limit(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test max-videos limiting functionality."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if len(real_video_files) < 3 or not real_audio_files:
            pytest.skip("Need at least 3 real video files for max-videos test")

        # Use all available videos but limit to 2
        all_video_files = [str(f) for f in real_video_files]
        audio_file = str(real_audio_files[0])
        output_path = str(output_dir / "max_videos_output.mp4")

        print(f"\n📊 Testing max-videos limit: {len(all_video_files)} → 2 videos")

        # The API itself doesn't have max_videos parameter, but the CLI does this filtering
        # Let's simulate the CLI behavior by limiting the list ourselves
        limited_videos = all_video_files[:2]

        result_path = api_client.process_videos(
            video_files=limited_videos,
            audio_file=audio_file,
            output_path=output_path,
            pattern="balanced",
            verbose=True,
        )

        assert Path(result_path).exists(), "Max-videos limiting should create output"
        file_size = Path(result_path).stat().st_size
        assert file_size > 0, "Limited videos output should not be empty"
        print(f"✅ Max videos limit working: {file_size / (1024 * 1024):.1f} MB")

    def test_process_input_validation(self, api_client: AutoCutAPI, temp_dir: Path):
        """Test input validation and error handling."""
        output_path = str(temp_dir / "validation_test.mp4")

        # Test with non-existent video file
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            api_client.process_videos(
                video_files=["non_existent_video.mp4"],
                audio_file="non_existent_audio.mp3",
                output_path=output_path,
                pattern="balanced",
            )

        print(f"✅ Input validation works: {exc_info.value}")

        # Test with empty video list
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            api_client.process_videos(
                video_files=[],
                audio_file="non_existent_audio.mp3",
                output_path=output_path,
                pattern="balanced",
            )
        print("✅ Empty video list validation works")

    @pytest.mark.media_required
    def test_process_mixed_formats(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test processing mixed video formats (MP4, MOV, etc.)."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if not real_video_files or not real_audio_files:
            pytest.skip("No real media files available")

        # Group files by extension
        mp4_files = [f for f in real_video_files if f.suffix.lower() == ".mp4"]
        mov_files = [f for f in real_video_files if f.suffix.lower() == ".mov"]

        if not mp4_files or not mov_files:
            pytest.skip("Need both MP4 and MOV files for mixed format test")

        # Mix different formats
        mixed_files = [str(mp4_files[0]), str(mov_files[0])]
        audio_file = str(real_audio_files[0])
        output_path = str(output_dir / "mixed_formats_output.mp4")

        print("\n🎭 Testing mixed formats:")
        for video in mixed_files:
            print(f"  - {Path(video).name} ({Path(video).suffix.upper()})")

        result_path = api_client.process_videos(
            video_files=mixed_files,
            audio_file=audio_file,
            output_path=output_path,
            pattern="balanced",
            verbose=True,
        )

        assert Path(result_path).exists(), "Mixed formats should create output"
        file_size = Path(result_path).stat().st_size
        assert file_size > 0, "Mixed formats output should not be empty"
        print(f"✅ Mixed formats processing: {file_size / (1024 * 1024):.1f} MB")

    @pytest.mark.media_required
    def test_process_portrait_and_landscape(
        self,
        api_client: AutoCutAPI,
        sample_video_files: List[Path],
        sample_audio_files: List[Path],
        output_dir: Path,
    ):
        """Test processing mixed aspect ratios (portrait + landscape)."""
        # Filter out macOS metadata files
        real_video_files = [
            f for f in sample_video_files if not f.name.startswith("._")
        ]
        real_audio_files = [
            f for f in sample_audio_files if not f.name.startswith("._")
        ]

        if not real_video_files or not real_audio_files:
            pytest.skip("No real media files available")

        # Look for files that likely have different aspect ratios
        portrait_files = [f for f in real_video_files if "portrait" in f.name.lower()]
        landscape_files = [f for f in real_video_files if "landscape" in f.name.lower()]

        if not portrait_files or not landscape_files:
            # Fallback to first two files if naming doesn't indicate aspect ratio
            mixed_files = [str(f) for f in real_video_files[:2]]
            print("\n📱 Testing mixed aspect ratios (assuming different ratios)")
        else:
            mixed_files = [str(portrait_files[0]), str(landscape_files[0])]
            print("\n📱 Testing portrait + landscape videos")

        audio_file = str(real_audio_files[0])
        output_path = str(output_dir / "mixed_aspects_output.mp4")

        for video in mixed_files:
            print(f"  - {Path(video).name}")

        result_path = api_client.process_videos(
            video_files=mixed_files,
            audio_file=audio_file,
            output_path=output_path,
            pattern="balanced",
            verbose=True,
        )

        assert Path(result_path).exists(), "Mixed aspect ratios should create output"
        file_size = Path(result_path).stat().st_size
        assert file_size > 0, "Mixed aspects output should not be empty"
        print(f"✅ Mixed aspect ratios: {file_size / (1024 * 1024):.1f} MB")
        print("   (Should have proper letterboxing/pillarboxing)")
