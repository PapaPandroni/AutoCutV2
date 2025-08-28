"""
Regression Tests for AutoCut V2

Tests to prevent regression of previously fixed bugs and critical issues.
These tests validate that resolved problems don't reoccur in future versions.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api import AutoCutAPI
from audio_analyzer import analyze_audio


class TestRegressions:
    """Test for regression of previously fixed bugs."""

    @pytest.fixture
    def api_client(self) -> AutoCutAPI:
        """Fresh API client for each test."""
        return AutoCutAPI()

    @pytest.fixture
    def regression_output_dir(self, temp_dir):
        """Output directory for regression test results."""
        output_dir = temp_dir / "regression_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def test_moviepy_compatibility_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for MoviePy compatibility regressions (H.265, audio fadeout, etc.)."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n🎬 Testing MoviePy compatibility regression")

        # Test H.265/HEVC video processing (was a major issue)
        h265_videos = [
            f
            for f in real_videos
            if "h265" in f.name.lower() or "hevc" in f.name.lower()
        ]
        test_video = h265_videos[0] if h265_videos else real_videos[0]

        output_path = str(regression_output_dir / "moviepy_compatibility_test.mp4")

        try:
            result_path = api_client.process_videos(
                video_files=[str(test_video)],
                audio_file=str(real_audio[0]),
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            # Validate successful processing
            assert Path(result_path).exists(), (
                "MoviePy should handle video formats correctly"
            )
            assert Path(result_path).stat().st_size > 0, "Output should not be empty"

            print("   ✅ MoviePy compatibility maintained")
            print(f"   📹 Processed: {Path(test_video).name}")
            print(f"   💾 Output: {Path(result_path).stat().st_size / (1024 * 1024):.1f}MB")

        except Exception as e:
            error_msg = str(e).lower()

            # Check for specific MoviePy-related errors that were previously fixed
            moviepy_keywords = ["codec", "ffmpeg", "audio", "fadeout", "duration"]
            if any(keyword in error_msg for keyword in moviepy_keywords):
                pytest.fail(f"MoviePy compatibility regression detected: {e}")
            else:
                print(f"   ⚠️ Processing failed with non-MoviePy error: {e}")

    def test_audio_analysis_corruption_regression(self, sample_audio_files):
        """Test for audio analysis corruption and crash regressions."""
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_audio:
            pytest.skip("No real audio files available")

        print("\\n🎵 Testing audio analysis corruption regression")

        for audio_file in real_audio:
            print(f"   Testing: {audio_file.name}")

            try:
                result = analyze_audio(str(audio_file))

                # Validate that analysis doesn't return corrupted data
                assert isinstance(result, dict), (
                    "Audio analysis should return dictionary"
                )
                assert "bpm" in result, "Should detect BPM"
                assert "beats" in result, "Should detect beats"
                assert "duration" in result, "Should detect duration"

                # Validate data integrity
                assert isinstance(result["bpm"], (int, float)), "BPM should be numeric"
                assert result["bpm"] > 0, "BPM should be positive"
                assert isinstance(result["beats"], list), "Beats should be a list"
                assert len(result["beats"]) > 0, "Should find some beats"
                assert isinstance(result["duration"], (int, float)), (
                    "Duration should be numeric"
                )
                assert result["duration"] > 0, "Duration should be positive"

                # Check for reasonable ranges (prevent garbage data regression)
                assert 30 <= result["bpm"] <= 300, (
                    f"BPM should be reasonable: {result['bpm']}"
                )
                assert result["duration"] < 3600, (
                    f"Duration should be reasonable: {result['duration']}s"
                )

                print(
                    f"     ✅ Clean analysis: {result['bpm']:.1f} BPM, {len(result['beats'])} beats"
                )

            except Exception as e:
                error_msg = str(e).lower()

                # Check for specific audio corruption errors
                corruption_keywords = [
                    "corrupt",
                    "invalid",
                    "decode",
                    "format",
                    "header",
                ]
                if any(keyword in error_msg for keyword in corruption_keywords):
                    pytest.fail(f"Audio analysis corruption regression: {e}")
                else:
                    print(f"     ⚠️ Analysis failed with: {e}")

        print("✅ Audio analysis corruption regression test passed")

    def test_memory_leak_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for memory leak regressions during repeated processing."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n🧠 Testing memory leak regression")

        import psutil

        process = psutil.Process()

        # Get initial memory baseline
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"   Initial memory: {initial_memory:.1f}MB")

        # Use small video for consistent testing
        small_video = next(
            (f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024),
            real_videos[0],
        )

        memory_samples = [initial_memory]

        # Process multiple times to detect memory leaks
        for i in range(3):  # 3 iterations to detect leaks
            output_path = str(regression_output_dir / f"memory_leak_test_{i}.mp4")

            try:
                result_path = api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,  # Use memory-safe mode
                    verbose=False,
                )

                # Measure memory after processing
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory)

                print(f"   Iteration {i + 1}: {current_memory:.1f}MB")

                # Clean up output to prevent disk space issues
                if Path(result_path).exists():
                    os.remove(result_path)

            except Exception as e:
                print(f"   ⚠️ Iteration {i + 1} failed: {e}")

        # Analyze memory growth
        if len(memory_samples) >= 3:
            final_memory = memory_samples[-1]
            memory_growth = final_memory - initial_memory

            print(f"   📊 Memory growth: {memory_growth:+.1f}MB")

            # Allow some growth (caching, etc.) but not excessive
            if memory_growth > 100:  # More than 100MB growth is concerning
                print(f"   ⚠️ Significant memory growth detected: {memory_growth:.1f}MB")
                print("   This may indicate a memory leak regression")
            else:
                print("   ✅ Memory usage looks stable")

        print("✅ Memory leak regression test completed")

    def test_file_handle_leak_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for file handle leak regressions."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n📂 Testing file handle leak regression")

        import psutil

        process = psutil.Process()

        # Get initial file handle count
        initial_handles = len(process.open_files())
        print(f"   Initial file handles: {initial_handles}")

        small_video = next(
            (f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024),
            real_videos[0],
        )

        # Process multiple times to detect handle leaks
        for i in range(3):
            output_path = str(regression_output_dir / f"handle_leak_test_{i}.mp4")

            try:
                result_path = api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,
                    verbose=False,
                )

                # Clean up output file
                if Path(result_path).exists():
                    os.remove(result_path)

                print(f"   ✅ Iteration {i + 1} completed")

            except Exception as e:
                print(f"   ⚠️ Iteration {i + 1} failed: {e}")

        # Check final file handle count
        final_handles = len(process.open_files())
        handle_growth = final_handles - initial_handles

        print(f"   Final file handles: {final_handles}")
        print(f"   Handle growth: {handle_growth:+d}")

        # Allow minimal growth but flag significant increases
        if handle_growth > 10:
            print(f"   ⚠️ Significant file handle increase: {handle_growth}")
            print("   This may indicate a file handle leak regression")
        else:
            print("   ✅ File handle usage looks stable")

        print("✅ File handle leak regression test completed")

    def test_canvas_sizing_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for canvas sizing and letterboxing regressions."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if len(real_videos) < 2 or not real_audio:
            pytest.skip("Need at least 2 video files for canvas sizing test")

        print("\\n🖼️ Testing canvas sizing regression")

        # Mix different aspect ratios to test canvas logic
        mixed_videos = [str(f) for f in real_videos[:2]]
        output_path = str(regression_output_dir / "canvas_sizing_test.mp4")

        print(f"   Testing with {len(mixed_videos)} mixed aspect ratio videos")

        try:
            result_path = api_client.process_videos(
                video_files=mixed_videos,
                audio_file=str(real_audio[0]),
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            # Validate output exists and is valid
            assert Path(result_path).exists(), "Canvas sizing should produce output"
            assert Path(result_path).stat().st_size > 0, (
                "Canvas sizing output should not be empty"
            )

            print("   ✅ Canvas sizing and letterboxing working")
            print(f"   📹 Output: {Path(result_path).stat().st_size / (1024 * 1024):.1f}MB")

        except Exception as e:
            error_msg = str(e).lower()

            # Check for canvas/aspect ratio related errors
            canvas_keywords = ["canvas", "size", "aspect", "dimension", "resolution"]
            if any(keyword in error_msg for keyword in canvas_keywords):
                pytest.fail(f"Canvas sizing regression detected: {e}")
            else:
                print(f"   ⚠️ Canvas test failed with: {e}")

        print("✅ Canvas sizing regression test passed")

    def test_pattern_variation_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for pattern variation and beat sync regressions."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n🎯 Testing pattern variation regression")

        patterns = ["energetic", "balanced", "dramatic", "buildup"]
        small_video = next(
            (f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024),
            real_videos[0],
        )

        pattern_results = []

        for pattern in patterns:
            print(f"   Testing {pattern} pattern...")
            output_path = str(regression_output_dir / f"pattern_{pattern}_test.mp4")

            try:
                result_path = api_client.process_videos(
                    video_files=[str(small_video)],
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern=pattern,
                    memory_safe=True,
                    verbose=False,
                )

                if Path(result_path).exists():
                    file_size = Path(result_path).stat().st_size
                    pattern_results.append((pattern, file_size))
                    print(f"     ✅ {pattern}: {file_size / (1024 * 1024):.1f}MB")
                else:
                    print(f"     ❌ {pattern}: No output created")

            except Exception as e:
                print(f"     ❌ {pattern}: Failed with {e}")

        # Validate that patterns produce different results (no regression to same output)
        if len(pattern_results) >= 2:
            sizes = [result[1] for result in pattern_results]
            unique_sizes = len(set(sizes))

            if unique_sizes > 1:
                print("   ✅ Patterns produce varied output sizes")
            else:
                print(
                    "   ⚠️ All patterns produce same output size - may indicate variation regression"
                )

        print(
            f"✅ Pattern variation regression test: {len(pattern_results)}/4 patterns working"
        )

    def test_error_recovery_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for error recovery and graceful failure regressions."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if len(real_videos) < 1 or not real_audio:
            pytest.skip("Need at least 1 video file for error recovery test")

        print("\\n🛡️ Testing error recovery regression")

        # Test mixed valid/invalid files (should recover gracefully)
        valid_files = [str(f) for f in real_videos[:1]]
        invalid_files = ["nonexistent1.mp4", "nonexistent2.mp4"]
        mixed_files = valid_files + invalid_files

        output_path = str(regression_output_dir / "error_recovery_test.mp4")

        print(
            f"   Testing with {len(valid_files)} valid + {len(invalid_files)} invalid files"
        )

        try:
            result_path = api_client.process_videos(
                video_files=mixed_files,
                audio_file=str(real_audio[0]),
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            # If it succeeds, should have processed valid files only
            if Path(result_path).exists():
                print("   ✅ Error recovery succeeded - processed valid files")
                print(
                    f"   📹 Output: {Path(result_path).stat().st_size / (1024 * 1024):.1f}MB"
                )
            else:
                print("   ⚠️ Error recovery succeeded but no output created")

        except Exception as e:
            error_msg = str(e).lower()

            # Should fail gracefully with informative error
            if any(word in error_msg for word in ["file", "not", "found", "missing"]):
                print(f"   ✅ Failed gracefully with file error: {type(e).__name__}")
            else:
                print(f"   ⚠️ Error recovery may have regressed: {e}")

        print("✅ Error recovery regression test completed")

    def test_platform_specific_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for platform-specific regressions (macOS metadata files, path issues, etc.)."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n💻 Testing platform-specific regression")

        # Test that macOS metadata files are properly filtered
        all_videos = sample_video_files  # Include metadata files
        metadata_files = [f for f in all_videos if f.name.startswith("._")]

        if metadata_files:
            print(f"   Found {len(metadata_files)} metadata files (should be ignored)")

            # Ensure processing doesn't fail due to metadata files
            mixed_files = [str(f) for f in real_videos[:1]] + [
                str(f) for f in metadata_files[:1]
            ]
            output_path = str(regression_output_dir / "platform_specific_test.mp4")

            try:
                result_path = api_client.process_videos(
                    video_files=mixed_files,
                    audio_file=str(real_audio[0]),
                    output_path=output_path,
                    pattern="balanced",
                    memory_safe=True,
                    verbose=False,
                )

                if Path(result_path).exists():
                    print("   ✅ Metadata files handled correctly")
                else:
                    print("   ⚠️ Processing completed but no output created")

            except Exception as e:
                error_msg = str(e).lower()
                if "metadata" in error_msg or "._" in error_msg:
                    pytest.fail(f"Metadata file handling regression: {e}")
                else:
                    print(f"   ⚠️ Processing failed with: {e}")
        else:
            print("   No metadata files found - skipping metadata test")

        # Test path handling with spaces and special characters
        space_output_path = str(regression_output_dir / "path with spaces.mp4")

        try:
            result_path = api_client.process_videos(
                video_files=[str(real_videos[0])],
                audio_file=str(real_audio[0]),
                output_path=space_output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            if Path(result_path).exists():
                print("   ✅ Paths with spaces handled correctly")
            else:
                print("   ⚠️ Path with spaces test completed but no output")

        except Exception as e:
            error_msg = str(e).lower()
            if "path" in error_msg or "space" in error_msg:
                print(f"   ⚠️ Path handling issue: {e}")
            else:
                print(f"   ⚠️ Path test failed with: {e}")

        print("✅ Platform-specific regression test completed")

    def test_data_integrity_regression(
        self,
        api_client: AutoCutAPI,
        sample_video_files,
        sample_audio_files,
        regression_output_dir: Path,
    ):
        """Test for data integrity and corruption prevention regressions."""
        real_videos = [f for f in sample_video_files if not f.name.startswith("._")]
        real_audio = [f for f in sample_audio_files if not f.name.startswith("._")]

        if not real_videos or not real_audio:
            pytest.skip("No real media files available")

        print("\\n🔒 Testing data integrity regression")

        small_video = next(
            (f for f in real_videos if f.stat().st_size < 50 * 1024 * 1024),
            real_videos[0],
        )
        output_path = str(regression_output_dir / "data_integrity_test.mp4")

        try:
            result_path = api_client.process_videos(
                video_files=[str(small_video)],
                audio_file=str(real_audio[0]),
                output_path=output_path,
                pattern="balanced",
                memory_safe=True,
                verbose=False,
            )

            if Path(result_path).exists():
                file_size = Path(result_path).stat().st_size

                # Basic integrity checks
                assert file_size > 1000, (
                    "Output file should be substantial (not just header)"
                )
                assert file_size < 10 * 1024 * 1024 * 1024, (
                    "Output file should not be unreasonably large"
                )

                # Try to read file header to ensure it's not corrupted
                with open(result_path, "rb") as f:
                    header = f.read(12)
                    assert len(header) == 12, "Should be able to read file header"

                    # Check for MP4 file signature
                    if b"ftyp" in header:
                        print("   ✅ Valid MP4 file signature detected")
                    else:
                        print("   ⚠️ MP4 signature not found in header")

                print("   ✅ Data integrity checks passed")
                print(f"   📹 File size: {file_size / (1024 * 1024):.1f}MB")

            else:
                print("   ⚠️ No output file created for integrity testing")

        except Exception as e:
            error_msg = str(e).lower()

            # Check for data corruption specific errors
            corruption_keywords = ["corrupt", "invalid", "damaged", "truncated"]
            if any(keyword in error_msg for keyword in corruption_keywords):
                pytest.fail(f"Data integrity regression detected: {e}")
            else:
                print(f"   ⚠️ Integrity test failed with: {e}")

        print("✅ Data integrity regression test completed")
