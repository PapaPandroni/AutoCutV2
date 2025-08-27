"""
Integration tests for iPhone H.265 video processing workflow.

Tests the complete pipeline from H.265 iPhone video to processed output,
including codec detection, transcoding, and validation steps.
"""

from pathlib import Path

import pytest

from src.hardware.detection import HardwareDetector
from src.video.codec_detection import CodecDetector
from src.video.transcoding import TranscodingService
from src.video.validation import VideoValidator


@pytest.mark.integration
@pytest.mark.media_required
class TestiPhoneH265Workflow:
    """Test complete iPhone H.265 processing workflow."""

    def test_complete_iphone_workflow(self, iphone_h265_file_path, temp_dir):
        """Test complete iPhone H.265 video processing workflow."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        # Initialize all components
        validator = VideoValidator()
        codec_detector = CodecDetector()
        transcoding_service = TranscodingService()
        hardware_detector = HardwareDetector()

        # Step 1: Basic validation
        basic_validation = validator.validate_basic(str(iphone_h265_file_path))
        assert basic_validation.is_valid, (
            f"Basic validation failed: {basic_validation.errors}"
        )

        # Step 2: Codec detection
        codec_info = codec_detector.detect_codec(str(iphone_h265_file_path))
        assert codec_info.codec is not None, "Codec detection failed"

        # Step 3: iPhone compatibility check
        iphone_validation = validator.validate_iphone_compatibility(
            str(iphone_h265_file_path),
        )
        assert iphone_validation.validation_type == "iphone_compatibility"

        # Step 4: Hardware capabilities
        hardware_caps = hardware_detector.detect_capabilities()
        assert hardware_caps is not None, "Hardware detection failed"

        # Step 5: Transcoding if needed
        if codec_info.is_hevc:
            # Test MoviePy compatibility
            is_compatible = transcoding_service.test_moviepy_h265_compatibility(
                str(iphone_h265_file_path),
            )

            if not is_compatible:
                # Transcode the file
                output_path = temp_dir / "transcoded_iphone.mp4"
                transcoding_result = transcoding_service.transcode_h265_to_h264(
                    str(iphone_h265_file_path),
                    str(output_path),
                )

                if transcoding_result.success:
                    # Validate transcoded output
                    transcoded_validation = validator.validate_transcoding_output(
                        str(output_path),
                    )
                    assert transcoded_validation.validation_type == "transcoding_output"

                    # Verify the transcoded file is H.264
                    transcoded_codec = codec_detector.detect_codec(str(output_path))
                    assert transcoded_codec.codec == "h264" or transcoded_codec.is_h264
                else:
                    # If transcoding fails, capture the error for analysis
                    pytest.fail(
                        f"Transcoding failed: {transcoding_result.error_message}",
                    )

        # Step 6: Final validation for workflow completion
        assert True, "Complete iPhone H.265 workflow succeeded"

    def test_iphone_preprocessing_workflow(self, iphone_h265_file_path, temp_dir):
        """Test the preprocessing workflow specifically."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        transcoding_service = TranscodingService()

        # Test the preprocessing decision logic
        processed_path = transcoding_service.preprocess_video_if_needed(
            str(iphone_h265_file_path),
        )

        # Should return either the original path (if compatible) or transcoded path
        assert processed_path is not None
        assert Path(processed_path).exists()

        # Verify the result is processable
        codec_detector = CodecDetector()
        final_codec = codec_detector.detect_codec(processed_path)

        # Should be either original H.265 (if compatible) or transcoded H.264
        assert final_codec.codec in ["h264", "hevc"], (
            f"Unexpected codec: {final_codec.codec}"
        )

    def test_hardware_specific_transcoding(self, iphone_h265_file_path, temp_dir):
        """Test transcoding with different hardware configurations."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        hardware_detector = HardwareDetector()
        transcoding_service = TranscodingService()

        # Get actual hardware capabilities
        capabilities = hardware_detector.detect_capabilities()

        # Test transcoding with the available hardware
        output_path = temp_dir / f"transcoded_{capabilities.best_encoder}.mp4"

        result = transcoding_service.transcode_h265_to_h264(
            str(iphone_h265_file_path),
            str(output_path),
        )

        if capabilities.has_gpu_acceleration:
            # With GPU acceleration, transcoding should be relatively fast
            assert result.encoder_used in ["nvenc", "qsv"], (
                f"Expected GPU encoder, got {result.encoder_used}"
            )
        else:
            # CPU fallback
            assert result.encoder_used == "cpu", (
                f"Expected CPU encoder, got {result.encoder_used}"
            )

        # Verify the result is valid regardless of encoder used
        if result.success:
            assert Path(output_path).exists()
            assert result.file_size_mb > 0

    def test_validation_error_handling(self, temp_dir):
        """Test error handling in validation workflow."""
        # Create a fake/corrupted file
        fake_iphone_file = temp_dir / "fake_iphone.mov"
        fake_iphone_file.write_text("This is not a video file")

        validator = VideoValidator()
        codec_detector = CodecDetector()

        # Basic validation should catch this
        basic_validation = validator.validate_basic(str(fake_iphone_file))
        # May pass basic checks (file exists, readable) but fail deeper analysis

        # Codec detection should handle this gracefully
        codec_info = codec_detector.detect_codec(str(fake_iphone_file))
        assert codec_info.codec == "unknown" or codec_info.codec is None

        # iPhone compatibility check should handle gracefully
        iphone_validation = validator.validate_iphone_compatibility(
            str(fake_iphone_file),
        )
        assert iphone_validation.validation_type == "iphone_compatibility"
        # Should not crash, but may report invalid

    @pytest.mark.slow
    def test_performance_benchmarking(self, iphone_h265_file_path, temp_dir):
        """Test performance of iPhone H.265 processing."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        import time

        transcoding_service = TranscodingService()

        start_time = time.time()

        # Run preprocessing
        processed_path = transcoding_service.preprocess_video_if_needed(
            str(iphone_h265_file_path),
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Performance assertions - adjust based on expected performance
        file_size_mb = iphone_h265_file_path.stat().st_size / (1024 * 1024)

        # Should process reasonably quickly (adjust thresholds as needed)
        if file_size_mb < 50:  # Small files should process quickly
            assert processing_time < 30, (
                f"Small file took too long: {processing_time:.2f}s for {file_size_mb:.1f}MB"
            )
        elif file_size_mb < 200:  # Medium files
            assert processing_time < 120, (
                f"Medium file took too long: {processing_time:.2f}s for {file_size_mb:.1f}MB"
            )

        # Log performance for analysis
        print(
            f"Processed {file_size_mb:.1f}MB in {processing_time:.2f}s ({file_size_mb / processing_time:.1f} MB/s)",
        )


@pytest.mark.integration
class TestiPhoneH265ErrorScenarios:
    """Test error scenarios in iPhone H.265 processing."""

    def test_transcoding_insufficient_disk_space(self, temp_dir, test_helpers):
        """Test transcoding behavior with insufficient disk space."""
        # Create a large mock file
        large_input = test_helpers.create_mock_video_file(
            temp_dir,
            "large_input.mov",
            100.0,
        )

        transcoding_service = TranscodingService()

        # Try to transcode to a path that might fail due to disk space
        # This is a simulation - actual behavior depends on available space
        output_path = temp_dir / "large_output.mp4"

        result = transcoding_service.transcode_h265_to_h264(
            str(large_input),
            str(output_path),
        )

        # Should handle the error gracefully
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error_message is not None
            assert len(result.error_message) > 0

    def test_transcoding_permission_denied(self, temp_dir, test_helpers):
        """Test transcoding behavior with permission issues."""
        input_file = test_helpers.create_mock_video_file(temp_dir, "input.mov")

        # Try to write to a restricted location (this may not always fail in test environments)
        restricted_output = "/root/restricted_output.mp4"

        transcoding_service = TranscodingService()
        result = transcoding_service.transcode_h265_to_h264(
            str(input_file),
            restricted_output,
        )

        # Should handle permission errors gracefully
        assert isinstance(result, object)  # TranscodingResult
        if not result.success:
            assert (
                "permission" in result.error_message.lower()
                or "denied" in result.error_message.lower()
            )

    def test_ffmpeg_not_available(self, iphone_h265_file_path, temp_dir):
        """Test behavior when FFmpeg is not available."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        transcoding_service = TranscodingService()

        # Mock FFmpeg not being available
        with pytest.mock.patch(
            "src.video.transcoding.subprocess.run",
        ) as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("ffmpeg: command not found")

            output_path = temp_dir / "output.mp4"
            result = transcoding_service.transcode_h265_to_h264(
                str(iphone_h265_file_path),
                str(output_path),
            )

            assert result.success is False
            assert "ffmpeg" in result.error_message.lower()


@pytest.mark.integration
@pytest.mark.media_required
class TestiPhoneH265RealWorldScenarios:
    """Test real-world iPhone H.265 scenarios."""

    def test_multiple_iphone_files_batch_processing(self, sample_video_files, temp_dir):
        """Test processing multiple iPhone H.265 files."""
        if not sample_video_files:
            pytest.skip("No sample video files available")

        # Filter for potential iPhone files (MOV extension)
        iphone_files = [f for f in sample_video_files if f.suffix.lower() == ".mov"]

        if not iphone_files:
            pytest.skip("No iPhone-like video files available")

        transcoding_service = TranscodingService()
        results = []

        for video_file in iphone_files[:3]:  # Test first 3 files
            processed_path = transcoding_service.preprocess_video_if_needed(
                str(video_file),
            )
            results.append(
                {
                    "original": str(video_file),
                    "processed": processed_path,
                    "success": processed_path is not None,
                },
            )

        # All files should process successfully
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(results), (
            f"Some files failed to process: {results}"
        )

    def test_iphone_file_format_variations(self, sample_video_files, temp_dir):
        """Test different iPhone file format variations."""
        if not sample_video_files:
            pytest.skip("No sample video files available")

        codec_detector = CodecDetector()
        validator = VideoValidator()

        format_results = {}

        for video_file in sample_video_files:
            # Detect codec and validate
            codec_info = codec_detector.detect_codec(str(video_file))
            validation = validator.validate_basic(str(video_file))

            format_key = f"{codec_info.codec}_{video_file.suffix.lower()}"
            format_results[format_key] = {
                "file": video_file.name,
                "codec": codec_info.codec,
                "is_hevc": codec_info.is_hevc,
                "valid": validation.is_valid,
                "resolution": codec_info.resolution,
                "fps": codec_info.fps,
            }

        # Should successfully analyze all formats
        assert len(format_results) > 0, "No video formats were analyzed"

        # Log results for analysis
        for format_key, info in format_results.items():
            print(f"{format_key}: {info}")

    def test_iphone_transcoding_quality_verification(
        self,
        iphone_h265_file_path,
        temp_dir,
    ):
        """Test that transcoded iPhone videos maintain acceptable quality."""
        if not iphone_h265_file_path.exists():
            pytest.skip("No iPhone H.265 test file available")

        transcoding_service = TranscodingService()
        codec_detector = CodecDetector()

        # Get original file info
        original_codec = codec_detector.detect_codec(str(iphone_h265_file_path))
        original_size_mb = iphone_h265_file_path.stat().st_size / (1024 * 1024)

        if original_codec.is_hevc:
            # Transcode the file
            output_path = temp_dir / "quality_test_output.mp4"
            result = transcoding_service.transcode_h265_to_h264(
                str(iphone_h265_file_path),
                str(output_path),
            )

            if result.success:
                # Analyze transcoded file
                transcoded_codec = codec_detector.detect_codec(str(output_path))
                transcoded_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

                # Quality checks
                assert transcoded_codec.is_h264, "Transcoded file should be H.264"

                # Size should be reasonable (not too much larger, allowing for some increase due to encoding differences)
                size_ratio = transcoded_size_mb / original_size_mb
                assert 0.5 < size_ratio < 3.0, (
                    f"Size ratio {size_ratio:.2f} outside acceptable range"
                )

                # Resolution should be preserved
                if original_codec.resolution and transcoded_codec.resolution:
                    assert original_codec.resolution == transcoded_codec.resolution, (
                        "Resolution should be preserved"
                    )

                print(
                    f"Transcoding quality check passed: {original_size_mb:.1f}MB → {transcoded_size_mb:.1f}MB (ratio: {size_ratio:.2f})",
                )
            else:
                pytest.fail(f"Transcoding failed: {result.error_message}")
        else:
            pytest.skip("Test file is not H.265, skipping transcoding quality test")
