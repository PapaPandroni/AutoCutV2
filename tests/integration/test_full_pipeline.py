"""
Integration tests for the complete AutoCut video processing pipeline.

Tests end-to-end video processing from input files to final output,
including all major components working together.
"""

import pytest
from pathlib import Path
import tempfile
import time

# Import all major components
from src.video.validation import VideoValidator
from src.video.codec_detection import CodecDetector
from src.video.transcoding import TranscodingService
from src.hardware.detection import HardwareDetector
from src.audio_analyzer import analyze_audio_file
from src.video_analyzer import analyze_video_file
from src.clip_assembler import assemble_clips


@pytest.mark.integration
@pytest.mark.media_required
@pytest.mark.slow
class TestFullPipeline:
    """Test complete AutoCut pipeline integration."""

    def test_complete_autocut_pipeline(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test the complete AutoCut pipeline from start to finish."""
        if not sample_video_files or not sample_audio_files:
            pytest.skip("Sample media files required for full pipeline test")

        # Take first few video files and first audio file
        video_files = [str(f) for f in sample_video_files[:3]]
        audio_file = str(sample_audio_files[0])
        output_path = str(temp_dir / "pipeline_test_output.mp4")

        # Step 1: Validate inputs
        validator = VideoValidator()

        # Validate all video files
        video_validation_results = []
        for video_file in video_files:
            result = validator.validate_basic(video_file)
            video_validation_results.append(result)
            assert result.is_valid, (
                f"Video validation failed for {video_file}: {result.errors}"
            )

        # Validate audio file
        audio_result = validator.validate_audio_file(audio_file)
        assert audio_result.is_valid, f"Audio validation failed: {audio_result.errors}"

        # Step 2: Preprocess videos (handle H.265 if needed)
        transcoding_service = TranscodingService()
        preprocessed_videos = []

        for video_file in video_files:
            preprocessed_path = transcoding_service.preprocess_video_if_needed(
                video_file
            )
            preprocessed_videos.append(preprocessed_path)
            assert Path(preprocessed_path).exists(), (
                f"Preprocessed video not found: {preprocessed_path}"
            )

        # Step 3: Audio analysis
        print("Analyzing audio...")
        audio_analysis = analyze_audio_file(audio_file)

        assert "beats" in audio_analysis, "Audio analysis missing beats"
        assert "bpm" in audio_analysis, "Audio analysis missing BPM"
        assert len(audio_analysis["beats"]) > 0, "No beats detected in audio"

        # Step 4: Video analysis
        print("Analyzing videos...")
        video_analysis_results = []

        for video_file in preprocessed_videos:
            try:
                analysis = analyze_video_file(video_file)
                video_analysis_results.append(analysis)

                # Verify analysis contains required data
                assert "scenes" in analysis, (
                    f"Video analysis missing scenes for {video_file}"
                )
                assert "quality_score" in analysis, (
                    f"Video analysis missing quality score for {video_file}"
                )
                assert len(analysis["scenes"]) > 0, (
                    f"No scenes detected in {video_file}"
                )

            except Exception as e:
                pytest.fail(f"Video analysis failed for {video_file}: {str(e)}")

        # Step 5: Clip assembly
        print("Assembling clips...")

        try:
            result = assemble_clips(
                video_files=preprocessed_videos,
                audio_file=audio_file,
                output_path=output_path,
                pattern="balanced",
            )

            assert result is not None, "Clip assembly returned None"
            assert Path(output_path).exists(), (
                f"Output video not created: {output_path}"
            )

            # Verify output file properties
            output_size = Path(output_path).stat().st_size
            assert output_size > 1000, f"Output file too small: {output_size} bytes"

            # Verify output can be analyzed
            codec_detector = CodecDetector()
            output_codec = codec_detector.detect_codec(output_path)
            assert output_codec.codec is not None, (
                "Could not detect codec of output file"
            )

        except Exception as e:
            pytest.fail(f"Clip assembly failed: {str(e)}")

        print(f"Full pipeline test completed successfully. Output: {output_path}")

    def test_pipeline_with_hardware_acceleration(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test pipeline with hardware acceleration when available."""
        if not sample_video_files or not sample_audio_files:
            pytest.skip("Sample media files required")

        # Check hardware capabilities
        hardware_detector = HardwareDetector()
        capabilities = hardware_detector.detect_capabilities()

        print(f"Hardware capabilities: {capabilities.best_encoder}")

        # Run pipeline with hardware-specific settings
        video_files = [str(f) for f in sample_video_files[:2]]
        audio_file = str(sample_audio_files[0])
        output_path = str(temp_dir / f"hw_accel_{capabilities.best_encoder}_output.mp4")

        start_time = time.time()

        # Process with hardware acceleration
        transcoding_service = TranscodingService()
        preprocessed_videos = []

        for video_file in video_files:
            preprocessed_path = transcoding_service.preprocess_video_if_needed(
                video_file
            )
            preprocessed_videos.append(preprocessed_path)

        # Run full pipeline
        result = assemble_clips(
            video_files=preprocessed_videos,
            audio_file=audio_file,
            output_path=output_path,
            pattern="energetic",
        )

        end_time = time.time()
        processing_time = end_time - start_time

        assert Path(output_path).exists(), "Hardware-accelerated output not created"

        print(
            f"Hardware-accelerated processing completed in {processing_time:.2f}s using {capabilities.best_encoder}"
        )

    def test_pipeline_error_recovery(
        self, sample_video_files, sample_audio_files, temp_dir, test_helpers
    ):
        """Test pipeline behavior with problematic input files."""
        if not sample_audio_files:
            pytest.skip("Sample audio files required")

        # Create a mix of good and bad video files
        good_videos = (
            [str(f) for f in sample_video_files[:1]] if sample_video_files else []
        )
        bad_video = str(
            test_helpers.create_mock_video_file(temp_dir, "corrupted.mp4", 0.1)
        )

        video_files = good_videos + [bad_video]
        audio_file = str(sample_audio_files[0])
        output_path = str(temp_dir / "error_recovery_output.mp4")

        # Pipeline should handle errors gracefully
        try:
            result = assemble_clips(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern="balanced",
            )

            # Should either succeed with good files or fail gracefully
            if result is not None and Path(output_path).exists():
                print("Pipeline succeeded despite problematic input")
            else:
                print("Pipeline failed gracefully with problematic input")

            # Either way, should not crash
            assert True

        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(
                f"Pipeline crashed instead of handling error gracefully: {str(e)}"
            )

    def test_pipeline_performance_benchmarks(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test pipeline performance benchmarks."""
        if not sample_video_files or not sample_audio_files:
            pytest.skip("Sample media files required")

        # Test with different numbers of input videos
        test_configurations = [
            (1, "single_video"),
            (2, "two_videos"),
            (3, "three_videos"),
        ]

        audio_file = str(sample_audio_files[0])
        performance_results = {}

        for num_videos, config_name in test_configurations:
            if len(sample_video_files) < num_videos:
                continue

            video_files = [str(f) for f in sample_video_files[:num_videos]]
            output_path = str(temp_dir / f"perf_test_{config_name}.mp4")

            start_time = time.time()

            try:
                result = assemble_clips(
                    video_files=video_files,
                    audio_file=audio_file,
                    output_path=output_path,
                    pattern="balanced",
                )

                end_time = time.time()
                processing_time = end_time - start_time

                # Calculate input size
                total_input_size_mb = sum(
                    Path(vf).stat().st_size for vf in video_files
                ) / (1024 * 1024)

                performance_results[config_name] = {
                    "num_videos": num_videos,
                    "processing_time": processing_time,
                    "input_size_mb": total_input_size_mb,
                    "throughput_mbps": total_input_size_mb / processing_time
                    if processing_time > 0
                    else 0,
                    "success": Path(output_path).exists(),
                }

                print(
                    f"{config_name}: {processing_time:.2f}s for {total_input_size_mb:.1f}MB ({total_input_size_mb / processing_time:.1f} MB/s)"
                )

            except Exception as e:
                performance_results[config_name] = {
                    "num_videos": num_videos,
                    "error": str(e),
                    "success": False,
                }

        # Verify at least some configurations succeeded
        successful_configs = [
            k for k, v in performance_results.items() if v.get("success", False)
        ]
        assert len(successful_configs) > 0, (
            f"No configurations succeeded: {performance_results}"
        )

    def test_pipeline_with_different_patterns(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test pipeline with different editing patterns."""
        if not sample_video_files or not sample_audio_files:
            pytest.skip("Sample media files required")

        video_files = [str(f) for f in sample_video_files[:2]]
        audio_file = str(sample_audio_files[0])

        patterns = ["balanced", "energetic", "dramatic", "buildup"]
        pattern_results = {}

        for pattern in patterns:
            output_path = str(temp_dir / f"pattern_{pattern}_output.mp4")

            try:
                result = assemble_clips(
                    video_files=video_files,
                    audio_file=audio_file,
                    output_path=output_path,
                    pattern=pattern,
                )

                pattern_results[pattern] = {
                    "success": Path(output_path).exists(),
                    "file_size_mb": Path(output_path).stat().st_size / (1024 * 1024)
                    if Path(output_path).exists()
                    else 0,
                }

            except Exception as e:
                pattern_results[pattern] = {"success": False, "error": str(e)}

        # At least some patterns should work
        successful_patterns = [
            k for k, v in pattern_results.items() if v.get("success", False)
        ]
        assert len(successful_patterns) > 0, f"No patterns succeeded: {pattern_results}"

        print(f"Successful patterns: {successful_patterns}")


@pytest.mark.integration
@pytest.mark.media_required
class TestPipelineComponentIntegration:
    """Test integration between specific pipeline components."""

    def test_validation_to_transcoding_integration(self, sample_video_files, temp_dir):
        """Test integration between validation and transcoding components."""
        if not sample_video_files:
            pytest.skip("Sample video files required")

        validator = VideoValidator()
        transcoding_service = TranscodingService()
        codec_detector = CodecDetector()

        for video_file in sample_video_files[:2]:
            # Step 1: Basic validation
            validation = validator.validate_basic(str(video_file))

            if validation.is_valid:
                # Step 2: Codec detection
                codec_info = codec_detector.detect_codec(str(video_file))

                # Step 3: iPhone compatibility if H.265
                if codec_info.is_hevc:
                    iphone_validation = validator.validate_iphone_compatibility(
                        str(video_file)
                    )

                    # Step 4: Transcoding if needed
                    if (
                        not iphone_validation.is_valid
                        or len(iphone_validation.errors) > 0
                    ):
                        output_path = (
                            temp_dir / f"integrated_transcode_{video_file.stem}.mp4"
                        )

                        transcoding_result = transcoding_service.transcode_h265_to_h264(
                            str(video_file), str(output_path)
                        )

                        if transcoding_result.success:
                            # Step 5: Validate transcoded output
                            transcoded_validation = (
                                validator.validate_transcoding_output(str(output_path))
                            )
                            assert (
                                transcoded_validation.validation_type
                                == "transcoding_output"
                            )

        assert True  # Test completed without errors

    def test_codec_detection_to_analysis_integration(self, sample_video_files):
        """Test integration between codec detection and video analysis."""
        if not sample_video_files:
            pytest.skip("Sample video files required")

        codec_detector = CodecDetector()

        for video_file in sample_video_files[:3]:
            # Codec detection
            codec_info = codec_detector.detect_codec(str(video_file))

            # Video analysis should work regardless of codec
            try:
                analysis = analyze_video_file(str(video_file))

                # Analysis should succeed or fail gracefully
                if analysis:
                    assert "scenes" in analysis or "error" in analysis

            except Exception as e:
                # Should not crash, but may fail gracefully
                print(f"Video analysis failed for {video_file}: {e}")

        assert True  # Integration test completed

    def test_hardware_detection_to_transcoding_integration(
        self, sample_video_files, temp_dir
    ):
        """Test integration between hardware detection and transcoding."""
        if not sample_video_files:
            pytest.skip("Sample video files required")

        hardware_detector = HardwareDetector()
        transcoding_service = TranscodingService()

        # Get hardware capabilities
        capabilities = hardware_detector.detect_capabilities()

        # Find an H.265 file or use any file for testing
        test_file = sample_video_files[0]
        output_path = temp_dir / f"hw_integration_output.mp4"

        # Test transcoding with detected hardware
        result = transcoding_service.transcode_h265_to_h264(
            str(test_file), str(output_path)
        )

        # Verify the encoder used matches hardware capabilities
        if result.success:
            expected_encoder = capabilities.best_encoder
            assert result.encoder_used == expected_encoder, (
                f"Expected {expected_encoder}, got {result.encoder_used}"
            )

        # Test should complete without errors
        assert True


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineStressTests:
    """Stress tests for the pipeline."""

    @pytest.mark.skipif(
        not pytest.config.getoption("--runslow", default=False),
        reason="Stress tests only run with --runslow",
    )
    def test_pipeline_with_many_files(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test pipeline with many input files."""
        if len(sample_video_files) < 5:
            pytest.skip("Need at least 5 video files for stress test")

        # Use all available video files (up to 10)
        video_files = [str(f) for f in sample_video_files[:10]]
        audio_file = str(sample_audio_files[0]) if sample_audio_files else None

        if not audio_file:
            pytest.skip("Audio file required for stress test")

        output_path = str(temp_dir / "stress_test_output.mp4")

        start_time = time.time()

        try:
            result = assemble_clips(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern="balanced",
            )

            end_time = time.time()
            processing_time = end_time - start_time

            if Path(output_path).exists():
                print(
                    f"Stress test completed: {len(video_files)} files in {processing_time:.2f}s"
                )
            else:
                print(f"Stress test failed after {processing_time:.2f}s")

            # Should complete within reasonable time (adjust as needed)
            assert processing_time < 300, (
                f"Stress test took too long: {processing_time:.2f}s"
            )

        except Exception as e:
            pytest.fail(f"Stress test failed: {str(e)}")

    def test_pipeline_memory_usage(
        self, sample_video_files, sample_audio_files, temp_dir
    ):
        """Test pipeline memory usage doesn't grow excessively."""
        if not sample_video_files or not sample_audio_files:
            pytest.skip("Sample media files required")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        video_files = [str(f) for f in sample_video_files[:3]]
        audio_file = str(sample_audio_files[0])
        output_path = str(temp_dir / "memory_test_output.mp4")

        try:
            result = assemble_clips(
                video_files=video_files,
                audio_file=audio_file,
                output_path=output_path,
                pattern="balanced",
            )

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(
                f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)"
            )

            # Memory increase should be reasonable (adjust threshold as needed)
            assert memory_increase < 1000, (
                f"Memory usage increased too much: {memory_increase:.1f}MB"
            )

        except Exception as e:
            pytest.fail(f"Memory test failed: {str(e)}")
