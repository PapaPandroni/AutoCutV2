"""
Clip Assembly Module for AutoCut

Handles the core logic of matching video clips to musical beats,
applying variety patterns, and rendering the final video.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import VideoChunk from canonical location
try:
    from video import VideoChunk
except ImportError:
    try:
        from video_analyzer import VideoChunk
    except ImportError:
        # Fallback if VideoChunk not available
        VideoChunk = None

# Import extracted classes from new modular structure
try:
    from video.encoder import (
        VideoEncoder,
        detect_optimal_codec_settings,
        detect_optimal_codec_settings_with_diagnostics,
    )
    from video.timeline_renderer import ClipTimeline
except ImportError:
    # Classes will be defined inline below for backward compatibility
    try:
        from video_analyzer import VideoChunk
    except ImportError:
        # Fallback if modules not available
        class VideoChunk:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


# Beat matching / clip selection extracted to src/beat_matching.py
try:
    from beat_matching import (
        VARIETY_PATTERNS,
        apply_variety_pattern,
        match_clips_to_beats,
        select_best_clips,
        _calculate_duration_fit,
        _clips_overlap,
        _fit_clip_to_duration,
    )
except ImportError:
    from .beat_matching import (
        VARIETY_PATTERNS,
        apply_variety_pattern,
        match_clips_to_beats,
        select_best_clips,
        _calculate_duration_fit,
        _clips_overlap,
        _fit_clip_to_duration,
    )


# VideoFormatAnalyzer class extracted to src/video/format_analyzer.py


# VideoNormalizationPipeline class extracted to src/video/normalization.py


# Video loading / preprocessing extracted to src/video_loading.py
try:
    from video_loading import (
        AdvancedMemoryManager,
        RobustVideoLoader,
        VideoPreprocessor,
        VideoResourceManager,
        _group_clips_by_file,
        load_video_clips_with_robust_error_handling,
        preprocess_videos_smart,
    )
except ImportError:
    from .video_loading import (
        AdvancedMemoryManager,
        RobustVideoLoader,
        VideoPreprocessor,
        VideoResourceManager,
        _group_clips_by_file,
        load_video_clips_with_robust_error_handling,
        preprocess_videos_smart,
    )


# ClipTimeline class extracted to src/video/timeline_renderer.py


# Rendering (uniformize/concat/encode) extracted to src/rendering.py
try:
    from rendering import add_transitions, render_video, uniformize_dimensions
except ImportError:
    from .rendering import add_transitions, render_video, uniformize_dimensions





def assemble_clips(
    video_files: List[str],
    audio_file: str,
    output_path: str,
    pattern: str = "balanced",
    max_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> str:
    """Main function to assemble clips into final video.

    Combines all steps:
    1. Analyze all video files
    2. Analyze audio file
    3. Determine optimal canvas format
    4. Match clips to beats
    5. Render final video

    Args:
        video_files: List of paths to video files
        audio_file: Path to music file
        output_path: Path for output video
        pattern: Variety pattern to use
        max_workers: Maximum parallel workers for video loading (None = auto-detect optimal)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to final rendered video

    Raises:
        FileNotFoundError: If any input file doesn't exist
        ValueError: If no suitable clips found or invalid audio file
        RuntimeError: If rendering fails
    """
    import logging
    import mimetypes

    # Dual import pattern for package/direct execution compatibility
    try:
        from audio_analyzer import analyze_audio
        from video.format_analyzer import VideoFormatAnalyzer
        from video_analyzer import analyze_video_file
    except ImportError:
        from .audio_analyzer import analyze_audio
        from .video.format_analyzer import VideoFormatAnalyzer
        from .video_analyzer import analyze_video_file

    def validate_audio_file_comprehensive(audio_path: str) -> tuple:
        """Comprehensive audio file validation before processing.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check 1: File existence
            if not Path(audio_path).exists():
                return False, f"Audio file not found: {audio_path}"

            # Check 2: File accessibility
            if not os.access(audio_path, os.R_OK):
                return False, f"Audio file is not readable: {audio_path}"

            # Check 3: File size validation
            try:
                file_size = Path(audio_path).stat().st_size
                if file_size == 0:
                    return False, f"Audio file is empty: {audio_path}"
                if file_size < 1024:  # Less than 1KB is suspicious
                    return (
                        False,
                        f"Audio file too small ({file_size} bytes), likely corrupted: {audio_path}",
                    )
                if file_size > 500 * 1024 * 1024:  # More than 500MB is excessive
                    return (
                        False,
                        f"Audio file too large ({file_size / (1024 * 1024):.1f}MB), may cause memory issues: {audio_path}",
                    )
            except OSError as e:
                return False, f"Cannot access audio file: {e}"

            # Check 4: File extension and MIME type validation
            audio_path_lower = audio_path.lower()
            valid_extensions = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"]
            has_valid_extension = any(
                audio_path_lower.endswith(ext) for ext in valid_extensions
            )

            if not has_valid_extension:
                return (
                    False,
                    f"Unsupported audio file extension. Supported: {', '.join(valid_extensions)}",
                )

            # Check 5: MIME type validation (if available)
            try:
                mime_type, _ = mimetypes.guess_type(audio_path)
                if mime_type and not mime_type.startswith("audio/"):
                    return (
                        False,
                        f"File does not appear to be an audio file (MIME type: {mime_type})",
                    )
            except Exception:
                # MIME type detection is optional, don't fail if it doesn't work
                pass

            # Check 6: Path complexity and character validation
            path_issues = []

            # Check for very long paths that might cause issues
            if len(audio_path) > 260:  # Windows MAX_PATH limit
                path_issues.append("Path length exceeds system limits")

            # Check for problematic characters that might cause FFmpeg issues
            problematic_chars = ["|", "<", ">", '"', "?", "*"]
            path_issues.extend(
                f"Contains problematic character '{char}'"
                for char in problematic_chars
                if char in audio_path
            )

            if path_issues:
                # These are warnings, not fatal errors
                logger = logging.getLogger("autocut.clip_assembler")
                logger.warning(
                    f"⚠️  Audio path has potential issues: {'; '.join(path_issues)}",
                )
                logger.warning(f"   Path: {audio_path}")
                logger.warning(
                    "   Will attempt to process but may encounter issues...",
                )

            # CRITICAL FIX: Return in try block, not orphaned else block
            return True, None
        except Exception as e:
            return False, f"Unexpected error validating audio file: {e!s}"

    # Set up detailed logging for the main pipeline
    logger = logging.getLogger("autocut.clip_assembler")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def report_progress(step: str, progress: float):
        """Helper to report progress if callback provided."""
        if progress_callback:
            progress_callback(step, progress)

    # Initialize comprehensive processing statistics
    processing_summary = {
        "total_videos": len(video_files),
        "videos_processed": 0,
        "videos_successful": 0,
        "videos_failed": 0,
        "total_chunks": 0,
        "file_results": [],  # Detailed per-file results
        "errors": [],
    }

    logger.info("=== AutoCut Video Processing Started ===")
    logger.info(f"Input videos: {len(video_files)} files")
    logger.info(f"Audio file: {Path(audio_file).name}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Pattern: {pattern}")

    # CRITICAL FIX: Enhanced comprehensive input validation
    logger.info("Validating input files...")

    # Validate audio file with comprehensive checks
    logger.info("🔍 Comprehensive audio file validation...")
    audio_valid, audio_error = validate_audio_file_comprehensive(audio_file)
    if not audio_valid:
        error_msg = f"Audio validation failed: {audio_error}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise ValueError(error_msg)

    logger.info("   ✅ Audio file validation passed")
    logger.info(f"   📁 File: {Path(audio_file).name}")
    logger.info(f"   📊 Size: {Path(audio_file).stat().st_size / (1024 * 1024):.2f}MB")

    # Validate video files
    missing_videos = [vf for vf in video_files if not Path(vf).exists()]
    if missing_videos:
        error_msg = f"Video files not found: {missing_videos}"
        logger.error(error_msg)
        processing_summary["errors"].append(error_msg)
        raise FileNotFoundError(error_msg)

    if not video_files:
        error_msg = "No video files provided"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("✅ All input files exist and are accessible")
    report_progress("Starting analysis", 0.0)

    # Step 1: Analyze audio file
    logger.info("=== Step 1: Audio Analysis ===")
    report_progress("Analyzing audio", 0.1)
    try:
        # CRITICAL FIX: Enhanced audio analysis with better error reporting
        try:
            audio_data = analyze_audio(audio_file)
        except FileNotFoundError as e:
            error_msg = f"Audio file access error during analysis: {e!s}"
            logger.exception(error_msg)
            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Audio analysis failed: {e!s}"
            logger.exception(error_msg)

            # Provide more helpful error messages for common issues
            if "No such file" in str(e) or "cannot find" in str(e).lower():
                error_msg += " (Check if audio file path is correct and accessible)"
            elif "format" in str(e).lower() or "codec" in str(e).lower():
                error_msg += " (Audio file may be corrupted or in unsupported format)"
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                error_msg += " (Check file permissions)"

            processing_summary["errors"].append(error_msg)
            raise ValueError(error_msg) from e

        # CRITICAL FIX: Use compensated beats instead of raw beats to fix sync issues
        beats = audio_data["compensated_beats"]  # Offset-corrected and filtered beats

        # Get musical timing information for professional synchronization
        musical_start_time = audio_data["musical_start_time"]
        intro_duration = audio_data["intro_duration"]
        allowed_durations = audio_data["allowed_durations"]

        if len(beats) < 2:
            error_msg = f"Insufficient beats detected in audio file: {len(beats)} beats"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ Audio analysis successful:")
        logger.info(f"  - Beats detected: {len(beats)}")
        logger.info(f"  - Musical start: {musical_start_time:.2f}s")
        logger.info(f"  - Intro duration: {intro_duration:.2f}s")
        report_progress("Audio analysis complete", 0.2)

    except Exception as e:
        error_msg = f"Failed to analyze audio file: {e!s}"
        logger.exception(error_msg)
        processing_summary["errors"].append(error_msg)
        raise RuntimeError(error_msg) from e

    # Step 2: Analyze all video files with detailed per-file tracking
    logger.info("=== Step 2: Video Analysis ===")
    report_progress("Analyzing videos", 0.3)
    all_video_chunks = []

    for i, video_file in enumerate(video_files):
        processing_summary["videos_processed"] += 1
        filename = Path(video_file).name

        file_result = {
            "file_path": video_file,
            "filename": filename,
            "index": i + 1,
            "status": "processing",
            "chunks_created": 0,
            "error_message": None,
            "processing_time": 0,
        }

        logger.info(f"--- Processing video {i + 1}/{len(video_files)}: {filename} ---")

        import time

        start_time = time.time()

        try:
            video_chunks = analyze_video_file(video_file, bpm=audio_data.get("bpm"))
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time

            if video_chunks:
                all_video_chunks.extend(video_chunks)
                file_result["chunks_created"] = len(video_chunks)
                file_result["status"] = "success"
                processing_summary["videos_successful"] += 1
                processing_summary["total_chunks"] += len(video_chunks)

                logger.info(
                    f"✅ {filename}: {len(video_chunks)} chunks created ({processing_time:.2f}s)",
                )

                # Log chunk quality summary
                if video_chunks:
                    scores = [chunk.score for chunk in video_chunks]
                    logger.info(
                        f"   Chunk scores: {min(scores):.1f}-{max(scores):.1f} (avg: {sum(scores) / len(scores):.1f})",
                    )
            else:
                file_result["status"] = "failed"
                file_result["error_message"] = (
                    "No chunks created - check logs above for detailed error analysis"
                )
                processing_summary["videos_failed"] += 1

                logger.error(
                    f"❌ {filename}: No chunks created ({processing_time:.2f}s)",
                )
                logger.error("   → This video will be excluded from the final output")

        except Exception as e:
            processing_time = time.time() - start_time
            file_result["processing_time"] = processing_time
            file_result["status"] = "failed"
            file_result["error_message"] = str(e)
            processing_summary["videos_failed"] += 1
            processing_summary["errors"].append(f"{filename}: {e!s}")

            logger.exception(
                f"❌ {filename}: Processing failed ({processing_time:.2f}s)"
            )
            logger.exception(f"   Error: {e!s}")
            logger.exception("   → This video will be excluded from the final output")

        processing_summary["file_results"].append(file_result)

        # Update progress for each video
        video_progress = 0.3 + (0.3 * (i + 1) / len(video_files))
        report_progress(f"Analyzed video {i + 1}/{len(video_files)}", video_progress)

    # Comprehensive processing summary
    logger.info("=== Video Processing Summary ===")
    logger.info(f"Total videos: {processing_summary['total_videos']}")
    logger.info(f"✅ Successful: {processing_summary['videos_successful']}")
    logger.info(f"❌ Failed: {processing_summary['videos_failed']}")
    logger.info(f"📊 Total chunks created: {processing_summary['total_chunks']}")

    # Detailed per-file results
    if processing_summary["videos_failed"] > 0:
        logger.warning("Failed video details:")
        for file_result in processing_summary["file_results"]:
            if file_result["status"] == "failed":
                logger.warning(
                    f"  - {file_result['filename']}: {file_result['error_message']}",
                )

    # Check if we have any usable content
    if not all_video_chunks:
        error_msg = (
            f"No suitable video clips found in any of the {len(video_files)} input files.\n"
            f"Processing results:\n"
            f"  - Successful videos: {processing_summary['videos_successful']}\n"
            f"  - Failed videos: {processing_summary['videos_failed']}\n"
            f"  - Total chunks created: {processing_summary['total_chunks']}\n"
            f"\nDetailed errors:\n"
            + "\n".join(
                [f"  - {error}" for error in processing_summary["errors"][-10:]],
            )  # Last 10 errors
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate success rate
    success_rate = (
        processing_summary["videos_successful"] / processing_summary["total_videos"]
    ) * 100
    logger.info(f"Processing success rate: {success_rate:.1f}%")

    if success_rate < 50:
        logger.warning(
            f"⚠️  Low success rate ({success_rate:.1f}%) - check video format compatibility",
        )

    report_progress(
        f"Video analysis complete: {len(all_video_chunks)} clips found",
        0.6,
    )

    # Step 2.5: CANVAS ANALYSIS - CRITICAL FIX for letterboxing issue
    logger.info("=== Step 2.5: Canvas Format Analysis ===")
    report_progress("Analyzing optimal canvas format", 0.65)

    try:
        # Initialize the VideoFormatAnalyzer
        format_analyzer = VideoFormatAnalyzer()

        # Analyze all video chunks to determine optimal canvas
        logger.info(
            f"🎯 Analyzing {len(all_video_chunks)} video clips for optimal canvas..."
        )

        canvas_format = format_analyzer.determine_optimal_canvas(all_video_chunks)

        # Log the canvas analysis results
        logger.info("✅ Canvas analysis complete:")
        logger.info(f"   🎬 Canvas type: {canvas_format['canvas_type']}")
        logger.info(
            f"   📐 Target dimensions: {canvas_format['target_width']}x{canvas_format['target_height']}"
        )
        logger.info("   📊 Content breakdown:")
        logger.info(
            f"      - Landscape: {canvas_format['aspect_ratio_analysis']['landscape_count']} clips ({(canvas_format['aspect_ratio_analysis']['landscape_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(
            f"      - Portrait: {canvas_format['aspect_ratio_analysis']['portrait_count']} clips ({(canvas_format['aspect_ratio_analysis']['portrait_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(
            f"      - Square: {canvas_format['aspect_ratio_analysis']['square_count']} clips ({(canvas_format['aspect_ratio_analysis']['square_count'] / len(all_video_chunks)) * 100:.1f}%)"
        )
        logger.info(f"   💡 Strategy: {canvas_format.get('description', canvas_format.get('aspect_ratio_analysis', {}).get('decision_rationale', 'Canvas analysis'))}")

        # Log letterboxing expectations
        if canvas_format.get("letterboxing_analysis"):
            logger.info("   📺 Letterboxing analysis:")
            for info in canvas_format["letterboxing_analysis"]:
                logger.info(f"      - {info}")
        else:
            logger.info(
                "   ✅ Minimal letterboxing expected - optimal aspect ratio match"
            )

        report_progress("Canvas analysis complete", 0.7)

    except Exception as e:
        error_msg = f"Canvas analysis failed: {e!s}"
        logger.exception(error_msg)
        logger.warning("Falling back to default 16:9 canvas (1920x1080)")

        # Fallback canvas format
        canvas_format = {
            "target_width": 1920,
            "target_height": 1080,
            "canvas_type": "fallback_16_9",
            "description": "Fallback 16:9 canvas due to analysis failure",
            "target_fps": 25,
            "aspect_ratio_analysis": {
                "landscape_count": 0,
                "portrait_count": 0,
                "square_count": 0,
                "dominant_orientation": "unknown",
                "decision_rationale": "Analysis failed, using safe fallback",
            },
            "letterboxing_analysis": [],
        }

    # Step 3: Match clips to beats
    logger.info("=== Step 3: Beat Matching ===")
    report_progress("Matching clips to beats", 0.75)
    try:
        timeline = match_clips_to_beats(
            video_chunks=all_video_chunks,
            beats=beats,
            allowed_durations=allowed_durations,
            pattern=pattern,
            musical_start_time=musical_start_time,  # Use musical intelligence for sync
        )

        if not timeline.clips:
            error_msg = f"No clips could be matched to the beat pattern using {len(all_video_chunks)} available chunks"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"✅ Beat matching successful: {len(timeline.clips)} clips selected",
        )

        # Log timeline statistics
        timeline_stats = timeline.get_summary_stats()
        logger.info("Timeline statistics:")
        logger.info(f"  - Total duration: {timeline_stats['total_duration']:.2f}s")
        logger.info(f"  - Average score: {timeline_stats['avg_score']:.1f}")
        logger.info(
            f"  - Score range: {timeline_stats['score_range'][0]:.1f}-{timeline_stats['score_range'][1]:.1f}",
        )
        logger.info(f"  - Unique videos used: {timeline_stats['unique_videos']}")

        report_progress(
            f"Beat matching complete: {len(timeline.clips)} clips selected",
            0.8,
        )

    except Exception as e:
        error_msg = f"Failed to match clips to beats: {e!s}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Step 4: Render final video
    logger.info("=== Step 4: Video Rendering ===")
    report_progress("Rendering video", 0.85)
    try:

        def render_progress(step_name: str, progress: float):
            # Scale render progress to final 15% of overall progress
            overall_progress = 0.85 + (0.15 * progress)
            report_progress(f"Rendering: {step_name}", overall_progress)

        # Calculate average beat interval for musical fade-out feature
        avg_beat_interval = None
        if len(beats) > 1:
            beat_intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
            avg_beat_interval = sum(beat_intervals) / len(beat_intervals)
            logger.info(f"Average beat interval calculated: {avg_beat_interval:.3f}s")

        # CRITICAL FIX: Pass canvas format to render_video function
        final_video_path = render_video(
            timeline=timeline,
            audio_file=audio_file,
            output_path=output_path,
            max_workers=max_workers,
            progress_callback=render_progress,
            bpm=audio_data.get("bpm"),
            avg_beat_interval=avg_beat_interval,
            canvas_format=canvas_format,  # NEW: Pass canvas format for optimal sizing
        )

        logger.info(f"✅ Video rendering complete: {final_video_path}")
        report_progress("Video rendering complete", 1.0)

        # Final success summary
        logger.info("=== AutoCut Processing Complete ===")
        logger.info(
            f"✅ Successfully created video: {Path(final_video_path).name}",
        )
        logger.info("📊 Processing summary:")
        logger.info(
            f"  - Videos processed: {processing_summary['videos_successful']}/{processing_summary['total_videos']}",
        )
        logger.info(
            f"  - Clips used: {len(timeline.clips)}/{processing_summary['total_chunks']}",
        )
        logger.info(
            f"  - Final video duration: {timeline_stats['total_duration']:.2f}s",
        )
        logger.info(
            f"  - Canvas format: {canvas_format['canvas_type']} ({canvas_format['target_width']}x{canvas_format['target_height']})"
        )
        # CRITICAL FIX: Return in try block, not orphaned else block
        return final_video_path
    except Exception as e:
        error_msg = f"Failed to render video: {e!s}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    # Export timeline JSON for debugging (optional)
    try:
        timeline_path = output_path.replace(".mp4", "_timeline.json")
        timeline.export_json(timeline_path)
        logger.info(f"Debug: Timeline exported to {timeline_path}")

        # Export processing summary for debugging
        summary_path = output_path.replace(".mp4", "_processing_summary.json")
        import json

        with Path(summary_path).open("w") as f:
            json.dump(processing_summary, f, indent=2)
        logger.info(f"Debug: Processing summary exported to {summary_path}")

    except Exception:
        pass  # Non-critical, ignore errors  # Non-critical, ignore errors  # Non-critical, ignore errors  # Non-critical, ignore errors


def detect_optimal_codec_settings() -> Tuple[Dict[str, Any], List[str]]:
    """Detect optimal codec settings for video encoding.

    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
    """
    try:
        # Try to use the extracted VideoEncoder class
        encoder = VideoEncoder()
        return encoder.detect_optimal_codec_settings()
    except Exception:
        # Fallback to safe default settings
        moviepy_params = {
            "codec": "libx264",
            "bitrate": "5000k",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
        }

        ffmpeg_params = [
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        return moviepy_params, ffmpeg_params


def detect_optimal_codec_settings_with_diagnostics() -> Tuple[
    Dict[str, Any],
    List[str],
    Dict[str, str],
]:
    """Enhanced codec settings detection with full diagnostic information.

    Returns:
        Tuple containing:
        - Dictionary of MoviePy parameters for write_videofile()
        - List of FFmpeg-specific parameters for ffmpeg_params argument
        - Dictionary of diagnostic information and capability details
    """
    try:
        # Try to use the extracted VideoEncoder class
        encoder = VideoEncoder()
        return encoder.detect_optimal_codec_settings_with_diagnostics()
    except Exception:
        # Fallback with basic diagnostics
        moviepy_params, ffmpeg_params = detect_optimal_codec_settings()
        diagnostics = {
            "encoder_type": "FALLBACK",
            "hardware_acceleration": "false",
            "platform": "unknown",
        }
        return moviepy_params, ffmpeg_params, diagnostics
