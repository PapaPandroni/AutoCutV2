"""
Video Analysis Module for AutoCut

Handles video processing including scene detection, quality scoring,
motion analysis, and face detection.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

# Import our domain-specific types
try:
    from video.types import PathLike, VideoAnalysisResult, SceneSegment
except ImportError:
    try:
        from .video.types import PathLike, VideoAnalysisResult, SceneSegment
    except ImportError:
        # Fallback type definitions if types module not available
        PathLike = Union[str, 'Path']
        VideoAnalysisResult = Dict[str, Any]
        SceneSegment = Dict[str, Any]

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        # Fallback for testing without full moviepy installation
        VideoFileClip = None


# Import VideoChunk from the canonical location with dual import pattern
try:
    # Try absolute import first for autocut.py execution context
    from video.assembly.clip_selector import VideoChunk
except ImportError:
    # Fallback to relative import for package execution context
    from .video import VideoChunk


def load_video(file_path: PathLike) -> Tuple[VideoFileClip, Dict[str, Any]]:
    """Load video file and extract basic metadata.

    Automatically preprocesses problematic video codecs (H.265/HEVC) for
    MoviePy compatibility by transcoding to H.264 when needed.

    Args:
        file_path: Path to the video file

    Returns:
        Tuple of (VideoFileClip object, metadata dictionary)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported
    """
    import os

    # Import video preprocessing from new transcoding module with dual import pattern
    try:
        # Try absolute import first for autocut.py execution context
        from video.transcoding import preprocess_video_if_needed
    except ImportError:
        try:
            # Try relative import for package execution context
            from .video.transcoding import preprocess_video_if_needed
        except ImportError:
            # Final fallback to utils module
            try:
                from utils import preprocess_video_if_needed
            except ImportError:
                from .utils import preprocess_video_if_needed

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    try:
        # Preprocess video if needed (handles H.265/HEVC transcoding)
        processed_file_path = preprocess_video_if_needed(file_path)

        # Load video with MoviePy (using processed file)
        video = VideoFileClip(processed_file_path)

        # Extract metadata
        metadata = {
            "duration": video.duration,
            "fps": video.fps,
            "size": video.size,  # (width, height)
            "width": video.w,
            "height": video.h,
            "filename": Path(file_path).name,  # Original filename
            "file_path": file_path,  # Original file path
            "processed_file_path": processed_file_path,  # May be different if transcoded
            "was_transcoded": processed_file_path != file_path,
        }

        return video, metadata

    except Exception as e:
        raise ValueError(f"Failed to load video file {file_path}: {e!s}") from e


def detect_scenes(
    video: VideoFileClip,
    threshold: float = 30.0,
) -> List[Tuple[float, float]]:
    """Detect scene changes in video based on frame differences.

    Args:
        video: VideoFileClip object
        threshold: Sensitivity threshold for scene detection

    Returns:
        List of (start_time, end_time) tuples for each scene
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    scenes = []
    duration = video.duration

    # Sample frames every 1.0 seconds for performance (was 0.5s)
    sample_interval = 1.0
    timestamps = np.arange(0, duration, sample_interval)

    if len(timestamps) < 2:
        # Video too short, return as single scene
        return [(0.0, duration)]

    # Get frames and calculate differences
    prev_frame = None
    scene_changes = [0.0]  # Always start with beginning

    def _safe_get_frame_diff(timestamp: float) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Safely get frame and calculate difference, returning (frame, diff)."""
        try:
            frame = video.get_frame(timestamp)
            if prev_frame is not None:
                diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                return frame, diff
            return frame, None
        except Exception:
            # Skip problematic frames
            return None, None

    for t in timestamps[1:]:  # Skip first timestamp
        frame, diff = _safe_get_frame_diff(t)
        
        if frame is not None:
            if diff is not None and diff > threshold:
                scene_changes.append(t)
            prev_frame = frame

    # Always end with video duration
    if scene_changes[-1] != duration:
        scene_changes.append(duration)

    # Convert to (start, end) tuples
    for i in range(len(scene_changes) - 1):
        start_time = scene_changes[i]
        end_time = scene_changes[i + 1]

        # Only include scenes longer than 1 second
        if end_time - start_time >= 1.0:
            scenes.append((start_time, end_time))

    # If no scenes found, return entire video as one scene
    if not scenes:
        scenes = [(0.0, duration)]

    return scenes


def score_scene(video: VideoFileClip, start_time: float, end_time: float) -> float:
    """Calculate quality score for a video scene.

    Combines multiple quality metrics:
    - Sharpness (Laplacian variance)
    - Brightness (mean pixel value)
    - Contrast (pixel value standard deviation)

    Args:
        video: VideoFileClip object
        start_time: Start time of scene in seconds
        end_time: End time of scene in seconds

    Returns:
        Quality score from 0-100 (higher is better)
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    # Sample 3-5 frames throughout the scene for analysis
    duration = end_time - start_time
    if duration < 0.1:
        return 0.0  # Too short to analyze

    # Calculate sample points
    if duration <= 1.0:
        sample_times = [start_time + duration / 2]  # Middle frame only
    elif duration <= 3.0:
        sample_times = [start_time + duration * 0.25, start_time + duration * 0.75]
    else:
        sample_times = [
            start_time + duration * 0.2,
            start_time + duration * 0.5,
            start_time + duration * 0.8,
        ]

    scores = []

    def _safe_score_frame(timestamp: float) -> Optional[float]:
        """Safely score a frame at given timestamp, returning score or None if failed."""
        try:
            # Get frame as RGB numpy array
            frame = video.get_frame(timestamp)

            # Convert to grayscale for sharpness calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 1. Sharpness (Laplacian variance) - higher is sharper
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            # 2. Brightness (mean pixel value) - prefer moderate brightness
            brightness = np.mean(gray)
            # Score brightness: peak at 128, drop off for too dark/bright
            brightness_score = 100 * (1 - abs(brightness - 128) / 128)

            # 3. Contrast (standard deviation) - higher is better
            contrast = np.std(gray)

            # Normalize and combine scores
            # Sharpness: log scale to handle wide range of values
            sharpness_score = min(100, max(0, 20 * np.log10(max(sharpness, 1))))

            # Contrast: linear scale, max around 60-80 std
            contrast_score = min(100, contrast * 1.5)

            # Weighted combination
            frame_score = (
                0.4 * sharpness_score + 0.3 * brightness_score + 0.3 * contrast_score
            )
            
            return frame_score

        except Exception:
            # Skip problematic frames
            return None

    for t in sample_times:
        frame_score = _safe_score_frame(t)
        if frame_score is not None:
            scores.append(frame_score)

    if not scores:
        return 0.0

    # Return average score
    return float(np.mean(scores))


def detect_motion(video: VideoFileClip, start_time: float, end_time: float) -> float:
    """Detect motion level in video segment using optical flow.

    Args:
        video: VideoFileClip object
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds

    Returns:
        Motion score (0-100, higher means more motion/activity)
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    duration = end_time - start_time
    if duration < 0.3:  # Need at least 0.3s for motion detection
        return 0.0

    try:
        # Sample frames for motion analysis (every 0.5 seconds, max 6 frames)
        sample_interval = max(0.5, duration / 6)
        sample_times = []
        current_time = start_time
        while current_time <= end_time - 0.1:  # Leave small buffer
            sample_times.append(current_time)
            current_time += sample_interval

        if len(sample_times) < 2:
            return 0.0

        motion_scores = []

        # Calculate optical flow between consecutive frames
        prev_frame = None
        for t in sample_times:
            # Get frame and convert to grayscale
            frame = video.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if prev_frame is not None:
                # Calculate optical flow using Lucas-Kanade method
                # First, detect corner points in previous frame
                corners = cv2.goodFeaturesToTrack(
                    prev_frame,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7,
                )

                if corners is not None and len(corners) > 10:
                    # Calculate optical flow
                    next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_frame,
                        gray,
                        corners,
                        None,
                    )

                    # Select good points
                    good_new = next_corners[status == 1]
                    good_old = corners[status == 1]

                    if len(good_new) > 5:
                        # Calculate motion vectors
                        motion_vectors = good_new - good_old

                        # Calculate motion magnitude
                        motion_magnitudes = np.sqrt(
                            motion_vectors[:, 0] ** 2 + motion_vectors[:, 1] ** 2,
                        )

                        # Average motion magnitude
                        avg_motion = np.mean(motion_magnitudes)
                        motion_scores.append(avg_motion)

            prev_frame = gray

        if not motion_scores:
            return 0.0

        # Calculate final motion score
        avg_motion = np.mean(motion_scores)

        # Normalize to 0-100 scale
        # Typical motion values range from 0-20 pixels per frame
        motion_score = min(100, avg_motion * 5)

        return float(motion_score)

    except Exception:
        # Return 0 if motion detection fails
        return 0.0


def detect_faces(video: VideoFileClip, start_time: float, end_time: float) -> int:
    """Detect faces in video segment using OpenCV cascade classifier.

    Args:
        video: VideoFileClip object
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds

    Returns:
        Number of faces detected (higher is better for family videos)
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    duration = end_time - start_time
    if duration < 0.1:
        return 0

    try:
        # Load the face cascade classifier
        # Try different possible locations for the cascade file
        cascade_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]

        face_cascade = None
        for path in cascade_paths:
            try:
                if Path(path).exists():
                    face_cascade = cv2.CascadeClassifier(path)
                    if not face_cascade.empty():
                        break
            except:
                continue

        if face_cascade is None or face_cascade.empty():
            # Fallback: return 0 if no face detection available
            return 0

        # Sample 3-5 frames throughout the segment for face detection
        if duration <= 1.0:
            sample_times = [start_time + duration / 2]
        elif duration <= 3.0:
            sample_times = [start_time + duration * 0.3, start_time + duration * 0.7]
        else:
            sample_times = [
                start_time + duration * 0.2,
                start_time + duration * 0.5,
                start_time + duration * 0.8,
            ]

        face_counts = []

        def _safe_detect_faces(timestamp: float) -> Optional[int]:
            """Safely detect faces at given timestamp, returning count or None if failed."""
            try:
                # Get frame and convert to grayscale
                frame = video.get_frame(timestamp)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                return len(faces)

            except Exception:
                # Skip problematic frames
                return None

        for t in sample_times:
            face_count = _safe_detect_faces(t)
            if face_count is not None:
                face_counts.append(face_count)

        if not face_counts:
            return 0

        # Return maximum faces detected in any frame
        # (assumption: family videos benefit from more faces visible)
        return int(max(face_counts))

    except Exception:
        # Return 0 if face detection fails
        return 0


def detect_camera_shake(
    video: VideoFileClip, start_time: float, end_time: float
) -> float:
    """Detect camera shake/instability using dense optical flow analysis.

    Uses dense optical flow to analyze global motion patterns and distinguish
    between camera shake (bad) and subject movement (good).

    Args:
        video: VideoFileClip object
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds

    Returns:
        Stability score (0-100, higher means more stable/less shaky)
    """
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")

    duration = end_time - start_time
    if duration < 0.3:  # Need at least 0.3s for motion analysis
        return 100.0  # Assume stable if too short to analyze

    try:
        # Sample frames for stability analysis (every 0.33 seconds, max 8 frames)
        sample_interval = max(0.33, duration / 8)
        sample_times = []
        current_time = start_time
        while current_time <= end_time - 0.1:  # Leave small buffer
            sample_times.append(current_time)
            current_time += sample_interval

        if len(sample_times) < 2:
            return 100.0  # Assume stable if insufficient frames

        stability_metrics = []

        # Calculate dense optical flow between consecutive frames
        prev_frame = None
        for t in sample_times:
            # Get frame and convert to grayscale
            frame = video.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Resize to speed up computation (maintain aspect ratio)
            height, width = gray.shape
            if width > 640:
                scale = 640.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))

            if prev_frame is not None:
                # Calculate dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    gray,
                    None,  # flow (output)
                    0.5,  # pyr_scale: image scale (<1) to build pyramids
                    3,  # levels: number of pyramid layers
                    15,  # winsize: averaging window size
                    3,  # iterations: number of iterations at each pyramid level
                    5,  # poly_n: size of pixel neighborhood for polynomial expansion
                    1.2,  # poly_sigma: standard deviation of Gaussian for smoothing
                    0,  # flags
                )

                # Analyze the flow for camera shake characteristics
                stability_score = _analyze_flow_stability(flow)
                stability_metrics.append(stability_score)

            prev_frame = gray

        if not stability_metrics:
            return 100.0  # Assume stable if no metrics calculated

        # Return average stability score
        avg_stability = np.mean(stability_metrics)
        return float(np.clip(avg_stability, 0.0, 100.0))

    except Exception:
        # Return neutral stability score if analysis fails
        return 50.0


def _analyze_flow_stability(flow: NDArray[np.floating[Any]]) -> float:
    """Analyze dense optical flow to determine camera stability.

    Args:
        flow: Dense optical flow array (H x W x 2)

    Returns:
        Stability score (0-100, higher is more stable)
    """
    if flow is None or flow.size == 0:
        return 100.0

    try:
        # Extract flow components
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        # Calculate flow magnitude for each pixel
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Ignore very small movements (likely noise)
        significant_motion_mask = flow_magnitude > 0.5

        if not np.any(significant_motion_mask):
            return 100.0  # No significant motion = very stable

        # Get flow vectors for pixels with significant motion
        significant_flow_x = flow_x[significant_motion_mask]
        significant_flow_y = flow_y[significant_motion_mask]

        # 1. Global Motion Consistency Analysis
        # Camera shake typically shows inconsistent/erratic global motion
        mean_flow_x = np.mean(significant_flow_x)
        mean_flow_y = np.mean(significant_flow_y)

        # Calculate how much each pixel's motion deviates from global mean
        deviation_x = significant_flow_x - mean_flow_x
        deviation_y = significant_flow_y - mean_flow_y
        deviation_magnitude = np.sqrt(deviation_x**2 + deviation_y**2)

        # High deviation indicates inconsistent motion (camera shake)
        motion_consistency = 100.0 - min(100.0, np.mean(deviation_magnitude) * 5)

        # 2. Global Motion Magnitude Analysis
        # Very high global motion often indicates camera shake or excessive movement
        global_motion_magnitude = np.sqrt(mean_flow_x**2 + mean_flow_y**2)

        # Apply penalties for different levels of global motion
        if global_motion_magnitude > 15:
            # Very high motion - likely shake or very unstable
            magnitude_penalty = min(60, (global_motion_magnitude - 15) * 3)
        elif global_motion_magnitude > 8:
            # Moderate-high motion - some penalty for excessive movement
            magnitude_penalty = min(25, (global_motion_magnitude - 8) * 2)
        elif global_motion_magnitude > 3:
            # Moderate motion - small penalty for global movement
            magnitude_penalty = min(15, (global_motion_magnitude - 3) * 1.5)
        else:
            # Low motion - no penalty
            magnitude_penalty = 0

        magnitude_score = 100.0 - magnitude_penalty

        # 3. Motion Direction Uniformity
        # Camera shake often has random directions, good motion has consistent direction
        flow_angles = np.arctan2(significant_flow_y, significant_flow_x)

        # Calculate circular variance of flow angles (0 = all same direction, high = random directions)
        mean_cos = np.mean(np.cos(flow_angles))
        mean_sin = np.mean(np.sin(flow_angles))
        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)

        # High circular variance indicates random motion directions (shake)
        direction_consistency = 100.0 - (circular_variance * 100)

        # 4. Spatial Distribution Analysis
        # Camera shake affects entire frame, localized motion affects specific regions
        motion_pixel_ratio = (
            np.sum(significant_motion_mask) / significant_motion_mask.size
        )

        # If most pixels have motion, it's likely camera movement - apply additional scrutiny
        if motion_pixel_ratio > 0.7:
            # For global motion, emphasize consistency and reasonable magnitude
            # If motion is very consistent but high magnitude, it might be camera movement
            if motion_consistency > 95 and global_motion_magnitude > 5:
                # Perfect consistency with high magnitude might indicate camera pan
                # Apply moderate penalty to prefer less camera movement
                spatial_consistency_score = 85.0 - min(
                    10, (global_motion_magnitude - 5) * 1
                )
            else:
                # Use average of motion and direction consistency for global motion
                spatial_consistency_score = (
                    motion_consistency + direction_consistency
                ) / 2
        else:
            # Localized motion is generally good (subject movement)
            spatial_consistency_score = 90.0

        # Combine all metrics with weights
        # Emphasize consistency metrics for shake detection
        final_stability = (
            0.30 * motion_consistency  # How consistent is the motion?
            + 0.30 * magnitude_score  # Is global motion magnitude reasonable?
            + 0.25 * direction_consistency  # Is motion direction consistent?
            + 0.15 * spatial_consistency_score  # Is spatial distribution reasonable?
        )

        return float(np.clip(final_stability, 0.0, 100.0))

    except Exception:
        # Return neutral score if analysis fails
        return 50.0


def analyze_video_file(
    file_path: PathLike,
    bpm: Optional[float] = None,
    min_beats: float = 1.0,
    min_scene_duration: Optional[float] = None,
) -> List[VideoChunk]:
    """Analyze video file and return scored chunks suitable for editing.

    Main function that combines all analysis methods:
    - Scene detection
    - Quality scoring
    - Motion analysis
    - Face detection (optional)

    Args:
        file_path: Path to video file
        bpm: Beats per minute of the music (for beat-based scene filtering)
        min_beats: Minimum duration for a scene in beats (default: 1.0)
        min_scene_duration: Minimum duration for a scene in seconds (fallback/override)

    Returns:
        List of VideoChunk objects sorted by quality score

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported
    """
    import logging

    # Set up detailed logging for video analysis
    logger = logging.getLogger("autocut.video_analyzer")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Calculate minimum scene duration using beat-based logic or fallback
    if min_scene_duration is not None:
        # Explicit override provided - use it
        calculated_min_duration = min_scene_duration
        logger.info(
            f"Using explicit minimum scene duration: {calculated_min_duration}s",
        )
    elif bpm is not None:
        # Calculate beat-based minimum duration
        beat_duration = 60.0 / bpm  # Duration of one beat in seconds
        calculated_min_duration = beat_duration * min_beats
        logger.info(
            f"Using beat-based minimum scene duration: {calculated_min_duration:.3f}s ({min_beats} beats at {bpm:.1f} BPM)",
        )
    else:
        # No BPM provided, use config default
        try:
            # Import config utility with dual import pattern
            try:
                from utils import get_config_value
            except ImportError:
                from .utils import get_config_value

            calculated_min_duration = get_config_value("min_clip_duration", 0.5)
        except ImportError:
            calculated_min_duration = 0.5
        logger.info(
            f"Using fallback minimum scene duration: {calculated_min_duration}s (no BPM available)",
        )

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    filename = Path(file_path).name
    logger.info(f"Starting analysis of video: {filename}")

    # Track processing statistics for detailed reporting
    processing_stats = {
        "file_path": file_path,
        "filename": filename,
        "load_success": False,
        "scene_detection_success": False,
        "scenes_found": 0,
        "valid_scenes": 0,
        "chunks_created": 0,
        "chunks_failed": 0,
        "errors": [],
    }

    try:
        # Step 1: Load video with detailed error tracking
        logger.info(f"Loading video file: {filename}")
        try:
            video, metadata = load_video(file_path)
            processing_stats["load_success"] = True
            logger.info(
                f"Video loaded successfully: {metadata['width']}x{metadata['height']}, {metadata['duration']:.2f}s, {metadata['fps']:.2f}fps",
            )

            # Log transcoding information if applicable
            if metadata.get("was_transcoded", False):
                logger.info(
                    f"Video was transcoded from: {file_path} -> {metadata['processed_file_path']}",
                )
        except Exception as e:
            error_msg = f"Failed to load video {filename}: {e!s}"
            processing_stats["errors"].append(error_msg)
            logger.exception(error_msg)
            raise ValueError(error_msg) from e

        # Step 2: Scene detection with detailed logging
        logger.info(f"Starting scene detection for: {filename}")
        try:
            scenes = detect_scenes(video, threshold=30.0)
            processing_stats["scene_detection_success"] = True
            processing_stats["scenes_found"] = len(scenes)
            logger.info(f"Scene detection complete: {len(scenes)} scenes found")

            if not scenes:
                error_msg = f"No scenes detected in {filename} - video may be too short or uniform"
                processing_stats["errors"].append(error_msg)
                logger.warning(error_msg)
        except Exception as e:
            error_msg = f"Scene detection failed for {filename}: {e!s}"
            processing_stats["errors"].append(error_msg)
            logger.exception(error_msg)
            # Don't raise here, try to continue with single scene
            scenes = [(0.0, metadata["duration"])]
            logger.info("Fallback: Using entire video as single scene")

        # Step 3: Filter scenes by minimum duration (using calculated beat-based duration)
        valid_scenes = [
            (start, end)
            for start, end in scenes
            if (end - start) >= calculated_min_duration
        ]
        processing_stats["valid_scenes"] = len(valid_scenes)

        if not valid_scenes:
            # If no scenes meet minimum duration, use the original scenes
            logger.warning(
                f"No scenes meet minimum duration {calculated_min_duration:.3f}s, using all {len(scenes)} scenes",
            )
            valid_scenes = scenes
            processing_stats["valid_scenes"] = len(valid_scenes)

        logger.info(
            f"Scene filtering complete: {len(valid_scenes)} valid scenes (min duration: {calculated_min_duration:.3f}s)",
        )

        # Step 4: Analyze each scene and create VideoChunk objects
        chunks = []
        scene_errors = []

        for scene_idx, (start_time, end_time) in enumerate(valid_scenes):
            scene_duration = end_time - start_time
            logger.debug(
                f"Analyzing scene {scene_idx + 1}/{len(valid_scenes)}: {start_time:.2f}-{end_time:.2f}s ({scene_duration:.2f}s)",
            )

            try:
                # Calculate basic quality score
                quality_score = score_scene(video, start_time, end_time)
                logger.debug(
                    f"Scene {scene_idx + 1} quality score: {quality_score:.2f}",
                )

                # Calculate motion score
                motion_score = detect_motion(video, start_time, end_time)
                logger.debug(f"Scene {scene_idx + 1} motion score: {motion_score:.2f}")

                # Calculate face count
                face_count = detect_faces(video, start_time, end_time)
                logger.debug(f"Scene {scene_idx + 1} face count: {face_count}")

                # Calculate camera stability score
                stability_score = detect_camera_shake(video, start_time, end_time)
                logger.debug(
                    f"Scene {scene_idx + 1} stability score: {stability_score:.2f}"
                )

                # Create enhanced metadata
                chunk_metadata = {
                    "source_file": Path(file_path).name,
                    "scene_index": scene_idx,
                    "video_width": metadata["width"],
                    "video_height": metadata["height"],
                    "video_fps": metadata["fps"],
                    "video_duration": metadata["duration"],
                    "quality_score": quality_score,
                    "motion_score": motion_score,
                    "face_count": face_count,
                    "stability_score": stability_score,
                }

                # Calculate enhanced combined score with stability penalty
                # Quality: 60% weight (image quality metrics)
                # Subject Motion: 15% weight (good localized movement)
                # Stability: 10% weight (penalizes camera shake)
                # Faces: 15% weight (cap at 100 for 4+ faces)
                face_score = min(100, face_count * 25)  # 4+ faces = 100 points

                enhanced_score = (
                    0.60 * quality_score
                    + 0.15 * motion_score
                    + 0.10 * stability_score
                    + 0.15 * face_score
                )

                # Create VideoChunk with canonical signature
                chunk = VideoChunk(
                    video_path=file_path,
                    start_time=start_time,
                    end_time=end_time,
                    score=enhanced_score,
                    motion_score=motion_score,
                    face_score=min(
                        100, face_count * 25
                    ),  # Convert face count to score (4+ faces = 100)
                    brightness_score=None,  # Individual components not extracted yet
                    sharpness_score=None,  # Individual components not extracted yet
                )

                chunks.append(chunk)
                processing_stats["chunks_created"] += 1
                logger.debug(
                    f"Scene {scene_idx + 1} chunk created: score={enhanced_score:.2f}",
                )

            except Exception as e:
                # Log detailed error for this specific scene but continue processing
                error_msg = f"Scene {scene_idx + 1} analysis failed: {e!s}"
                scene_errors.append(error_msg)
                processing_stats["chunks_failed"] += 1
                logger.warning(
                    f"Scene {scene_idx + 1} failed ({start_time:.2f}-{end_time:.2f}s): {e!s}",
                )
                continue

        # Clean up video object
        video.close()

        # Final processing statistics and warnings
        logger.info(f"Video analysis complete for {filename}:")
        logger.info(f"  - Scenes detected: {processing_stats['scenes_found']}")
        logger.info(f"  - Valid scenes: {processing_stats['valid_scenes']}")
        logger.info(f"  - Chunks created: {processing_stats['chunks_created']}")
        logger.info(f"  - Chunks failed: {processing_stats['chunks_failed']}")

        if scene_errors:
            logger.warning(f"Scene processing errors for {filename}:")
            for error in scene_errors:
                logger.warning(f"  - {error}")

        if not chunks:
            error_msg = f"No usable chunks created from {filename} - all {len(valid_scenes)} scenes failed analysis"
            processing_stats["errors"].append(error_msg)
            logger.error(error_msg)
            logger.error(f"Scene errors: {scene_errors}")

            # Provide detailed diagnosis
            if processing_stats["scenes_found"] == 0:
                logger.error(
                    "Root cause: No scenes detected - check if video is valid and not corrupted",
                )
            elif processing_stats["valid_scenes"] == 0:
                logger.error(
                    f"Root cause: No scenes meet minimum duration {calculated_min_duration:.3f}s",
                )
            else:
                logger.error(
                    "Root cause: All scene analysis steps failed - check video codec compatibility",
                )

            return []  # Return empty list instead of raising exception

        # Sort chunks by score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Successfully created {len(chunks)} chunks from {filename} (scores: {chunks[0].score:.1f}-{chunks[-1].score:.1f})",
        )
        return chunks

    except Exception as e:
        error_msg = f"Critical error analyzing {filename}: {e!s}"
        processing_stats["errors"].append(error_msg)
        logger.exception(error_msg)
        logger.exception(f"Processing stats: {processing_stats}")
        raise ValueError(error_msg) from e
