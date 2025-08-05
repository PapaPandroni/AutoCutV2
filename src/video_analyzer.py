"""
Video Analysis Module for AutoCut

Handles video processing including scene detection, quality scoring,
motion analysis, and face detection.
"""

from typing import Dict, List, Tuple, Optional
import os
import cv2
import numpy as np
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        # Fallback for testing without full moviepy installation
        VideoFileClip = None


class VideoChunk:
    """Represents a scored video segment."""
    
    def __init__(self, start_time: float, end_time: float, score: float, 
                 video_path: str, metadata: Optional[Dict] = None):
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
        self.video_path = video_path
        self.metadata = metadata or {}
        
    @property
    def duration(self) -> float:
        """Duration of the video chunk in seconds."""
        return self.end_time - self.start_time
        
    def __repr__(self) -> str:
        return f"VideoChunk({self.start_time:.1f}-{self.end_time:.1f}, score={self.score:.1f})"


def load_video(file_path: str) -> Tuple[VideoFileClip, Dict]:
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
    from .utils import preprocess_video_if_needed
    
    if not os.path.exists(file_path):
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
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,  # (width, height)
            'width': video.w,
            'height': video.h,
            'filename': os.path.basename(file_path),  # Original filename
            'file_path': file_path,  # Original file path
            'processed_file_path': processed_file_path,  # May be different if transcoded
            'was_transcoded': processed_file_path != file_path
        }
        
        return video, metadata
        
    except Exception as e:
        raise ValueError(f"Failed to load video file {file_path}: {str(e)}")


def detect_scenes(video: VideoFileClip, threshold: float = 30.0) -> List[Tuple[float, float]]:
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
    
    for t in timestamps[1:]:  # Skip first timestamp
        try:
            # Get frame as numpy array
            frame = video.get_frame(t)
            
            if prev_frame is not None:
                # Calculate frame difference (mean absolute difference)
                diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                
                # If difference exceeds threshold, mark as scene change
                if diff > threshold:
                    scene_changes.append(t)
            
            prev_frame = frame
            
        except Exception:
            # Skip problematic frames
            continue
    
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
            start_time + duration * 0.8
        ]
    
    scores = []
    
    for t in sample_times:
        try:
            # Get frame as RGB numpy array
            frame = video.get_frame(t)
            
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
                0.4 * sharpness_score +
                0.3 * brightness_score +
                0.3 * contrast_score
            )
            
            scores.append(frame_score)
            
        except Exception:
            # Skip problematic frames
            continue
    
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
                    blockSize=7
                )
                
                if corners is not None and len(corners) > 10:
                    # Calculate optical flow
                    next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_frame, gray, corners, None
                    )
                    
                    # Select good points
                    good_new = next_corners[status == 1]
                    good_old = corners[status == 1]
                    
                    if len(good_new) > 5:
                        # Calculate motion vectors
                        motion_vectors = good_new - good_old
                        
                        # Calculate motion magnitude
                        motion_magnitudes = np.sqrt(
                            motion_vectors[:, 0] ** 2 + motion_vectors[:, 1] ** 2
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
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        face_cascade = None
        for path in cascade_paths:
            try:
                if os.path.exists(path):
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
                start_time + duration * 0.8
            ]
        
        face_counts = []
        
        for t in sample_times:
            try:
                # Get frame and convert to grayscale
                frame = video.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                face_counts.append(len(faces))
                
            except Exception:
                # Skip problematic frames
                continue
        
        if not face_counts:
            return 0
        
        # Return maximum faces detected in any frame
        # (assumption: family videos benefit from more faces visible)
        return int(max(face_counts))
        
    except Exception:
        # Return 0 if face detection fails
        return 0


def analyze_video_file(file_path: str, min_scene_duration: float = 2.0) -> List[VideoChunk]:
    """Analyze video file and return scored chunks suitable for editing.
    
    Main function that combines all analysis methods:
    - Scene detection
    - Quality scoring
    - Motion analysis
    - Face detection (optional)
    
    Args:
        file_path: Path to video file
        min_scene_duration: Minimum duration for a scene in seconds
        
    Returns:
        List of VideoChunk objects sorted by quality score
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported
    """
    import logging
    
    # Set up detailed logging for video analysis
    logger = logging.getLogger('autocut.video_analyzer')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    filename = os.path.basename(file_path)
    logger.info(f"Starting analysis of video: {filename}")
    
    # Track processing statistics for detailed reporting
    processing_stats = {
        'file_path': file_path,
        'filename': filename,
        'load_success': False,
        'scene_detection_success': False,
        'scenes_found': 0,
        'valid_scenes': 0,
        'chunks_created': 0,
        'chunks_failed': 0,
        'errors': []
    }
    
    try:
        # Step 1: Load video with detailed error tracking
        logger.info(f"Loading video file: {filename}")
        try:
            video, metadata = load_video(file_path)
            processing_stats['load_success'] = True
            logger.info(f"Video loaded successfully: {metadata['width']}x{metadata['height']}, {metadata['duration']:.2f}s, {metadata['fps']:.2f}fps")
            
            # Log transcoding information if applicable
            if metadata.get('was_transcoded', False):
                logger.info(f"Video was transcoded from: {file_path} -> {metadata['processed_file_path']}")
        except Exception as e:
            error_msg = f"Failed to load video {filename}: {str(e)}"
            processing_stats['errors'].append(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 2: Scene detection with detailed logging
        logger.info(f"Starting scene detection for: {filename}")
        try:
            scenes = detect_scenes(video, threshold=30.0)
            processing_stats['scene_detection_success'] = True
            processing_stats['scenes_found'] = len(scenes)
            logger.info(f"Scene detection complete: {len(scenes)} scenes found")
            
            if not scenes:
                error_msg = f"No scenes detected in {filename} - video may be too short or uniform"
                processing_stats['errors'].append(error_msg)
                logger.warning(error_msg)
        except Exception as e:
            error_msg = f"Scene detection failed for {filename}: {str(e)}"
            processing_stats['errors'].append(error_msg)
            logger.error(error_msg)
            # Don't raise here, try to continue with single scene
            scenes = [(0.0, metadata['duration'])]
            logger.info(f"Fallback: Using entire video as single scene")
        
        # Step 3: Filter scenes by minimum duration
        valid_scenes = [
            (start, end) for start, end in scenes 
            if (end - start) >= min_scene_duration
        ]
        processing_stats['valid_scenes'] = len(valid_scenes)
        
        if not valid_scenes:
            # If no scenes meet minimum duration, use the original scenes
            logger.warning(f"No scenes meet minimum duration {min_scene_duration}s, using all {len(scenes)} scenes")
            valid_scenes = scenes
            processing_stats['valid_scenes'] = len(valid_scenes)
        
        logger.info(f"Scene filtering complete: {len(valid_scenes)} valid scenes (min duration: {min_scene_duration}s)")
        
        # Step 4: Analyze each scene and create VideoChunk objects
        chunks = []
        scene_errors = []
        
        for scene_idx, (start_time, end_time) in enumerate(valid_scenes):
            scene_duration = end_time - start_time
            logger.debug(f"Analyzing scene {scene_idx+1}/{len(valid_scenes)}: {start_time:.2f}-{end_time:.2f}s ({scene_duration:.2f}s)")
            
            try:
                # Calculate basic quality score
                quality_score = score_scene(video, start_time, end_time)
                logger.debug(f"Scene {scene_idx+1} quality score: {quality_score:.2f}")
                
                # Calculate motion score
                motion_score = detect_motion(video, start_time, end_time)
                logger.debug(f"Scene {scene_idx+1} motion score: {motion_score:.2f}")
                
                # Calculate face count
                face_count = detect_faces(video, start_time, end_time)
                logger.debug(f"Scene {scene_idx+1} face count: {face_count}")
                
                # Create enhanced metadata
                chunk_metadata = {
                    'source_file': os.path.basename(file_path),
                    'scene_index': scene_idx,
                    'video_width': metadata['width'],
                    'video_height': metadata['height'],
                    'video_fps': metadata['fps'],
                    'video_duration': metadata['duration'],
                    'quality_score': quality_score,
                    'motion_score': motion_score,
                    'face_count': face_count
                }
                
                # Calculate enhanced combined score
                # Base quality: 60% weight
                # Motion: 25% weight (normalized to 0-100)
                # Faces: 15% weight (cap at 100 for 4+ faces)
                face_score = min(100, face_count * 25)  # 4+ faces = 100 points
                
                enhanced_score = (
                    0.60 * quality_score +
                    0.25 * motion_score +
                    0.15 * face_score
                )
                
                # Create VideoChunk
                chunk = VideoChunk(
                    start_time=start_time,
                    end_time=end_time,
                    score=enhanced_score,
                    video_path=file_path,
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                processing_stats['chunks_created'] += 1
                logger.debug(f"Scene {scene_idx+1} chunk created: score={enhanced_score:.2f}")
                
            except Exception as e:
                # Log detailed error for this specific scene but continue processing
                error_msg = f"Scene {scene_idx+1} analysis failed: {str(e)}"
                scene_errors.append(error_msg)
                processing_stats['chunks_failed'] += 1
                logger.warning(f"Scene {scene_idx+1} failed ({start_time:.2f}-{end_time:.2f}s): {str(e)}")
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
            processing_stats['errors'].append(error_msg)
            logger.error(error_msg)
            logger.error(f"Scene errors: {scene_errors}")
            
            # Provide detailed diagnosis
            if processing_stats['scenes_found'] == 0:
                logger.error(f"Root cause: No scenes detected - check if video is valid and not corrupted")
            elif processing_stats['valid_scenes'] == 0:
                logger.error(f"Root cause: No scenes meet minimum duration {min_scene_duration}s")
            else:
                logger.error(f"Root cause: All scene analysis steps failed - check video codec compatibility")
            
            return []  # Return empty list instead of raising exception
        
        # Sort chunks by score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Successfully created {len(chunks)} chunks from {filename} (scores: {chunks[0].score:.1f}-{chunks[-1].score:.1f})")
        return chunks
        
    except Exception as e:
        error_msg = f"Critical error analyzing {filename}: {str(e)}"
        processing_stats['errors'].append(error_msg)
        logger.error(error_msg)
        logger.error(f"Processing stats: {processing_stats}")
        raise ValueError(error_msg)


if __name__ == "__main__":
    # Test script for video analysis
    print("AutoCut Video Analyzer - Test Mode")
    print("TODO: Add test with sample video files")