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
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Tuple of (VideoFileClip object, metadata dictionary)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is unsupported
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    if VideoFileClip is None:
        raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")
    
    try:
        # Load video with MoviePy
        video = VideoFileClip(file_path)
        
        # Extract metadata
        metadata = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,  # (width, height)
            'width': video.w,
            'height': video.h,
            'filename': os.path.basename(file_path),
            'file_path': file_path
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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    try:
        # Load video
        video, metadata = load_video(file_path)
        
        # Detect scenes
        scenes = detect_scenes(video, threshold=30.0)
        
        # Filter scenes by minimum duration
        valid_scenes = [
            (start, end) for start, end in scenes 
            if (end - start) >= min_scene_duration
        ]
        
        if not valid_scenes:
            # If no scenes meet minimum duration, use the original scenes
            valid_scenes = scenes
        
        # Analyze each scene and create VideoChunk objects
        chunks = []
        
        for start_time, end_time in valid_scenes:
            try:
                # Calculate basic quality score
                quality_score = score_scene(video, start_time, end_time)
                
                # Calculate motion score
                motion_score = detect_motion(video, start_time, end_time)
                
                # Calculate face count
                face_count = detect_faces(video, start_time, end_time)
                
                # Create enhanced metadata
                chunk_metadata = {
                    'source_file': os.path.basename(file_path),
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
                
            except Exception:
                # Skip problematic scenes but continue processing
                continue
        
        # Clean up video object
        video.close()
        
        # Sort chunks by score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        return chunks
        
    except Exception as e:
        raise ValueError(f"Failed to analyze video file {file_path}: {str(e)}")


if __name__ == "__main__":
    # Test script for video analysis
    print("AutoCut Video Analyzer - Test Mode")
    print("TODO: Add test with sample video files")