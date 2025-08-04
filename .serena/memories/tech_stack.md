# AutoCut V2 Technical Stack

## Core Dependencies (requirements.txt)
```python
moviepy>=1.0.3          # Video editing - frame-accurate cuts, prevents stuttering
librosa>=0.10.1         # Audio analysis - industry-standard BPM detection  
opencv-python>=4.8.0    # Video frame analysis - fast computer vision
numpy>=1.24.0           # Numerical operations - matrix processing
scipy>=1.11.0           # Signal processing - audio/video algorithms
Pillow>=10.0.0         # Image processing - frame manipulation
# tkinter - GUI framework (built into Python)
```

## Version Compatibility Issues Resolved
- **MoviePy Import**: Fixed compatibility with newer versions using fallback imports
  ```python
  try:
      from moviepy import VideoFileClip  # Direct import works
  except ImportError:
      from moviepy.editor import VideoFileClip  # Fallback
  ```
- **OpenCV**: Updated from exact version (4.8.1) to flexible (>=4.8.0) 
- **NumPy**: Resolved formatting issues with numpy arrays in f-strings

## Architecture Decisions & Rationale

### Audio Processing: Librosa
**Why Chosen**: Industry standard for music information retrieval
**Key Features Used**:
- BPM detection with `librosa.beat.beat_track()`
- Harmonic-percussive separation for better beat detection
- Frame-to-time conversion for precise timestamps
- Tempo validation and correction for edge cases

**Performance**: 
- Processes 3 music files (74-193 seconds) quickly
- Accurate BPM detection (99-123 range tested)
- Proper beat extraction (134-368 beats per song)

### Video Processing: MoviePy + OpenCV
**MoviePy**: High-level video editing
- Frame-accurate cuts without stuttering
- Metadata extraction (duration, fps, resolution)
- Future: Rendering and composition

**OpenCV**: Low-level frame analysis
- Scene detection via frame differences
- Quality scoring (Laplacian variance for sharpness)
- Brightness/contrast calculations
- Future: Motion detection, face detection

**Performance**:
- ~8 videos/minute processing speed
- Handles multiple formats (MP4, AVI, MOV, MKV, WEBM)
- Multi-resolution support (720p to 4K)
- Optimized sampling (1.0s intervals) for 2x speed improvement

### Data Structures

#### VideoChunk Class
```python
class VideoChunk:
    def __init__(self, start_time, end_time, score, video_path, metadata=None)
    # Properties: duration, __repr__
```

#### ClipTimeline Class
```python
class ClipTimeline:
    def add_clip(video_file, start, end, beat_position, score)
    def export_json(file_path)  # For debugging
    def get_total_duration()
```

## Algorithm Implementations

### Audio Analysis Algorithm
```python
def analyze_audio(file_path):
    # 1. Load with librosa
    y, sr = librosa.load(file_path)
    
    # 2. Harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 3. Beat tracking on percussive component  
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    
    # 4. Convert to timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
```

### Video Quality Scoring Algorithm
```python
def score_scene(video, start_time, end_time):
    # Sample 1-3 frames per scene
    # For each frame:
    #   1. Sharpness: Laplacian variance (40% weight)
    #   2. Brightness: Mean pixel value, optimal ~128 (30% weight)  
    #   3. Contrast: Std deviation (30% weight)
    # Return weighted average 0-100
```

### Scene Detection Algorithm
```python
def detect_scenes(video, threshold=30.0):
    # 1. Sample frames every 1.0 seconds
    # 2. Calculate mean absolute difference between consecutive frames
    # 3. Mark scene change if difference > threshold
    # 4. Filter scenes < 1 second duration
    # 5. Return (start_time, end_time) tuples
```

## Performance Optimizations Applied

### Frame Sampling Optimization
- **Before**: 0.5s intervals (slow, detailed)
- **After**: 1.0s intervals (2x faster, sufficient accuracy)
- **Result**: 16 videos processed in ~2 minutes vs. >2 minute timeout

### User Experience Improvements
- Progress indicators: `[X/Y]` format shows processing status
- Quick testing mode: `--quick` flag processes first 3 videos only
- Error handling: Graceful fallbacks for problematic frames/files
- Clean output: Fixed numpy formatting issues in test scripts

## Testing Infrastructure

### Real Media Testing
- **Audio**: 3 music files (soft-positive-summer-pop, this-heart-is-yours, upbeat-summer-pop)
- **Video**: 16 diverse files (various stock footage + Crowd.mp4)
- **Formats**: Mixed resolutions (720p-4K), frame rates (24-30fps)

### Performance Metrics Achieved
- **Processing Speed**: ~8 videos per minute
- **Quality Range**: 38.0-79.4/100 (good distribution)
- **Scene Detection**: 1-14 scenes per video (adaptive)
- **Memory Efficiency**: No memory issues with large files

## Development Tools & Workflow

### Environment Setup
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Testing Commands
```bash
python test_real_audio.py                    # Test audio analysis
python test_video_analysis.py               # Test all videos  
python test_video_analysis.py --quick       # Test first 3 videos
python test_video_analysis.py --mock        # Test without real files
```

### Git Workflow
- Atomic commits for each major step
- Comprehensive commit messages with context
- Documentation updates committed separately
- Test before commit principle