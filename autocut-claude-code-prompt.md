# AutoCut - Claude Code Development Prompt

## 🎯 Project Overview

### What is AutoCut?
AutoCut is a desktop application that automatically creates beat-synced highlight videos from raw footage and music. It analyzes video quality, detects the music's rhythm, and intelligently assembles clips that match the beat - all without requiring any video editing knowledge.

### Who is it for?
- **Primary Users**: Families, travelers, and hobbyists with lots of unedited video footage
- **Use Cases**: Family vacations, birthday parties, weddings, sports events, travel compilations
- **User Profile**: Non-technical users who want professional-looking results with minimal effort

### Core Value Proposition
Transform hours of raw footage into a polished, music-synced highlight reel in minutes, not hours.

---

## ⚠️ CRITICAL DEVELOPMENT INSTRUCTIONS FOR CLAUDE CODE

1. **ALWAYS test each implementation step before proceeding**
2. **STOP after each major step for manual testing and git commit**
3. **Create small, atomic commits with clear messages**
4. **If something doesn't work, debug it completely before moving forward**
5. **Keep the implementation SIMPLE - avoid over-engineering**
6. **Test with real video files after each step**

---

## 🛠️ Technical Stack

### Core Dependencies
```python
# requirements.txt
moviepy==1.0.3          # Video editing (better than raw ffmpeg for precise cuts)
librosa==0.10.1         # Audio analysis and beat detection
opencv-python==4.8.1    # Video frame analysis
numpy==1.24.3           # Numerical operations
scipy==1.11.4           # Signal processing
Pillow==10.1.0         # Image processing
tkinter                 # GUI (comes with Python)
```

### Why These Choices?
- **MoviePy**: Handles frame-accurate cuts, prevents stuttering, built-in transitions
- **Librosa**: Industry-standard for music analysis, reliable BPM detection
- **OpenCV**: Fast video frame analysis
- **Tkinter**: Simple GUI, no installation issues

---

## 📋 IMPLEMENTATION CHECKLIST

### ✅ Step 0: Project Setup - COMPLETED ✅
```bash
✅ Create project directory structure:
  autocut/
  ├── src/
  │   ├── __init__.py
  │   ├── audio_analyzer.py
  │   ├── video_analyzer.py
  │   ├── clip_assembler.py
  │   ├── gui.py
  │   └── utils.py
  ├── tests/
  ├── test_media/
  ├── output/
  ├── requirements.txt
  ├── README.md
  └── .gitignore

✅ Initialize git repository
✅ Create virtual environment
✅ Install dependencies
□ Create test_media folder with sample videos and music
✅ Test that all imports work

**COMPLETED** - Committed: "Initial project setup" (commit 111035f)
```

### ✅ Step 1: Audio Analysis Module - COMPLETED ✅
```python
# src/audio_analyzer.py

✅ Implement analyze_audio() function:
  - Load audio file using librosa
  - Detect BPM (tempo)
  - Extract beat timestamps
  - Calculate allowed clip durations based on BPM
  - Return structured data

✅ Implement get_cut_points() function:
  - Convert beats to potential cut points
  - Filter beats to avoid cuts too close together
  
✅ Create test script:
  - Comprehensive test suite with unit tests
  - BPM validation and edge case testing
  - Cut point generation testing

✅ Handle edge cases:
  - BPM range validation (30-300)
  - Empty beat sequences
  - Tempo correction for extreme values

**COMPLETED** - All functions implemented and tested successfully!
```

### ✅ Step 2: Basic Video Analysis - COMPLETED ✅
```python
# src/video_analyzer.py

✅ Implement load_video() function:
  - Use MoviePy to load video
  - Extract basic metadata (duration, fps, resolution)
  - Handle missing files and import errors

✅ Implement detect_scenes() function:
  - Simple scene detection based on frame differences
  - Sample frames every 0.5s for performance
  - Return list of (start_time, end_time) tuples
  - Filter scenes minimum 1 second length

✅ Implement score_scene() function:
  - Calculate sharpness (Laplacian variance)
  - Calculate brightness (mean pixel value) 
  - Calculate contrast (pixel value std deviation)
  - Return combined weighted score (0-100)
  - Sample multiple frames per scene

✅ Create test script:
  - Mock tests working without real files  
  - Real video testing with 16 diverse files (720p-4K, 24-30fps)
  - Performance optimizations: 1.0s sampling, progress indicators
  - Processing speed: ~8 videos/minute, ~2 minutes for all files
  - Quality scores: 38.0-79.4/100 range with good distribution

✅ Test with multiple videos:
  - 16 videos successfully processed (Crowd.mp4, various stock footage)
  - Scene detection: 1-14 scenes per video (adaptive threshold)
  - MoviePy import compatibility fixed for newer versions
  - Added --quick mode for development testing (first 3 videos)

**COMPLETED** - All basic video analysis functions implemented, optimized, and thoroughly tested!
```

### ✅ Step 3: Advanced Video Scoring
```python
# Continue in src/video_analyzer.py

□ Add motion detection:
  - Optical flow between frames
  - Higher motion = more interesting

□ Add face detection (optional):
  - Use OpenCV's cascade classifier
  - More faces = higher score

□ Implement analyze_video_file() main function:
  - Combines all scoring methods
  - Returns list of scored video chunks

□ Test with family videos:
  - Verify face detection works
  - Check that shaky footage gets low scores
  - Ensure dark/blurry scenes score low

**STOP HERE** - Test and refine scoring, commit: "Add advanced video scoring"
```

### ✅ Step 4: Clip Assembly Logic
```python
# src/clip_assembler.py

□ Implement basic assembly:
  - Match video chunks to beat grid
  - Respect minimum clip duration
  - Fill entire song duration

□ Add variety patterns:
  - Define 3-4 rhythm patterns
  - Prevent repetitive cutting
  - Mix short and long clips

□ Implement select_best_clips() function:
  - Sort clips by score
  - Ensure variety in source videos
  - Avoid using same scene twice

□ Create timeline data structure:
  - List of (video_file, start, end, beat_position)
  - Export as JSON for debugging

**STOP HERE** - Test assembly logic, commit: "Add clip assembly engine"
```

### ✅ Step 5: Video Rendering
```python
# src/clip_assembler.py (continued)

□ Implement render_video() function:
  - Use MoviePy CompositeVideoClip
  - Add music track
  - NO audio manipulation (prevent choppiness)

□ Add transitions:
  - Simple crossfade between clips
  - Fade in/out at start/end

□ Optimize rendering:
  - Use appropriate codec
  - Maintain source quality
  - Progress callback for GUI

□ Test full pipeline:
  - 3 videos + 1 song = 1 output
  - Check for stuttering
  - Verify music stays in sync

**STOP HERE** - Test rendering quality, commit: "Add video rendering"
```

### ✅ Step 6: Simple GUI
```python
# src/gui.py

□ Create basic Tkinter window:
  - File selection for videos (multiple)
  - File selection for music
  - Output location
  - "Generate" button
  - Progress bar

□ Add settings panel:
  - Tempo preference (Fast/Normal/Slow)
  - Target duration
  - Face priority (checkbox)

□ Implement threading:
  - Process in background
  - Update progress bar
  - Keep GUI responsive

□ Add preview capability:
  - Show first 10 seconds
  - Play with sound

**STOP HERE** - Test GUI thoroughly, commit: "Add basic GUI"
```

### ✅ Step 7: Polish and Packaging
```
□ Add error handling:
  - Unsupported file formats
  - Corrupted videos
  - No good clips found

□ Create presets:
  - "Action" - fast cuts
  - "Cinematic" - slow, dramatic
  - "Musical" - perfectly on beat

□ Write user documentation:
  - README with examples
  - Troubleshooting guide

□ Package application:
  - PyInstaller configuration
  - Test on fresh system
  - Create installer

**STOP HERE** - Final testing, commit: "Polish and package v1.0"
```

---

## 🎯 Core Algorithm Detail

### Beat-to-Clip Matching Logic
```python
def calculate_clip_constraints(bpm):
    """
    For a given BPM, calculate allowed clip durations
    
    Examples:
    - 60 BPM = 1 beat/second → clips: 1s, 2s, 4s, 8s
    - 120 BPM = 2 beats/second → clips: 0.5s, 1s, 2s, 4s
    - 90 BPM = 1.5 beats/second → clips: 0.67s, 1.33s, 2.67s, 5.33s
    """
    beat_duration = 60.0 / bpm
    
    # Minimum clip is 1 beat (but at least 0.5 seconds)
    min_duration = max(beat_duration, 0.5)
    
    # Allowed durations are musical multiples
    multipliers = [1, 2, 4, 8, 16]
    allowed_durations = [beat_duration * m for m in multipliers]
    
    # Filter out clips longer than 8 seconds
    allowed_durations = [d for d in allowed_durations if d <= 8.0]
    
    return min_duration, allowed_durations
```

### Variety Pattern System
```python
# Prevent monotony by varying clip lengths
VARIETY_PATTERNS = {
    'energetic': [1, 1, 2, 1, 1, 4],  # Mostly fast with occasional pause
    'buildup': [4, 2, 2, 1, 1, 1],    # Start slow, increase pace
    'balanced': [2, 1, 2, 4, 2, 1],   # Mixed pacing
    'dramatic': [1, 1, 1, 1, 8],      # Fast cuts then long hold
}
```

---

## 🐛 Common Issues and Solutions

### Stuttering/Freezing
- **Cause**: Imprecise cut points, codec issues
- **Solution**: Use MoviePy's `subclip()` method, always cut on keyframes

### Choppy Audio
- **Cause**: Trying to manipulate audio track
- **Solution**: NEVER modify audio, only sync video cuts to it

### Memory Issues
- **Cause**: Loading entire videos into memory
- **Solution**: Process videos in chunks, use generators

### Slow Processing
- **Cause**: Inefficient scene detection
- **Solution**: Skip frames (analyze every 10th frame), use threading

---

## 📊 Testing Checklist

Before considering any step complete:

□ **Unit Test**: Individual functions work correctly
□ **Integration Test**: Modules work together
□ **Real Media Test**: Use actual family videos
□ **Edge Cases**: Empty videos, silent music, single scene videos
□ **Performance**: Can handle 1 hour of footage
□ **Quality Check**: Output looks professional

---

## 🚀 Success Metrics

The implementation is successful when:

1. **Processing Speed**: 10 minutes of footage processes in <2 minutes
2. **Sync Accuracy**: Cuts align with beats >95% of the time
3. **Visual Quality**: No stuttering, smooth transitions
4. **Variety**: No more than 3 consecutive cuts of same duration
5. **Ease of Use**: Grandma can use it without help

---

## 💡 Remember

- **KISS Principle**: Keep It Simple, Stupid
- **Test Early, Test Often**: Don't write 100 lines without testing
- **User First**: Every decision should make it easier for non-technical users
- **Iterate**: Version 1 doesn't need to be perfect

**Most importantly**: After each step, STOP and wait for confirmation before proceeding. Each step should result in working, tested code that can be committed to git.