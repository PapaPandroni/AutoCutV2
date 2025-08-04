# AutoCut V2 Codebase Structure

## Project Directory Layout
```
AutoCutV2/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization (v2.0.0)
│   ├── audio_analyzer.py        # ✅ COMPLETE - Music analysis & BPM detection
│   ├── video_analyzer.py        # ✅ COMPLETE - Scene detection & advanced scoring
│   ├── clip_assembler.py        # 🔜 NEXT - Beat matching & rendering
│   ├── gui.py                   # 🔜 PLANNED - Tkinter interface
│   └── utils.py                 # 🔜 PLANNED - Helper functions
├── test_media/                   # Test files (16 videos + 3 music files)
├── output/                       # Generated highlight videos
├── tests/                        # Unit tests directory
├── test_real_audio.py           # ✅ Audio analysis testing script
├── test_video_analysis.py       # ✅ Basic video analysis testing script  
├── test_step3_complete.py       # ✅ Advanced video analysis testing script
├── requirements.txt             # Dependencies (moviepy, librosa, opencv, etc)
├── README.md                    # ✅ Comprehensive project documentation
├── CLAUDE.md                    # ✅ Project memory and development notes
├── autocut-claude-code-prompt.md # ✅ Implementation checklist (Steps 1-3 complete)
└── .gitignore                   # Git ignore patterns

Git History:
- b001338: Initial commit
- 111035f: Initial project setup (Step 0)
- 72c0dae: Add audio analysis module - Step 1 complete
- d46c721: Add basic video analysis - Step 2 complete
- f98c1a6: Optimize video analysis for multiple files
- 7b8beec: Update all project documentation with current progress
- 5343c94: Update all Serena memory files with comprehensive current status
- [CURRENT]: Step 3 advanced video scoring completion
```

## Module Details

### ✅ src/audio_analyzer.py (COMPLETE)
**Purpose**: Music analysis and BPM detection
**Key Functions**:
- `analyze_audio(file_path)` - Main analysis with librosa
- `calculate_clip_constraints(bpm)` - Musical timing calculation
- `get_cut_points(beats, duration)` - Beat-to-cut conversion
- `test_audio_analyzer()` - Built-in testing

**Status**: Fully implemented and tested with 3 real music files
**Performance**: Accurate BPM detection (99-123 range), proper beat extraction

### ✅ src/video_analyzer.py (COMPLETE - Step 3 Finished)
**Purpose**: Video processing and advanced quality analysis
**Key Classes/Functions**:
- `VideoChunk` class - Represents scored video segments with rich metadata
- `load_video(file_path)` - MoviePy loading with metadata extraction
- `detect_scenes(video, threshold)` - Frame difference scene detection
- `score_scene(video, start, end)` - Basic quality scoring (sharpness/brightness/contrast)
- `detect_motion(video, start, end)` - ✅ COMPLETE - Optical flow motion detection (Lucas-Kanade)
- `detect_faces(video, start, end)` - ✅ COMPLETE - OpenCV Haar cascade face detection  
- `analyze_video_file(file_path)` - ✅ COMPLETE - Full pipeline with enhanced scoring

**Status**: FULLY COMPLETE including advanced features
**Enhanced Scoring**: Quality (60%) + Motion (25%) + Faces (15%) = 0-100 scale
**Performance**: Basic ~8 videos/minute, enhanced scoring 68-73/100 range
**Motion Detection**: Detects both camera and object motion using optical flow
**Face Detection**: Successfully detects 1-3 faces per video segment for family video prioritization

### 🔜 src/clip_assembler.py (NEXT PRIORITY)
**Purpose**: Beat matching and video rendering
**Key Components**:
- `VARIETY_PATTERNS` - Rhythm patterns for cutting variety
- `ClipTimeline` class - Timeline data structure
- `match_clips_to_beats()` - Beat synchronization logic
- `select_best_clips()` - Quality-based clip selection
- `render_video()` - MoviePy rendering with music sync
- `add_transitions()` - Crossfade implementation

**Status**: Structure defined, functions need implementation

### 🔜 src/gui.py (PLANNED)
**Purpose**: Tkinter desktop interface
**Planned Features**:
- File selection (videos + music)
- Settings panel (tempo, duration, face priority)
- Progress bar with threading
- Preview capability

### 🔜 src/utils.py (PLANNED)
**Purpose**: Shared utilities and helpers

## Testing Infrastructure

### ✅ test_real_audio.py
- Tests audio analysis with real music files
- Supports MP3, WAV, M4A, FLAC formats
- Shows BPM, beats, duration, allowed clip durations
- Fixed numpy formatting issues for clean output

### ✅ test_video_analysis.py  
- Tests basic video analysis with real video files
- Supports MP4, AVI, MOV, MKV, WEBM formats
- Shows scenes, quality scores, best clips
- Performance optimizations: --quick mode, progress indicators
- Successfully tested with 16 diverse video files (Steps 1-2)

### ✅ test_step3_complete.py
- Tests complete advanced video analysis pipeline
- Uses analyze_video_file() with enhanced scoring
- Shows motion scores, face counts, enhanced combined scores
- Tests Step 3 functionality: motion detection + face detection
- Successfully tested with 3 videos showing 68-73/100 enhanced scores

## Key Implementation Patterns

### Error Handling
- Try/catch blocks with meaningful error messages
- FileNotFoundError for missing files
- ImportError fallbacks for optional dependencies
- Graceful degradation for problematic frames
- Silent exception handling in analyze_video_file() to skip bad scenes

### Performance Optimizations
- Frame sampling (1.0s intervals vs 0.5s for 2x speed)
- Chunk processing to avoid memory issues
- Progress indicators for user feedback
- Optional quick testing modes
- Motion detection sampling (every 0.5s, max 6 frames)
- Face detection fallback handling for missing cascade files

### Testing Philosophy
- Real media testing (not just mock data)
- Performance measurement and optimization
- Edge case handling (empty files, extreme values)
- User-friendly output formatting
- Step-by-step validation of each implementation phase

## Current Implementation Status
- **Steps 1-3: COMPLETE** - Audio analysis, basic video analysis, advanced video scoring
- **Step 4: NEXT** - Clip assembly logic with beat matching
- **Steps 5-7: PLANNED** - Video rendering, GUI, polish/packaging