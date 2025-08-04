# AutoCut V2 Codebase Structure

## Project Directory Layout
```
AutoCutV2/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization (v2.0.0)
│   ├── audio_analyzer.py        # ✅ COMPLETE - Music analysis & BPM detection
│   ├── video_analyzer.py        # ✅ COMPLETE - Scene detection & quality scoring
│   ├── clip_assembler.py        # 🚧 IN PROGRESS - Beat matching & rendering
│   ├── gui.py                   # 🔜 PLANNED - Tkinter interface
│   └── utils.py                 # 🔜 PLANNED - Helper functions
├── test_media/                   # Test files (16 videos + 3 music files)
├── output/                       # Generated highlight videos
├── tests/                        # Unit tests directory
├── test_real_audio.py           # ✅ Audio analysis testing script
├── test_video_analysis.py       # ✅ Video analysis testing script  
├── requirements.txt             # Dependencies (moviepy, librosa, opencv, etc)
├── README.md                    # ✅ Comprehensive project documentation
├── CLAUDE.md                    # ✅ Project memory and development notes
├── autocut-claude-code-prompt.md # ✅ Implementation checklist (Steps 1-2 complete)
└── .gitignore                   # Git ignore patterns

Git History:
- b001338: Initial commit
- 111035f: Initial project setup (Step 0)
- 72c0dae: Add audio analysis module - Step 1 complete
- d46c721: Add basic video analysis - Step 2 complete
- f98c1a6: Optimize video analysis for multiple files
- 7b8beec: Update all project documentation with current progress
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

### ✅ src/video_analyzer.py (COMPLETE)
**Purpose**: Video processing and quality analysis
**Key Classes/Functions**:
- `VideoChunk` class - Represents scored video segments
- `load_video(file_path)` - MoviePy loading with metadata
- `detect_scenes(video, threshold)` - Frame difference scene detection
- `score_scene(video, start, end)` - Quality scoring (sharpness/brightness/contrast)
- `detect_motion()` - 🚧 Motion detection (placeholder)
- `detect_faces()` - 🚧 Face detection (placeholder)
- `analyze_video_file()` - 🚧 Complete pipeline (placeholder)

**Status**: Basic functions complete, advanced features in progress
**Performance**: ~8 videos/minute, scores 38-79/100, optimized 1.0s sampling

### 🚧 src/clip_assembler.py (IN PROGRESS)
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
- Tests video analysis with real video files
- Supports MP4, AVI, MOV, MKV, WEBM formats
- Shows scenes, quality scores, best clips
- Performance optimizations: --quick mode, progress indicators
- Successfully tested with 16 diverse video files

## Key Implementation Patterns

### Error Handling
- Try/catch blocks with meaningful error messages
- FileNotFoundError for missing files
- ImportError fallbacks for optional dependencies
- Graceful degradation for problematic frames

### Performance Optimizations
- Frame sampling (1.0s intervals vs 0.5s for 2x speed)
- Chunk processing to avoid memory issues
- Progress indicators for user feedback
- Optional quick testing modes

### Testing Philosophy
- Real media testing (not just mock data)
- Performance measurement and optimization
- Edge case handling (empty files, extreme values)
- User-friendly output formatting