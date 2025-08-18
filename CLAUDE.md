# AutoCut V2 - Project Memory

## Project Overview

AutoCut is a desktop application that automatically creates beat-synced highlight videos from raw footage and music. It analyzes video quality, detects music rhythm, and intelligently assembles clips that match the beat.

### Target Users
- **Primary**: Families, travelers, hobbyists with unedited video footage
- **Use Cases**: Family vacations, birthday parties, weddings, sports events, travel compilations

### Core Value Proposition
Transform hours of raw footage into polished, music-synced highlight reel in minutes.

## 🎯 Current Status (January 2025)

**Status**: Core functionality complete with production-ready system
**Current Focus**: GUI development (Step 6)
**Architecture**: Modular, optimized, memory-safe, cross-platform compatible

### ✅ Core Features Complete

**Steps 1-5: Production-Ready Pipeline**
- ✅ Audio Analysis: >95% sync accuracy with musical intelligence
- ✅ Video Analysis: Multi-format support, quality scoring, motion/face detection  
- ✅ Clip Assembly: Beat-synced editing with variety patterns
- ✅ Video Rendering: Hardware-accelerated, visual artifact elimination
- ✅ System Optimization: Dynamic worker detection, memory safety

### 🎯 Next Steps

**Step 6: GUI Development** (Current Focus)
- Tkinter interface with file selection
- Settings panel and progress bar  
- Threading for responsive UI

**Step 7: Polish & Packaging**
- Error handling and presets
- Documentation and PyInstaller packaging

## Technical Architecture

### Core Dependencies
- moviepy==2.2.1 (video editing with compatibility layer)
- librosa>=0.10.1 (audio analysis)  
- opencv-python>=4.8.1 (video analysis)
- psutil>=5.9.0 (system monitoring)
- click>=8.0.0 (CLI framework)
- pytest>=7.0.0 (testing)

### Project Structure
```
src/
  audio_analyzer.py      # BPM detection, beat timestamps
  video_analyzer.py      # Scene detection, quality scoring  
  clip_assembler.py      # Beat matching, rendering
  gui.py                 # Tkinter interface
tests/
test_media/              # Sample videos and music
autocut.py               # Main CLI entry point
```

## Core Algorithms

### Beat-to-Clip Matching
- Musical timing constraints (1, 2, 4, 8, 16 beat multipliers)
- Minimum 0.5s clips, maximum 8s clips
- Align cuts to beat grid for natural flow

### Video Quality Scoring
- Quality (60%): sharpness, brightness, contrast
- Motion (25%): optical flow detection  
- Face Detection (15%): OpenCV cascade classifier

### Variety Patterns
- energetic, buildup, balanced, dramatic patterns
- Prevents repetitive same-duration cuts

## Development Workflow

### Environment Setup
- Virtual environment: `source env/bin/activate`
- Install dependencies from requirements.txt
- Create test_media folder with sample content

### Performance Achievements
- ✅ >95% sync accuracy 
- ✅ 2-3 minute processing (was 20+ minutes)
- ✅ Multi-format support (20+ video formats)
- ✅ Hardware acceleration (NVENC/QSV)
- ✅ Memory safety with dynamic worker detection

## Technical Constraints

### Audio Handling
- **NEVER manipulate audio track** (prevents choppiness)
- Only sync video cuts to existing music

### Video Processing  
- Frame-accurate cuts with MoviePy
- Process videos in chunks to avoid memory issues

## File Formats & Compatibility

### Supported Formats
- **Video**: MP4, MOV, AVI, MKV, WEBM, H.265/iPhone footage
- **Audio**: MP3, WAV, M4A, FLAC

## Development Commands

### Main CLI Commands
```bash
python autocut.py demo              # Run demonstration with test media
python autocut.py process *.mov --audio music.mp3 --pattern dramatic
python autocut.py validate video.mp4 --detailed
python autocut.py benchmark --detailed
```

### Development & Testing
```bash
make demo               # Run AutoCut demo
make test-unit          # Run unit tests  
make test-integration   # Run integration tests
make benchmark          # System performance test
make info               # Show project status
```

## Code Duplication Cleanup - August 2025

### Completed Cleanups ✅
- **Memory monitoring**: Consolidated to memory.monitor module (removed 12-line duplicate)
- **Audio loading**: Simplified compatibility wrappers (reduced from 11 to 3 lines)  
- **VideoChunk classes**: Unified to enhanced clip_selector version (5 classes → 1 canonical)
- **Import paths**: Standardized through video module exports (`from src.video import VideoChunk`)

### Preserved Functionality ✅
- **Backward compatibility**: All existing APIs maintained 100%
- **Performance**: No performance regressions introduced
- **Error handling**: All existing error handling preserved
- **Fallback mechanisms**: Multiple import fallbacks maintained for robustness

### Technical Results
- **Lines reduced**: ~60 lines of true duplication eliminated
- **Classes consolidated**: 5 VideoChunk definitions → 1 enhanced canonical version
- **Import complexity**: Simplified while maintaining compatibility
- **Module structure**: Enhanced separation of concerns

### Architecture Improvements
- **Single source of truth**: VideoChunk now has one authoritative implementation
- **Enhanced features**: Quality breakdown methods available across all modules
- **Cleaner imports**: Canonical path `from src.video import VideoChunk`
- **Maintainability**: Easier to modify VideoChunk behavior going forward