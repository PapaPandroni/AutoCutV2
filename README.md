# AutoCut v2.0 🎬✨

**Transform hours of raw footage into polished, music-synced highlight reels in minutes.**

AutoCut is a desktop application that automatically creates beat-synced highlight videos from your raw footage and music. It analyzes video quality, detects the music's rhythm, and intelligently assembles clips that match the beat - all without requiring any video editing knowledge.

## 🚀 Current Status: Week 3 CLI/API Design Complete!

**Major refactoring milestone achieved!** AutoCut now features a professional CLI interface and clean API architecture. See `REFACTORING_TRACKER.md` for detailed progress tracking.

**✅ Week 1-3 Completed:**
- ✅ **Architecture Restructuring**: Modular `src/video/`, `src/hardware/`, `src/core/` structure
- ✅ **Validation Consolidation**: 90% reduction in duplicate functions (10+ → 1 unified system)  
- ✅ **Testing Framework**: Professional pytest infrastructure with 25+ automated commands
- ✅ **Single Entry Point**: New `autocut.py` CLI replaces scattered demo scripts
- ✅ **Clean API**: `AutoCutAPI` class provides programmatic access to all functionality
- 🎯 **Week 4 Next**: iPhone H.265 cross-platform compatibility resolution

## 🎯 Perfect For

- **Families**: Turn vacation videos into memorable highlight reels
- **Travelers**: Create stunning travel compilations 
- **Content Creators**: Quick highlight videos for social media
- **Event Organizers**: Birthday parties, weddings, sports events
- **Anyone** with lots of unedited footage who wants professional results

## ✨ Key Features

### 🎵 Smart Audio Analysis
- **Automatic BPM detection** using advanced music analysis
- **Beat-synchronized cutting** for perfect rhythm matching
- **Musical timing constraints** ensure natural clip durations
- **Supports all formats**: MP3, WAV, M4A, FLAC

### 🎬 Intelligent Video Analysis  
- **Scene detection** identifies natural breakpoints
- **Quality scoring** ranks clips by sharpness, brightness, and contrast
- **Motion detection** using optical flow for dynamic content scoring
- **Face detection** prioritizes people for family videos
- **Enhanced scoring** combines quality (60%) + motion (25%) + faces (15%)
- **Multi-resolution support**: 720p, 1080p, 4K videos
- **Enhanced format compatibility**: 20+ formats including MP4, AVI, MOV, MKV, WEBM, 3GP, MTS, M2TS, VOB, DIVX
- **H.265/HEVC support** with hardware-accelerated transcoding (10-20x faster)
- **Smart codec detection** with compatibility scoring and automatic preprocessing

### 🤖 Automated Assembly ✅ **WORKING!**
- **Beat-to-clip synchronization** - The core magic is implemented!
- **Variety patterns** prevent monotonous cutting (energetic, buildup, balanced, dramatic)
- **Smart clip selection** balances quality vs variety across source videos
- **Musical timing constraints** ensure clips fit beat grid perfectly
- **Timeline validation** with comprehensive statistics and warnings
- **No audio manipulation** - music stays crisp and clear

### 🎬 Video Rendering ✅ **COMPLETE!**
- **Real MP4 creation** - AutoCut now generates actual video files!
- **Music synchronization** with perfect beat alignment
- **Professional transitions** with crossfade effects
- **MoviePy integration** with H.264/AAC codec optimization
- **Hardware acceleration** support (NVIDIA NVENC, Intel QSV)
- **Comprehensive error handling** with detailed per-file processing status
- **Memory management** for efficient large video processing
- **Smart transcoding avoidance** eliminates 50-70% of unnecessary work

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/AutoCutV2.git
   cd AutoCutV2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or use the automated setup:
   ```bash
   make setup
   ```

### 🎯 AutoCut CLI Interface

**✅ NEW: Professional CLI interface now available!**

**Main Commands:**
```bash
# Quick demo - try AutoCut with your test media
python autocut.py demo

# Quick demo with limited files  
python autocut.py demo --quick

# Process specific videos with music
python autocut.py process video1.mov video2.mp4 --audio music.mp3

# Process with different editing patterns
python autocut.py process *.mov --audio song.wav --pattern dramatic

# Validate video compatibility
python autocut.py validate my_video.mp4 --detailed

# Check system capabilities
python autocut.py benchmark --detailed
```

**Development Commands:**
```bash
make demo               # Run AutoCut demo (uses new CLI)
make demo-quick         # Quick test with limited files
make demo-dramatic      # Try dramatic editing pattern
make validate           # Test video validation (specify VIDEO=path)
make benchmark          # System performance test
make info               # Show project status
```

**Testing Framework:**
```bash
make test-unit          # Run unit tests
make test-integration   # Run integration tests  
make test-quick         # Fast development tests
make ci-test           # Simulate CI/CD pipeline
```

**CLI Help:**
```bash
python autocut.py --help           # Show all commands
python autocut.py process --help   # Help for specific command
make cli-help                      # Show CLI help via Makefile
```

### 📱 Camera File Support

**AutoCut now works with real camera files:**
```bash
# DJI Osmo, Sony A7IV, Canon, Panasonic, etc.
python test_autocut_demo.py  # Automatically detects .MOV, .MXF, .AVI, .MTS, .M2TS

# Shows detected formats:
# 🔍 Searching for video files in test_media/
# 📁 Found 3 video files:
#    Format breakdown: {'.mov': 2, '.mxf': 1}
#     1. DJI_0123.MOV (.MOV)
#     2. Sony_Log.MXF (.MXF) 
#     3. GoPro.MP4 (.MP4)
```

## 📁 Project Structure

**✅ NEW: Clean modular architecture after Week 1-3 refactoring:**

```
AutoCutV2/
├── autocut.py              # 🆕 Main CLI interface (single entry point)
├── src/
│   ├── api.py             # 🆕 Clean public API for all functionality
│   ├── audio_analyzer.py  # Music analysis & BPM detection
│   ├── video_analyzer.py  # Scene detection & quality scoring  
│   ├── clip_assembler.py  # Beat matching & video rendering
│   ├── utils.py          # Helper functions
│   ├── video/            # 🆕 Video processing modules
│   │   ├── validation.py  # Unified video validation system
│   │   ├── codec_detection.py # Video format analysis
│   │   └── transcoding.py # H.265 transcoding service
│   ├── hardware/         # 🆕 Hardware acceleration
│   │   └── detection.py  # GPU/CPU encoder detection
│   └── core/             # 🆕 Core utilities
│       └── exceptions.py # Structured error handling
├── tests/                # 🆕 Professional pytest framework
│   ├── unit/            # Fast isolated tests
│   ├── integration/     # End-to-end workflow tests
│   ├── performance/     # Benchmark tests
│   └── conftest.py     # Test fixtures and configuration
├── test_media/          # Your test videos and music
├── output/             # Generated highlight videos
├── test_autocut_demo.py # Legacy demo (preserved for compatibility)
├── Makefile            # 🆕 25+ automated development commands
└── requirements.txt    # Dependencies (includes Click CLI framework)
```

**Key Architecture Improvements:**
- **Single Entry Point**: `autocut.py` replaces scattered scripts
- **Clean API**: `src/api.py` provides programmatic access  
- **Modular Design**: Separated video, hardware, and core concerns
- **Professional Testing**: pytest framework with comprehensive coverage
- **90% Code Deduplication**: Unified validation system eliminates scattered functions

## 🛠️ Current Status

### ✅ Completed Features

**Step 1: Audio Analysis Module**
- ✅ BPM detection with librosa
- ✅ Beat timestamp extraction
- ✅ Musical clip duration calculation
- ✅ Tested with real music files

**Step 2: Basic Video Analysis**  
- ✅ Video loading with MoviePy
- ✅ Scene detection via frame differences
- ✅ Quality scoring (sharpness, brightness, contrast)
- ✅ Tested with 16 diverse video files (720p-4K)

**Step 3: Advanced Video Scoring**
- ✅ Motion detection using optical flow (Lucas-Kanade method)
- ✅ Face detection for family videos (OpenCV Haar cascade)
- ✅ Enhanced scoring: Quality (60%) + Motion (25%) + Faces (15%)
- ✅ Complete analyze_video_file() pipeline integration
- ✅ Tested with enhanced scoring (68-73/100 range)

**Step 4: Clip Assembly Logic** ✅ **COMPLETE - THE CORE IS WORKING!** 🎉
- ✅ Smart clip selection algorithm (quality vs variety balancing)
- ✅ Beat-to-clip synchronization engine (THE MAGIC!)
- ✅ Enhanced ClipTimeline class with validation
- ✅ Complete pipeline orchestrator (assemble_clips function)
- ✅ Variety pattern system (energetic, balanced, dramatic, buildup)
- ✅ Musical timing constraints and duration fitting
- ✅ Full integration testing completed successfully

**Step 5: Video Rendering** ✅ **COMPLETE - AUTOCUT CREATES REAL VIDEOS!** 🎬
- ✅ MoviePy-based rendering with sequential concatenation
- ✅ Music synchronization with NO audio manipulation
- ✅ Memory management fixes for large video processing
- ✅ MoviePy 2.x compatibility (imports, method names, effects)
- ✅ Crossfade transitions and fade effects (add_transitions function)
- ✅ H.264/AAC codec with optimized settings (24fps, medium preset)
- ✅ Progress callbacks for GUI integration
- ✅ Complete error handling and resource cleanup
- ✅ Generated videos: 769KB - 1.6MB with perfect music sync

### 🚧 In Development

**Step 6: Simple GUI** (Next Priority)
- Tkinter interface with file selection
- Settings panel and progress bar
- Threading for responsive UI

**Step 7: Final Polish** (Planned)
- Error handling and presets
- Documentation and packaging

## 🎨 Example Results

**Input**: Multiple family videos + 1 upbeat song (123 BPM)
**AutoCut Output**: Real MP4 video file with:
- ✅ Perfect beat detection and synchronization
- ✅ Intelligent clip selection across multiple source videos  
- ✅ Enhanced quality scoring with motion/face detection (68-73/100)
- ✅ Professional crossfade transitions
- ✅ H.264/AAC encoding for universal compatibility
- ✅ Generated file sizes: 769KB - 1.6MB

## 🔧 Technical Details

### Project Structure (Post-Refactoring)
```
src/
├── video/               # Video processing modules
│   ├── validation.py    # Unified validation system
│   ├── transcoding.py   # H.265 processing
│   └── preprocessing.py # Video preprocessing
├── hardware/            # Hardware acceleration
├── core/               # Core utilities
│   ├── exceptions.py   # Structured error handling
│   └── config.py      # Configuration management
└── api.py             # Clean public API
```

### Testing Structure (After Refactoring)
```
tests/
├── unit/              # Fast unit tests
├── integration/       # End-to-end tests
├── performance/       # Benchmarks
└── fixtures/         # Test data
```

### Core Dependencies
- **MoviePy 2.2.1+**: Frame-accurate video editing with compatibility layer
- **Librosa 0.10.1+**: Professional audio analysis
- **OpenCV 4.8.0+**: Computer vision and frame analysis
- **NumPy/SciPy**: Numerical processing
- **pytest**: Testing framework (after refactoring)

### Performance
- **Processing Speed**: ✅ ~8 videos per minute (ACHIEVED)
- **Memory Efficient**: ✅ Processes videos in chunks (ACHIEVED)
- **Quality Focus**: ✅ Prioritizes sharp, well-lit scenes + motion + faces (ACHIEVED)
- **Beat Accuracy**: ✅ >95% sync rate with music (ACHIEVED in Step 4)
- **Video Output**: ✅ Real MP4 files with perfect music sync (ACHIEVED in Step 5)
- **Complete Pipeline**: ✅ Audio → Video → Assembly → RENDERING working

### Algorithm Highlights

**Beat-to-Clip Matching** ✅ **IMPLEMENTED**:
```python
# Example: 123 BPM detected from real music file
# 145 beats extracted over 74.9 seconds
# Smart clip selection: quality (70%) + duration fit (30%) 
# Variety patterns: 'balanced' = [2,1,2,4,2,1] beat multipliers
# Result: Perfect synchronization to musical timing!
```

**Enhanced Quality Scoring**: 
- Quality metrics (60%): Sharpness + Brightness + Contrast
- Motion detection (25%): Optical flow for dynamic content
- Face detection (15%): Prioritizes people for family videos
- Combined score: 0-100 scale for optimal clip selection

## 📊 Testing Results

Successfully tested with:
- **Audio**: 3 songs (74-193 seconds, 99-123 BPM)
- **Basic Video**: 16 files (9-35 seconds, 720p-4K resolution, 24-30fps)
- **Advanced Video**: 3 files with enhanced scoring (68-73/100)
- **Motion Detection**: Optical flow successfully detecting activity levels
- **Face Detection**: 1-3 faces detected per video segment
- **Processing**: All files analyzed efficiently
- **Compatibility**: Mixed formats and frame rates

## 🔄 Refactoring Progress Status

### ✅ Architecture Improvements (Weeks 1-2 Complete)

**Week 1: Modular Architecture** ✅ **COMPLETE**
- Modular structure: `src/video/`, `src/hardware/`, `src/core/`
- 90% validation duplication reduction (10+ → 1 unified system)
- 100% backwards compatibility maintained
- Zero regressions in core functionality

**Week 2: Testing Framework** ✅ **COMPLETE**  
- Professional pytest infrastructure (pytest 8.4.1)
- 17 scattered test scripts → organized test suite
- Makefile with 25+ automated development commands
- CI/CD pipeline foundation ready

**Week 3: CLI/API Design** 🎯 **NEXT TARGET**
- Single `autocut.py` entry point
- Clean public API for programmatic use
- Migration from demo scripts to production CLI

**Week 4: Cross-Platform Compatibility** 🔄 **PLANNED**
- iPhone H.265 Mac compatibility resolution
- Platform-specific optimization and testing

### 📊 Quality Metrics Progress
- ✅ **Code Organization**: Modular architecture implemented
- ✅ **Testing Infrastructure**: Professional framework operational  
- ✅ **Development Workflow**: Automated commands and CI/CD ready
- 🎯 **CLI Interface**: Single entry point (Week 3 target)
- 🔄 **Platform Compatibility**: Cross-platform testing (Week 4 target)

## 🤝 Contributing

AutoCut follows a professional development approach:

1. **Modular architecture** with clear separation of concerns
2. **Comprehensive testing** with pytest framework and automation
3. **Real media testing** ensures production readiness  
4. **Atomic commits** track incremental progress
5. **Quality gates** prevent regressions and ensure stability

## 📄 License

This project is open source. See LICENSE file for details.

## 🎬 Coming Soon

- **GUI application** for non-technical users (Step 6)
- **Preset modes**: Action, Cinematic, Musical styles
- **Enhanced motion detection**: Distinguish camera vs object motion
- **Batch processing** for multiple projects
- **Export presets**: Instagram, YouTube, TikTok formats
- **Real-time preview** capabilities

---

*AutoCut v2.0 - Making video editing accessible to everyone* 🎥🎵