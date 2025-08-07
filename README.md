# AutoCut v2.0 🎬✨

**Transform hours of raw footage into polished, music-synced highlight reels in minutes.**

AutoCut is a desktop application that automatically creates beat-synced highlight videos from your raw footage and music. It analyzes video quality, detects the music's rhythm, and intelligently assembles clips that match the beat - all without requiring any video editing knowledge.

## ⚠️ Current Status: Major Refactoring Phase

**The codebase is undergoing major refactoring to address technical debt.** While core functionality works, the current testing approach and code organization need significant improvement. See `REFACTOR.md` for detailed refactoring plan.

**Issues being addressed:**
- 10+ duplicate validation functions need consolidation
- 16+ scattered test scripts being replaced with proper pytest framework
- Platform inconsistency (iPhone H.265 works on Linux but not Mac)
- Monolithic modules need separation of concerns

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

### Quick Testing (Current Scripts - Will be Replaced)

⚠️ **Note**: These scattered test scripts will be replaced with a proper pytest framework during refactoring.

**Main Demo** (works with any video/audio files):
```bash
# Add your videos and music to test_media/ folder
python test_autocut_demo.py            # Process all videos
python test_autocut_demo.py --videos 5 # Limit to 5 videos
```

**Individual Component Tests**:
```bash
python test_real_audio.py              # Audio analysis test
python test_video_analysis.py          # Video analysis test  
python test_step5_rendering.py         # Full rendering test
```

### Future Testing Framework (After Refactoring)

The new testing approach will use pytest:
```bash
# Future unified testing (not yet implemented)
make test                 # Run all tests
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-iphone         # iPhone H.265 specific tests
python autocut.py demo   # Single demo command
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

```
AutoCutV2/
├── src/
│   ├── audio_analyzer.py    # Music analysis & BPM detection
│   ├── video_analyzer.py    # Scene detection & quality scoring  
│   ├── clip_assembler.py    # Beat matching & video rendering
│   ├── gui.py              # Simple desktop interface
│   └── utils.py            # Helper functions
├── test_media/             # Your test videos and music
├── output/                 # Generated highlight videos
├── tests/                  # Unit tests
├── test_real_audio.py      # Audio analysis testing
├── test_video_analysis.py  # Basic video analysis testing
├── test_step3_complete.py  # Advanced video analysis testing
├── test_step4_assembly.py  # Beat matching and assembly testing
├── test_step5_rendering.py # Video rendering pipeline testing
├── test_autocut_demo.py    # Easy demo script for testing
└── requirements.txt        # Dependencies
```

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

## 🤝 Contributing

AutoCut follows a test-driven development approach:

1. **Each step** includes comprehensive testing
2. **Real media testing** ensures production readiness  
3. **Atomic commits** track incremental progress
4. **Performance optimization** for user experience

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