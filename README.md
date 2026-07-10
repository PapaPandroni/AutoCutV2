# AutoCut v2.0 🎬✨

**Transform hours of raw footage into polished, music-synced highlight reels in minutes.**

AutoCut is a desktop application that automatically creates beat-synced highlight videos from your raw footage and music. It analyzes video quality, detects the music's rhythm, and intelligently assembles clips that match the beat - all without requiring any video editing knowledge.

## 🚀 Current Status: Core Pipeline Complete, GUI In Progress

The full pipeline — audio analysis, video analysis, beat matching, and
rendering — works end-to-end via the `autocut.py` CLI and produces real MP4
output. Current focus is the Tkinter GUI (`src/gui.py`) and continued
beat-sync accuracy work.

- ✅ **Beat-synced assembly**: matches scored video clips to the beat grid
  with variety patterns (energetic, buildup, balanced, dramatic)
- ✅ **Timeline anchoring fix (2026-07)**: the renderer concatenates clips
  sequentially, so cut timing depends on getting cumulative clip durations
  to line up exactly with the beat grid — not just approximately. Recent
  work anchored the first clip to the true start of the song, made clip
  trims exact instead of allowing small slack, computed cut-to-cut spans
  from actual beat gaps instead of a song-wide average, added phase/downbeat
  detection, and added a `ClipTimeline.get_alignment_report()` to measure
  drift directly. Real-world accuracy validation is ongoing.
- ✅ **Modular architecture**: `src/video/`, `src/hardware/`, `src/core/`
  with a unified validation system and hardware-accelerated H.265 transcoding
- ✅ **Testing framework**: pytest suite (`tests/unit`, `tests/smoke`,
  `tests/integration`) runs against synthetic media on a clean checkout, no
  real footage required
- 🚧 **In progress**: Tkinter GUI (`src/gui.py`), continued sync-accuracy
  validation against real songs/footage

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

### 🤖 Automated Assembly
- **Beat-to-clip synchronization** - clips are matched to the beat grid, not just cut on a timer
- **Variety patterns** prevent monotonous cutting (energetic, buildup, balanced, dramatic)
- **Smart clip selection** balances quality vs variety across source videos
- **Musical timing constraints** ensure clips fit beat grid durations
- **Timeline validation** with comprehensive statistics and warnings
- **No audio manipulation** - music stays crisp and clear, never trimmed/stretched to fit

### 🎬 Video Rendering
- **Real MP4 creation** with music attached and trimmed to match video length
- **MoviePy integration** with H.264/AAC codec optimization
- **Hardware acceleration** support (NVIDIA NVENC, Intel QSV)
- **Comprehensive error handling** with detailed per-file processing status
- **Memory management** for efficient large video processing
- **Smart transcoding avoidance** eliminates unnecessary H.265→H.264 work when the source is already compatible

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

### 🖥️ GUI

```bash
python autocut_gui.py     # or: python autocut.py gui / make gui
```

Opens a window for picking videos, music, and output location — no
command-line knowledge needed.

### 📦 Share as a macOS app

Build a self-contained `AutoCut.app` that runs without Python, Homebrew, or
FFmpeg installed (for non-technical recipients):

```bash
make app                  # or: bash scripts/build_app.sh
```

This fetches static FFmpeg binaries into `vendor/ffmpeg/` (once), runs
PyInstaller with `autocut.spec`, and produces `dist/AutoCut-mac.zip` — send
that file.

**Instructions for recipients:**

1. Unzip and drag **AutoCut** to Applications.
2. First launch: **right-click → Open → Open** (the app is not notarized by
   Apple, so double-clicking shows a warning instead).
3. On Apple Silicon (M1/M2/M3...) Macs, macOS may offer to install
   **Rosetta 2** on first launch — click Install (the app is built for Intel
   and runs through Rosetta).
4. The first click on **Generate Video** pauses a few seconds at
   "Loading processing engine..." — that's normal, once per launch.

**Troubleshooting:** the app writes a log to
`~/Library/Logs/AutoCut/autocut.log` — if something fails, send that file
along with the error message. Builds can be verified headlessly with:

```bash
dist/AutoCut.app/Contents/MacOS/AutoCut --selftest video.mp4 --audio music.wav --output out.mp4
```

### 🎯 AutoCut CLI Interface

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
make demo               # Run AutoCut demo (uses the CLI)
make demo-quick         # Quick test with limited files
make validate-video     # Validate a video file (specify VIDEO=path)
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

Handles real camera output directly — DJI, Sony, Canon, Panasonic, GoPro,
iPhone (including H.265/HEVC, with automatic hardware-accelerated
transcoding to H.264 when needed). Supports MP4, MOV, AVI, MKV, WEBM, MXF,
MTS, M2TS, and more. Run `python autocut.py validate <file> --detailed` to
check compatibility for a specific file.

## 📁 Project Structure

```
AutoCutV2/
├── autocut.py              # Main CLI interface (single entry point)
├── src/
│   ├── api.py             # Clean public API for all functionality
│   ├── audio_analyzer.py  # Music analysis, BPM/beat detection
│   ├── video_analyzer.py  # Scene detection & quality scoring
│   ├── beat_matching.py   # Beat-to-clip matching (timeline planning)
│   ├── video_loading.py   # Video loading/preprocessing
│   ├── rendering.py       # Final video rendering (concat, audio, encode)
│   ├── clip_assembler.py  # Orchestrates the full pipeline
│   ├── audio_loader.py    # Robust audio loading with fallbacks
│   ├── utils.py           # Helper functions
│   ├── gui.py             # Tkinter GUI (in progress)
│   ├── video/             # Video processing modules
│   │   ├── validation.py    # Unified video validation system
│   │   ├── codec_detection.py # Video format analysis
│   │   ├── transcoding.py   # H.265 transcoding service
│   │   ├── encoder.py       # Encoding settings/hardware detection
│   │   └── timeline_renderer.py # ClipTimeline data structure
│   ├── hardware/           # Hardware acceleration
│   │   └── detection.py    # GPU/CPU encoder detection
│   └── core/               # Core utilities
│       ├── exceptions.py   # Structured error handling
│       └── logging_config.py # Structured logging configuration
├── tests/                # pytest framework, synthetic-media-based
│   ├── unit/             # Fast isolated tests
│   ├── smoke/            # End-to-end pipeline tests (synthetic media)
│   ├── integration/      # Media-gated integration tests
│   ├── performance/      # Benchmark tests
│   ├── synthetic_media.py # Generates test video/audio (no real footage needed)
│   └── conftest.py       # Test fixtures and configuration
├── test_media/          # Your test videos and music (not checked in)
├── output/             # Generated highlight videos
├── Makefile            # Automated development commands
└── requirements.txt    # Dependencies (includes Click CLI framework)
```

**Key Architecture Improvements:**
- **Single Entry Point**: `autocut.py` replaces scattered scripts
- **Clean API**: `src/api.py` provides programmatic access  
- **Modular Design**: Separated video, hardware, and core concerns
- **Professional Testing**: pytest framework with comprehensive coverage
- **90% Code Deduplication**: Unified validation system eliminates scattered functions

## 🔧 Technical Details

### Core Dependencies
- **MoviePy 2.2.1+**: Frame-accurate video editing with a compatibility layer for the 1.x/2.x API differences
- **Librosa 0.10.1+**: BPM/beat detection and onset-strength analysis
- **OpenCV 4.8.0+**: Computer vision and frame analysis
- **NumPy/SciPy**: Numerical processing
- **Click**: CLI framework
- **pytest**: Testing framework

### Performance
- Processes videos in chunks to keep memory bounded on large source libraries
- Dynamic worker-count detection based on available system resources
- Smart transcoding avoidance: skips H.265→H.264 re-encoding when the source is already MoviePy-compatible

### Algorithm Highlights

**Beat-to-Clip Matching**:
```python
# Variety patterns define beat multipliers per cut, e.g.
# 'balanced' = [4, 4, 4, 8, 4, 4] beats per clip
# Each clip's target duration is the actual gap between its start and end
# beat (not an average), so tempo drift doesn't accumulate into timeline drift.
# Clip selection scores candidates on quality (70%) + duration fit (30%).
```

**Enhanced Quality Scoring**:
- Quality metrics (60%): Sharpness + Brightness + Contrast
- Motion detection (25%): Optical flow for dynamic content
- Face detection (15%): Prioritizes people for family videos
- Combined score: 0-100 scale for optimal clip selection

## 📊 Testing

The pytest suite (`tests/unit`, `tests/smoke`) runs against generated
synthetic video/audio (`tests/synthetic_media.py`), so `pytest tests/unit
tests/smoke -m "not slow"` passes on a clean checkout with no real footage
required. `tests/integration` and other media-gated tests run against
real files when `test_media/` is present locally.

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

- **Finished GUI application** for non-technical users (Tkinter GUI in progress, `src/gui.py`)
- **Preset modes**: Action, Cinematic, Musical styles
- **Enhanced motion detection**: Distinguish camera vs object motion
- **Batch processing** for multiple projects
- **Export presets**: Instagram, YouTube, TikTok formats
- **Real-time preview** capabilities

---

*AutoCut v2.0 - Making video editing accessible to everyone* 🎥🎵