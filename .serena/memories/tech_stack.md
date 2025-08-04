# AutoCut V2 Tech Stack

## Programming Language
- **Python 3.12.3**
- Virtual environment located at `env/`
- Activation: `source env/bin/activate`

## Core Dependencies (Planned)
```python
moviepy==1.0.3          # Video editing - frame-accurate cuts, prevents stuttering
librosa==0.10.1         # Audio analysis - industry-standard BPM detection
opencv-python==4.8.1    # Video frame analysis - fast processing
numpy==1.24.3           # Numerical operations
scipy==1.11.4           # Signal processing
Pillow==10.1.0         # Image processing
tkinter                 # GUI - simple, no installation issues (comes with Python)
```

## Rationale for Tech Choices
- **MoviePy**: Handles frame-accurate cuts, prevents stuttering, built-in transitions
- **Librosa**: Industry-standard for music analysis, reliable BPM detection
- **OpenCV**: Fast video frame analysis and computer vision
- **Tkinter**: Simple GUI, no installation issues, comes with Python
- **Scientific Python Stack**: NumPy/SciPy for numerical operations and signal processing

## Architecture Pattern
- **Modular Design**: Separate modules for audio analysis, video analysis, clip assembly, and GUI
- **Pipeline Architecture**: Sequential processing stages with clear interfaces
- **Event-Driven GUI**: Tkinter with threading for responsive user interface

## File Formats Support
### Video Formats
- MP4 (H.264) - primary target
- MOV, AVI, MKV - secondary support
- Handle variable framerates gracefully

### Audio Formats
- MP3, WAV, M4A, FLAC
- Automatic format detection
- Graceful degradation for unsupported formats

## Development Environment
- **Git**: Version control
- **Virtual Environment**: Python venv
- **Platform**: Linux (Ubuntu/Debian-based)
- **Editor**: Any Python-compatible IDE/editor