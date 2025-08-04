# AutoCut V2 Codebase Structure

## Current Directory Structure
```
AutoCutV2/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization (created)
│   ├── audio_analyzer.py        # BPM detection, beat timestamps (planned)
│   ├── video_analyzer.py        # Scene detection, quality scoring (planned)
│   ├── clip_assembler.py        # Beat matching, rendering (planned)
│   ├── gui.py                   # Tkinter interface (planned)
│   └── utils.py                 # Utility functions (planned)
├── tests/                       # Unit and integration tests (planned)
├── test_media/                  # Sample videos and music (planned)
├── output/                      # Generated highlight videos (planned)
├── env/                         # Python virtual environment
├── .serena/                     # Serena MCP configuration
├── .claude/                     # Claude Code configuration
├── CLAUDE.md                    # Project memory and documentation
├── autocut-claude-code-prompt.md # Original project specification
├── README.md                    # Basic project description
├── .gitignore                   # Git ignore patterns (comprehensive Python)
├── requirements.txt             # Python dependencies (to be created)
└── LICENSE                      # Project license
```

## Module Responsibilities

### src/audio_analyzer.py (Planned)
- Load audio files using Librosa
- Detect BPM (tempo) and extract beat timestamps
- Calculate allowed clip durations based on BPM
- Handle edge cases (slow/fast songs, tempo changes)

### src/video_analyzer.py (Planned)
- Load videos using MoviePy
- Scene detection based on frame differences
- Quality scoring: sharpness, brightness, contrast, motion
- Face detection using OpenCV
- Return scored video chunks for selection

### src/clip_assembler.py (Planned)
- Match video chunks to beat grid
- Apply variety patterns to prevent monotony
- Select best clips based on scores and variety
- Render final video with MoviePy
- Add transitions and music synchronization

### src/gui.py (Planned)
- Tkinter-based user interface
- File selection for videos and music
- Settings panel (tempo preference, target duration)
- Progress tracking with threading
- Preview capability

### src/utils.py (Planned)
- Common utility functions
- File format validation
- Error handling helpers
- Configuration management

## Development Phases
1. **Project Setup** - Directory structure, dependencies ✓
2. **Audio Analysis** - BPM detection, beat timestamps
3. **Basic Video Analysis** - Scene detection, quality scoring
4. **Advanced Video Scoring** - Motion detection, face detection
5. **Clip Assembly** - Beat matching, timeline creation
6. **Video Rendering** - MoviePy composition, transitions
7. **GUI & Polish** - Tkinter interface, error handling