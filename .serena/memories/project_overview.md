# AutoCut V2 Project Overview

## Project Purpose
AutoCut is a desktop application that automatically creates beat-synced highlight videos from raw footage and music. It analyzes video quality, detects music rhythm, and intelligently assembles clips that match the beat - all without requiring video editing knowledge.

## Target Users
- **Primary**: Families, travelers, hobbyists with unedited video footage
- **Use Cases**: Family vacations, birthday parties, weddings, sports events, travel compilations
- **Profile**: Non-technical users wanting professional results with minimal effort

## Core Value Proposition
Transform hours of raw footage into polished, music-synced highlight reel in minutes, not hours.

## Development Status
- **Current Phase**: Project setup and planning
- **Version**: 2.0.0
- **Language**: Python 3.12.3
- **Virtual Environment**: Located at `env/` (activate with `source env/bin/activate`)

## Key Features (Planned)
1. **Audio Analysis**: BPM detection, beat timestamps using Librosa
2. **Video Quality Scoring**: Sharpness, brightness, contrast, motion, face detection
3. **Beat-to-Clip Matching**: Intelligent synchronization with musical timing
4. **Variety Patterns**: Prevent monotonous cutting with rhythm patterns
5. **Simple GUI**: Tkinter-based interface for non-technical users
6. **Professional Output**: Frame-accurate cuts, smooth transitions, no stuttering

## Success Metrics
- Processing Speed: 10 minutes footage → <2 minutes processing
- Sync Accuracy: >95% cuts align with beats
- Visual Quality: No stuttering, smooth transitions
- Variety: Max 3 consecutive same-duration cuts
- Usability: Non-technical users succeed without help