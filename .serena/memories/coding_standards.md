# AutoCut V2 Coding Standards and Conventions

## Python Style Guidelines

### Code Style
- **PEP 8**: Follow Python Enhancement Proposal 8 for code style
- **Line Length**: Maximum 88 characters (Black formatter standard)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes for strings consistently

### Naming Conventions
- **Functions**: snake_case (e.g., `analyze_audio`, `detect_scenes`)
- **Variables**: snake_case (e.g., `beat_timestamps`, `video_clips`)
- **Classes**: PascalCase (e.g., `AudioAnalyzer`, `VideoProcessor`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MIN_CLIP_DURATION`, `MAX_BPM`)
- **Private methods**: Leading underscore (e.g., `_calculate_score`)

### Type Hints
- **Required**: Use type hints for all function parameters and return values
- **Import**: `from typing import List, Dict, Tuple, Optional, Union`
- **Example**: 
```python
def analyze_audio(file_path: str) -> Dict[str, Union[float, List[float]]]:
    """Analyze audio file and return BPM and beat timestamps."""
    pass
```

### Docstrings
- **Format**: Google-style docstrings
- **Required**: All public functions, classes, and modules
- **Example**:
```python
def calculate_clip_constraints(bpm: float) -> Tuple[float, List[float]]:
    """Calculate allowed clip durations based on BPM.
    
    Args:
        bpm: Beats per minute of the music track
        
    Returns:
        Tuple containing minimum duration and list of allowed durations
        
    Raises:
        ValueError: If BPM is not within valid range (30-300)
    """
```

### Error Handling
- **Use specific exceptions**: `ValueError`, `FileNotFoundError`, etc.
- **Custom exceptions**: Create domain-specific exceptions when needed
- **Logging**: Use Python's `logging` module instead of print statements
- **Graceful degradation**: Handle unsupported formats gracefully

### Import Organization
1. Standard library imports
2. Third-party imports (moviepy, librosa, opencv, etc.)
3. Local application imports
4. Separate groups with blank lines

### Code Organization
- **Single Responsibility**: Each function should do one thing well
- **Pure Functions**: Prefer functions without side effects when possible
- **Constants**: Define magic numbers as named constants
- **Configuration**: Use configuration files or environment variables

## Project-Specific Guidelines

### Performance Considerations
- **Memory Efficiency**: Process videos in chunks, use generators
- **Frame Skipping**: Analyze every 10th frame for scene detection
- **Threading**: Use threading for GUI responsiveness
- **Lazy Loading**: Load media files only when needed

### Audio/Video Processing Rules
- **NEVER manipulate audio track**: Only sync video cuts to existing music
- **Frame-accurate cuts**: Use MoviePy's precise timing methods
- **Keyframe alignment**: Cut on keyframes to prevent stuttering
- **Quality preservation**: Maintain source video quality

### Testing Standards
- **Unit Tests**: Test individual functions with various inputs
- **Integration Tests**: Test module interactions
- **Real Media Tests**: Use actual video and audio files
- **Edge Cases**: Test with corrupted files, extreme values
- **Performance Tests**: Verify processing speed requirements