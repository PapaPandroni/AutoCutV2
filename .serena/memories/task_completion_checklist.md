# AutoCut V2 Task Completion Checklist

## After Implementing Any Feature

### 1. Code Quality Checks
```bash
# Activate virtual environment first
source env/bin/activate

# Format code (when Black is configured)
black src/

# Lint code (when Flake8 is configured)
flake8 src/

# Type checking (when MyPy is configured)
mypy src/
```

### 2. Testing Requirements
```bash
# Run unit tests for the module
python -m pytest tests/test_[module_name].py -v

# Run integration tests if applicable
python -m pytest tests/ -k "integration"

# Test with real media files
python -m src.[module_name] --test-mode

# Check test coverage
python -m pytest tests/ --cov=src/
```

### 3. Manual Testing Checklist
- [ ] **Functionality**: Feature works as specified
- [ ] **Edge Cases**: Handles invalid inputs gracefully
- [ ] **Performance**: Meets speed requirements (if applicable)
- [ ] **Memory Usage**: No memory leaks with large files
- [ ] **Error Handling**: Provides clear error messages
- [ ] **Real Media**: Works with actual video/audio files

### 4. Documentation Updates
- [ ] **Docstrings**: All new functions have proper docstrings
- [ ] **Type Hints**: All parameters and return values typed
- [ ] **CLAUDE.md**: Update project memory if architecture changes
- [ ] **Comments**: Complex algorithms explained with comments

### 5. Version Control
```bash
# Check what has changed
git status
git diff

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add [feature_name]: [brief_description]

- Specific change 1
- Specific change 2
- Fixes #issue_number (if applicable)"

# Push to remote (when ready)
git push origin main
```

## Phase-Specific Completion Requirements

### Audio Analysis Module
- [ ] **BPM Detection**: Accurate for various music genres
- [ ] **Beat Timestamps**: Precise timing extraction
- [ ] **Edge Cases**: Handle tempo changes, very slow/fast songs
- [ ] **File Formats**: Support MP3, WAV, M4A, FLAC
- [ ] **Performance**: Process 5-minute song in <10 seconds

### Video Analysis Module
- [ ] **Scene Detection**: Identify distinct scenes accurately
- [ ] **Quality Scoring**: Consistent scoring across video types
- [ ] **Face Detection**: Reliable face recognition (if implemented)
- [ ] **Motion Analysis**: Detect activity levels correctly
- [ ] **File Formats**: Support MP4, MOV, AVI at minimum

### Clip Assembly Module
- [ ] **Beat Synchronization**: >95% accuracy in alignment
- [ ] **Variety Patterns**: No more than 3 consecutive same-duration cuts
- [ ] **Rendering Quality**: No stuttering or artifacts
- [ ] **Transitions**: Smooth crossfades between clips
- [ ] **Audio Sync**: Perfect synchronization with music

### GUI Module
- [ ] **Usability**: Non-technical users can operate without help
- [ ] **Responsiveness**: No GUI freezing during processing
- [ ] **Progress Feedback**: Clear indication of processing status
- [ ] **Error Messages**: User-friendly error reporting
- [ ] **File Selection**: Intuitive file browser integration

## Critical Don'ts
- [ ] **DON'T** manipulate audio track (causes choppiness)
- [ ] **DON'T** proceed without testing each step
- [ ] **DON'T** commit broken code
- [ ] **DON'T** skip real media testing
- [ ] **DON'T** ignore memory leaks with large files

## Definition of Done
A task is complete when:
1. **Code works**: Feature functions as specified
2. **Tests pass**: All relevant tests green
3. **Quality checks**: Linting and type checking clean
4. **Real testing**: Verified with actual media files
5. **Documentation**: Code is properly documented
6. **Committed**: Changes saved to version control
7. **Performance**: Meets specified speed/quality targets

## Success Metrics Reminder
- **Processing Speed**: 10 minutes footage → <2 minutes processing
- **Sync Accuracy**: >95% cuts align with beats
- **Visual Quality**: No stuttering, smooth transitions
- **Variety**: Max 3 consecutive same-duration cuts
- **Usability**: Grandma can use it without help