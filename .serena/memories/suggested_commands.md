# AutoCut V2 - Suggested Development Commands

## Environment Management
```bash
# Activate virtual environment (ALWAYS do this first)
source env/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Check installed packages
pip freeze

# Deactivate when done
deactivate
```

## Testing Commands

### Audio Analysis Testing
```bash
# Test audio analysis with real music files
python test_real_audio.py

# Expected output: BPM detection, beat timestamps, clip durations
# Files tested: soft-positive-summer-pop.mp3, this-heart-is-yours.mp3, upbeat-summer-pop.mp3
# Results: 99-123 BPM range, 134-368 beats detected
```

### Video Analysis Testing  
```bash
# Test all videos (takes ~2 minutes for 16 files)
python test_video_analysis.py

# Quick test mode (first 3 videos only, for development)
python test_video_analysis.py --quick

# Mock test mode (no real files needed, for CI/development)
python test_video_analysis.py --mock

# Expected output: Scene detection, quality scores 38-79/100
# Performance: ~8 videos per minute processing speed
```

### Individual Module Testing
```bash
# Test specific functions interactively
python -c "
import sys
sys.path.append('src')
from audio_analyzer import analyze_audio
result = analyze_audio('test_media/your-song.mp3')
print(f'BPM: {result[\"bpm\"]}, Duration: {result[\"duration\"]}s')
"

# Test video loading
python -c "
import sys
sys.path.append('src') 
from video_analyzer import load_video
video, metadata = load_video('test_media/your-video.mp4')
print(f'Resolution: {metadata[\"width\"]}x{metadata[\"height\"]}, FPS: {metadata[\"fps\"]}')
"
```

## Development Workflow

### Git Commands
```bash
# Check status and stage changes
git status
git add -A

# Commit with descriptive message
git commit -m "Description of changes

- Specific detail 1
- Specific detail 2
- Test results or performance notes

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"

# View commit history
git log --oneline -10
```

### Code Quality Checks
```bash
# Check Python syntax
python -m py_compile src/*.py

# Run module directly to test
python src/audio_analyzer.py
python src/video_analyzer.py

# Import test (useful for debugging)
python -c "import sys; sys.path.append('src'); import audio_analyzer, video_analyzer; print('✅ All imports successful')"
```

## File Management

### Adding Test Media
```bash
# Create test media directory structure
mkdir -p test_media

# Copy your files (examples)
cp ~/Music/your-song.mp3 test_media/
cp ~/Videos/your-video.mp4 test_media/

# Verify files are detected
ls -la test_media/
python test_real_audio.py      # Should find your music
python test_video_analysis.py  # Should find your videos
```

### Project Structure Check
```bash
# Verify project structure
tree . -I 'env|__pycache__|*.pyc'

# Expected structure:
# ├── src/
# │   ├── audio_analyzer.py (✅ complete)
# │   ├── video_analyzer.py (✅ complete) 
# │   ├── clip_assembler.py (🚧 in progress)
# │   └── ...
# ├── test_media/ (your files here)
# ├── requirements.txt
# └── README.md
```

## Performance Monitoring

### Timing Tests
```bash
# Time audio analysis
time python test_real_audio.py

# Time video analysis  
time python test_video_analysis.py

# Monitor system resources during processing
htop  # or top on some systems
```

### File Size Management
```bash
# Check test media sizes
du -sh test_media/*

# Clean up large temporary files if needed
find . -name "*.tmp" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

## Debugging Commands

### Import and Dependency Issues
```bash
# Test MoviePy import specifically
python -c "
try:
    from moviepy import VideoFileClip
    print('✅ Direct MoviePy import works')
except ImportError:
    try:
        from moviepy.editor import VideoFileClip  
        print('✅ MoviePy editor import works')
    except ImportError as e:
        print(f'❌ MoviePy import failed: {e}')
"

# Test librosa import
python -c "
try:
    import librosa
    print(f'✅ Librosa version: {librosa.__version__}')
except ImportError as e:
    print(f'❌ Librosa import failed: {e}')
"

# Test OpenCV import
python -c "
try:
    import cv2
    print(f'✅ OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV import failed: {e}')
"
```

### Memory and Performance Debugging
```bash
# Check Python memory usage during processing
python -c "
import tracemalloc
tracemalloc.start()
# ... run your test here ...
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
"
```

## Next Development Steps

### Step 3: Advanced Video Scoring (Current Priority)
```bash
# Work on motion detection implementation
cd src/
# Edit video_analyzer.py, implement detect_motion() function
# Test with: python test_video_analysis.py --quick

# Work on face detection
# Edit video_analyzer.py, implement detect_faces() function  
# Test with family videos containing faces
```

### Documentation Updates
```bash
# Update progress in documentation
# Edit: README.md, CLAUDE.md, autocut-claude-code-prompt.md
# Commit documentation changes separately from code

# Update Serena memories when major milestones are reached
# Use mcp__serena__write_memory for updated status
```

## Useful Debugging/Development Tips

### Quick Function Testing
```bash
# Test single function changes quickly
python -c "
import sys; sys.path.append('src')
from video_analyzer import score_scene, load_video
video, _ = load_video('test_media/sample.mp4')
score = score_scene(video, 0.0, 2.0)
print(f'Quality score: {score:.1f}/100')
video.close()
"
```

### Performance Comparison
```bash
# Before/after performance comparison
echo "Before optimization:"
time python test_video_analysis.py --quick

# Make changes...

echo "After optimization:"  
time python test_video_analysis.py --quick
```