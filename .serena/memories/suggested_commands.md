# AutoCut V2 Development Commands

## Virtual Environment Management
```bash
# Activate virtual environment (ALWAYS use before development)
source env/bin/activate

# Deactivate virtual environment
deactivate

# Check Python version
python --version

# List installed packages
pip list
```

## Package Management
```bash
# Install dependencies from requirements.txt (when created)
pip install -r requirements.txt

# Install specific packages for development
pip install moviepy==1.0.3 librosa==0.10.1 opencv-python==4.8.1
pip install numpy==1.24.3 scipy==1.11.4 Pillow==10.1.0

# Update requirements.txt
pip freeze > requirements.txt

# Install development dependencies (when added)
pip install pytest black flake8 mypy
```

## Code Quality and Formatting
```bash
# Format code with Black (when configured)
black src/

# Lint code with Flake8 (when configured)
flake8 src/

# Type checking with MyPy (when configured)
mypy src/

# Run all quality checks
black src/ && flake8 src/ && mypy src/
```

## Testing Commands
```bash
# Run all tests (when test suite is created)
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_audio_analyzer.py

# Run tests with coverage
python -m pytest tests/ --cov=src/
```

## Running the Application
```bash
# Run main application (when GUI is implemented)
python -m src.gui

# Run specific modules for testing
python -m src.audio_analyzer
python -m src.video_analyzer
python -m src.clip_assembler
```

## Development Workflow Commands
```bash
# Create new module with basic structure
touch src/new_module.py
echo '"""New module docstring."""' > src/new_module.py

# Create test file for module
touch tests/test_new_module.py

# Create test media directory
mkdir -p test_media
mkdir -p output
```

## Git Commands
```bash
# Check status
git status

# Add and commit changes
git add .
git commit -m "Descriptive commit message"

# Push changes
git push origin main

# Create feature branch
git checkout -b feature/audio-analysis
```

## System Utilities (Linux)
```bash
# List files and directories
ls -la

# Find files by pattern
find . -name "*.py" -type f

# Search for text in files
grep -r "function_name" src/

# Check disk space
df -h

# Check memory usage
free -h

# Monitor processes
htop
```

## Media Processing Utilities
```bash
# Check video file information (if ffmpeg installed)
ffprobe video_file.mp4

# Check audio file information
ffprobe audio_file.mp3

# List supported codecs
ffmpeg -codecs
```

## Debugging and Profiling
```bash
# Run with Python debugger
python -m pdb src/module.py

# Profile code execution
python -m cProfile src/module.py

# Memory profiling (when memory_profiler installed)
python -m memory_profiler src/module.py
```

## Project Structure Commands
```bash
# Create directory structure
mkdir -p src tests test_media output

# View project tree (if tree installed)
tree -I '__pycache__|*.pyc|env'

# Count lines of code
find src -name "*.py" | xargs wc -l
```