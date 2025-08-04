# AutoCut V2 - Coding Standards & Patterns

## Code Style Guidelines

### Python Style
- **PEP 8 compliance**: Follow standard Python style guidelines
- **Type hints**: Use for all function parameters and return values
- **Docstrings**: Google-style docstrings for all functions and classes
- **Import organization**: Standard library, third-party, local imports
- **Line length**: Prefer 88 characters (Black formatter default), max 100

### Example Function Pattern
```python
def analyze_audio(file_path: str) -> Dict[str, Union[float, List[float]]]:
    """Analyze audio file and extract tempo and beat information.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary containing:
        - 'bpm': Detected beats per minute
        - 'beats': List of beat timestamps in seconds
        - 'duration': Total audio duration in seconds
        - 'allowed_durations': List of musically appropriate clip durations
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If BPM is outside valid range (30-300)
    """
    # Implementation here
```

## Error Handling Patterns

### File Operations
```python
# Pattern: Check file existence before processing
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Audio file not found: {file_path}")

try:
    # File processing
    pass
except Exception as e:
    raise ValueError(f"Failed to process file {file_path}: {str(e)}")
```

### Import Handling
```python
# Pattern: Graceful import fallbacks for optional dependencies
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        # Fallback for testing without full installation
        VideoFileClip = None

# Later in code:
if VideoFileClip is None:
    raise ImportError("MoviePy not available. Please install moviepy>=1.0.3")
```

### Processing Errors
```python
# Pattern: Skip problematic items, continue processing
for item in items:
    try:
        result = process_item(item)
        results.append(result)
    except Exception:
        # Skip problematic items, continue with others
        continue

if not results:
    raise ValueError("No valid items could be processed")
```

## Performance Patterns

### Frame Sampling Optimization
```python
# Pattern: Sample frames at intervals for performance
sample_interval = 1.0  # seconds
timestamps = np.arange(0, duration, sample_interval)

# Only process sampled frames
for t in timestamps:
    try:
        frame = video.get_frame(t)
        # Process frame
    except Exception:
        # Skip problematic frames
        continue
```

### Memory Management
```python
# Pattern: Close resources properly
try:
    video = VideoFileClip(file_path)
    # Process video
    return results
finally:
    if 'video' in locals():
        video.close()
```

### Chunked Processing
```python
# Pattern: Process data in chunks to avoid memory issues
def process_large_dataset(items, chunk_size=100):
    results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_results = process_chunk(chunk)
        results.extend(chunk_results)
    return results
```

## Testing Patterns

### Real Data Testing
```python
# Pattern: Test with real files when available, mock otherwise
def test_with_real_files():
    files = find_test_files()
    if not files:
        print("❌ No test files found")
        return
    
    for file in files:
        try:
            result = process_file(file)
            print(f"✅ {file}: Success")
        except Exception as e:
            print(f"❌ {file}: {e}")

def test_with_mock_data():
    # Mock testing for CI/development
    mock_data = create_mock_data()
    assert process_data(mock_data) is not None
```

### Performance Testing
```python
# Pattern: Measure and report performance
import time

def test_performance():
    start_time = time.time()
    
    # Test operation
    results = process_files(test_files)
    
    end_time = time.time()
    duration = end_time - start_time
    rate = len(test_files) / duration
    
    print(f"Processed {len(test_files)} files in {duration:.1f}s")
    print(f"Rate: {rate:.1f} files/minute")
```

### User-Friendly Output
```python
# Pattern: Progress indicators and clear status
def process_multiple_items(items):
    print(f"Processing {len(items)} items:")
    print("-" * 30)
    
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {item.name}")
        
        try:
            result = process_item(item)
            print(f"   ✅ Success: {result.summary}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Processing complete!")
```

## Data Structure Patterns

### Configuration with Defaults
```python
# Pattern: Use dataclasses or dictionaries with defaults
VARIETY_PATTERNS = {
    'energetic': [1, 1, 2, 1, 1, 4],  # Fast with occasional pause
    'buildup': [4, 2, 2, 1, 1, 1],    # Start slow, increase pace
    'balanced': [2, 1, 2, 4, 2, 1],   # Mixed pacing
    'dramatic': [1, 1, 1, 1, 8],      # Fast cuts then long hold
}

def apply_pattern(pattern_name: str, beat_count: int) -> List[int]:
    if pattern_name not in VARIETY_PATTERNS:
        pattern_name = 'balanced'  # Safe default
    
    pattern = VARIETY_PATTERNS[pattern_name]
    # Apply pattern logic
```

### Class Design
```python
# Pattern: Simple classes with clear responsibilities
class VideoChunk:
    """Represents a scored video segment."""
    
    def __init__(self, start_time: float, end_time: float, score: float, 
                 video_path: str, metadata: Optional[Dict] = None):
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
        self.video_path = video_path
        self.metadata = metadata or {}
        
    @property
    def duration(self) -> float:
        """Duration of the video chunk in seconds."""
        return self.end_time - self.start_time
        
    def __repr__(self) -> str:
        return f"VideoChunk({self.start_time:.1f}-{self.end_time:.1f}, score={self.score:.1f})"
```

## Algorithm Implementation Patterns

### Scoring and Ranking
```python
# Pattern: Normalize scores to 0-100 scale with weights
def calculate_combined_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Combine multiple metrics into single score."""
    total_score = 0.0
    total_weight = 0.0
    
    for metric, value in metrics.items():
        if metric in weights:
            # Normalize value to 0-100 range if needed
            normalized_value = normalize_metric(value, metric)
            total_score += normalized_value * weights[metric]
            total_weight += weights[metric]
    
    return total_score / total_weight if total_weight > 0 else 0.0

# Usage:
weights = {'sharpness': 0.4, 'brightness': 0.3, 'contrast': 0.3}
score = calculate_combined_score(metrics, weights)
```

### Filtering and Selection
```python
# Pattern: Multi-criteria filtering with fallbacks
def select_best_items(items: List[Item], criteria: Dict) -> List[Item]:
    """Select best items based on multiple criteria."""
    # Primary filtering
    filtered = [item for item in items if meets_criteria(item, criteria)]
    
    # Fallback if too few results
    if len(filtered) < criteria.get('min_required', 1):
        # Relax criteria and try again
        relaxed_criteria = relax_criteria(criteria)
        filtered = [item for item in items if meets_criteria(item, relaxed_criteria)]
    
    # Sort by quality and return top N
    filtered.sort(key=lambda x: x.score, reverse=True)
    return filtered[:criteria.get('max_results', len(filtered))]
```

## Development Workflow Patterns

### Test-Driven Development
```python
# Pattern: Write tests first, implement after
def test_audio_analyzer():
    """Test audio analyzer with known inputs."""
    # Test with mock data first
    test_bpms = [60, 120, 90, 140]
    for bpm in test_bpms:
        min_dur, allowed_durs = calculate_clip_constraints(bpm)
        assert min_dur > 0
        assert len(allowed_durs) > 0
    
    # Test with real files if available
    if real_files_available():
        test_with_real_files()
```

### Git Commit Patterns
```python
# Pattern: Atomic commits with descriptive messages
"""
git commit -m "Add video quality scoring algorithm

- Implement sharpness detection using Laplacian variance
- Add brightness scoring with optimal middle-gray preference  
- Include contrast scoring via standard deviation
- Combine metrics with weighted average (40/30/30)
- Test with sample video files, scores range 38-79/100

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"
"""
```

## Quality Assurance Patterns

### Input Validation
```python
# Pattern: Validate inputs early and clearly
def process_video(file_path: str, start_time: float, end_time: float) -> float:
    # Validate file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    # Validate timing
    if start_time < 0 or end_time <= start_time:
        raise ValueError(f"Invalid time range: {start_time}-{end_time}")
    
    # Validate duration
    if end_time - start_time < 0.1:
        return 0.0  # Too short to analyze
    
    # Process with validated inputs
    return analyze_segment(file_path, start_time, end_time)
```

### Edge Case Handling
```python
# Pattern: Handle edge cases gracefully
def detect_scenes(video, threshold=30.0):
    duration = video.duration
    
    # Edge case: Very short video
    if duration < 2.0:
        return [(0.0, duration)]
    
    # Edge case: No scene changes detected
    scenes = find_scene_changes(video, threshold)
    if not scenes:
        # Return entire video as single scene
        return [(0.0, duration)]
    
    # Edge case: Filter out very short scenes
    min_scene_length = 1.0
    filtered_scenes = [(start, end) for start, end in scenes 
                      if end - start >= min_scene_length]
    
    return filtered_scenes if filtered_scenes else [(0.0, duration)]
```

## Documentation Patterns

### Function Documentation
- Always include purpose, parameters, return values, and exceptions
- Use concrete examples in docstrings
- Document performance characteristics for expensive operations
- Include usage examples for complex functions

### Code Comments
- Explain *why* not *what* (code should be self-explanatory)
- Document non-obvious algorithms or calculations
- Mark TODO/FIXME items with context
- Explain performance optimizations and trade-offs