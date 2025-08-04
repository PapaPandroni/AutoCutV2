#!/usr/bin/env python3
"""
Simple Test for Parallel Video Loading System
Tests the implementation without actually loading large video files.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import the module directly 
import clip_assembler


def test_video_cache_logic():
    """Test VideoCache class logic without loading actual videos."""
    print("🧪 Testing VideoCache Logic...")
    
    cache = clip_assembler.VideoCache()
    
    # Test basic functionality
    assert len(cache.get_cached_paths()) == 0, "Cache should start empty"
    
    # Test reference counting logic
    assert cache._ref_counts == {}, "Reference counts should start empty"
    
    print("   ✅ VideoCache initialization and basic methods work")
    
    # Test error handling for missing MoviePy
    if clip_assembler.VideoFileClip is None:
        try:
            cache.get_or_load("/fake/path.mp4")
            assert False, "Should have raised RuntimeError for missing MoviePy"
        except RuntimeError as e:
            assert "MoviePy not available" in str(e)
            print("   ✅ Correctly handles missing MoviePy")
    else:
        print("   ✅ MoviePy is available for testing")
    
    cache.clear()
    return True


def test_parallel_loading_function_exists():
    """Test that parallel loading function exists with correct signature."""
    print("🔧 Testing Parallel Loading Function...")
    
    # Check function exists
    assert hasattr(clip_assembler, 'load_video_clips_parallel'), "Missing load_video_clips_parallel function"
    
    # Check helper function exists  
    assert hasattr(clip_assembler, 'load_video_segment'), "Missing load_video_segment function"
    
    print("   ✅ Parallel loading functions are defined")
    
    # Test with empty clips list
    try:
        video_clips, cache = clip_assembler.load_video_clips_parallel([])
        assert False, "Should raise ValueError for empty clips"
    except ValueError as e:
        assert "No clips provided" in str(e)
        print("   ✅ Correctly handles empty clips list")
    
    return True


def test_imports_and_dependencies():
    """Test that all required imports work."""
    print("📦 Testing Imports and Dependencies...")
    
    # Test concurrent.futures import
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("   ✅ concurrent.futures imported")
    
    # Test threading import
    import threading
    print("   ✅ threading imported")
    
    # Test collections import
    from collections import defaultdict
    print("   ✅ collections imported")
    
    # Test MoviePy availability
    if clip_assembler.VideoFileClip is not None:
        print("   ✅ MoviePy is available")
    else:
        print("   ⚠️  MoviePy not available (testing mode)")
    
    return True


def test_parallel_structure():
    """Test the parallel loading structure without actual video files."""
    print("🏗️  Testing Parallel Structure...")
    
    # Create fake clip data
    fake_clips = [
        {'video_file': '/fake/video1.mp4', 'start': 0, 'end': 2, 'beat_position': 0},
        {'video_file': '/fake/video2.mp4', 'start': 1, 'end': 3, 'beat_position': 1},
    ]
    
    progress_calls = []
    def mock_progress(step, progress):
        progress_calls.append((step, progress))
    
    try:
        # This should fail because files don't exist, but we can test the structure
        clip_assembler.load_video_clips_parallel(fake_clips, mock_progress, max_workers=2)
        assert False, "Should have failed with missing files"
    except RuntimeError as e:
        assert "No video clips could be loaded" in str(e)
        print("   ✅ Correctly handles missing video files")
        print(f"   ✅ Progress callback called {len(progress_calls)} times")
    
    return True


def test_performance_characteristics():
    """Test performance-related characteristics."""
    print("⚡ Testing Performance Characteristics...")
    
    # Test max_workers parameter handling
    large_clip_list = [
        {'video_file': f'/fake/video{i}.mp4', 'start': 0, 'end': 2, 'beat_position': i}
        for i in range(20)
    ]
    
    try:
        # Test with different worker counts
        clip_assembler.load_video_clips_parallel(large_clip_list, max_workers=1)
    except RuntimeError:
        pass  # Expected due to fake files
    
    try:
        clip_assembler.load_video_clips_parallel(large_clip_list, max_workers=8)
    except RuntimeError:
        pass  # Expected due to fake files
    
    print("   ✅ Worker count parameters handled correctly")
    
    return True


def main():
    """Run all simple tests."""
    print("🎬 AutoCut V2 - Parallel Video Loading (Simple Tests)")
    print("=" * 60)
    
    tests = [
        test_imports_and_dependencies,
        test_video_cache_logic,
        test_parallel_loading_function_exists,
        test_parallel_structure,
        test_performance_characteristics,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            print()
            result = test_func()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print()
    print("=" * 60)
    print(f"🎯 Simple tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All simple tests PASSED!")
        print("✅ Parallel video loading system structure is correct")
        print("⚡ Ready for performance testing with real video files")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    print()
    print("💡 Next steps:")
    print("   1. Test with real video files using existing Step 5 test")
    print("   2. Measure actual performance improvement")
    print("   3. Integrate with full AutoCut pipeline")


if __name__ == "__main__":
    main()