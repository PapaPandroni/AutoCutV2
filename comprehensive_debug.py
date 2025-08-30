#!/usr/bin/env python3
"""
Comprehensive debug script to isolate video loading failures

This script tests each component in the video loading pipeline step by step
to identify exactly where failures are occurring.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment_setup():
    """Test 1: Basic environment and imports"""
    print("=== Test 1: Environment Setup ===")
    
    try:
        # Check test media
        test_media = Path("test_media")
        if not test_media.exists():
            print("❌ test_media directory not found")
            return False
            
        video_files = list(test_media.glob("*.mov")) + list(test_media.glob("*.MP4")) + list(test_media.glob("*.mp4"))
        video_files = [f for f in video_files if not f.name.startswith('._')]
        
        if not video_files:
            print("❌ No valid video files found")
            return False
            
        test_file = video_files[0]
        print(f"✅ Using test file: {test_file}")
        
        return str(test_file)
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        traceback.print_exc()
        return False

def test_moviepy_imports():
    """Test 2: MoviePy import functionality"""
    print("\n=== Test 2: MoviePy Import Testing ===")
    
    try:
        print("Testing import_moviepy_safely...")
        from clip_assembler import import_moviepy_safely
        
        result = import_moviepy_safely()
        if len(result) == 4:
            VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip = result
            print(f"✅ import_moviepy_safely succeeded")
            print(f"  VideoFileClip: {VideoFileClip}")
            print(f"  AudioFileClip: {AudioFileClip}")
            print(f"  concatenate_videoclips: {concatenate_videoclips}")
            print(f"  CompositeVideoClip: {CompositeVideoClip}")
            return True
        else:
            print(f"❌ import_moviepy_safely returned wrong number of items: {len(result)}")
            return False
            
    except Exception as e:
        print(f"❌ MoviePy import failed: {e}")
        traceback.print_exc()
        return False

def test_resource_manager(test_file):
    """Test 3: VideoResourceManager"""
    print("\n=== Test 3: VideoResourceManager ===")
    
    try:
        from clip_assembler import VideoResourceManager
        
        resource_manager = VideoResourceManager()
        print("✅ VideoResourceManager created")
        
        print("Testing load_video_with_delayed_cleanup...")
        video = resource_manager.load_video_with_delayed_cleanup(test_file)
        
        if video is not None:
            print(f"✅ Video loaded: {video.w}x{video.h}, {video.duration:.2f}s")
            return resource_manager
        else:
            print("❌ load_video_with_delayed_cleanup returned None")
            return False
            
    except Exception as e:
        print(f"❌ VideoResourceManager failed: {e}")
        traceback.print_exc()
        return False

def test_strategy_functions(test_file):
    """Test 4: Individual strategy functions"""
    print("\n=== Test 4: Individual Strategy Functions ===")
    
    try:
        from clip_assembler import RobustVideoLoader, VideoResourceManager
        
        loader = RobustVideoLoader()
        resource_manager = VideoResourceManager()
        
        clip_data = {
            "video_file": test_file,
            "start": 0.0,
            "end": 2.0
        }
        
        # Test each strategy individually
        strategies = [
            ("direct_moviepy", loader._load_direct_moviepy),
            ("format_conversion", loader._load_with_format_conversion),
            ("quality_reduction", loader._load_with_quality_reduction),
            ("emergency_minimal", loader._load_emergency_minimal)
        ]
        
        results = {}
        
        for name, strategy_func in strategies:
            print(f"Testing {name}...")
            try:
                result = strategy_func(clip_data, resource_manager, None)
                if result is not None:
                    print(f"  ✅ {name}: {result.duration:.2f}s")
                    results[name] = True
                else:
                    print(f"  ❌ {name}: returned None")
                    results[name] = False
            except Exception as e:
                print(f"  ❌ {name}: exception {e}")
                results[name] = False
        
        return results
        
    except Exception as e:
        print(f"❌ Strategy function testing failed: {e}")
        traceback.print_exc()
        return False

def test_try_loading_strategy_wrapper(test_file):
    """Test 5: _try_loading_strategy wrapper logic"""
    print("\n=== Test 5: Strategy Wrapper Logic ===")
    
    try:
        from clip_assembler import RobustVideoLoader, VideoResourceManager
        
        loader = RobustVideoLoader()
        resource_manager = VideoResourceManager()
        
        clip_data = {
            "video_file": test_file,
            "start": 0.0,
            "end": 2.0
        }
        
        print("Testing load_clip_with_fallbacks...")
        result = loader.load_clip_with_fallbacks(clip_data, resource_manager, None)
        
        if result is not None:
            print(f"✅ load_clip_with_fallbacks: {result.duration:.2f}s")
            
            # Get detailed error report
            error_report = loader.get_error_report()
            print("Error report:", error_report)
            return True
        else:
            print("❌ load_clip_with_fallbacks returned None")
            error_report = loader.get_error_report()
            print("Error report:", error_report)
            return False
            
    except Exception as e:
        print(f"❌ Strategy wrapper testing failed: {e}")
        traceback.print_exc()
        return False

def test_full_loading_pipeline(test_file):
    """Test 6: Full loading pipeline"""
    print("\n=== Test 6: Full Loading Pipeline ===")
    
    try:
        from clip_assembler import load_video_clips_with_robust_error_handling
        
        sorted_clips = [
            {
                "video_file": test_file,
                "start": 0.0,
                "end": 2.0,
                "original_index": 0
            },
            {
                "video_file": test_file,
                "start": 2.0,
                "end": 4.0,
                "original_index": 1
            }
        ]
        
        video_files = [test_file]
        
        print(f"Testing load_video_clips_with_robust_error_handling with {len(sorted_clips)} clips...")
        
        video_clips, failed_indices, error_report, resource_manager = load_video_clips_with_robust_error_handling(
            sorted_clips=sorted_clips,
            video_files=video_files,
            canvas_format=None,
            progress_callback=lambda step, prog: print(f"  Progress: {step} {prog:.1%}")
        )
        
        if video_clips:
            print(f"✅ Pipeline SUCCESS: {len(video_clips)} clips loaded")
            print(f"  Failed indices: {failed_indices}")
            print(f"  Error report: {error_report}")
            
            for i, clip in enumerate(video_clips):
                if clip is not None:
                    print(f"  Clip {i}: {clip.duration:.2f}s")
                else:
                    print(f"  Clip {i}: None!")
            return True
        else:
            print(f"❌ Pipeline FAILED: No clips loaded")
            print(f"  Failed indices: {failed_indices}")
            print(f"  Error report: {error_report}")
            return False
            
    except Exception as e:
        print(f"❌ Full pipeline testing failed: {e}")
        traceback.print_exc()
        return False

def test_git_status():
    """Test 7: Check git status for latest fixes"""
    print("\n=== Test 7: Git Status Check ===")
    
    try:
        import subprocess
        
        # Get latest commit
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True,
            text=True,
            check=True
        )
        
        commits = result.stdout.strip().split('\n')
        print("Recent commits:")
        for commit in commits:
            print(f"  {commit}")
            
        # Check if we have the critical fixes
        critical_fixes = [
            "Fix video loading failures with macOS resource fork files",
            "Critical fix: Restore video loading functionality broken in Phase 21"
        ]
        
        all_commits = result.stdout
        fixes_found = []
        for fix in critical_fixes:
            if any(fix in commit for commit in commits):
                fixes_found.append(fix)
                print(f"✅ Found fix: {fix}")
            else:
                print(f"❌ Missing fix: {fix}")
        
        return len(fixes_found) == len(critical_fixes)
        
    except Exception as e:
        print(f"❌ Git status check failed: {e}")
        return False

if __name__ == "__main__":
    print("Comprehensive Video Loading Debug Script")
    print("=" * 50)
    
    # Run all tests
    test_file = test_environment_setup()
    if not test_file:
        sys.exit(1)
    
    results = {}
    results["environment"] = bool(test_file)
    results["moviepy_imports"] = test_moviepy_imports()
    results["resource_manager"] = test_resource_manager(test_file)
    results["strategy_functions"] = test_strategy_functions(test_file)
    results["strategy_wrapper"] = test_try_loading_strategy_wrapper(test_file)
    results["full_pipeline"] = test_full_loading_pipeline(test_file)
    results["git_status"] = test_git_status()
    
    print("\n" + "=" * 50)
    print("COMPREHENSIVE TEST RESULTS:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        if isinstance(passed, dict):
            # Strategy functions return a dict
            status = "✅ PASS" if any(passed.values()) else "❌ FAIL"
            print(f"{test_name:20}: {status}")
            for strategy, result in passed.items():
                print(f"  {strategy:18}: {'✅' if result else '❌'}")
        else:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:20}: {status}")
            
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Video loading should work")
    else:
        print("💥 SOME TESTS FAILED - This explains the video loading errors")
        print("\nDiagnosis:")
        
        if not results["git_status"]:
            print("• Missing recent fixes - please pull latest changes")
        if not results["moviepy_imports"]:
            print("• MoviePy import issues")
        if not results["resource_manager"]:
            print("• Video resource manager problems")
        if not results["strategy_functions"]:
            print("• Strategy function failures")
        if not results["strategy_wrapper"]:
            print("• Strategy wrapper logic issues")
        if not results["full_pipeline"]:
            print("• Full pipeline integration problems")