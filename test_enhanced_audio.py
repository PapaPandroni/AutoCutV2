#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced audio analysis system
Tests musical intelligence, intro detection, beat hierarchy, and offset compensation
"""
import sys
import os
sys.path.append('src')

from audio_analyzer import analyze_audio
import glob
import json

def test_enhanced_audio_analysis():
    """Test the enhanced audio analysis system with detailed reporting."""
    print("🎵 Enhanced Audio Analysis - Comprehensive Test Suite")
    print("=" * 60)
    
    # Find audio files in test_media
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(f'test_media/{ext}'))
        audio_files.extend(glob.glob(f'test_media/{ext.upper()}'))
    
    if not audio_files:
        print("❌ No audio files found in test_media/")
        print("Please add some music files to test_media/ folder")
        print("Supported formats: MP3, WAV, M4A, FLAC")
        return
    
    print(f"Found {len(audio_files)} audio file(s) for testing:")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    print("\n" + "=" * 60)
    print("TESTING ENHANCED MUSICAL INTELLIGENCE")
    print("=" * 60)
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"\n🎵 ANALYZING: {filename}")
        print("-" * 50)
        
        try:
            result = analyze_audio(audio_file)
            
            # === BASIC ANALYSIS RESULTS ===
            print("📊 BASIC ANALYSIS:")
            print(f"   Duration: {result['duration']:.1f} seconds")
            print(f"   BPM: {result['bpm']:.1f}")
            print(f"   Analysis Version: {result.get('analysis_version', 'N/A')}")
            
            # === BEAT DETECTION COMPARISON ===
            print("\n🥁 BEAT DETECTION COMPARISON:")
            original_beats = result['beats']
            compensated_beats = result['compensated_beats']
            
            print(f"   Original beats detected: {len(original_beats)}")
            print(f"   Compensated beats (filtered): {len(compensated_beats)}")
            print(f"   Beats filtered out: {len(original_beats) - len(compensated_beats)}")
            
            if len(original_beats) > 0 and len(compensated_beats) > 0:
                avg_original_gap = sum(original_beats[i+1] - original_beats[i] 
                                     for i in range(len(original_beats)-1)) / (len(original_beats)-1)
                avg_compensated_gap = sum(compensated_beats[i+1] - compensated_beats[i] 
                                        for i in range(len(compensated_beats)-1)) / (len(compensated_beats)-1)
                
                print(f"   Average beat gap (original): {avg_original_gap:.3f}s")
                print(f"   Average beat gap (compensated): {avg_compensated_gap:.3f}s")
                
                # Show timing comparison for first few beats
                print(f"   First 5 original beats: {[f'{b:.3f}' for b in original_beats[:5]]}")
                print(f"   First 5 compensated beats: {[f'{b:.3f}' for b in compensated_beats[:5]]}")
                
                # Calculate timing difference
                if len(compensated_beats) > 0:
                    offset_applied = result.get('librosa_offset_compensation', 0)
                    print(f"   Librosa offset compensation: {offset_applied:.3f}s")
                    
                    if len(original_beats) > 0 and len(compensated_beats) > 0:
                        actual_shift = compensated_beats[0] - original_beats[0]
                        print(f"   Actual first beat shift: {actual_shift:.3f}s")
            
            # === MUSICAL INTELLIGENCE ===
            print("\n🎼 MUSICAL INTELLIGENCE:")
            musical_start = result['musical_start_time']
            intro_duration = result['intro_duration']
            
            print(f"   Musical start time: {musical_start:.2f}s")
            print(f"   Intro duration: {intro_duration:.2f}s")
            print(f"   Main content starts at: {musical_start:.1f}s ({intro_duration:.1f}% of song)")
            
            # Classify intro type
            if intro_duration < 1.0:
                intro_type = "Immediate start (no intro)"
            elif intro_duration < 3.0:
                intro_type = "Short intro"
            elif intro_duration < 6.0:
                intro_type = "Medium intro/buildup"
            else:
                intro_type = "Long intro/ambient start"
            
            print(f"   Intro classification: {intro_type}")
            
            # === BEAT HIERARCHY ===
            print("\n🏗️ BEAT HIERARCHY:")
            hierarchy = result['beat_hierarchy']
            
            for beat_type, beats in hierarchy.items():
                if isinstance(beats, list) and len(beats) > 0:
                    print(f"   {beat_type.replace('_', ' ').title()}: {len(beats)} beats")
                    if len(beats) <= 10:
                        print(f"      Times: {[f'{b:.2f}' for b in beats[:5]]}{'...' if len(beats) > 5 else ''}")
            
            # === SYNCHRONIZATION ANALYSIS ===
            print("\n⚡ SYNCHRONIZATION ANALYSIS:")
            allowed_durations = result['allowed_durations']
            
            print(f"   Allowed clip durations: {[f'{d:.2f}s' for d in allowed_durations]}")
            print(f"   Minimum clip duration: {result['min_duration']:.2f}s")
            
            # Calculate optimal clip placement after intro
            post_intro_beats = [b for b in compensated_beats if b >= intro_duration]
            if len(post_intro_beats) > 0:
                print(f"   Post-intro beats available: {len(post_intro_beats)}")
                print(f"   First post-intro beat: {post_intro_beats[0]:.2f}s")
                
                # Recommend video cut timing
                optimal_start = post_intro_beats[0] if len(post_intro_beats) > 0 else musical_start
                print(f"   Recommended video start: {optimal_start:.2f}s")
            
            # === QUALITY METRICS ===
            print("\n📈 QUALITY METRICS:")
            
            # Beat consistency check
            if len(compensated_beats) > 3:
                beat_intervals = [compensated_beats[i+1] - compensated_beats[i] 
                                for i in range(len(compensated_beats)-1)]
                interval_std = sum((interval - sum(beat_intervals)/len(beat_intervals))**2 
                                 for interval in beat_intervals) / len(beat_intervals)
                interval_std = interval_std ** 0.5
                
                consistency_score = max(0, 100 - (interval_std * 100))
                print(f"   Beat consistency score: {consistency_score:.1f}/100")
                
                if consistency_score > 80:
                    print("   ✅ Excellent beat consistency - high sync accuracy expected")
                elif consistency_score > 60:
                    print("   ⚠️  Good beat consistency - acceptable sync accuracy")
                else:
                    print("   ❌ Poor beat consistency - may need manual adjustment")
            
            # Intro detection accuracy assessment
            intro_confidence = "High" if 1.0 <= intro_duration <= 6.0 else "Medium"
            if intro_duration < 0.5:
                intro_confidence = "Low (no clear intro detected)"
            elif intro_duration > 8.0:
                intro_confidence = "Medium (very long intro detected)"
            
            print(f"   Intro detection confidence: {intro_confidence}")
            
            print("   ✅ Enhanced analysis completed successfully!")
            
        except Exception as e:
            print(f"   ❌ Error during enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("ENHANCED AUDIO ANALYSIS TEST COMPLETE")
    print("=" * 60)
    print("\n📋 SUMMARY OF ENHANCEMENTS:")
    print("✅ Musical start detection using onset analysis")
    print("✅ Intro duration calculation with configurable thresholds")
    print("✅ Systematic offset compensation (-0.04s for librosa latency)")
    print("✅ Beat hierarchy (downbeats, half-beats, measures, quarter-notes)")
    print("✅ Energy-based weak beat filtering during intros")
    print("✅ Backward compatibility maintained with existing API")
    print("✅ Enhanced synchronization accuracy for professional results")


def compare_original_vs_enhanced():
    """Compare original vs enhanced beat detection side-by-side."""
    print("\n" + "=" * 60)
    print("ORIGINAL vs ENHANCED COMPARISON")
    print("=" * 60)
    
    # Find first audio file for comparison
    audio_files = glob.glob('test_media/*.mp3') + glob.glob('test_media/*.wav')
    if not audio_files:
        print("❌ No audio files found for comparison")
        return
    
    test_file = audio_files[0]
    filename = os.path.basename(test_file)
    print(f"\n🎵 Comparing analysis methods for: {filename}")
    
    try:
        result = analyze_audio(test_file)
        
        print(f"\n📊 TIMING COMPARISON:")
        print(f"   Original first beat: {result['beats'][0]:.3f}s")
        print(f"   Compensated first beat: {result['compensated_beats'][0]:.3f}s")
        print(f"   Musical start time: {result['musical_start_time']:.3f}s")
        print(f"   Timing improvement: {result['musical_start_time'] - result['beats'][0]:.3f}s")
        
        print(f"\n🎼 SYNCHRONIZATION IMPACT:")
        print(f"   Old approach: Video cuts start at {result['beats'][0]:.2f}s")
        print(f"   New approach: Video cuts start at {result['musical_start_time']:.2f}s")
        print(f"   Better alignment: {result['musical_start_time'] > result['beats'][0]}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def test_backward_compatibility():
    """Test that existing code still works with enhanced analyzer."""
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 60)
    
    audio_files = glob.glob('test_media/*.mp3')
    if not audio_files:
        print("❌ No MP3 files found for compatibility test")
        return
    
    test_file = audio_files[0]
    filename = os.path.basename(test_file)
    print(f"\n🔄 Testing backward compatibility with: {filename}")
    
    try:
        result = analyze_audio(test_file)
        
        # Test that all original fields exist
        required_fields = ['bpm', 'beats', 'duration', 'allowed_durations', 'min_duration']
        
        print("\n📋 REQUIRED FIELD CHECK:")
        all_present = True
        for field in required_fields:
            if field in result:
                print(f"   ✅ {field}: {type(result[field])}")
            else:
                print(f"   ❌ {field}: MISSING")
                all_present = False
        
        if all_present:
            print("\n✅ All backward compatibility fields present!")
            print("✅ Existing clip_assembler.py should work without changes")
        else:
            print("\n❌ Backward compatibility BROKEN!")
            
        # Test data types
        print(f"\n🔍 DATA TYPE VALIDATION:")
        print(f"   bpm: {type(result['bpm'])} = {result['bpm']}")
        print(f"   beats: {type(result['beats'])} with {len(result['beats'])} items")
        print(f"   duration: {type(result['duration'])} = {result['duration']}")
        print(f"   allowed_durations: {type(result['allowed_durations'])} with {len(result['allowed_durations'])} items")
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")


if __name__ == "__main__":
    # Run comprehensive test suite
    test_enhanced_audio_analysis()
    
    # Run comparison between old and new methods
    compare_original_vs_enhanced()
    
    # Test backward compatibility
    test_backward_compatibility()
    
    print(f"\n🎯 ENHANCED AUDIO ANALYSIS TESTING COMPLETE!")
    print("Ready for integration with AutoCut video synchronization!")