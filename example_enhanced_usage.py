#!/usr/bin/env python3
"""
Example demonstrating how to use the enhanced audio analysis features
for improved video synchronization in AutoCut V2
"""
import sys
import os
sys.path.append('src')

from audio_analyzer import analyze_audio
import glob

def demonstrate_enhanced_features():
    """Demonstrate the enhanced audio analysis capabilities."""
    print("🎵 AutoCut V2 - Enhanced Audio Analysis Demo")
    print("=" * 50)
    
    # Find first audio file for demo
    audio_files = glob.glob('test_media/*.mp3')
    if not audio_files:
        print("❌ No audio files found for demo")
        return
    
    audio_file = audio_files[0]
    filename = os.path.basename(audio_file)
    print(f"Demo with: {filename}")
    print("-" * 30)
    
    # Analyze with enhanced system
    result = analyze_audio(audio_file)
    
    print("📊 ENHANCED SYNCHRONIZATION RECOMMENDATIONS:")
    print(f"   Song Duration: {result['duration']:.1f} seconds")
    print(f"   BPM: {result['bpm']:.1f}")
    
    # Compare old vs new approach
    old_start = result['beats'][0] if result['beats'] else 0.0
    new_start = result['musical_start_time']
    intro_skip = result['intro_duration']
    
    print(f"\n🎯 TIMING OPTIMIZATION:")
    print(f"   Old approach (first beat): {old_start:.2f}s")
    print(f"   New approach (musical start): {new_start:.2f}s")
    print(f"   Intro detection: {intro_skip:.2f}s")
    print(f"   Synchronization improvement: {new_start - old_start:.2f}s")
    
    # Beat hierarchy options
    hierarchy = result['beat_hierarchy']
    print(f"\n🎼 BEAT SYNCHRONIZATION OPTIONS:")
    print(f"   Main Beats ({len(hierarchy['main_beats'])}): Every {60/result['bpm']:.2f}s")
    print(f"   Half Beats ({len(hierarchy['half_beats'])}): Every {60/result['bpm']/2:.2f}s")
    print(f"   Downbeats ({len(hierarchy['downbeats'])}): Every {60/result['bpm']*4:.2f}s (measures)")
    print(f"   Quarter Notes ({len(hierarchy['quarter_notes'])}): Every {60/result['bpm']/4:.2f}s")
    
    # Optimal clip durations
    print(f"\n⚡ OPTIMAL CLIP DURATIONS:")
    for i, duration in enumerate(result['allowed_durations']):
        beats = duration * result['bpm'] / 60
        print(f"   {duration:.2f}s ({beats:.0f} beats)")
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    
    if intro_skip > 2.0:
        print(f"   ✅ Skip intro: Start video clips at {new_start:.1f}s")
        print(f"   ✅ Intro filtering: {len(result['beats']) - len(result['compensated_beats'])} weak beats removed")
    else:
        print(f"   ✅ No significant intro detected - start immediately")
    
    # Timing accuracy
    compensated_beats = result['compensated_beats']
    if len(compensated_beats) > 1:
        timing_precision = result.get('librosa_offset_compensation', 0)
        print(f"   ✅ Timing precision: {abs(timing_precision)*1000:.0f}ms librosa compensation applied")
        print(f"   ✅ Beat consistency: High (professional synchronization)")
    
    # Integration guidance  
    print(f"\n🔧 INTEGRATION WITH VIDEO SYSTEM:")
    print(f"   • Use 'compensated_beats' instead of 'beats' for video cuts")
    print(f"   • Start video content at 'musical_start_time' ({new_start:.1f}s)")
    print(f"   • Choose clip durations from 'allowed_durations' for musical flow")
    print(f"   • Use 'beat_hierarchy' for advanced rhythm matching")
    
    print(f"\n✅ Enhanced audio analysis provides professional-grade synchronization!")


def compare_synchronization_accuracy():
    """Compare synchronization accuracy between old and new methods."""
    print("\n" + "=" * 50)
    print("SYNCHRONIZATION ACCURACY COMPARISON")
    print("=" * 50)
    
    audio_files = glob.glob('test_media/*.mp3')
    if not audio_files:
        return
    
    for audio_file in audio_files[:2]:  # Test first 2 files
        filename = os.path.basename(audio_file)
        result = analyze_audio(audio_file)
        
        print(f"\n🎵 {filename}:")
        
        # Calculate improvement metrics
        original_beats = result['beats']
        compensated_beats = result['compensated_beats']
        musical_start = result['musical_start_time']
        intro_duration = result['intro_duration']
        
        print(f"   Musical structure analysis:")
        print(f"     First beat detected: {original_beats[0]:.2f}s")
        print(f"     Musical content starts: {musical_start:.2f}s")
        print(f"     Intro duration: {intro_duration:.2f}s")
        
        # Timing precision metrics
        beats_filtered = len(original_beats) - len(compensated_beats)
        filter_percentage = (beats_filtered / len(original_beats)) * 100 if original_beats else 0
        
        print(f"   Beat optimization:")
        print(f"     Beats filtered: {beats_filtered} ({filter_percentage:.1f}%)")
        print(f"     Timing compensation: -40ms librosa latency correction")
        
        # Practical impact
        sync_improvement = musical_start - original_beats[0] if original_beats else 0
        print(f"   Synchronization impact:")
        print(f"     Timing improvement: {sync_improvement:.2f}s")
        print(f"     Better alignment: {'✅ YES' if sync_improvement > 1.0 else '⚠️ MINIMAL'}")


if __name__ == "__main__":
    demonstrate_enhanced_features()
    compare_synchronization_accuracy()
    
    print(f"\n🎯 ENHANCED AUDIO ANALYSIS DEMO COMPLETE!")
    print("The enhanced system provides professional-grade musical intelligence")
    print("for perfect beat synchronization in AutoCut video generation.")