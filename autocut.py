#!/usr/bin/env python3
"""
AutoCut V2 - Unified Command Line Interface

AutoCut automatically creates beat-synced highlight videos from raw footage and music.
It analyzes video quality, detects music rhythm, and intelligently assembles clips
that match the beat - all without requiring video editing knowledge.
"""

import sys
import time
from pathlib import Path

import click

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the API - no src. prefix needed since src is in path
from api import AutoCutAPI

# Global API instance
api = AutoCutAPI()


@click.group()
@click.version_option(version="2.0.0", message="AutoCut %(version)s")
def cli():
    """
    AutoCut V2 - Automated Beat-Synced Video Creation

    Transform hours of raw footage into polished, music-synced highlight reels
    in minutes. Professional results with minimal effort.
    """


@cli.command()
@click.argument("video_files", nargs=-1, required=True)
@click.option(
    "--audio",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Audio file for beat synchronization",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output video path (default: auto-generated)",
)
@click.option(
    "--pattern",
    "-p",
    type=click.Choice(["energetic", "balanced", "dramatic", "buildup"]),
    default="balanced",
    help="Editing pattern style",
)
@click.option("--max-videos", type=int, help="Maximum number of videos to use")
@click.option(
    "--memory-safe",
    is_flag=True,
    help="Enable memory-safe processing (reduces parallel workers)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def process(video_files, audio, output, pattern, max_videos, memory_safe, verbose):
    """
    Process videos to create beat-synced compilation

    Takes multiple video files and an audio track, then creates a professionally
    edited video where cuts are synchronized to the music beats.

    Examples:
    \b
        autocut process video1.mov video2.mp4 --audio music.mp3
        autocut process *.mov --audio song.wav --pattern dramatic
        autocut process folder/*.mp4 --audio track.mp3 --max-videos 5
    """
    try:
        # Convert video_files tuple to list and validate
        video_list = []
        for pattern_or_file in video_files:
            # Handle glob patterns and individual files
            if "*" in pattern_or_file:
                from pathlib import Path as PathlibPath

                matched_files = [str(p) for p in PathlibPath().glob(pattern_or_file)]
                video_list.extend(matched_files)
            elif Path(pattern_or_file).exists():
                video_list.append(pattern_or_file)
            else:
                click.echo(f"❌ File not found: {pattern_or_file}", err=True)
                sys.exit(1)

        if not video_list:
            click.echo("❌ No valid video files provided", err=True)
            sys.exit(1)

        # Limit videos if requested
        if max_videos and len(video_list) > max_videos:
            video_list = video_list[:max_videos]

        # Display processing info
        if verbose:
            click.echo(f"🎬 AutoCut V2 - Processing {len(video_list)} videos")
            click.echo(f"🎵 Audio: {Path(audio).name}")
            click.echo(f"🎯 Pattern: {pattern}")

        # Create output path if not specified
        if not output:
            timestamp = int(time.time())
            output = f"output/autocut_{pattern}_{timestamp}.mp4"

        # Create output directory
        output_dir = Path(output).parent if Path(output).parent != Path() else Path()
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            click.echo(f"📄 Output: {output}")
            click.echo("\n🚀 Processing...")

        # Memory optimization info
        if memory_safe:
            click.echo("🧠 Memory-safe mode enabled: Using single-threaded processing")

        # Process videos using API
        result_path = api.process_videos(
            video_files=video_list,
            audio_file=audio,
            output_path=output,
            pattern=pattern,
            memory_safe=memory_safe,
            verbose=verbose,
        )

        # Show results
        click.echo(f"\n✅ Success! Created: {result_path}")

        if Path(result_path).exists():
            file_size = Path(result_path).stat().st_size / (1024 * 1024)  # MB
            click.echo(f"   Size: {file_size:.1f} MB")

        click.echo("\n🎬 Your beat-synced video is ready to watch!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed validation information",
)
def validate(video_path, detailed):
    """
    Validate video compatibility and quality

    Analyzes a video file for AutoCut compatibility, including codec support,
    iPhone H.265 compatibility, and processing requirements.

    Examples:
    \b
        autocut validate my_video.mp4
        autocut validate iPhone_footage.mov --detailed
    """
    try:
        click.echo(f"🔍 Validating: {Path(video_path).name}")

        result = api.validate_video(video_path, detailed=detailed)

        # Display validation results
        if result.is_valid:
            click.echo("✅ Video is compatible with AutoCut")
        else:
            click.echo("⚠️  Video has compatibility issues:")
            for issue in result.issues:
                click.echo(f"   • {issue}")

        if detailed:
            click.echo("\n📊 Video Information:")
            for key, value in result.metadata.items():
                click.echo(f"   {key}: {value}")

            if result.suggestions:
                click.echo("\n💡 Suggestions:")
                for suggestion in result.suggestions:
                    click.echo(f"   • {suggestion}")

    except Exception as e:
        click.echo(f"❌ Validation error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed system information")
def benchmark(detailed):
    """
    Run system performance benchmarks

    Tests hardware acceleration capabilities, processing speed, and system
    compatibility to help optimize AutoCut performance.

    Examples:
    \b
        autocut benchmark
        autocut benchmark --detailed
    """
    try:
        click.echo("🏃 Running AutoCut system benchmarks...")

        system_info = api.get_system_info()

        click.echo("\n💻 System Capabilities:")
        click.echo(
            f"   Hardware Acceleration: {'✅' if system_info.has_hardware_acceleration else '❌'}",
        )
        click.echo(
            f"   Supported Encoders: {', '.join(system_info.available_encoders)}",
        )
        click.echo(f"   CPU Cores: {system_info.cpu_cores}")

        if detailed:
            diagnostics = api.run_diagnostics()

            click.echo("\n🔧 Detailed Diagnostics:")
            click.echo(f"   FFmpeg Version: {diagnostics.ffmpeg_version}")
            click.echo(f"   MoviePy Version: {diagnostics.moviepy_version}")
            click.echo(f"   Platform: {diagnostics.platform}")

            if diagnostics.performance_metrics:
                click.echo("\n⚡ Performance Metrics:")
                for metric, value in diagnostics.performance_metrics.items():
                    click.echo(f"   {metric}: {value}")

    except Exception as e:
        click.echo(f"❌ Benchmark error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--quick", "-q", is_flag=True, help="Quick demo with limited files")
@click.option(
    "--pattern",
    "-p",
    type=click.Choice(["energetic", "balanced", "dramatic", "buildup"]),
    default="balanced",
    help="Editing pattern to demonstrate",
)
def demo(quick, pattern):
    """
    Run AutoCut demonstration

    Demonstrates AutoCut functionality using sample files from the test_media
    directory. Great for testing the system and seeing AutoCut in action.

    Examples:
    \b
        autocut demo
        autocut demo --quick --pattern dramatic
    """
    try:
        click.echo("🎬 AutoCut V2 Demonstration")
        click.echo("=" * 40)

        # Find test media
        test_media_dir = Path("test_media")
        if not test_media_dir.exists():
            click.echo("❌ test_media directory not found")
            click.echo("   Create test_media/ and add video/audio files for demo")
            sys.exit(1)

        # Use the API's demo functionality
        result = api.run_demo(
            quick=quick,
            pattern=pattern,
            test_media_dir=str(test_media_dir),
        )

        if result.success:
            click.echo("\n🎉 Demo completed successfully!")
            click.echo(f"   Created: {result.output_path}")
            click.echo(f"   Processing time: {result.processing_time:.1f}s")
            click.echo("\n🎬 Watch the video to see AutoCut's beat-sync magic!")
        else:
            click.echo(
                "❌ Demo failed - check test_media directory has video/audio files",
            )
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Demo error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
