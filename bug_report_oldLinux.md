(env) peremil@OldBook:~/Documents/repos/AutoCutV2$ python3 autocut.py process /home/peremil/Documents/repos/AutoCutV2/test_media/*.mov -a /home/peremil/Documents/repos/AutoCutV2/test_media/'ES_Sunset Beach - PW.mp3' -o /home/peremil/Downloads/test.mp4
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - === AutoCut Video Processing Started ===
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - Input videos: 10 files
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - Audio file: ES_Sunset Beach - PW.mp3
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - Output path: /home/peremil/Downloads/test.mp4
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - Pattern: balanced
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - Validating input files...
2025-08-07 16:22:12,634 - autocut.clip_assembler - INFO - ✅ All input files exist and are accessible
2025-08-07 16:22:12,635 - autocut.clip_assembler - INFO - === Step 1: Audio Analysis ===
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO - ✅ Audio analysis successful:
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO -   - Beats detected: 374
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO -   - Musical start: 3.18s
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO -   - Intro duration: 1.83s
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO - === Step 2: Video Analysis ===
2025-08-07 16:22:28,363 - autocut.clip_assembler - INFO - --- Processing video 1/10: IMG_0431.mov ---
2025-08-07 16:22:28,364 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:22:28,364 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0431.mov
2025-08-07 16:22:28,364 - autocut.video_analyzer - INFO - Loading video file: IMG_0431.mov
🔍 Enhanced preprocessing: IMG_0431.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.53s)
      📹 Video: 1080x1920 @ 30.0fps, 3.9s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 0.7s
2025-08-07 16:22:29,268 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 3.94s, 29.97fps
2025-08-07 16:22:29,272 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0431.mov
2025-08-07 16:22:30,322 - autocut.video_analyzer - INFO - Scene detection complete: 2 scenes found
2025-08-07 16:22:30,322 - autocut.video_analyzer - INFO - Scene filtering complete: 2 valid scenes (min duration: 0.511s)
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0431.mov:
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO -   - Scenes detected: 2
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO -   - Valid scenes: 2
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO -   - Chunks created: 2
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:22:35,875 - autocut.video_analyzer - INFO - Successfully created 2 chunks from IMG_0431.mov (scores: 75.4-67.6)
2025-08-07 16:22:35,875 - autocut.clip_assembler - INFO - ✅ IMG_0431.mov: 2 chunks created (7.51s)
2025-08-07 16:22:35,875 - autocut.clip_assembler - INFO -    Chunk scores: 67.6-75.4 (avg: 71.5)
2025-08-07 16:22:35,875 - autocut.clip_assembler - INFO - --- Processing video 2/10: IMG_0472.mov ---
2025-08-07 16:22:35,876 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:22:35,876 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0472.mov
2025-08-07 16:22:35,876 - autocut.video_analyzer - INFO - Loading video file: IMG_0472.mov
🔍 Enhanced preprocessing: IMG_0472.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.78s)
      📹 Video: 1080x1920 @ 30.0fps, 10.1s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 0.9s
2025-08-07 16:22:37,046 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 10.14s, 29.97fps
2025-08-07 16:22:37,046 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0472.mov
2025-08-07 16:22:40,423 - autocut.video_analyzer - INFO - Scene detection complete: 9 scenes found
2025-08-07 16:22:40,423 - autocut.video_analyzer - INFO - Scene filtering complete: 9 valid scenes (min duration: 0.511s)
2025-08-07 16:22:57,988 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0472.mov:
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO -   - Scenes detected: 9
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO -   - Valid scenes: 9
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO -   - Chunks created: 9
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO - Successfully created 9 chunks from IMG_0472.mov (scores: 80.6-60.3)
2025-08-07 16:22:57,989 - autocut.clip_assembler - INFO - ✅ IMG_0472.mov: 9 chunks created (22.11s)
2025-08-07 16:22:57,989 - autocut.clip_assembler - INFO -    Chunk scores: 60.3-80.6 (avg: 69.7)
2025-08-07 16:22:57,989 - autocut.clip_assembler - INFO - --- Processing video 3/10: IMG_0488.mov ---
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0488.mov
2025-08-07 16:22:57,989 - autocut.video_analyzer - INFO - Loading video file: IMG_0488.mov
🔍 Enhanced preprocessing: IMG_0488.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.62s)
      📹 Video: 1080x1920 @ 30.0fps, 5.9s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 0.8s
2025-08-07 16:22:58,985 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 5.91s, 29.97fps
2025-08-07 16:22:58,985 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0488.mov
2025-08-07 16:23:00,640 - autocut.video_analyzer - INFO - Scene detection complete: 4 scenes found
2025-08-07 16:23:00,640 - autocut.video_analyzer - INFO - Scene filtering complete: 4 valid scenes (min duration: 0.511s)
2025-08-07 16:23:14,819 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0488.mov:
2025-08-07 16:23:14,820 - autocut.video_analyzer - INFO -   - Scenes detected: 4
2025-08-07 16:23:14,820 - autocut.video_analyzer - INFO -   - Valid scenes: 4
2025-08-07 16:23:14,820 - autocut.video_analyzer - INFO -   - Chunks created: 4
2025-08-07 16:23:14,820 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:23:14,820 - autocut.video_analyzer - INFO - Successfully created 4 chunks from IMG_0488.mov (scores: 88.2-50.2)
2025-08-07 16:23:14,820 - autocut.clip_assembler - INFO - ✅ IMG_0488.mov: 4 chunks created (16.83s)
2025-08-07 16:23:14,820 - autocut.clip_assembler - INFO -    Chunk scores: 50.2-88.2 (avg: 66.0)
2025-08-07 16:23:14,820 - autocut.clip_assembler - INFO - --- Processing video 4/10: IMG_0502.mov ---
2025-08-07 16:23:14,821 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:23:14,821 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0502.mov
2025-08-07 16:23:14,821 - autocut.video_analyzer - INFO - Loading video file: IMG_0502.mov
🔍 Enhanced preprocessing: IMG_0502.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (1.63s)
      📹 Video: 1080x1920 @ 30.0fps, 19.3s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 2.0s
2025-08-07 16:23:17,236 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 19.28s, 29.97fps
2025-08-07 16:23:17,236 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0502.mov
2025-08-07 16:23:29,341 - autocut.video_analyzer - INFO - Scene detection complete: 10 scenes found
2025-08-07 16:23:29,341 - autocut.video_analyzer - INFO - Scene filtering complete: 10 valid scenes (min duration: 0.511s)
2025-08-07 16:24:19,893 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0502.mov:
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO -   - Scenes detected: 10
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO -   - Valid scenes: 10
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO -   - Chunks created: 10
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO - Successfully created 10 chunks from IMG_0502.mov (scores: 75.1-61.7)
2025-08-07 16:24:19,894 - autocut.clip_assembler - INFO - ✅ IMG_0502.mov: 10 chunks created (65.07s)
2025-08-07 16:24:19,894 - autocut.clip_assembler - INFO -    Chunk scores: 61.7-75.1 (avg: 67.1)
2025-08-07 16:24:19,894 - autocut.clip_assembler - INFO - --- Processing video 5/10: IMG_0511.mov ---
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0511.mov
2025-08-07 16:24:19,894 - autocut.video_analyzer - INFO - Loading video file: IMG_0511.mov
🔍 Enhanced preprocessing: IMG_0511.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.96s)
      📹 Video: 1080x1920 @ 30.0fps, 18.9s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 1.2s
2025-08-07 16:24:21,386 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 18.95s, 29.97fps
2025-08-07 16:24:21,387 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0511.mov
2025-08-07 16:24:30,763 - autocut.video_analyzer - INFO - Scene detection complete: 1 scenes found
2025-08-07 16:24:30,763 - autocut.video_analyzer - INFO - Scene filtering complete: 1 valid scenes (min duration: 0.511s)
2025-08-07 16:24:46,281 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0511.mov:
2025-08-07 16:24:46,281 - autocut.video_analyzer - INFO -   - Scenes detected: 1
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO -   - Valid scenes: 1
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO -   - Chunks created: 1
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO - Successfully created 1 chunks from IMG_0511.mov (scores: 58.7-58.7)
2025-08-07 16:24:46,282 - autocut.clip_assembler - INFO - ✅ IMG_0511.mov: 1 chunks created (26.39s)
2025-08-07 16:24:46,282 - autocut.clip_assembler - INFO -    Chunk scores: 58.7-58.7 (avg: 58.7)
2025-08-07 16:24:46,282 - autocut.clip_assembler - INFO - --- Processing video 6/10: IMG_0525.mov ---
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0525.mov
2025-08-07 16:24:46,282 - autocut.video_analyzer - INFO - Loading video file: IMG_0525.mov
🔍 Enhanced preprocessing: IMG_0525.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.69s)
      📹 Video: 1080x1920 @ 30.0fps, 3.3s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 0.9s
2025-08-07 16:24:47,448 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 3.33s, 30.00fps
2025-08-07 16:24:47,452 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0525.mov
2025-08-07 16:24:48,695 - autocut.video_analyzer - INFO - Scene detection complete: 2 scenes found
2025-08-07 16:24:48,695 - autocut.video_analyzer - INFO - Scene filtering complete: 2 valid scenes (min duration: 0.511s)
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0525.mov:
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO -   - Scenes detected: 2
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO -   - Valid scenes: 2
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO -   - Chunks created: 2
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:24:55,068 - autocut.video_analyzer - INFO - Successfully created 2 chunks from IMG_0525.mov (scores: 78.1-76.1)
2025-08-07 16:24:55,068 - autocut.clip_assembler - INFO - ✅ IMG_0525.mov: 2 chunks created (8.79s)
2025-08-07 16:24:55,068 - autocut.clip_assembler - INFO -    Chunk scores: 76.1-78.1 (avg: 77.1)
2025-08-07 16:24:55,069 - autocut.clip_assembler - INFO - --- Processing video 7/10: IMG_0579.mov ---
2025-08-07 16:24:55,069 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:24:55,069 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0579.mov
2025-08-07 16:24:55,069 - autocut.video_analyzer - INFO - Loading video file: IMG_0579.mov
🔍 Enhanced preprocessing: IMG_0579.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: h264 (85/100 compatibility)
      ⚠️  High resolution may require more processing power
      ⚠️  Pixel format 'yuvj420p' may need conversion
   ✅ h264 format compatible, no processing needed
   ⏱️ Preprocessing completed in 0.2s
2025-08-07 16:24:55,490 - autocut.video_analyzer - INFO - Video loaded successfully: 720x1280, 25.70s, 30.00fps
2025-08-07 16:24:55,490 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0579.mov
2025-08-07 16:24:58,185 - autocut.video_analyzer - INFO - Scene detection complete: 7 scenes found
2025-08-07 16:24:58,185 - autocut.video_analyzer - INFO - Scene filtering complete: 7 valid scenes (min duration: 0.511s)
2025-08-07 16:25:06,127 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0579.mov:
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO -   - Scenes detected: 7
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO -   - Valid scenes: 7
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO -   - Chunks created: 7
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO - Successfully created 7 chunks from IMG_0579.mov (scores: 76.6-67.9)
2025-08-07 16:25:06,128 - autocut.clip_assembler - INFO - ✅ IMG_0579.mov: 7 chunks created (11.06s)
2025-08-07 16:25:06,128 - autocut.clip_assembler - INFO -    Chunk scores: 67.9-76.6 (avg: 70.9)
2025-08-07 16:25:06,128 - autocut.clip_assembler - INFO - --- Processing video 8/10: IMG_0596.mov ---
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0596.mov
2025-08-07 16:25:06,128 - autocut.video_analyzer - INFO - Loading video file: IMG_0596.mov
🔍 Enhanced preprocessing: IMG_0596.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (25/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.83s)
      📹 Video: 1920x1080 @ 30.0fps, 10.1s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 1.0s
2025-08-07 16:25:07,350 - autocut.video_analyzer - INFO - Video loaded successfully: 1920x1080, 10.14s, 29.97fps
2025-08-07 16:25:07,350 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0596.mov
2025-08-07 16:25:11,223 - autocut.video_analyzer - INFO - Scene detection complete: 9 scenes found
2025-08-07 16:25:11,224 - autocut.video_analyzer - INFO - Scene filtering complete: 9 valid scenes (min duration: 0.511s)
2025-08-07 16:25:32,720 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0596.mov:
2025-08-07 16:25:32,720 - autocut.video_analyzer - INFO -   - Scenes detected: 9
2025-08-07 16:25:32,720 - autocut.video_analyzer - INFO -   - Valid scenes: 9
2025-08-07 16:25:32,720 - autocut.video_analyzer - INFO -   - Chunks created: 9
2025-08-07 16:25:32,721 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:25:32,721 - autocut.video_analyzer - INFO - Successfully created 9 chunks from IMG_0596.mov (scores: 76.4-59.1)
2025-08-07 16:25:32,721 - autocut.clip_assembler - INFO - ✅ IMG_0596.mov: 9 chunks created (26.59s)
2025-08-07 16:25:32,721 - autocut.clip_assembler - INFO -    Chunk scores: 59.1-76.4 (avg: 66.1)
2025-08-07 16:25:32,721 - autocut.clip_assembler - INFO - --- Processing video 9/10: IMG_0604.mov ---
2025-08-07 16:25:32,721 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:25:32,721 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0604.mov
2025-08-07 16:25:32,721 - autocut.video_analyzer - INFO - Loading video file: IMG_0604.mov
🔍 Enhanced preprocessing: IMG_0604.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (0.97s)
      📹 Video: 1080x1920 @ 30.0fps, 8.5s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 1.2s
2025-08-07 16:25:34,186 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 8.47s, 30.00fps
2025-08-07 16:25:34,187 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0604.mov
2025-08-07 16:25:37,622 - autocut.video_analyzer - INFO - Scene detection complete: 7 scenes found
2025-08-07 16:25:37,622 - autocut.video_analyzer - INFO - Scene filtering complete: 7 valid scenes (min duration: 0.511s)
2025-08-07 16:25:54,494 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0604.mov:
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO -   - Scenes detected: 7
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO -   - Valid scenes: 7
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO -   - Chunks created: 7
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO - Successfully created 7 chunks from IMG_0604.mov (scores: 79.0-62.5)
2025-08-07 16:25:54,495 - autocut.clip_assembler - INFO - ✅ IMG_0604.mov: 7 chunks created (21.77s)
2025-08-07 16:25:54,495 - autocut.clip_assembler - INFO -    Chunk scores: 62.5-79.0 (avg: 69.0)
2025-08-07 16:25:54,495 - autocut.clip_assembler - INFO - --- Processing video 10/10: IMG_0608.mov ---
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO - Using beat-based minimum scene duration: 0.511s (1.0 beats at 117.5 BPM)
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO - Starting analysis of video: IMG_0608.mov
2025-08-07 16:25:54,495 - autocut.video_analyzer - INFO - Loading video file: IMG_0608.mov
🔍 Enhanced preprocessing: IMG_0608.mov
   📊 Step 1: Analyzing video format...
   ✅ Codec analysis: hevc (20/100 compatibility)
      ⚠️  H.265/HEVC may require transcoding for optimal compatibility
      ⚠️  High resolution may require more processing power
      ⚠️  10-bit video may have limited compatibility
   📱 Step 2: Testing H.265/iPhone compatibility...
   🧪 Testing MoviePy H.265 compatibility (timeout: 10.0s)...
   ✅ Compatibility test passed (1.04s)
      📹 Video: 1080x1920 @ 30.0fps, 20.8s
      🎯 Frame decode: successful
   ✅ H.265 compatible with MoviePy, skipping transcoding
   ⏱️ Preprocessing completed in 1.2s
2025-08-07 16:25:56,045 - autocut.video_analyzer - INFO - Video loaded successfully: 1080x1920, 20.78s, 29.97fps
2025-08-07 16:25:56,045 - autocut.video_analyzer - INFO - Starting scene detection for: IMG_0608.mov
2025-08-07 16:26:04,960 - autocut.video_analyzer - INFO - Scene detection complete: 7 scenes found
2025-08-07 16:26:04,960 - autocut.video_analyzer - INFO - Scene filtering complete: 7 valid scenes (min duration: 0.511s)
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO - Video analysis complete for IMG_0608.mov:
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO -   - Scenes detected: 7
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO -   - Valid scenes: 7
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO -   - Chunks created: 7
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO -   - Chunks failed: 0
2025-08-07 16:26:39,020 - autocut.video_analyzer - INFO - Successfully created 7 chunks from IMG_0608.mov (scores: 79.0-65.6)
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - ✅ IMG_0608.mov: 7 chunks created (44.53s)
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO -    Chunk scores: 65.6-79.0 (avg: 73.7)
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - === Video Processing Summary ===
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - Total videos: 10
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - ✅ Successful: 10
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - ❌ Failed: 0
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - 📊 Total chunks created: 58
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - Processing success rate: 100.0%
2025-08-07 16:26:39,020 - autocut.clip_assembler - INFO - === Step 3: Beat Matching ===
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO - ✅ Beat matching successful: 58 clips selected
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO - Timeline statistics:
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO -   - Total duration: 58.89s
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO -   - Average score: 69.1
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO -   - Score range: 50.2-88.2
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO -   - Unique videos used: 10
2025-08-07 16:26:39,024 - autocut.clip_assembler - INFO - === Step 4: Video Rendering ===
Debug: MoviePy API compatibility detected: new
Debug: Method mappings - subclip: subclipped, set_audio: with_audio
Debug: Original audio duration: 199.800000s
Debug: Timeline has 58 clips to load
   🧠 Initial memory: 2.7GB used (41.6%), 4.5GB available
🔍 System Analysis:
   💾 Memory: 4.5GB available / 7.7GB total (41.6% used)
   🖥️  CPU: 4 cores @ 3.3GHz (Linux x86_64)
   ⚡ Hardware Acceleration: ❌ CPU only
   📊 Performance Score: 65.4/100

🎬 Video Analysis:
   📹 Estimated memory per video: 31MB
   🧮 Codec complexity factor: 1.0x
   📐 Resolution factor: 1.0x
   📈 Analysis confidence: 1.0/1.0

⚙️  Worker Calculation:
   📊 Memory allows: 64 workers
   🖥️  CPU allows: 3 workers
   ⚡ HW acceleration bonus: 1.0x
   🚀 Performance bonus: 1.1x
   ✅ Optimal workers: 3

🧠 Predicted Memory Usage:
   📈 Expected usage: 0.1GB (2.1% of available)
   ✅ LOW: Plenty of headroom
   🔍 Started adaptive monitoring (thresholds: 85.0%/95.0%)
/home/peremil/Documents/repos/AutoCutV2/env/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:190: UserWarning: In file /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0502.mov, 6220800 bytes wanted but 0 bytes read at frame index 269 (out of a total 577 frames), at time 8.98/19.28 sec. Using the last valid frame instead.
  warnings.warn(
/home/peremil/Documents/repos/AutoCutV2/env/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:190: UserWarning: In file /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0472.mov, 6220800 bytes wanted but 0 bytes read at frame index 239 (out of a total 303 frames), at time 7.97/10.14 sec. Using the last valid frame instead.
  warnings.warn(
Warning: Failed to load clip /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0472.mov: PyMemoryView_FromBuffer(): info->buf must not be NULL
Warning: Failed to load clip /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0596.mov: PyMemoryView_FromBuffer(): info->buf must not be NULL
Warning: Failed to load clip /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0596.mov: read of closed file
Warning: Failed to load clip /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0502.mov: PyMemoryView_FromBuffer(): info->buf must not be NULL
/home/peremil/Documents/repos/AutoCutV2/env/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:190: UserWarning: In file /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0502.mov, 6220800 bytes wanted but 0 bytes read at frame index 359 (out of a total 577 frames), at time 11.98/19.28 sec. Using the last valid frame instead.
  warnings.warn(
Proc not detected
Proc not detected
/home/peremil/Documents/repos/AutoCutV2/env/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:190: UserWarning: In file /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0502.mov, 6220800 bytes wanted but 0 bytes read at frame index 299 (out of a total 577 frames), at time 9.98/19.28 sec. Using the last valid frame instead.
  warnings.warn(
Warning: Failed to load clip /home/peremil/Documents/repos/AutoCutV2/test_media/IMG_0502.mov: PyMemoryView_FromBuffer(): info->buf must not be NULL
   🛑 Stopped adaptive monitoring
Warning: 5 clips failed to load (indices: [20, 31, 34, 38, 53])
   🧠 Final memory: 6.0GB used (+3.4GB), 1.3GB available
   📦 Video cache: 10 unique files loaded
Debug: Loaded 53 video clips successfully
SYNC FIX: Removing 5 failed clips from timeline
Debug: Timeline synchronized - 58 -> 53 clips
Debug: Final clip count - video_clips: 53, timeline.clips: 53
Format Analysis: Dominant 1080x1920 @ 30.0fps
Format Diversity: 3 resolutions, 1 frame rates
Normalization Required: True
FORMAT ISSUES DETECTED:
  - resolution_mismatch: Mixed resolutions detected: {(1920, 1080), (720, 1280), (1080, 1920)}
    Artifacts: flashing up/down, scaling artifacts, centering issues
  - framerate_mismatch: Mixed frame rates detected: {29.97002997002997, 30.0}
    Artifacts: VHS-like wrap around, temporal stuttering, motion artifacts
  - aspect_ratio_mismatch: Mixed aspect ratios detected: {0.562, 1.778}
    Artifacts: letterboxing inconsistency, stretching artifacts
Format normalization: Normalizing 53 clips to 1080x1920 @ 30.0fps
Normalized clip 1/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 2/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 3/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 4/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 5/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 6/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 7/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 8/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 9/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 10/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 11/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 12/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 13/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 14/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 15/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 16/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 17/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 18/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 19/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 20/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 21/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 22/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 23/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 24/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 25/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 26/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 27/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 28/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 29/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 30/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 31/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 32/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 33/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 34/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 35/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 36/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 37/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 38/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 39/53: 1080x1920@30.0fps -> 1080x1920@30.0fps
Normalized clip 40/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 41/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 42/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 43/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 44/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 45/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 46/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 47/53: 1920x1080@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 48/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 49/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 50/53: 720x1280@30.0fps -> 1080x1920@30.0fps
Normalized clip 51/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 52/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Normalized clip 53/53: 1080x1920@29.97002997002997fps -> 1080x1920@29.97002997002997fps
Debug: Normalized clip 1: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 2: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 3: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 4: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 5: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 6: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 7: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 8: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 9: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 10: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 11: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 12: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 13: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 14: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 15: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 16: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 17: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 18: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 19: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 20: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 21: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 22: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 23: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 24: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 25: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 26: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 27: actual=2.040119s, expected=2.040119s
Debug: Normalized clip 28: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 29: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 30: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 31: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 32: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 33: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 34: actual=1.020059s, expected=1.020059s
Debug: Normalized clip 35: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 36: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 37: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 38: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 39: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 40: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 41: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 42: actual=2.000000s, expected=2.000000s
Debug: Normalized clip 43: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 44: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 45: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 46: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 47: actual=1.020059s, expected=1.020059s
Debug: Normalized clip 48: actual=2.040119s, expected=2.040119s
Debug: Normalized clip 49: actual=1.000000s, expected=1.000000s
Debug: Normalized clip 50: actual=1.020059s, expected=1.020059s
Debug: Normalized clip 51: actual=0.510030s, expected=0.510030s
Debug: Normalized clip 52: actual=1.020059s, expected=1.020059s
Debug: Normalized clip 53: actual=2.040119s, expected=2.040119s
Debug: Total expected video duration: 54.871100s
Debug: Using 'compose' method for 53 normalized clips
Debug: Concatenation successful, final video duration: 54.871100s
Debug: Target FPS: 30.0, Frame duration: 0.033333s
Debug: Adding sync buffer: 0.066667s to prevent audio cutoff
Debug: Frame-accurate calculation: 1638 frames @ 30.0fps = 54.600000s
Debug: Concatenation duration: 54.871100s
Debug: FRAME-ACCURATE audio trim: 199.800000s -> 54.666667s
Debug: Audio buffer added: 0.066667s to prevent cutoff
Debug: Final audio duration: 54.666667s
Debug: Audio-video sync difference: 0.204433s
Warning: Significant audio-video sync difference: 0.204433s
Debug: Attaching audio using method: with_audio
🔍 Enhanced hardware capability detection...
   🧪 Fast-testing NVENC with iPhone parameters...
   ❌ NVENC fast test failed: driver_version
   🧪 Fast-testing QSV with iPhone parameters...
   ❌ QSV fast test failed: device_capability
⚡ Using optimized CPU encoding (hardware acceleration not available)
📋 Codec settings (legacy interface): UNKNOWN encoder
Debug: Enhanced FFmpeg params: ['-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p', '-vsync', 'cfr', '-async', '1', '-r', '30.0', '-s', '1080x1920']
Debug: Rendering final video to: /home/peremil/Downloads/test.mp4
Debug: Write parameters: ['codec', 'audio_codec', 'threads', 'ffmpeg_params', 'temp_audiofile', 'remove_temp', 'fps', 'logger', 'audio_fps', 'audio_bitrate']
Killed
