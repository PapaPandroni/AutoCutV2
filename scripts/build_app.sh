#!/usr/bin/env bash
# Build dist/AutoCut.app - a self-contained macOS app for non-technical users.
#
# Fetches static FFmpeg binaries (once), runs PyInstaller with autocut.spec,
# and zips the result for sharing. Recipients need nothing installed; on
# first launch they must right-click > Open (the app is not notarized).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-env/bin/python}
ARCH=$(uname -m)

if [ "$ARCH" != "x86_64" ]; then
    echo "This script fetches x86_64 FFmpeg builds from evermeet.cx." >&2
    echo "For arm64, place static ffmpeg + ffprobe in vendor/ffmpeg/ manually" >&2
    echo "(e.g. from https://www.osxexperts.net) and re-run." >&2
    if [ ! -x vendor/ffmpeg/ffmpeg ] || [ ! -x vendor/ffmpeg/ffprobe ]; then
        exit 1
    fi
fi

# 1. Static FFmpeg binaries (bundled into the app so recipients don't need
#    Homebrew). Downloaded once, cached in vendor/ (gitignored).
mkdir -p vendor/ffmpeg
fetch_tool() {
    local tool=$1 url=$2
    if [ -x "vendor/ffmpeg/$tool" ]; then
        echo "vendor/ffmpeg/$tool already present, skipping download"
        return
    fi
    echo "Downloading static $tool..."
    curl -fL "$url" -o "vendor/ffmpeg/$tool.zip"
    unzip -o -q "vendor/ffmpeg/$tool.zip" "$tool" -d vendor/ffmpeg
    rm "vendor/ffmpeg/$tool.zip"
    chmod +x "vendor/ffmpeg/$tool"
    "vendor/ffmpeg/$tool" -version | head -1
}
if [ "$ARCH" = "x86_64" ]; then
    fetch_tool ffmpeg "https://evermeet.cx/ffmpeg/getrelease/zip"
    fetch_tool ffprobe "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
fi

# 2. PyInstaller
if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    "$PYTHON" -m pip install --quiet pyinstaller
fi

# 3. Build the .app (PyInstaller ad-hoc code-signs it, required on macOS)
rm -rf build/autocut dist/AutoCut dist/AutoCut.app
"$PYTHON" -m PyInstaller --noconfirm autocut.spec

# 4. Zip for sharing (ditto preserves the bundle metadata Finder needs)
echo "Zipping..."
ditto -c -k --keepParent dist/AutoCut.app dist/AutoCut-mac.zip

echo
echo "Done:"
echo "  App: dist/AutoCut.app"
echo "  Share this file: dist/AutoCut-mac.zip ($(du -h dist/AutoCut-mac.zip | cut -f1))"
echo
echo "Tell recipients: unzip, move AutoCut to Applications, then"
echo "right-click > Open > Open the first time (unsigned app warning)."
