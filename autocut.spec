# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the AutoCut macOS app.

Build with scripts/build_app.sh (fetches the static FFmpeg binaries and
runs PyInstaller). Produces dist/AutoCut.app.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

vendor = Path("vendor/ffmpeg")
for tool in ("ffmpeg", "ffprobe"):
    if not (vendor / tool).exists():
        raise SystemExit(
            f"Missing {vendor / tool} - run scripts/build_app.sh to fetch "
            "the static FFmpeg builds first"
        )

# Bundled next to the executable; src/ffmpeg_paths.py resolves them via
# sys._MEIPASS at runtime.
binaries = [
    (str(vendor / "ffmpeg"), "."),
    (str(vendor / "ffprobe"), "."),
]

# librosa loads submodules through lazy_loader and ships data files
# (example audio, registry files) that static analysis misses.
datas = collect_data_files("librosa")
hiddenimports = collect_submodules("librosa")

a = Analysis(
    ["autocut_gui.py"],
    # Mirrors the sys.path.insert(0, "src") convention used by the entry
    # points: src modules are imported as top-level packages (api, gui, ...).
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # Listed in requirements.txt but never imported by src/
        "deffcode",
        "av",
        # Heavyweight optional extras of scipy/librosa we don't use
        "matplotlib",
        "pandas",
        "IPython",
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoCut",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # windowed app: no terminal window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AutoCut",
)

app = BUNDLE(
    coll,
    name="AutoCut.app",
    icon=None,
    bundle_identifier="com.autocut.app",
    info_plist={
        "CFBundleName": "AutoCut",
        "CFBundleDisplayName": "AutoCut",
        "CFBundleShortVersionString": "2.0.0",
        "NSHighResolutionCapable": True,
    },
)
