#!/usr/bin/env bash
# ------------------------------------------------------------
# SceneIQ — one-shot environment bootstrap
# Usage: bash setup.sh
# ------------------------------------------------------------
set -euo pipefail

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==> Checking for ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "    ffmpeg found: $(command -v ffmpeg)"
else
    echo "    WARNING: ffmpeg not found on PATH."
    echo "    Whisper ASR fallback will be unavailable."
    echo "    Install via: https://ffmpeg.org/download.html"
fi

echo ""
echo "==> Setup complete. Run the app with:"
echo "    python run.py"
echo ""
