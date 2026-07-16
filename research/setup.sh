#!/usr/bin/env bash
#
# One-shot environment setup for SoccerNet/sn-gamestate on Google Colab.
#
# Why a script instead of notebook cells?
#   - Google Drive mounting is currently broken in Colab: it invokes the wrong
#     OAuth client ("Google Drive for desktop" instead of "Google Colaboratory"),
#     failing with "credential propagation was unsuccessful".
#     See https://github.com/googlecolab/colabtools/issues/5944
#   - Because Drive can't be mounted, we cannot persist installs across Colab VM
#     restarts. Every fresh VM (restart/disconnect) needs a full re-install, so we
#     make that a single, unattended, re-runnable script rather than a cell dance.
#
# Recommended invocation (survives notebook disconnects):
#     nohup bash research/setup.sh > /content/install_log.txt 2>&1 &
#     tail -f /content/install_log.txt
#
# Idempotent where it can be: the clone is skipped if present, and apt/pip/uv
# steps are safe to re-run.

set -euo pipefail

# Absolute path to this script's directory (the football-tactical-engine clone),
# so we can find patch_sn_gamestate.py after cd-ing into the sn-gamestate tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/8] Installing Python 3.9 via deadsnakes PPA ==="
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.9 python3.9-venv python3.9-dev python3.9-distutils

echo "=== [2/8] Bootstrapping pip + registering Jupyter kernel for Python 3.9 ==="
# Without this the Colab kernel picker shows "Python 3.9 (sn-gamestate)" but hangs
# forever on "Connecting", because ipykernel isn't actually installed in this interpreter.
python3.9 -m ensurepip --upgrade
python3.9 -m pip install ipykernel
python3.9 -m ipykernel install \
  --name sn-gamestate-py39 \
  --display-name "Python 3.9 (sn-gamestate)" \
  --user

echo "=== [3/8] Installing uv ==="
# Colab presets UV_SYSTEM_PYTHON=true, which silently forces uv to ignore any venv
# and install into the system Python (3.12). That breaks sn-gamestate's
# torch==1.13.1 dependency, which has no cp312 wheels. Unset it before doing anything.
unset UV_SYSTEM_PYTHON
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [4/8] Cloning SoccerNet/sn-gamestate ==="
cd /content
if [ ! -d sn-gamestate ]; then
  git clone https://github.com/SoccerNet/sn-gamestate.git
else
  echo "sn-gamestate already present, skipping clone."
fi
cd sn-gamestate

echo "=== [5/8] Creating Python 3.9 venv ==="
# Explicit interpreter path (not just --python 3.9) so uv can't resolve to the
# wrong interpreter.
uv venv --python /usr/bin/python3.9

echo "=== [6/8] Installing sn-gamestate + TrackLab ==="
uv pip install -e . --python .venv/bin/python

echo "=== [7/8] Installing mmcv ==="
uv run --python .venv/bin/python mim install mmcv==2.0.1

echo "=== [8/8] Patching sn-gamestate/TrackLab for custom-video inference ==="
# Three fixes needed to run the pipeline on our own broadcast clips rather than
# SoccerNet sequences (see research/patch_sn_gamestate.py for the full rationale):
#   1. calibration keypoint indexing (KeyError: 0 on ExternalVideo metadata)
#   2. radar minimap hardcoded to 1920x1080 (empty addWeighted on 720p video)
#   3. O(n^2) vid:// single-file decode -> folder-of-frames fast path
# Idempotent, so re-running setup.sh (e.g. after a VM restart) is safe.
uv run --python .venv/bin/python "$SCRIPT_DIR/patch_sn_gamestate.py" /content/sn-gamestate

echo ""
echo "=================================================================="
echo " DONE — environment ready (with custom-video patches applied)."
echo " Switch this notebook's kernel to 'Python 3.9 (sn-gamestate)' and"
echo " continue from the 'Run the demo' cell."
echo ""
echo " To run on a custom clip, extract frames first (decode once), then"
echo " point the pipeline at the FOLDER:"
echo "   ffmpeg -i clip.mp4 -vf fps=25 /content/frames/%06d.jpg"
echo "   uv run --python .venv/bin/python tracklab -cn video_demo \\"
echo "     dataset.video_path=/content/frames dataset.nframes=-1 num_cores=0"
echo "=================================================================="
