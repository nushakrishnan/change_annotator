#!/usr/bin/env bash
# One-env setup for the SceneDiff change annotator.
#
# Builds a SINGLE Python 3.10 virtualenv that runs every stage of the pipeline
# (SAM3 click/segment + the lamar geometry "seeds" stage). Replaces the old
# two-env setup (~/sam3_env on 3.12 + ~/lamar_env on 3.10).
#
# Why 3.10: SAM3 supports 3.8-3.12, but the lamar geometry stack
# (scantools/raybender/pycolmap/open3d) is built for 3.10 here, so 3.10 is the
# common denominator. numpy is pinned <2 (1.26.x), which both stacks accept.
#
# The only codebase dependency kept external is lamaria-indoor's `scantools`,
# which is pure-python and used via PYTHONPATH (no install) — clone it and point
# LAMARIA_INDOOR at it (see the env block this script prints at the end).
#
# Usage:
#   bash setup.sh                 # create ~/annotator_env and install everything
#   ENV_DIR=~/foo bash setup.sh   # custom env location
#
# Reproduces the exact stack validated on the original account:
#   torch 2.10.0+cu128 / torchvision 0.25.0+cu128 (CUDA 12.8)
#   sam3 0.1.0 (editable), open3d 0.19.0, pycolmap 4.1.0, raybender 0.0.1
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/annotator_env}"
PY310="${PY310:-python3.10}"

# Source repos this setup needs present (clone these on a new account first):
SAM3_SRC="${SAM3_SRC:-$HOME/repos/refs/sam3}"          # Meta SAM 3.1 (editable)
RAYBENDER_SRC="${RAYBENDER_SRC:-$HOME/repos/geom/raybender}"  # custom C++ raycaster (bundles embree)
COLMAP_SRC="${COLMAP_SRC:-$HOME/repos/refs/colmap}"    # only used if PyPI pycolmap wheel is unavailable
LAMARIA_INDOOR="${LAMARIA_INDOOR:-$HOME/repos/lamaria-indoor}"  # provides scantools (PYTHONPATH only)

echo "== creating venv at $ENV_DIR (python: $PY310) =="
"$PY310" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"
PY="$ENV_DIR/bin/python"
# setuptools must stay <81: SAM3 imports the (removed-in-81) `pkg_resources`.
"$PIP" install --upgrade pip wheel "setuptools<81"

# 1) Torch FIRST so SAM3 doesn't drag in an unpinned/CPU build. CUDA 12.8 wheels.
echo "== torch / torchvision (cu128) =="
"$PIP" install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) SAM3 (editable) — brings opencv, einops, decord, scikit-image, torchmetrics, etc.
echo "== SAM3 (editable) from $SAM3_SRC =="
"$PIP" install -e "${SAM3_SRC}[notebooks]"

# 3) Geometry stack used by the seeds stage (scantools' runtime deps).
echo "== geometry deps (open3d / pycolmap / plyfile / rawpy / scipy) =="
"$PIP" install open3d==0.19.0 plyfile==1.1.3 rawpy scipy==1.15.3
# pycolmap: try the prebuilt PyPI wheel (fast). Falls back to building the local
# COLMAP source (slow, needs the full C++ toolchain) only if no wheel is found.
if ! "$PIP" install pycolmap==4.1.0; then
  echo "!! PyPI pycolmap unavailable — building from source at $COLMAP_SRC (this is slow)"
  "$PIP" install -e "$COLMAP_SRC"
fi

# 4) raybender — custom C++ raycaster. Its cmake does find_package(embree),
# so point embree_DIR at the prebuilt embree bundled in the raybender repo.
echo "== raybender (editable, compiles) from $RAYBENDER_SRC =="
EMBREE_CMAKE="$RAYBENDER_SRC/embree-3.12.2/lib/cmake/embree-3.12.2"
if [ ! -f "$EMBREE_CMAKE/embree-config.cmake" ]; then
  echo "!! embree not found at $EMBREE_CMAKE — download the v3.12.2 linux release into"
  echo "   $RAYBENDER_SRC/embree-3.12.2 (see that repo's README), then re-run."
  exit 1
fi
embree_DIR="$EMBREE_CMAKE" "$PIP" install -e "$RAYBENDER_SRC"

# 5) GUI server.
echo "== flask =="
"$PIP" install flask

# 6) Validate: every stage's imports resolve in ONE interpreter.
echo "== validating single-env imports =="
PYTHONPATH="$LAMARIA_INDOOR" "$PY" - <<'PYEOF'
import importlib, sys
mods = ["torch", "torchvision", "sam3", "cv2", "numpy",
        "open3d", "pycolmap", "raybender",
        "scantools.capture", "scantools.proc.rendering", "scantools.utils.io"]
import numpy
assert numpy.__version__.startswith("1.26"), f"numpy must be <2, got {numpy.__version__}"
for m in mods:
    importlib.import_module(m)
    print(f"  ok  {m}")
import torch
print(f"  cuda available: {torch.cuda.is_available()} (cuda {torch.version.cuda})")
print("ALL IMPORTS OK — single env is ready")
PYEOF

cat <<EOF

== done ==
Single env: $ENV_DIR

Add to your shell profile (or the run commands) so scantools resolves:
  export ANNOTATOR_PY="$ENV_DIR/bin/python"
  export LAMARIA_INDOOR="$LAMARIA_INDOOR"
  export PYTHONPATH="\$LAMARIA_INDOOR\${PYTHONPATH:+:\$PYTHONPATH}"

Then run anything with the one interpreter, e.g.:
  PYTHONPATH="$LAMARIA_INDOOR" "$ENV_DIR/bin/python" gui.py --capture /path/to/capture
  PYTHONPATH="$LAMARIA_INDOOR" "$ENV_DIR/bin/python" annotate.py --config configs/cnb_e100.json
EOF
