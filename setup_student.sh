#!/usr/bin/env bash
# Bootstrap the SceneDiff annotator for a SECOND user on this same machine.
#
# Run this AS THE NEW USER (e.g. sangwooyu) in their own shell:
#   bash /local/home/akrishnan/repos/change_annotator/setup_student.sh
#
# It pulls the source dependencies from akrishnan's world-readable clones into
# the new user's own ~/repos, then builds a self-contained ~/annotator_env via
# setup.sh. Nothing here touches akrishnan's account — only reads from it.
#
# Result for the new user:
#   ~/repos/change_annotator   (navvis, with the single-env consolidation)
#   ~/repos/lamaria-indoor     (changes branch; provides scantools via PYTHONPATH)
#   ~/repos/refs/sam3          (SAM 3.1, editable build)
#   ~/repos/geom/raybender     (custom raycaster + its bundled embree)
#   ~/annotator_env            (the one Python 3.10 env that runs everything)
set -euo pipefail

AK="${AK_REPOS:-/local/home/akrishnan/repos}"   # akrishnan's readable repos
DEST="$HOME/repos"

# Idempotent: each step is skipped if its destination already exists, so this is
# safe to re-run after a partial setup.
[ -r "$AK/geom/raybender/embree-3.12.2/lib/cmake/embree-3.12.2/embree-config.cmake" ] \
  || { echo "!! cannot read akrishnan's raybender/embree at $AK — check the path/perms"; exit 1; }

mkdir -p "$DEST/geom" "$DEST/refs"

# git refuses to read another user's repo ("dubious ownership"). Whitelist
# akrishnan's source repos in THIS user's global git config (specific paths, not
# a wildcard). Idempotent.
for src in "$AK/lamaria-indoor" "$AK/refs/sam3"; do
  for p in "$src" "$src/.git"; do
    git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$p" \
      || git config --global --add safe.directory "$p"
  done
done

# 1) raybender — COPY (its embree-3.12.2 is gitignored, so a clone would miss it).
if [ -e "$DEST/geom/raybender" ]; then
  echo "-- raybender exists, skipping"
else
  echo "== copying raybender (+ bundled embree, ~250M) =="
  cp -r "$AK/geom/raybender" "$DEST/geom/raybender"
  # drop akrishnan's stale compiled artifacts so setup.sh rebuilds against MY embree
  rm -rf "$DEST/geom/raybender/build" \
         "$DEST/geom/raybender/raybender/"_raybender*.so \
         "$DEST/geom/raybender/raybender.egg-info"
fi

# 2) sam3 — code only; clone from akrishnan's local repo (fast, hardlinked).
if [ -e "$DEST/refs/sam3" ]; then
  echo "-- sam3 exists, skipping"
else
  echo "== cloning sam3 =="
  git clone "$AK/refs/sam3" "$DEST/refs/sam3"
fi

# 3) lamaria-indoor @ changes — scantools is committed; clone + checkout the branch.
if [ -d "$DEST/lamaria-indoor/.git" ]; then
  echo "-- lamaria-indoor exists, skipping (ensure it is on the 'changes' branch)"
else
  echo "== cloning lamaria-indoor @ changes =="
  rm -rf "$DEST/lamaria-indoor"          # clear any partial/aborted clone
  git clone "$AK/lamaria-indoor" "$DEST/lamaria-indoor"
  git -C "$DEST/lamaria-indoor" checkout changes
fi

# 4) change_annotator @ navvis — the consolidation work is UNCOMMITTED in akrishnan's
#    tree, so copy the working tree (not a clone) to get setup.sh + the edited files.
if [ -e "$DEST/change_annotator" ]; then
  echo "-- change_annotator exists, skipping"
else
  echo "== copying change_annotator (navvis working tree) =="
  cp -r "$AK/change_annotator" "$DEST/change_annotator"
  rm -rf "$DEST/change_annotator/__pycache__"
fi

# 5) Ensure a private Python 3.10 exists for THIS user (the validated version).
#    akrishnan's python3.10 lives in a private ~/.local/share and is NOT reusable,
#    and the only system python is 3.12 — so install our own 3.10 via uv (no sudo).
PY310_BIN="$(command -v python3.10 2>/dev/null || true)"
# ignore a python3.10 that resolves into another user's (unreadable) home
case "$PY310_BIN" in /local/home/akrishnan/*) PY310_BIN="";; esac
if [ -z "$PY310_BIN" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv (user-local, no sudo) =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  echo "== installing Python 3.10 via uv =="
  uv python install 3.10
  PY310_BIN="$(uv python find 3.10)"
fi
echo "using Python 3.10: $PY310_BIN"

# 6) Build the single env. setup.sh defaults already point at the paths above:
#    ENV_DIR=~/annotator_env, SAM3_SRC=~/repos/refs/sam3,
#    RAYBENDER_SRC=~/repos/geom/raybender, LAMARIA_INDOOR=~/repos/lamaria-indoor
echo "== building ~/annotator_env (torch download + raybender compile; takes a while) =="
cd "$DEST/change_annotator"
PY310="$PY310_BIN" bash setup.sh

cat <<EOF

== bootstrap complete ==
Single env: ~/annotator_env   (validated by setup.sh)

Last step — SAM 3.1 weights (gated 'facebook/sam3.1'). Pick ONE:
  (a) authenticate and let it download (~3.3GB on first gui.py run):
        ~/annotator_env/bin/hf auth login      # or: huggingface-cli login
  (b) reuse akrishnan's already-downloaded weights (no auth, no re-download) by
      copying the snapshot into your own cache:
        mkdir -p ~/.cache/huggingface/hub
        cp -r /local/home/akrishnan/.cache/huggingface/hub/models--facebook--sam3* \\
              ~/.cache/huggingface/hub/

Then run the annotator (scantools goes on PYTHONPATH for the seeds stage):
  PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \\
      --capture /path/to/captures/changes/cnb_e100
EOF
