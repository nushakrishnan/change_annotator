# SceneDiff Annotator

A video annotation tool for logging down changes between paired video sequences. 

🔗 Check out the [project page](https://yuqunw.github.io/SceneDiff) for more details.

## Overview

The SceneDiff Annotator is built on top of [SAM 3.1](https://github.com/facebookresearch/sam3) and provides a geometry-assisted workflow for annotating changes between a paired "pre" and "post" capture of the same scene:

- **Click to Segment**: Click point prompts on a source frame and get a live SAM 3.1 mask preview; add objects with a label and deformability.
- **Geometry-Assisted Propagation**: LaMAR poses and per-state mesh depth seed each object across every frame of the camera walk, then SAM 3.1 produces a per-frame mask.
- **Video Refinement (optional)**: Sharpen the per-frame masks with the SAM 3.1 video tracker over each visible span.
- **Review & Refine**: Scrub the full walk to catch missed frames, then re-click, brush, erase, or delete masks before exporting.

### Demo
https://github.com/user-attachments/assets/1779894f-f843-4e9b-a651-3fcb0ae43166

*Watch the video above to see the annotation tool in action.*

## Installation

The annotator runs inside a **single** Python 3.10 environment that carries the
whole pipeline — SAM 3.1, the Flask GUI, **and** the LaMAR geometry stack
(`scantools` / `raybender` / `pycolmap` / `open3d`) used by the propagation
(`seeds`) step. `setup.sh` builds it from scratch; by default it lives at
`~/annotator_env`. (Earlier versions split this across a `sam3_env` on 3.12 and a
separate `lamar_env` on 3.10 — that is no longer needed.)

The only codebase dependency kept *outside* the env is lamaria-indoor's
`scantools`, which is pure-python and used via `PYTHONPATH` (no install). Clone
it once and point `LAMARIA_INDOOR` at it.

### Prerequisites
- **Python 3.10** (the common denominator: SAM 3.1 supports 3.8–3.12, but the geometry stack is built for 3.10 here)
- A CUDA-capable GPU with CUDA 12.8 (required — SAM 3.1 loads itself on CUDA)
- A HuggingFace account with access to [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)
- Local checkouts of the source-built dependencies (the install script expects these paths, override via env vars):
  - SAM 3.1 — `~/repos/refs/sam3` (`SAM3_SRC`)
  - `raybender` (custom C++ raycaster, bundles embree) — `~/repos/geom/raybender` (`RAYBENDER_SRC`)
  - `lamaria-indoor` (provides `scantools`) — `~/repos/lamaria-indoor` (`LAMARIA_INDOOR`)
  - `colmap` source — `~/repos/refs/colmap` (`COLMAP_SRC`) — only needed if the prebuilt `pycolmap` wheel is unavailable

### Setup Instructions

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yuqunw/scenediff_annotator
   cd scenediff_annotator
   ```

2. **Clone the source dependencies** (skip any you already have):
   ```bash
   git clone git@github.com:facebookresearch/sam3.git ~/repos/refs/sam3
   git clone <lamaria-indoor-url>                      ~/repos/lamaria-indoor
   git clone <raybender-url>                           ~/repos/geom/raybender
   ```

3. **Build the single environment**:
   ```bash
   bash setup.sh
   ```
   This creates `~/annotator_env` (Python 3.10), installs torch 2.10/cu128,
   SAM 3.1 (editable, with the `notebooks` extras), the geometry stack
   (`open3d`, `pycolmap`, `plyfile`, `rawpy`, `scipy`, `raybender`), and Flask,
   then validates that every stage's imports resolve in the one interpreter.
   Override locations with env vars, e.g. `ENV_DIR=~/foo SAM3_SRC=... bash setup.sh`.

4. **Authenticate with HuggingFace** and accept the model access request for `facebook/sam3.1`:
   ```bash
   hf auth login   # or: huggingface-cli login
   ```
   The SAM 3.1 checkpoint is downloaded automatically on first run of `gui.py`.

5. **Verify the install** (also done automatically at the end of `setup.sh`):
   ```bash
   PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python -c \
     "import torch, flask, scantools.proc.rendering; from sam3 import build_sam3_image_model; print('cuda:', torch.cuda.is_available())"
   ```
   This should print `cuda: True` with no import errors.

For detailed SAM 3.1 installation instructions (including optional Flash
Attention 3 support), refer to the [official SAM 3 repository](https://github.com/facebookresearch/sam3).

## Usage

### Starting the Application

The GUI runs against a single capture directory holding a `pre` and a `post` session of the same scene (by default the Aria RGB sessions `aria_a_rgb` / `aria_b_rgb`). Launch it with the one env, putting `scantools` (lamaria-indoor) on `PYTHONPATH` so the `Propagate` step can run:

```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
    --capture /path/to/captures/changes/cnb_e100
```

Then open `http://127.0.0.1:5000` in your browser. Useful flags:

- `--pre-session` / `--post-session` — session folder names (default `aria_a_rgb` / `aria_b_rgb`)
- `--pre-ref` / `--post-ref` — reference reconstruction per state (default `navvis_a` / `navvis_b`)
- `--n` — frames to seed per object (`0` = every frame, the default; `N>0` evenly subsamples N for a quick coarse pass)
- `--host` / `--port` — bind address (default `127.0.0.1:5000`)

The geometry-seeding step shells out to a separate environment that has the LaMAR / `scantools` stack; point to it with the `LAMAR_PY` and `LAMAR_PYTHONPATH` environment variables (default `~/lamar_env/bin/python` and `~/repos/lamaria-indoor`).

### Annotation Workflow

1. **Segment an object**: Choose the `pre` or `post` state, scrub to a frame, and click positive (and optional negative) points. SAM 3.1 returns a live mask preview; brush or erase to clean it up.

2. **Add the object**: Give it an id, label, and deformability (`rigid` / `deformable`). The same id used in both `pre` and `post` marks a *moved* object; an id in only one state is *added* or *removed* — the change type is derived from presence, not set by hand.

3. **Propagate**: Click `Propagate`. Geometry seeds (LaMAR poses + per-state mesh depth) carry the object across every frame of that state's camera walk, and SAM 3.1 produces a per-frame mask. A contact-sheet preview is generated for review.

4. **Refine (optional)**: Run `Refine` to sharpen the per-frame masks with the SAM 3.1 video tracker over each visible span.

5. **Review & edit**: Scrub the whole walk to spot frames the propagation missed. Re-click, brush/erase, or delete the mask on any frame.

6. **Export**: Click `Export` to write `changes/segments.json` (see [Output Format](#output-format)).

## End-to-End: a Fresh Sequence to Symmetric GT

The recipe for a capture that has **no masks at all yet** — neither per-sequence
object masks nor ghost masks on the inference frames. Only step 2 involves a
human; everything after is generated from those masks plus the scan geometry.

### 0. What the capture must already contain

```
<capture>/                                  e.g. /media/lamaria_indoor/captures/changes/<scene>
├── sessions/
│   ├── aria_a_rgb/                         "pre" walk
│   │   ├── raw_data/images/cam0/*.jpg      the frames
│   │   └── proc/navvis_a/colmap_model_aligned/   aligned poses
│   ├── aria_b_rgb/                         "post" walk (same layout, ref navvis_b)
│   ├── navvis_a/
│   │   ├── raw_data/pointcloud.ply         RAW lidar scan (keeps the changed objects!)
│   │   └── proc/.../mesh                   NavVis mesh (occlusion testing)
│   └── navvis_b/                           same layout
└── changes/navvis_b_to_navvis_a/T_navvis_a_from_navvis_b.txt   rigid NavVis↔NavVis bridge
```

Different session/ref names? Every command below accepts
`--pre-session/--pre-ref/--post-session/--post-ref` (defaults shown above).

### 1. Environments

| env | used by | contents |
|---|---|---|
| `~/annotator_env` | `gui.py` (steps 2 and 5) | SAM 3.1 + Flask + geometry stack (`setup.sh`) |
| `~/lamar_env` | `point_ghost_prototype.py`, `change_mask.py --symmetric` | scantools/raybender/open3d, no GPU needed |

Both need lamaria-indoor's `scantools` on `PYTHONPATH` (shown inline below).

### 2. Annotate the per-sequence masks (manual, the only annotation step)

```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
    --capture /media/lamaria_indoor/captures/changes/<scene>
```

For every changed object, **in each state where it is physically visible**
(a moved chair gets `chair` in *pre* AND `chair` in *post*; an added bottle only
in *post*): click → add object → `+ seed` from 1-3 spread-out views →
`Propagate` → review/edit (brush, erase, `d` to drop a bad frame) → next object.
Finish with `Export`. This writes the verified per-object masks
(`changes/geom_sam_out/<id>__<state>/masks_index.json`), `segments.json`, and
the native `change_mask/` GT. Do NOT annotate where an object is absent — the
ghost there is generated in step 3.

### 3. Generate the symmetric GT (automatic ghosts on the inference frames)

```bash
CAP=/media/lamaria_indoor/captures/changes/<scene>
# per-object 3D meshes from the RAW lidar cloud + the step-2 masks (~minutes/object)
PYTHONPATH=~/repos/lamaria-indoor:. ~/lamar_env/bin/python point_ghost_prototype.py objects   --capture $CAP
# render each mesh into the OTHER state's frames, occlusion-tested -> the symmetric GT
PYTHONPATH=~/repos/lamaria-indoor:. ~/lamar_env/bin/python point_ghost_prototype.py symmetric --capture $CAP --viz
```

Output: `changes/change_mask_symmetric_points/<state>/*.png` (+ `_index.json`,
`<state>_viz/` panels with native = green, ghost = red). Sanity-check the
`objects` stage per object in `changes/point_ghost/<id>__<state>/`
(`mesh.ply`, `points.ply`, `stats.json`) — a degenerate object (too few lidar
points) is logged and simply has no ghost.

### 4. (Optional) benchmark against the voted-faces baseline

```bash
PYTHONPATH=~/repos/lamaria-indoor ~/lamar_env/bin/python change_mask.py --capture $CAP --symmetric --viz
PYTHONPATH=~/repos/lamaria-indoor:. ~/lamar_env/bin/python point_ghost_prototype.py compare --capture $CAP
```

Writes `[RGB | old | new]` panels + a source-frame IoU table under
`changes/point_ghost_compare/`.

### 5. (Optional) hand-correct the generated ghosts in the GUI

```bash
# each object's ghost as a GUI-editable 👻 pseudo-object (<id>_ghost__<state>)
PYTHONPATH=~/repos/lamaria-indoor:. ~/lamar_env/bin/python point_ghost_prototype.py ghosts --capture $CAP
# restart gui.py (step 2 command) -> 👻 objects appear in the sidebar:
#   review/edit -> brush/erase the ghost region (SAM clicks can't help: nothing
#   visible to segment where an object *used to be*), `d` drops a frame
# then rebuild the symmetric GT from the corrected masks:
PYTHONPATH=~/repos/lamaria-indoor:. ~/lamar_env/bin/python point_ghost_prototype.py merge --capture $CAP --viz
```

Ghost pseudo-objects are derived data: they never enter `segments.json` or the
native `change_mask/`, and `propagate`/`+ seed` are disabled for them. Re-running
`ghosts` regenerates them (overwriting hand edits — correct AFTER the geometry
is final).

## Output Format

All outputs are written into the capture directory under `changes/`:

- **`changes/geom_sam_out/<id>__<state>/`** — per-object working data: the source mask (`src_mask.png`), geometry seeds (`seeds.json`), per-frame masks (`masks/`), and an index (`masks_index.json`).
- **`changes/gui_objects.json`** — the object sources you added; reloaded on the next launch so you can resume.
- **`changes/segments.json`** — the final export.
- **`changes/change_mask/<state>/<frame>.png`** — per-frame binary change mask (255 = changed), the **union of all object masks** on that frame. This is the per-pixel ground truth a method is scored against (`annotation_spec.md` §6/§7), the metric MV3DCD / SceneDiff report. Produced automatically on `Export`, indexed in `changes/change_mask_index.json`, and referenced from `segments.json` under a top-level `change_mask: {state: {frame: relpath}}` block.

- **`changes/change_mask_symmetric/<state>/<frame>.png`** — the **MV3DCD-suited symmetric** change mask (`--symmetric`). Each frame carries changes from **both** directions: the native object masks **plus** the cross-projected footprint of the other state's changes (where an object *was* / *will be*). This matches what a multi-view 3D detector outputs — every changed region, both directions, in every image. The cross-projection is mesh-anchored and multi-view-consistent: each object's verified masks are **voted onto the NavVis mesh faces**, carried across the rigid NavVis→NavVis bridge, and the labeled sub-mesh is **rendered** into the other sequence with the target mesh providing occlusion (no per-frame 2D warp, no SAM). Indexed in `change_mask_symmetric_index.json`. Native per-sequence masks stay the authoritative human-verified GT; this is a derived view.

To (re)generate change masks for a capture without re-running the GUI:

```bash
# native (pure cv2, any env)
~/lamar_env/bin/python change_mask.py --capture /path/to/captures/changes/cnb_e100

# + symmetric MV3DCD GT (needs lamar_env: scantools/raybender/open3d)
PYTHONPATH=~/repos/lamaria-indoor ~/lamar_env/bin/python change_mask.py \
    --capture /path/to/captures/changes/cnb_e100 --symmetric --render-scale 1.0
```

Useful `--symmetric` knobs: `--ratio` (min inside/seen vote per face, raise to reject stray faces), `--min-views` (min source views to keep a face), `--vote-scale` / `--render-scale` (speed vs. crispness), `--occ-tol` (target-mesh occlusion tolerance, metres).

The NavVis meshing step drops most changed objects, so the voted-face footprint can come out holey or empty. Two prototypes rebuild the missing geometry: `lidar_tsdf_prototype.py` (DA3 mono depth anchored to the raw lidar, TSDF-fused per state, output in `change_mask_symmetric_tsdf/`) and `point_ghost_prototype.py` (no mono depth: per object, the verified masks select the object's own points from the **raw lidar cloud** — z-buffered multi-view point voting — which are alpha-shape meshed, mask-exterior trimmed, and rendered as the ghost; output in `change_mask_symmetric_points/`). `point_ghost_prototype.py compare` writes `[RGB | old | new]` panels and a source-frame IoU table under `changes/point_ghost_compare/` to judge the variants against each other.

The generated ghosts can be **hand-corrected in the GUI**: `point_ghost_prototype.py ghosts` renders each object's ghost separately into the other state's frames and registers it as a 👻 pseudo-object (`<id>_ghost__<state>` in `geom_sam_out/`, flagged `ghost` in `gui_objects.json`). Restart `gui.py` and the ghosts appear in the sidebar — review/edit them with the brush/eraser (SAM clicks won't help: there is nothing visible to segment where an object *used to be*), delete bad frames, then run `point_ghost_prototype.py merge --viz` to rebuild `change_mask_symmetric_points/` from the corrected masks. Ghost pseudo-objects never enter `segments.json` or the native `change_mask/` export.

### `segments.json` Structure

```json
{
  "scene": "cnb_e100",
  "tier": "instance",
  "camera": "cam0",
  "pre":  {"session": "aria_a_rgb", "ref": "navvis_a"},
  "post": {"session": "aria_b_rgb", "ref": "navvis_b"},
  "objects": {
    "<object_id>": {
      "label": "chair",
      "deformability": "rigid",
      "in_pre": true,
      "in_post": false,
      "change_type": "removed",
      "masks": {
        "pre": {"images/cam0/<frame>.jpg": "geom_sam_out/<id>__pre/masks/images_cam0_<frame>.jpg"}
      }
    }
  }
}
```

`change_type` is derived from presence: `pre` only → `removed`, `post` only → `added`, both states → `moved`. Only the states an object appears in show up under `masks`. Each mask path is relative to the capture's `changes/` directory (i.e. resolve as `<capture>/changes/<path>`) and points to a binary PNG (white = object).

### Loading Masks

The masks are plain PNGs, so decode with any image library:

```python
import cv2
mask = cv2.imread("changes/geom_sam_out/chair__pre/masks/images_cam0_0001.jpg", 0) > 127
```

<!-- ## Citation

If you use this annotation tool in your research, please cite the SceneDiff project:

```bibtex
@misc{scenediff2024,
  title={SceneDiff: Scene Change Detection and Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://yuqunw.github.io/SceneDiff}}
}
``` -->

## Acknowledgements

This project is built upon the excellent [SAM 3 repository](https://github.com/facebookresearch/sam3) (Segment Anything Model 3). We gratefully acknowledge their contributions to the computer vision community.

## License

See [LICENSE](LICENSE) for more information.
