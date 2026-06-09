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

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (required — SAM 3.1 loads itself on CUDA)
- A HuggingFace account with access to [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)
- A capture directory with LaMAR poses + per-state mesh, and a separate Python environment carrying the `scantools` / LaMAR stack (used for the geometry-seeding step; see `LAMAR_PY` under [Usage](#usage))

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yuqunw/scenediff_annotator
   cd scenediff_annotator
   ```

2. **Create conda environment and install dependencies**:
   ```bash
   conda create -n scenediff_annotator python=3.10 -y
   conda activate scenediff_annotator
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

3. **Install SAM 3.1** from a local checkout (assumes the repo is cloned at `~/repos/refs/sam3`):
   ```bash
   pip install -e ~/repos/refs/sam3
   ```

4. **Authenticate with HuggingFace** and accept the model access request for `facebook/sam3.1`:
   ```bash
   hf auth login   # or: huggingface-cli login
   ```
   The SAM 3.1 checkpoint is downloaded automatically on first run of `gui.py`.

For detailed SAM 3.1 installation instructions, refer to the [official SAM 3 repository](https://github.com/facebookresearch/sam3).

## Usage

### Starting the Application

The GUI runs against a single capture directory holding a `pre` and a `post` session of the same scene (by default the Aria RGB sessions `aria_a_rgb` / `aria_b_rgb`). Launch it in the SAM 3.1 environment:

```bash
~/sam3_env/bin/python gui.py --capture /path/to/captures/changes/cnb_e100
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

## Output Format

All outputs are written into the capture directory under `changes/`:

- **`changes/geom_sam_out/<id>__<state>/`** — per-object working data: the source mask (`src_mask.png`), geometry seeds (`seeds.json`), per-frame masks (`masks/`), and an index (`masks_index.json`).
- **`changes/gui_objects.json`** — the object sources you added; reloaded on the next launch so you can resume.
- **`changes/segments.json`** — the final export.

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
        "pre": {"images/cam0/<frame>.jpg": "masks/images_cam0_<frame>.jpg"}
      }
    }
  }
}
```

`change_type` is derived from presence: `pre` only → `removed`, `post` only → `added`, both states → `moved`. Only the states an object appears in show up under `masks`. Each mask path is relative to that object's `changes/geom_sam_out/<id>__<state>/` directory and points to a binary PNG (white = object).

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
