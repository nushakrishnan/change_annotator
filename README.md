# SceneDiff Annotator

A tool for building **ground truth about what changed** between two visits to the same
place. You get two lidar scans (NavVis) and two head-mounted camera walks (Aria) of the
same scene, recorded weeks or months apart. The tool helps you produce, for every video
frame, a mask marking the pixels that changed — plus, where it makes sense, which object
each mask belongs to.

🔗 [Project page](https://yuqunw.github.io/SceneDiff)

---

## 1. The idea in plain English

Two things look at the scene, and they are good at opposite jobs:

- **The lidar scans** know *geometry*. Comparing them tells you, in 3D, where the world
  physically differs — regardless of lighting, shadows, or how dark the room was. But
  lidar is blobby: it cannot draw a crisp outline on a photo, and it can't see a poster
  swapped for another poster (same shape, different content).
- **The video frames** know *appearance*. SAM (Segment Anything) draws beautiful pixel
  boundaries — but it has no idea what "changed" means, and it fails where contrast dies
  (a black chair in a shadow).

So the tool always uses them in the same order: **geometry says where to look, the image
draws the outline, and a human vouches for the result.** Everything else in this README is
a consequence of that sentence.

### The four colours

Turn on the scene view (press **`u`**) and every frame is painted with the current state
of knowledge:

| Colour | Meaning | What you do |
|---|---|---|
| 🟩 **Green** | Geometry certifies **no change** here — the two scans match within ~3 cm (and the floor, which is static by definition), plus anything a human dismissed | Nothing. Skip it. |
| 🟥 **Red** | An object mask exists here — someone or something claims this changed | Review it, fix it, or dismiss it |
| 🟪 **Purple** | Geometry says **something changed** and *nobody has claimed it yet* | Resolve it: annotate it, or decide it's noise |
| ⬜ **Uncoloured** | No lidar evidence either way — glass, unscanned surfaces, your own hands | Nothing. This is the "ignore" class. |

Green and purple are **born together**: one measurement (how far each scan point is from
the other scan) split at a threshold. Red is *added on top* — first by the machine
(rough masks from detection), then upgraded by you.

A frame is done when no purple is left unexplained. A scene is done when the purple-blob
list is empty.

### Who vouches for what

Not all red is equal, and the tool tracks this per mask (`src` in `masks_index.json`):

| Tier | Where it came from | Trust |
|---|---|---|
| `hand` | You drew or fixed it | Ground truth. Never overwritten by anything. |
| `prop` | Propagated from your hand masks by the video tracker, or rebuilt from 3D consensus | High — measured ~0.97 IoU against hand masks |
| `concept` | `label → remask` (SAM segmenting by the word you typed) | Good on sharp frames, spills on motion blur |
| `geom` | Auto-generated when an object was detected (lidar box → SAM) | Starting point only. Freely regenerated. |

Plus one line you control: **the green frontier**. Press **`g`** on a frame and everything
up to it is marked "a human has checked this" — no automatic process may ever write there
again, whatever its tier.

---

## 2. Install

Everything runs in **one** Python 3.10 environment (`~/annotator_env`): SAM 3.1, the Flask
GUI, and the LaMAR geometry stack (`scantools` / `raybender` / `pycolmap` / `open3d`).

**You need:** Python 3.10, a CUDA 12.8 GPU, a HuggingFace account with access to
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1), and local checkouts of
SAM 3.1 (`~/repos/refs/sam3`), `raybender` (`~/repos/geom/raybender`), and
`lamaria-indoor` (`~/repos/lamaria-indoor`, provides `scantools` — used via `PYTHONPATH`,
never installed).

```bash
bash setup.sh          # builds ~/annotator_env and validates every import
hf auth login          # accept model access; the checkpoint downloads on first run
```

Verify:

```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python -c \
  "import torch, flask, scantools.proc.rendering; from sam3 import build_sam3_image_model; print('cuda:', torch.cuda.is_available())"
```

---

## 3. Launching

```bash
cd ~/repos/change_annotator-sangwoo
GEOM_OUT=changes/geom_sam_out_1_2_sangwoo PYTHONPATH=~/repos/lamaria-indoor \
  ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/<scene> \
  --pre-session <walk>_rgb  --pre-ref navvis_1 \
  --post-session <walk>_rgb --post-ref navvis_2 \
  [--port 5001]
```

Then open `http://127.0.0.1:5000`.

**`GEOM_OUT` is the workspace** — the folder holding everything you produce. Give each
state-pair (and each annotator) its own: `geom_sam_out_1_2_sangwoo`, `geom_sam_out_2_3_...`.
Workspaces are independent and disposable; the raw capture is never touched.

**Depth mode** — add `--depth` and drop `GEOM_OUT`; the workspace is derived from the
sessions automatically. See §5.

**Naming varies by scene.** Check what exists first:

```bash
ls /media/lamaria_indoor/captures/changes/<scene>/sessions/
```

Some scenes number the scans differently from the walks (climate_day's states 1 and 2 use
`navvis_2` and `navvis_3`), and some have several walk repetitions per state
(`climate_day_1_1_rgb`, `_1_2_rgb`, …) — each walk pair deserves its own workspace.

If port 5000 is taken by a stale server: `ss -tlnp | grep 5000`, then `kill <pid>`.

---

## 4. Two ways to work

### A. Object-first — for ordinary room-scale scenes

Best when a handful of discrete things changed (a room, an office, a lab).

1. **🔍 detect changes** — diffs the two scans, keeps what the walk actually looked at, and
   drops each survivor in as an object with rough masks.
2. **Triage** — for each proposal: real change → keep; junk (glass noise, a wall sliver) →
   **dismiss ⇢ green**.
3. **Annotate** — open an object, fix its mask on a good frame (`x` to edit, tools below,
   `z` to save), then **propagate this fix** to carry it across the walk. Press **`g`** as
   you verify to advance the frontier.
4. **Link moved objects** — give a `pre` object and a `post` object the *same id* and they
   become one physical object that moved. Presence decides the change type: pre only =
   removed, post only = added, both = moved. You never set it by hand.
5. **Export** — writes `segments.json` and the per-frame change masks.

### B. Scene-first — for mass-change scenes (trade fairs, event halls)

When *most* of the scene was reconfigured, per-object triage stops being an aid and becomes
a burden. Here the object is just a tool for making masks — the deliverable is the pixels.

1. **🔍 detect changes** at a coarse preset (see §8).
2. **Sieve** — dismiss the proposals you don't care about; press **🟣 find unresolved
   blobs** and resolve each leftover (promote to an object / merge into an existing one /
   dismiss to green).
3. **🧱 rebuild all from 3D** — every object's masks are re-derived from multi-view
   consensus and re-rendered consistently across all frames (see §7).
4. **Spot-fix** what consensus can't: draw the correction on one frame and press
   **propagate in 3D** — it applies to every frame at once.
5. Work with the scene view on (`u`), driving purple to zero.

**Multi-SAM** is useful here: click ten separate things, they all land in one mask, save as
one "blob" object. Identity is optional when nobody can meaningfully name 200 shuffled chairs.

---

## 5. Depth mode

A second product, kept strictly apart from the change ground truth: **masks of where the
NavVis-derived depth cannot be trusted** in an Aria frame. The lidar goes straight through
glass, so at a window the mesh "surface" is the building across the courtyard — a depth
value that is confidently wrong. Unobserved regions and mirrors are the same story. A depth
method trained or scored against NavVis depth must ignore those pixels; depth mode is how
you mark them.

The object machinery is only a tool here: the deliverable is one per-frame mask, and an
"object" is just a convenient bag of blobs (e.g. all the windows on a wall).

### Launch

Your change command, minus `GEOM_OUT`, plus `--depth`:

```bash
cd ~/repos/change_annotator-sangwoo
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/<scene> \
  --pre-session <walk>_rgb  --pre-ref navvis_<a> \
  --post-session <walk>_rgb --post-ref navvis_<b> \
  --depth
```

The workspace is created for you at `<capture>/depth/<pre-session>__<post-session>/`
— one per walk pair, nothing to name. `GEOM_OUT` is ignored in depth mode (a note is printed),
so a stale variable in a saved command can never redirect writes into a change workspace.
A purple **DEPTH MODE** badge at the top of the page confirms which mode you are in.

### Workflow

1. **Draw** — on a frame where the glass is visible, pick **multi-SAM** and click each
   window until it is solidly red. Separate clicks stay separate blobs; that is what you
   want. **add object**, give it any id (`windows`).
2. **Seed a few more frames** — open the object, scrub to a different viewpoint, `x` to edit,
   multi-SAM the windows there, `z` to save. Two to five frames spread along the walk, and
   between them they should see *every* window you care about (a lower row only visible
   from one spot needs a seed from that spot).
3. **Propagate** — the object card's **propagate** tracks every blob forward through the
   walk. Each blob gets its own tracker memory, so a small window cannot be swallowed by a
   big one, and blobs are run in batches to stay inside GPU memory. Scrub the preview
   (red = proposed, green = existing), `v` to veto a frame, **apply**.
4. **Verify and lock** — walk forward; where the masks are right, press **`g`** so nothing
   automatic can touch them again.
5. **Fix** — on a frame where a blob drifted, `x`, correct it, `z`, then **propagate this
   fix**. This is *surgical*: it compares your saved mask with the one you replaced, re-tracks
   only the regions you actually changed, and leaves every other blob's masks byte-for-byte
   as they were. Fixing two windows out of twenty costs two tracks, not twenty.
6. **Export depth masks** — writes the per-frame union of all depth objects to
   `depth/masks/<state>/<frame>.jpg` (white = depth unreliable) plus
   `depth/depth_index.json`. `segments.json` and `change_mask/` are never written in
   depth mode.

Panels that only make sense for change annotation — detect changes, change fields, purple
blobs, rebuild-from-3D, propagate in 3D — are hidden in depth mode. Glass has no lidar
points to lift onto, so the 3D tools have nothing to work with here.

### What to expect, honestly

- **Runtime scales with blob count.** Twenty windows seeded from five frames is on the order
  of a hundred tracks; a full-walk propagate on a long walk can take ten-plus minutes. The
  card `propagate` therefore runs **forward only**; tick **whole span** in edit mode when
  you also want frames before the seed filled. The surgical fix is fast because it tracks
  only what you changed.
- **Pending previews live in server memory.** A propagation you have not applied is lost
  if the server stops (or the machine reboots). Apply before you walk away.
- **Each walk is annotated on its own.** We measured every way of carrying window masks
  from one walk to another automatically — mesh holes, wall-plane projection, RGB
  feature matching — and none is reliable on curved glass walls the lidar pierces. Windows
  sit in fixed places, so seeding the second walk is a few clicks; there is no
  "port and it's done".
- **The mask format is provisional** — per-frame binary "depth unreliable" masks today;
  classes (glass / unobserved / mirror) are a possible later refinement.

### Where things live

| Path | What |
|---|---|
| `depth/<pre>__<post>/<id>__<state>/` | the working object (seeds, masks, index) — same layout as a change object |
| `depth/masks/<state>/*.jpg` | **the deliverable**: per-frame depth-unreliable mask |
| `depth/depth_index.json` | which frames have a mask |

Depth mode and change mode never write to each other's folders — `<capture>/depth/` and
`<capture>/changes/` are siblings; both read the raw capture read-only.

---

## 6. The tools

Everything below is display-and-canvas work — nothing is written to disk until you press
save (`z`), or apply a preview.

### Drawing (edit mode: press `x`, or `z` from browsing)

| Tool | What it does |
|---|---|
| **point (SAM)** | Click = positive, shift-click = negative. Live mask preview. |
| **multi-SAM** | Every click segments *independently* and adds to the mask — stamp many separate things into one object without them bleeding together. Shift-click removes one. |
| **brush** | Paint freehand. When a stroke closes off an area, the enclosed region fills automatically. |
| **erase** | Paint to remove. |
| **smart brush** | Rough scribble to include (shift = exclude); SAM refines to the object edge on release. |
| **line** / **line fill** | Click start, click end — a straight stroke at brush width, chained. `line fill` also fills a loop when you close one. Good for thin legs and cables. |
| **erase line** / **erase fill** | Same, but removing. `erase fill` deletes an island the moment your cut separates it — carve off a wrongly-included neighbour and it vanishes. |
| **geom brush** | Paint over a wrong region and only there the mask is replaced by the lidar geometry's answer. |
| **propagate in 3D** | Lifts your current mask onto the 3D point cloud → applies to *every* frame and walk (§7). |

### Keys

| Key | Action |
|---|---|
| `←` `→` | Scrub frames |
| `x` | Enter edit mode |
| `z` | Enter edit / save mask |
| `d` | Delete this frame's mask |
| `g` | **Good up to here** — advance the verified frontier |
| `q` (hold) | Peek the clean original frame (all overlays off) |
| `u` | Cycle the scene view (off → red → +green → +purple) |
| `i` | Cycle display filters (normal → shadow lift → invert) — helps in dark scenes |
| `e` | Toggle mesh depth edges — geometry outlines that survive any lighting |
| `v` | Veto the current frame in a preview |
| `m` | Confirm an armed action |
| `ctrl+z` / `ctrl+y` | Undo (both keys, for QWERTZ) |
| `Esc` / `Enter` | Finish a line, or cancel an armed action |

---

## 7. How masks spread across frames

Four mechanisms, from most human to most automatic.

**Propagate this fix** — you fixed a frame; this carries it forward with the SAM video
tracker. It fills gaps *between* your hand masks, continues into frames that never had a
mask, tolerates a few blurry frames before giving up, bridges stretches where the object
is off-screen, and re-acquires the object at a later visit when the geometry confirms it.
It never writes behind your frontier or over a hand mask. Everything arrives as a preview
you scrub and veto before applying.

**Propagate (re-seed)** — for detected objects: unions the object's lidar cluster with all
your hand masks into completed 3D geometry, re-projects it, and regenerates masks. Only
`geom`-tier and empty frames are touched.

**Rebuild from 3D** — the consensus mechanism. Every existing mask *votes* for the 3D
points it covers (hand masks count triple). A point is kept when ≥60% of the frames that
could see it also masked it. Then the surviving points are rendered back into every frame.
This fixes three problems at once:

- **jitter** — masks stop wobbling frame to frame, because they all render one 3D thing;
- **misses** — a part most frames dropped survives on the votes of the few that caught it,
  and reappears everywhere;
- **inconsistent overflow** — a spill that only one viewpoint made falls below the vote
  ratio and disappears from every frame.

It cannot fix *systematic* overflow (a neighbour swallowed from every angle) — that's the
next tool's job.

**Propagate in 3D** — edit once, apply everywhere. Draw the correction on the frame in
front of you and press the button: your stroke is lifted onto the point cloud (the points
it images, line-of-sight tested), grown a little in 3D to cover parts this view can't see,
shown to you in orange, and committed with **`m`**. From then on it renders into every
frame of that state. This is how you fix a poster change the geometry can't see, or carve
off a neighbour the consensus keeps. Human 3D labels are stored separately and **a rebuild
never erases them**.

---

## 8. Detection settings

The **🔍 detect changes (cloud diff + texture diff)** button runs three things: the
geometric scan diff (proposals named `cd_*`), the change fields that power the scene view,
and a DINOv3 appearance check on geometrically-static surfaces (proposals named `tex_*`)
for content changes like swapped posters.

| Setting | What it means | Room scene | Mass scene |
|---|---|---|---|
| `voxel` | Cloud downsampling (m) — smaller sees smaller things, slower | 0.015 | 0.02 |
| `tau` | How far apart two scans must be to call it a change (m) | 0.10 | 0.10 |
| `tau-lo` | The "same" threshold — below this, geometry certifies *unchanged* (green). Empty = tau/3 ≈ 3.3 cm | empty | empty |
| `min-cluster` | Smallest proposal, in points | 150 | 300 |
| `eps` | Cluster linkage (m) — lower splits touching objects apart | 0.10 | 0.08 |
| `min-frames` | How many frames the walk must have seen it in | 5 | 8 |
| remove floor | Floor is static by definition | ✓ | ✓ |
| strict occlusion | Only log changes the glasses actually saw | ✓ | ✓ |

**A hard-won lesson:** if a change seems "missed", check the size gate before blaming the
diff. A tray on a table was once dropped simply because it produced ~450 points against a
`min-cluster` of 500 — the diff had seen it perfectly.

**Detect once per workspace.** Re-running it on a workspace you've already annotated can
collide with existing object ids. To start over, use **☢ nuke workspace** (below) or launch
with a new `GEOM_OUT`.

---

## 9. Deciding what *didn't* change

Two buttons, easy to confuse, opposite meanings:

- **delete** — "this object entry was wrong." The masks go to `.trash`, and the region
  **returns to purple**: the question is re-opened and the blob finder will ask again.
- **dismiss ⇢ green** — "this change claim was wrong." The masks go to `.trash` *and* the
  object's geometry is recorded as human-certified no-change: the region turns **green**
  and stops being asked about. Requires typing `g` to confirm, because it applies to the
  whole sequence.

When in doubt, **delete** — the system will re-ask.

---

## 10. Your work is safe

- **The raw capture is never modified.** Everything under `sessions/` — the point clouds,
  the meshes, the images — is opened read-only. Every write goes into your `changes/<workspace>/`
  folder. Even the clouds that carry 3D labels are *derived copies* inside the workspace.
- **Snapshots** are taken automatically before anything destructive (detect, re-seed,
  texture check, batch rebuild): hardlink copies under `<workspace>/snapshots/`, last 10 kept.
  Restore with `rsync -a <snapshot>/ <workspace>/`.
- **Deletions are moves**, not deletions: `.trash/` for objects, `masks/.deleted/` and
  `masks/.bak_*/` for masks.
- **☢ nuke workspace** starts a workspace over: everything is renamed to
  `<workspace>.nuked_<timestamp>` next to it — recoverable with a `mv`. Requires typing
  `proceed`.
- **Hand masks and the frontier** are respected by every automatic writer, including
  mid-job: a job that runs for minutes re-checks them right before each write, so a mask
  you save while it runs is never clobbered.

---

## 11. Output

Written under `<capture>/changes/`:

| Path | What |
|---|---|
| `segments.json` | The export: objects, labels, change types, per-frame mask paths |
| `change_mask/<state>/*.png` | Per-frame binary change mask (union of all object masks) — the pixel-level GT |
| `<workspace>/<id>__<state>/masks/` | Per-object per-frame mask PNGs |
| `<workspace>/<id>__<state>/masks_index.json` | Which frames have masks, their tier (`src`) and pixel counts |
| `<workspace>/gui_objects.json` | Object list, labels, done/reviewed flags, frontiers |
| `<workspace>/fields/` | The green/purple point fields, dismissals, and 3D labels |
| `depth/masks/<state>/*.jpg` + `depth/depth_index.json` | Depth mode only: per-frame depth-unreliable masks (see §5) |

`segments.json` in short:

```json
{
  "scene": "climate_day", "tier": "instance", "camera": "cam0",
  "pre":  {"session": "climate_day_1_1_rgb", "ref": "navvis_2"},
  "post": {"session": "climate_day_2_1_rgb", "ref": "navvis_3"},
  "objects": {
    "chair_01": {
      "label": "chair", "deformability": "rigid",
      "in_pre": true, "in_post": false, "change_type": "removed",
      "masks": {"pre": {"images/cam0/123.jpg": "geom_sam_out_.../masks/images_cam0_123.jpg"}}
    }
  }
}
```

Mask paths are relative to `<capture>/changes/`. Masks are plain PNGs — `cv2.imread(path, 0) > 127`.

Note: working state (`done`, `reviewed`, the frontier) deliberately stays out of the
export; it is annotation bookkeeping, not ground truth.

---

## 12. What this tool is good at — and where it struggles

**Designed for:** a room-scale scene where 5–30 discrete, rigid, opaque objects changed
between two well-registered scans, filmed by a walk that revisits them in decent light.

**Degrades gracefully:** many small objects (use the fine preset), long multi-visit walks,
objects that grow hugely as you approach, scenes needing heavy hand annotation.

**Genuinely hard:**

- **Mass reconfiguration** (event halls) — hundreds of proposals; use workflow B.
- **Glass** — lidar barely sees it, so it produces both false proposals and uncoloured gaps.
- **Two changed objects touching** — the diff fuses them into one proposal; separate them
  by hand (erase fill + propagate in 3D).
- **Changes thinner than ~3 cm** — a swapped plate of the same shape is below any safe
  threshold. Appearance changes of that kind need the texture check or a manual 3D label.
- **Deformables** (curtains, bags) — geometry-based helpers stand down; annotate by hand.
- **Boundaries in deep shadow or motion blur** — the `i` and `e` display aids help you see;
  the masks still need care.

---

## 13. File map

| File | Role |
|---|---|
| `gui.py` | The web GUI and every endpoint — the thing you run |
| `geom_sam_prototype.py` | Core geometry↔SAM pipeline: source masks, seeds, per-frame masks |
| `cloud_diff_prototype.py` | Scan-to-scan change detection, hysteresis, the change fields |
| `appearance_check.py` | DINOv3 texture/appearance change detection on static surfaces |
| `point_labels.py` | The 3D point-label store (auto vs human tiers, region growing) |
| `propagate_fix.py` | The video-tracker propagation between human anchors |
| `change_mask.py` | Per-frame change-mask GT generation |
| `point_ghost_prototype.py` | Symmetric GT: renders an object's footprint into the *other* sequence |
| `annotate.py` | Headless config-driven runner (no GUI) |
| `templates/gui.html` | The entire front end |

---

## Appendix — depth mode commands

Run from `~/repos/change_annotator-sangwoo`. Each writes to `<capture>/depth/<pre>__<post>/`. Add `--port 5001` if 5000 is taken.

**climate_day — walk 1**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/climate_day \
  --pre-session climate_day_1_1_rgb --pre-ref navvis_2 \
  --post-session climate_day_2_1_rgb --post-ref navvis_3 --depth
```

**climate_day — walk 2**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/climate_day \
  --pre-session climate_day_1_2_rgb --pre-ref navvis_2 \
  --post-session climate_day_2_2_rgb --post-ref navvis_3 --depth
```

**climate_day — walk 3**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/climate_day \
  --pre-session climate_day_1_3_rgb --pre-ref navvis_2 \
  --post-session climate_day_2_3_rgb --post-ref navvis_3 --depth
```

**cvg_kitchen**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/cvg_kitchen \
  --pre-session cvg_kitchen_1_rgb --pre-ref navvis_1 \
  --post-session cvg_kitchen_2_rgb --post-ref navvis_2 --depth
```

**billiards — 1↔2**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/billiards \
  --pre-session billiards_1_1_rgb --pre-ref navvis_1 \
  --post-session billiards_2_1_rgb --post-ref navvis_2 --depth
```

**cab_kitchen**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/cab_kitchen \
  --pre-session cab_kitchen_1_rgb --pre-ref navvis_1 \
  --post-session cab_kitchen_2_rgb --post-ref navvis_2 --depth
```

**g66**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/g66 \
  --pre-session g66_1_rgb --pre-ref navvis_1 \
  --post-session g66_2_rgb --post-ref navvis_2 --depth
```

**dlab_open_space — 1↔2**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/dlab_open_space \
  --pre-session dlab_open_space_1_rgb --pre-ref navvis_1 \
  --post-session dlab_open_space_2_rgb --post-ref navvis_2 --depth
```

**dlab_open_space — 2↔3**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/dlab_open_space \
  --pre-session dlab_open_space_2_rgb --pre-ref navvis_2 \
  --post-session dlab_open_space_3_rgb --post-ref navvis_3 --depth
```

**no_sofa_area — 1↔2**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/no_sofa_area \
  --pre-session no_sofa_1_rgb --pre-ref navvis_1 \
  --post-session no_sofa_2_rgb --post-ref navvis_2 --depth
```

**no_sofa_area — 2↔3**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/no_sofa_area \
  --pre-session no_sofa_2_rgb --pre-ref navvis_2 \
  --post-session no_sofa_3_rgb --post-ref navvis_3 --depth
```

**no_sofa_area — 3↔4**
```bash
PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
  --capture /media/lamaria_indoor/captures/changes/no_sofa_area \
  --pre-session no_sofa_3_rgb --pre-ref navvis_3 \
  --post-session no_sofa_4_rgb --post-ref navvis_4 --depth
```

Not sure of a scene's names? `ls /media/lamaria_indoor/captures/changes/<scene>/sessions/`

---

## Acknowledgements

Built on [SAM 3](https://github.com/facebookresearch/sam3) and the LaMAR geometry stack.

## License

See [LICENSE](LICENSE).
