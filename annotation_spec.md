# Change-Detection Benchmark — Annotation Specification

Source of truth for how we annotate the LaMAria-indoor `changes/` recordings and what
ground truth the annotation tool in this repo must produce. Read this before touching
`gui.py` / `geom_sam_prototype.py`.

This is **not** SceneDiff and **not** PASLCD — it borrows the interface and instance
semantics of [SceneDiff](https://yuqunw.github.io/SceneDiff) and the pixel-mask tier of
PASLCD, but generates ground truth from our own 3D-grounded pipeline (LaMAR poses +
per-state NavVis mesh). See `~/datasets/dataset_notes.md` for the reference-dataset
contrast.

---

## 1. Problem definition

A **paired-video change-detection benchmark**.

- **Input to a method under test:** a **video pair** `(pre, post)` — two Aria RGB clips of
  the same physical space in two different states.
- **Ground truth:** lives in **image space** (per-frame masks on the 10 Hz RGB frames of
  both clips). Never in NavVis/3D space — see §7.
- **Two annotation tiers**, chosen per scene:
  - **Instance tier** — clean object-level changes (objects added / removed / moved).
    Per-object masks with a shared **instance id** across the pair.
  - **Pixel tier** — amorphous change (event setup: half-built stalls, construction,
    clutter) where instances can't be cleanly enumerated. A binary change region only.
- **Universal payload:** *every* sample carries a binary **pixel change mask** (for the
  instance tier it's the union of the object masks). Instance samples carry object
  masks + ids *in addition*. This is what makes one leaderboard possible across both
  tiers — see §6, §8.

---

## 2. Source data

Aria Gen-? recordings under `/media/lamaria_indoor/recordings/changes/<scene>/`.

- **Modality used for GT:** the **pinhole RGB** stream. Each state is registered as a
  separate `aria_*_rgb` session (single RGB camera, ~2560×1920) produced by
  `run_change_capture.py --camera rgb`; the eval pipelines consume these RGB frames, not
  grayscale. The session stays single-camera, and for layout compatibility the RGB frames
  live under the session's `images/cam0/` folder (sole sensor `cam0`). The earlier SLAM
  `cam0` sessions (`aria_a`/`aria_b`, ~724×725 grayscale) were an interim stand-in and are
  no longer used for annotation. IMU/etc. feed the localization pipeline, not annotation.
- **Localization:** each `.vrs` is processed by the `lamaria-indoor` LaMAR pipeline
  (`scantools.run_aria_to_capture` → `run_sequence_aligner`) into a `TimedReconstruction`
  (COLMAP model + `timestamps.txt`), aligned to a **per-state NavVis** reference session.
  Result: every RGB frame has a 6-DoF pose in a common world frame, and there is a NavVis
  **mesh + depth** for that state.
- **Each `.vrs` clip is internally static** — the scene is frozen for the duration of a
  recording; only the camera moves. Change happens strictly *between* clips, never within
  one. (Confirmed; the schema relies on it.)

### Naming convention

| pattern | meaning |
|---|---|
| `<scene>_<state>_<walk>` | e.g. `climate_day_1_2` → scene `climate_day`, **state 1**, **walk 2**. Multiple walk-throughs per state. |
| `<scene>_<walk>` (single index) | **only one state captured so far**; the second state (often the empty capture) is **pending**. Not change-annotatable yet. |

A **state** is one scene configuration (empty / partial / built / arrangement-N). A
**walk** is one static recording of that state; multiple walks per state give viewpoint
coverage.

### How the data was collected

- **Instance scenes:** `(Aria, NavVis)` at state 0 → physically rearrange objects →
  `(Aria, NavVis)` at state 1, etc.
- **Event scenes:** pick a university space, capture `(Aria, NavVis)` when an event is set
  up; also capture the **empty** space. Sometimes intermediate states are captured
  (e.g. `polymesse`: partially-built stalls → fully-built stalls).

### Role of NavVis (important)

NavVis is **infrastructure, not a change source**: it provides poses, the mesh for
occlusion tests, and the surface for lifting a 2D mask to 3D and reprojecting it into
other frames (`geom_sam_prototype.py`). We do **not** diff NavVis-vs-NavVis geometry —
inter-state scans carry spurious extra points in unchanged areas that would manufacture
false-positive change. All change originates from annotation on Aria frames.

---

## 3. Sample definition (what becomes a benchmark pair)

A **sample** = one `pre` clip + one `post` clip + its GT.

**State pairing rules:**
- Scene with multiple captured states → **consecutive** state pairs `(state_i, state_{i+1})`
  (e.g. polymesse `empty→partial`, `partial→built`; optionally also `empty→built`).
- Event scene with a single non-empty state → pair it against the **empty** state
  `(empty, event)`, i.e. a walk that saw the empty space vs. a walk after setup.

**Walk selection:** each state has a pool of walks (`_<state>_1`, `_<state>_2`, …). A
sample draws one walk per state from the relevant pools. Walk choice is a viewpoint-
coverage decision (pick the pre/post walks that maximize co-visible overlap); record the
exact walk ids in the sample so it's reproducible.

---

## 4. Annotation tiers

Set `tier` per **scene** (a scene is wholly instance or wholly pixel for now).

### 4a. Instance tier

For each changed object:
- a free-text `label` (`noun_instanceidx`, SceneDiff-style, e.g. `chair_2`),
- a stable **`instance_id`** used to link the *same physical object* across `pre` and `post`,
- per-frame masks in whichever clip(s) the object is visible,
- `deformability ∈ {rigid, deformable}` (carried over from SceneDiff).

**Change type is implicit** from the presence pattern (no explicit label needed):

| in `pre` | in `post` | meaning |
|---|---|---|
| ✓ | ✗ | removed |
| ✗ | ✓ | added |
| ✓ | ✓ | persisted — *moved* iff masks disagree beyond a displacement threshold (derived downstream) |

Because both clips share one world frame, `geom_sam` can reproject a removed/added
object's 3D footprint **across** clips — so the "where it used to be / will be" region gets
marked without re-clicking in the other clip.

### 4b. Pixel tier

One **binary change mask per frame** over the co-visible region — "this pixel images a
surface that differs between the two states." No instance id, no object decomposition.
Optionally tag a coarse category per region (`construction` / `clutter` / `crowd` /
`signage` / …) as scene-level metadata; not required for scoring.

---

## 5. Ground-truth generation pipeline

Annotation happens on Aria frames; masks are **propagated densely** across the 10 Hz clip
(we annotate keyframes, not every frame by hand).

1. **Seed** — annotator clicks (instance tier) or scribbles a region (pixel tier) on one
   or a few keyframes in the `pre` and/or `post` clip → SAM 3.1 image mask.
2. **Propagate** through the clip — either SAM 3.1 video tracking, or the geometry route
   in `geom_sam_prototype.py` (lift mask to the NavVis mesh by raycast → reproject into
   every frame with a mesh-depth occlusion test → per-frame seed → SAM image predict).
   The geometry route is preferred for long clips and for crossing between `pre`/`post`.
3. **Link across the pair** (instance tier) — assign the same `instance_id` to the object
   in both clips so presence/movement is derivable.
4. **Ignore mask** (see below).
5. **Review/refine** — QC the *propagation* in the review flow (`review*.html`), not by
   redrawing frames. `review_meta.json` records sign-off.

### Ignore / don't-care mask

The `ignore_mask` channel excludes pixels that shouldn't be scored. The dominant nuisance —
**non-co-visibility** (a pixel imaging a surface the other state's clip never saw) — is
always present and is the reason this channel exists even on clean scenes.

**Assumption (current): all `changes/` scenes were captured with no people present.** So the
people/dynamics source is deferred. For now `ignore_mask` is sourced from:
- **non-co-visible pixels** → from poses + mesh (a pixel with no corresponding surface
  observed in the other state's clip),
- optionally **global lighting-only** regions.

Deferred — wire only when a sequence with humans turns up:
- **people / transient dynamics** → SAM 3.1 person/dynamic segmentation.

---

## 6. Ground-truth schema (data contract)

Extends the existing `segments.pkl` (see `CLAUDE.md` → Data contract) rather than replacing
it. Masks are **COCO-RLE** (`pycocotools.mask`), keyed by frame name. `pre`/`post` map to
the legacy `video1`/`video2`.

```python
sample = {
    'scene':       str,
    'tier':        'instance' | 'pixel',
    'pre':  {'session': str, 'state': str, 'walk': str},   # e.g. state='1', walk='climate_day_1_2'
    'post': {'session': str, 'state': str, 'walk': str},
    'category':    str | None,            # pixel-tier coarse tag: construction/clutter/...

    # ── universal: present for EVERY sample ──
    'change_mask': {'pre': {frame: rle}, 'post': {frame: rle}},   # binary changed pixels
    'ignore_mask': {'pre': {frame: rle}, 'post': {frame: rle}},   # don't-care (people/lighting/non-covisible)

    # ── instance tier only ──
    'objects': {
        instance_id: {
            'label':         str,
            'deformability': 'rigid' | 'deformable',
            'in_pre':        bool,
            'in_post':       bool,
            'masks':         {'pre': {frame: rle}, 'post': {frame: rle}},
        },
    },
}
```

Invariant: for an instance sample, `change_mask[clip][frame]` == union over objects of
`masks[clip][frame]` (minus `ignore_mask`). The tool should derive it, not ask the
annotator for it twice.

---

## 7. Evaluation protocol

- **Primary metric (all scenes):** per-pixel change detection — **IoU / F1** of predicted
  vs. GT `change_mask`, with `ignore_mask` pixels removed from both prediction and GT
  before scoring. One number, one leaderboard, instance + pixel scenes together.
- **Secondary metric (instance subset only):** per-object change detection —
  precision/recall of detected changed instances (mask IoU matching), plus change-type
  accuracy over {added, removed, moved} derived from the presence bits + mask displacement.
- Methods only ever output a change map; instance scoring is an *additional* lens on the
  scenes that support it. An event-scene method is never penalized for not emitting
  instances.

---

## 8. Explicit decisions / non-goals

- **GT is image-space**, generated from 3D but delivered as per-frame masks. No 3D GT.
- **No NavVis-vs-NavVis geometry diff** as a change source (false positives from scan noise).
- **No explicit change-type label on the instance tier** — it's implied by presence bits.
- **Dense per-frame masks** on the 10 Hz video (not a sparse held-out test-view set);
  density comes from propagation, QC is at the propagation level.
- **A scene is single-tier** (wholly instance or wholly pixel) for now.

---

## 9. Open items / pending

- **Empty-state captures pending** for some event spaces. Until collected, single-state
  scenes (single-index filenames) are **not** change-annotatable. Scenes with ≥2 captured
  states (e.g. `climate_day` states 1 & 2) can be annotated between those states now, and
  re-paired against empty once it lands.
- **Per-scene tier assignment** table — to be filled (instance vs pixel) as scenes are
  triaged.
- **Pixel-tier category taxonomy** — finalize the coarse tag set if we decide to score by
  category.
- **Moved-object displacement threshold** for deriving "moved" vs "persisted-unchanged".
- **Walk-selection policy** — heuristic for picking the pre/post walks with best overlap.

### Observed scene inventory (file counts, June 2026)

| scene | `.vrs` | inferred decomposition | likely tier |
|---|---|---|---|
| `climate_day` | 6 | states 1,2 × 3 walks; empty pending | pixel (event) |
| `cnb_e100_5` | 2 | 2 states | instance |
| `construction_e` | 4 | 1 state × 4 walks; empty pending | pixel (event) |
| `construction_f` | 2 | 1 state × 2 walks; empty pending | pixel (event) |
| `dlab_open_space` | 3 | 3 states | instance |
| `g68` | 2 | 2 states | instance |
| `learning_fair` | 5 | three states including empty, only state 2 available available; state 1 and empty pending | pixel (event) |
| `nexus` | 6 | 6 walks covering the area; empty pending | pixel (event) |
| `polymesse` | 13 | partial + built states; empty pending | pixel (event) |

Counts/decompositions are observed from filenames and must be confirmed per scene.
</content>
</invoke>
