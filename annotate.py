"""Headless change-annotation orchestrator (RGB, single-camera).

Reads a per-scene annotation config (JSON) and, for each declared object, runs the
geometry-assisted propagation pipeline across the two envs:

  src-mask  click(s) on a source frame        -> SAM3 mask
  seeds     lift onto the source-state mesh,
            reproject within the source walk
            (mesh-depth occlusion test)        -> per-frame seeds
  perframe  SAM3 per-frame masks               -> saved binary masks

All three stages run in ONE env (see setup.sh) — `seeds` only needs scantools on
PYTHONPATH (lamaria-indoor), which is pure-python; SAM3 and the geometry stack
coexist in the same interpreter.

then assembles all objects into one GT record (`changes/segments.json`) following
`annotation_spec.md` §6. Visual overlays for each object are copied to
`changes/geom_sam_out/_review/<id>__<state>/` (under the workspace) for review.

v1 propagates WITHIN each object's source state (one click -> all cam0 frames of that
walk). Cross-sequence vacated/arrival footprints (change_mask in the *other* clip) are a
documented phase-2 — see annotation_spec.md §5. `change_type`/`in_pre`/`in_post` come from
the config (annotator-declared), not inferred.

Run (single env from setup.sh; scantools on PYTHONPATH for the seeds stage):
  PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python annotate.py \
      --config configs/cnb_e100.json [--skip-existing]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEOM = HERE / "geom_sam_prototype.py"
# One interpreter for every stage. Defaults to the interpreter running this
# script, so launching from the single env (setup.sh) just works with no paths
# to set. Override with ANNOTATOR_PY only to point at a different env.
ANNOTATOR_PY = Path(os.environ.get("ANNOTATOR_PY", sys.executable))
# scantools lives in lamaria-indoor and is used via PYTHONPATH (not installed).
LAMARIA_PYTHONPATH = os.environ.get(
    "LAMARIA_INDOOR",
    os.environ.get("LAMAR_PYTHONPATH", str(Path.home() / "repos/lamaria-indoor")))


def run(cmd, env=None):
    print("›", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def stage_src_mask(cap, session, src_name, clicks, obj):
    pts = [str(v) for xy in clicks for v in xy]
    run([ANNOTATOR_PY, GEOM, "src-mask", "--capture", cap, "--session", session,
         "--src-name", src_name, "--obj", obj, "--points", *pts])


def stage_seeds(cap, session, ref, src_name, obj, n, min_vis):
    env = dict(os.environ, PYTHONPATH=LAMARIA_PYTHONPATH)  # scantools on path
    run([ANNOTATOR_PY, GEOM, "seeds", "--capture", cap, "--session", session, "--ref", ref,
         "--src-name", src_name, "--obj", obj, "--n", str(n), "--min-vis", str(min_vis),
         "--no-cross"], env=env)  # v1: within-source propagation only


def stage_perframe(cap, obj):
    run([ANNOTATOR_PY, GEOM, "perframe", "--capture", cap, "--obj", obj, "--save-masks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a stage whose output already exists (resume / re-assemble)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cap = cfg["capture"]
    states = {"pre": cfg["pre"], "post": cfg["post"]}
    n = cfg.get("seeds", {}).get("n", 60)
    min_vis = cfg.get("seeds", {}).get("min_vis", 10)
    geom_out = Path(cap) / "changes" / "geom_sam_out"

    objects_out = {}
    for obj in cfg["objects"]:
        oid = obj["id"]
        present, per_state_masks = set(), {}
        for src in obj["sources"]:
            st = src["state"]                       # 'pre' | 'post'
            session, ref = states[st]["session"], states[st]["ref"]
            objdir = f"{oid}__{st}"
            od = geom_out / objdir

            if not (args.skip_existing and (od / "src_mask.png").exists()):
                stage_src_mask(cap, session, src["frame"], src["clicks"], objdir)
            if not (args.skip_existing and (od / "seeds.json").exists()):
                stage_seeds(cap, session, ref, src["frame"], objdir, n, min_vis)
            if not (args.skip_existing and (od / "masks_index.json").exists()):
                stage_perframe(cap, objdir)

            mi = json.load(open(od / "masks_index.json"))
            per_state_masks[st] = {name: m["mask_file"] for name, m in mi.items()}
            present.add(st)

            # collect review visualizations (under the workspace, no external dir)
            viz = geom_out / "_review" / objdir
            viz.mkdir(parents=True, exist_ok=True)
            for f in ("src_mask_overlay.png", "result_contact.png", "result.mp4"):
                if (od / f).exists():
                    shutil.copy(od / f, viz / f)

        in_pre, in_post = "pre" in present, "post" in present
        # change type is derived from presence (annotation_spec.md §4a), not annotator-set;
        # in-both = a change ⇒ moved (downstream mask-displacement refines moved vs static)
        change_type = ("removed" if in_pre and not in_post else
                       "added" if in_post and not in_pre else
                       "moved" if in_pre and in_post else None)
        objects_out[oid] = {
            "label": obj["label"],
            "deformability": obj.get("deformability", "rigid"),
            "in_pre": in_pre,
            "in_post": in_post,
            "change_type": change_type,
            # masks: {state: {frame_name: mask_file relative to geom_sam_out/<id>__<state>/}}
            "masks": per_state_masks,
        }

    segments = {
        "scene": cfg["scene"],
        "tier": cfg["tier"],
        "camera": "cam0",
        "pre": cfg["pre"],
        "post": cfg["post"],
        "objects": objects_out,
        # NOTE (v1): change_mask = per-state union of object masks is derivable from
        # `objects`; cross-clip vacated/arrival footprints are phase-2 (spec §5).
    }
    out = Path(cap) / "changes" / "segments.json"
    json.dump(segments, open(out, "w"), indent=1)
    print(f"\nwrote {out}  ({len(objects_out)} objects)")
    print(f"viz -> {geom_out / '_review'}")


if __name__ == "__main__":
    main()
