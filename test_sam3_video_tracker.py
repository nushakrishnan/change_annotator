"""Validate the SAM2-compatible SAM3 video tracker (the correct SAM-only path).

build_sam3_video_model().tracker exposes init_state / add_new_points_or_box /
propagate_in_video (SAM2-style). This is what the broken multiplex migration
should have used. Success = a point mask actually tracks across many frames.
"""
import os
import sys
import numpy as np
import torch
from PIL import Image

frames_dir = sys.argv[1]
names = sorted((f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg"))),
               key=lambda p: int(os.path.splitext(p)[0]))
with Image.open(os.path.join(frames_dir, names[0])) as im:
    W, H = im.size
print(f"frames={len(names)} size={W}x{H}")

from sam3.model_builder import build_sam3_video_model
print("building video model...")
m = build_sam3_video_model()
pred = m.tracker
pred.backbone = m.detector.backbone
print("model ready")

ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else None
with torch.inference_mode(), (ctx if ctx else torch.no_grad()):
    st = pred.init_state(video_path=frames_dir)
    # pixel-coordinate center click, obj 1, frame 0 (normalize_coords handles px->model)
    out = pred.add_new_points_or_box(
        st, frame_idx=0, obj_id=1,
        points=np.array([[W * 0.5, H * 0.55]], np.float32),
        labels=np.array([1], np.int32),
        clear_old_points=True, normalize_coords=True, rel_coordinates=False,
    )
    vr = out[3]  # video_res_masks
    a0 = int((vr[0] > 0).sum())
    print(f"frame0 add_new_points -> objs={list(np.asarray(out[1]))} mask_px={a0}")

    n = nm = 0
    for fi, oids, lr, vrm, sc in pred.propagate_in_video(
            st, start_frame_idx=0, max_frame_num_to_track=len(names),
            propagate_preflight=True, reverse=False):
        n += 1
        if (vrm[0] > 0).sum() > 0:
            nm += 1
    print(f"propagated {n} frames; {nm} carried a non-empty mask")
print("DONE")
