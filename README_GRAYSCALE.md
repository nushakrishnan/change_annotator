# Grayscale depth mask review: cnb_e100_1

Run these commands on the machine with the dataset and the existing
`~/annotator_env` environment. SAM brushes require the installed SAM model and a
CUDA GPU; see [installation](README.md#2-install) if setting up a new machine.

```bash
cd ~/repos/change_annotator
git switch depth_mode_grayscale
~/annotator_env/bin/python gui.py \
  --grayscale-sequence /media/lamaria_indoor/depth_gt/v3/sequences/cnb_e100_1
```

Open **http://127.0.0.1:5000** in your browser. If running on a remote machine,
forward port 5000 through SSH or your editor, then open that address locally.
If port 5000 is occupied, add `--port 5001` and open/forward port 5001 instead.

## Review and export

- The editor loads grayscale keyframes from `stereo/left/` and projected masks
  from `stereo/masks/`, combining all masked regions into one **Invalid depth**
  object. Saved drafts and previous exports are loaded when available.
- Select **Multi smart brush** and click a blob to fill small holes and refine its
  edges with SAM. Each click acts independently. **Shift-click** disables trimming;
  **Ctrl-z** undoes the last click. The existing manual brushes also work.
- Press **z** to save a draft, or move to another frame to save it automatically.
- Click **Export corrected masks** to save the current canvas and publish a
  complete mask set for all keyframes to:

  ```text
  /media/lamaria_indoor/depth_gt/v3/sequences/cnb_e100_1/grayscale_mask_export/
  ```

  Export replaces the previous contents of this export folder. The original
  `stereo/masks/` files stay unchanged; draft saves do not publish exports.

To review another sequence, replace `cnb_e100_1` in the launch command. For manual
brushes without SAM or a GPU, append `--no-sam`; Multi smart brush then becomes
unavailable.

See the [main README](README.md#3-launching) for mask formats and pipeline details.
