#!/usr/bin/env python3
"""
Post-install patches to sn-gamestate / TrackLab for running the game-state
reconstruction pipeline on custom (non-SoccerNet) broadcast video.

Run by research/setup.sh after `uv pip install -e .`, and safe to run by hand:

    /content/sn-gamestate/.venv/bin/python research/patch_sn_gamestate.py

Idempotent: every patch checks for a sentinel (or for the original text) before
editing, so re-running is a no-op. Pass one or more search roots as argv; the
script globs each for the target files. Default root: /content/sn-gamestate
(covers both the editable sn_gamestate source and the tracklab install under
.venv/lib/.../site-packages).

Why these three patches — all found while running the pipeline on a 1280x720
broadcast clip (OFI vs Olympiacos), where the stock code either crashed or
silently dropped output:

  1. calibration (nbjw_calib.py, pnlcalib.py)
     `predictions = metadatas["keypoints"][0]` uses *label* indexing and throws
     `KeyError: 0` whenever the per-image metadata index does not start at label
     0 — which is exactly what ExternalVideo produces. The very next lines use
     `metadatas.iloc[0].name` (positional), so the intent was positional.
     Fix: `metadatas["keypoints"][0]` -> `metadatas["keypoints"].iloc[0]`.

  2. radar minimap (pitch.py :: draw_radar_view)
     The bird's-eye radar is hardcoded to a 1920x1080 frame:
         radar_center_x = int(1920/2 - ...)
         radar_top_y    = int(1080 - pitch_height*scale - y_delta)   # = 765
     On a 1280x720 video that rectangle (rows 765..1077) is entirely below the
     frame, so the destination slice is empty and cv2.addWeighted raises
     `(-215) src1.empty() == src2.empty()`. The Radar visualizer is caught per
     frame, so the video still renders but WITHOUT the minimap panel.
     Fix: place the radar relative to the real frame size (`patch.shape[1]` /
     `patch.shape[0]`) instead of the 1920 / 1080 literals.

  3. video decode (external_video.py :: ExternalVideo.__init__)
     A single video file is exposed to the engine as `vid://<path>:<i>`, which
     re-opens/re-seeks the file for frame i -> O(n^2). It is fine for ~60 frames
     and stalls for minutes at a few hundred. The wrapper's existing directory
     branch treats a folder as *many* videos (resetting the tracker every file),
     which breaks tracking. Fix: add an early return that treats a directory of
     image frames as ONE video sequence, each frame a plain image path decoded
     once (O(n)), keeping a single video_id so StrongSORT tracks across frames.
     Pair with:  ffmpeg -i clip.mp4 -vf fps=25 frames/%06d.jpg
     then run:   tracklab ... dataset.video_path=/path/to/frames
"""

import sys
from pathlib import Path

DEFAULT_ROOTS = ["/content/sn-gamestate"]


def find_one(roots, rel_glob):
    """Return the first file matching rel_glob under any root, or None."""
    for root in roots:
        for p in sorted(Path(root).rglob(rel_glob)):
            # Skip anything inside a build/ or .git/ tree.
            if "/.git/" in str(p) or "/build/" in str(p):
                continue
            return p
    return None


def apply(path, replacements, sentinel):
    """
    Apply a list of (old, new) string replacements to `path`.

    `sentinel` is a string that is present once the patch has been applied; if it
    is already in the file we skip. Each `old` must be found exactly once (after
    the sentinel check) or we report and skip that replacement, so a drifted
    upstream file fails loudly instead of silently half-patching.
    """
    if path is None:
        print(f"  [skip] target not found")
        return False
    text = path.read_text()
    if sentinel in text:
        print(f"  [ok]   already patched: {path}")
        return True
    new_text = text
    for old, new in replacements:
        count = new_text.count(old)
        if count != 1:
            print(f"  [WARN] expected exactly 1 occurrence of {old!r} in "
                  f"{path.name}, found {count} — skipping this file")
            return False
        new_text = new_text.replace(old, new)
    path.write_text(new_text)
    print(f"  [done] patched: {path}")
    return True


def main():
    roots = sys.argv[1:] or DEFAULT_ROOTS
    print(f"Applying sn-gamestate custom-video patches (roots: {roots})")

    # --- 1. calibration: label -> positional indexing -----------------------
    print("[1/3] calibration keypoints indexing")
    for rel in ("**/sn_gamestate/calibration/nbjw_calib.py",
                "**/sn_gamestate/calibration/pnlcalib.py"):
        f = find_one(roots, rel)
        apply(
            f,
            [('metadatas["keypoints"][0]', 'metadatas["keypoints"].iloc[0]')],
            sentinel='metadatas["keypoints"].iloc[0]',
        )

    # --- 2. radar minimap: use real frame size, not 1920x1080 ----------------
    print("[2/3] radar minimap frame-size")
    pitch = find_one(roots, "**/sn_gamestate/visualization/pitch.py")
    apply(
        pitch,
        [
            # radar_center_x
            ("int(1920/2 - pitch_width * scale / 2 * sign - delta * sign)",
             "int(patch.shape[1]/2 - pitch_width * scale / 2 * sign - delta * sign)"),
            # radar_center_y
            ("int(1080 - pitch_height * scale / 2 - y_delta)",
             "int(patch.shape[0] - pitch_height * scale / 2 - y_delta)"),
            # radar_top_y
            ("int(1080 - pitch_height * scale - y_delta)",
             "int(patch.shape[0] - pitch_height * scale - y_delta)"),
        ],
        sentinel="int(patch.shape[0] - pitch_height * scale - y_delta)",
    )

    # --- 3. decode: folder-of-frames as a single video (O(n)) ----------------
    print("[3/3] external_video folder-of-frames fast path")
    ev = find_one(roots, "**/tracklab/wrappers/dataset/external_video.py")
    # Insert an early return right after the existence assert. Keyed on the
    # assert so it lands before the original is_dir() (folder-of-videos) branch.
    anchor = ('assert self.video_path.exists(), '
              '"Video does not exist (\'{}\')".format(self.video_path)')
    early_return = anchor + "\n" + (
        "        # >>> football-tactical-engine patch: folder-of-frames = ONE video\n"
        "        # A directory of image frames is decoded once each (O(n)) instead\n"
        "        # of re-seeking a single .mp4 per frame via vid:// (O(n^2)).\n"
        "        if self.video_path.is_dir():\n"
        "            _exts = {'.jpg', '.jpeg', '.png', '.bmp'}\n"
        "            _frames = sorted(p for p in self.video_path.iterdir()\n"
        "                             if p.suffix.lower() in _exts)\n"
        "            if _frames:\n"
        "                _md = pd.DataFrame([\n"
        "                    {'id': i, 'name': f'{video_name}_{i}', 'frame': i,\n"
        "                     'nframes': len(_frames), 'video_id': 0,\n"
        "                     'file_path': str(_frames[i])}\n"
        "                    for i in range(len(_frames))\n"
        "                ])\n"
        "                _vm = pd.DataFrame([{'id': video_name, 'name': video_name}])\n"
        "                _vs = TrackingSet(_vm, _md, None, _md)\n"
        "                super().__init__(dataset_path, dict(val=_vs), *args, **kwargs)\n"
        "                return\n"
        "        # <<< end patch\n"
    )
    apply(
        ev,
        [(anchor, early_return)],
        sentinel="football-tactical-engine patch: folder-of-frames",
    )

    print("Done.")


if __name__ == "__main__":
    main()
