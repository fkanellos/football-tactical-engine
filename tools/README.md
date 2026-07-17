# tools

Small standalone utilities supporting the pipeline's research/validation loop.

`annotator.html` carries two unrelated jobs behind a mode switch — labelling
tactical patterns in a clip, and labelling the ball's pixel position in
extracted frames. Same constraints either way: one file, double-click to open,
no server, no install, no network, video and frames never leave the machine.

## annotator.html — tactical pattern annotation mode

The clip-annotation loop is the highest-leverage Phase 4 investment (design doc §7):
~50 analyst-labelled clips per motif turn threshold tuning from vibes into
measurement. This is that loop's tool — usable **today**, on any match video you
have, independent of the rest of the pipeline.

**Usage:** open `annotator.html` directly in a browser (double-click; no server, no
install, no network — the video never leaves your machine). Load a clip, scrub,
mark a time range (`[` and `]`), pick a pattern label (`1`–`6`, including `none`
for reviewed-negative spans), team, confidence (certain/likely/borderline), add an
optional note, hit Enter.

**Output:** Export JSON produces a `pattern-events/v1` document — the same schema
Phase 4's detectors emit (design doc §4.3) — so labelled clips plug straight into
the future evaluation harness as ground truth. `intensity` is `null` for manual
annotations; confidence encodes the certain/likely/borderline choice.

Annotations autosave to browser localStorage per match id; Import JSON resumes a
previous session from an exported file.

## annotator.html — ball ground-truth mode

The ball channel's headline unknown is its detection rate, and nothing in
`pipeline/ball/` can measure recall or precision without labels
(ball-tracking-design.md §6.4). This mode makes them: ~30 minutes of clicking
the ball centre in ~107 uniformly sampled frames turns the probe's "detection
rate" into a measured number.

**Usage:** switch to *Ball ground truth*, open the **folder of extracted
frames** — the same folder `research/ball_probe.py` reads, because the frame
index both sides join on is sorted position in it. Set a stride (default 8 →
~107 of 858 frames). Then, per frame: click the ball roughly, click it exactly
in the Zoom panel, `→`. `x` marks *ball not visible* and moves on; `u` undoes;
`j` jumps to the next unlabelled frame; `shift`+arrows nudge 1 px. The
magnifier and Zoom panel are the point of the mode, not decoration — the ball
is 4–10 px, and an unaided click at this framing measures ~9 px off versus
~0.5 px through the Zoom panel.

Three things this mode is opinionated about, all for the same reason (a
measurement you can't trust is worse than no measurement):

- **Uniform stride, not cherry-picking** — label only the frames where the ball
  is easy to see and recall measures you, not the detector.
- **"Ball not visible" is a label, not a skip** — an absent-ball frame is
  ground truth: any detection there is a false positive. The export keeps
  adjudicated absences (`status: "absent"`) and skips (simply not present)
  distinct, and the evaluator counts skips without scoring them.
- **Native pixel coordinates, resolution recorded** — never pitch coordinates,
  which would fold calibration error into a detector measurement. Frames that
  decode at more than one resolution block the export rather than emitting
  pixel labels with no stated pixel frame.

**Output:** `ball-ground-truth/v1` JSON, consumed directly by
`pipeline/ball/evaluation.py`:

```bash
python -m pipeline.ball.evaluation ofi.ball-gt.json \
    ball_probe_640.jsonl ball_probe_1280.jsonl --conf 0.1 0.2 0.3 0.4
```

The evaluator verifies the join rather than trusting it: it refuses to score
when the labels and the probe dump disagree on the frame count or on which
filename an index names (both files record the stem), and refuses labels whose
click-time resolution differs from the document's. A mis-joined folder produces
a plausible ~2% recall table, which is a conclusion someone would act on.

Labels autosave per clip — keyed by a fingerprint of the frames, not the folder
name, since everyone extracts to `frames/`. Import JSON resumes, and works
before the folder is open. The full extract → label → probe → evaluate
workflow, with the exact ffmpeg command, is ball-tracking-design.md §6.6.
