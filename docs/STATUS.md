# STATUS — where the project actually stands

_Snapshot: 2026-09-13. Read this before trusting any "done" claim elsewhere._

**Change since 2026-07-16:** the project has its **first real perception measurement**. The
ball channel was probed against 108 hand-labelled frames — 30.4% raw recall, 93.8% precision
at a usable operating point, and the design doc's "raise resolution to 1280" prescription
measured as 6× *worse* than the stock 640. Details in the green section; full tables in
docs/ball-tracking-design.md §11. Everything else below is unchanged: **no tactical component
has still ever consumed real tracking output.**

The one-line honest version: **the perception front-end (stages 1–4) is proven end-to-end on a
full real broadcast clip — detection, tracking, team ID, pitch mapping, and the 2D minimap all
rendering — while everything downstream of tracking (stages 4b–6) is implemented and green, but
only against synthetic fixtures we wrote ourselves.** No tactical detector, event inference,
scouting, or streaming component has ever consumed real tracking output. The join between "real
perception" and "real tactics" has not happened yet.

Test suite: **199 passed** (`python -m pytest -q`, ~2 s) as of this snapshot — all synthetic.

---

## ✅ Proven on real data — the OFI vs Olympiacos run

Ran the **full 858-frame clip** of **real broadcast footage** (OFI–Olympiacos, 1280×720, ~34 s
@ 25 fps) through the sn-gamestate/TrackLab stack in Colab, end-to-end with no per-frame
visualizer failures. Run metrics: **7,813 detections carrying a track_id**, **85 total track
IDs** across the clip (down from 825 on the earlier fragmented 150-frame pass), **VIZ_ERR: 0**.
Every stage produced sane output on real pixels:

- **Detection** — players and referees detected per frame. **Not the ball**: the deployed
  detector keeps only the person class (correction 2026-07-16 — the original "ball detected
  per frame" claim here was wrong; see docs/ball-tracking-design.md §2. There is no ball
  detector anywhere in the deployed pipeline, by upstream design).
- **Tracking** — identities tracked across frames (StrongSORT), holding through the full clip.
- **Team classification** — clean two-way split (red Olympiacos / blue OFI).
- **Jersey OCR** — jersey numbers read (aids long-term track stitching); e.g. JN 18/19 labelled.
- **Calibration & pitch mapping** — homography estimated, positions projected to the 2D pitch.
- **2D minimap (radar)** — **confirmed rendering fully on-frame** on the full clip (the "Predictions"
  bird's-eye panel with per-team dots), after the 720p radar fix. This was the payoff of the run:
  the earlier stage-1–4 pass predated the radar/decode patches, so the minimap had never actually
  rendered on real footage until now.

Getting there required three patches to stock sn-gamestate for custom (non-SoccerNet) video plus
a pipeline config that upstream doesn't ship, all now committed and auto-applied by
`research/setup.sh` so a fresh Colab VM reproduces this with zero manual work:
- `research/patch_sn_gamestate.py` (step 9): calibration `.iloc[0]` positional indexing, radar
  minimap sized to the real frame instead of a hardcoded 1920×1080, and an O(n²)→O(n)
  folder-of-frames decode path.
- `research/configs/video_demo.yaml` (installed in step 8): the `tracklab -cn video_demo` config
  for arbitrary clips, including the `use_wandb`/`use_rich`/`eval_tracking: False` keys the newer
  TrackLab requires.

### Ball detection, measured against hand labels (2026-09-13)

The first per-stage precision/recall number this project has ever had. 108 frames of the same
OFI clip hand-labelled at stride 8 (102 visible + 6 adjudicated absent), scored against
`research/ball_probe.py` dumps at six inference sizes. Full analysis:
docs/ball-tracking-design.md §11; reproduce with `python research/ball_analysis.py`.

| imgsz | 320 | 480 | **640** | 800 | 960 | 1280 |
|---|---|---|---|---|---|---|
| raw recall | 1.0% | 16.7% | **30.4%** | 22.5% | 22.5% | 4.9% |

At **640 / conf 0.20**: precision **93.8%**, recall 14.7%, and zero false positives on the six
ball-absent frames. At 640 / conf 0.10: precision 70.3%, recall 25.5% (best F1).

Four things this settled, all of which were open guesses on 2026-07-16:
- **The "raise inference to imgsz 1280" plan (ball-tracking-design §7.2) was wrong** — 1280 is
  6× worse than the stock 640. Higher resolution makes the detector more talkative, not more
  correct (candidates 10 → 1,968 as frames-with-nothing fall 101 → 11).
- **The ball channel is sparse but clean, not dead.** Gaps at 640/0.10: median 3 frames, 65%
  bridgeable, but 5 blackouts over 1 s covering 23% of the clip that must gate consumers off.
- **The confetti (H2) is not the dominant failure.** A confidence floor alone reaches 93.8%
  precision; the surviving false positives sit in the stands (22–31% of candidates above the
  pitch line, ~2× a real ball's width), removable by a pitch mask.
- **The blackouts are genuine detector blindness** — not cuts (a scene-change pass finds the
  clip is one continuous take) and not replays (no frame has zero persons).

Bounds, because the table is quotable and these are not: 102 labels and 15–31 true positives
(±~9 points), one 34 s clip, one camera, one confetti-covered pitch, one stock COCO checkpoint.
A clean-pitch control passage (BvB–PSG, identical 1280×720/25 fps, cut-verified) is extracted
and probed, pending labels.

### Shot-boundary detection, built and validated on real broadcast (2026-09-13)

`pipeline/video/` closes red-team §1.1/§3.1: the batch stack asserted everywhere that
"broadcast cuts leave gaps in the frame stream" and **nothing produced those gaps**. Now
something does. 32 tests; pure stdlib (the decode half lives in
`research/frame_signatures.py` and uses ffmpeg raw bytes — no numpy, PIL, or OpenCV).

Measured over the full BvB–PSG broadcast (100 min, 150,455 frames):

| | cuts | one every |
|---|---|---|
| ffmpeg scene filter | 340 | 17.7 s |
| `pipeline/video/shots.py` | 525 | 11.5 s |

93.5% agreement on ffmpeg's calls. Three of our 209 extra calls were sampled and eyeballed:
all genuine cuts ffmpeg missed (close-up→wide mid-play, a shot change during a run, a
star-wipe into a replay). Neither tool is ground truth — but **a randomly chosen 34-second
broadcast clip spans two or three shot changes**, which is the number that matters: every
passage fed to the pipeline must be cut inside a verified continuous shot, and the OFI clip
all our measurements rest on was single-shot by luck.

Known limitation, tested and pinned rather than hidden: **slow dissolves are missed.** A
15-frame cross-fade moves too little mass per frame to clear any threshold that leaves a
camera pan alone. A learned detector (TransNetV2-class) is the known-better replacement and
slots in behind the same interface.

**Caveat, stated plainly:** for the *tracking* stages above, "worked" means *ran without
crashing and produced qualitatively reasonable output*. Apart from the ball numbers just
given, we have **not** measured accuracy against ground truth (no tracking-quality metrics,
no calibration-error numbers, no per-stage precision/recall for detection or tracking).
The `boundary_noise_m = 0.5` used throughout the event layer is still an engineering guess,
not a measured value from this footage.

**Known defect in the OFI run's pitch coordinates (found 2026-07-16, fixed in config):**
the calibration module's hardcoded 1920×1080 pixel frame vs the clip's real 1280×720 means
**every `bbox_pitch` of that run is systematically distorted** (positions compressed toward
the world region at the frame's top-left). `research/configs/video_demo.yaml` now pins the
module image size to the clip resolution; already-exported homographies are repairable via
`pipeline.calibration.rescale_homography`. Full story: docs/calibration-design.md §3.4.
Separately, the projected pitch overlays visibly wobble/skew (worst on near-goal framings) —
diagnosed with ranked hypotheses in calibration-design.md §4, measurable next GPU session.

## 🟡 Implemented, but only validated against synthetic fixtures

All of this is real, tested code — and every test builds its own scripted 22-player scenarios.
**None of it has seen a single frame of real tracking output.** The thresholds are grounded in
literature (see the provenance tables) and pinned to hand-authored positives/negatives, which
proves the *semantics of the heuristics*, not their behaviour on noisy real data.

- **Phase 4 tactical detectors** — feature layer + all five detectors (high press, low block,
  flank overload, offside trap, counter-attack) + runner/aggregation. 42 tests incl. a
  cross-scenario silence matrix. (`pipeline/patterns/`)
- **Event inference** — Tier 1 restarts, Tier 2 passes/shots/set-pieces, Tier 3 stoppage
  signal. Full implementations with semantics tests. (`pipeline/events/`)
- **Opponent scouting** — multi-match aggregation, recency/Wilson/n_eff math, live priors,
  report schema + adapter. Known-value tests. (`pipeline/scouting/`)
- **Streaming causal core** — delayed-exact smoothing, causal possession machine, episode
  lifecycle machines, live event/WS schema, `StreamingHighPressDetector`. Batch-parity tested
  (7-seed randomized). (`pipeline/patterns/streaming/`)
- **Calibration measurement layer + smoother** — no-ground-truth per-frame
  `calibration_quality` (residual/conditioning/plausibility/temporal components), static-span
  jitter and teleport metrics, and quality-gated pose-space temporal smoothing, all validated
  against synthetic broadcast-camera trajectories with injected detector pathologies. 44
  tests. Has never seen a real exported homography — that export is the next-session
  priority. (`pipeline/calibration/`, design: docs/calibration-design.md)
- **Ball measurement layer + motion model** — coverage/gap statistics, physical-plausibility
  and player-consistency checks, per-frame `ball_quality` gate, and a gated alpha-beta ball
  smoother with honest short-gap interpolation, validated against synthetic trajectories with
  injected dropouts and false positives. 42 tests. **Partially exercised on real data as of
  2026-09-13**: `evaluation.py` has now scored real probe dumps against real hand labels (see
  the ball section above), and the gap/coverage statistics have been computed over the real
  clip. The `ball_quality` gate and `BallSmoother` themselves remain synthetic-only — they
  need a ball *stream*, and the pipeline still has no ball detector wired in (see 🔴 below).
  (`pipeline/ball/`, design: docs/ball-tracking-design.md §11)

## 🔵 Design-only (specified, not built)

- **Phase 5 recommendation engine** — rule matching/scoring/conflict resolution. Skeleton
  only; scouting reports currently take pre-rendered recommendations as input because the
  engine isn't wired. (`pipeline/recommendations/`)
- **Event → Phase 4 possession refactor** — the `event-inference-design.md` §8 seam
  (event-backed DEAD spans and turnover anchors). Designed and flagged, deliberately not
  implemented.
- **`pipeline/live/` service layer** — stream capture, cut detection, GPU live-profile wrapper,
  FastAPI/WebSocket endpoint, post-match reconciliation. Design-only.
- **Streaming feature-extractor assembly** and the low-block/flank/offside streaming wrappers —
  skeletons pending the same scorer extraction as high press.
- **`ui/`** — the Kotlin + Compose Desktop client. Not started.

## 🔴 Blocked

- **Colab free-tier GPU quota is currently exhausted** (transient — resets on cooldown). No
  further GPU runs until it does.
- **Real tracking output has not been exported to disk yet.** The full-clip run proved stages
  1–4 *visually* (annotated video + minimap), but we have not serialized the per-frame tracking
  state that would let Phase 4 finally eat real data. That export is the next GPU-session
  priority (see recipe below), not a code blocker.
- **The pipeline has no ball detector — at all (found 2026-07-16).** The deployed TrackLab
  detector wrapper keeps only the person class from a stock COCO checkpoint; upstream GSR
  excludes the ball by design (dataset, baseline, and metric). Every ball detection rate we
  have ever had is exactly 0, and everything ball-dependent (possession, pressing, counters,
  Tier 1–2 events) currently has no input. **Still true as of 2026-09-13** — but the ball is
  now *measured* rather than merely absent: see the green section below.

---

## The single most important gap

**Nothing beyond stages 1–4 has touched real tracking data.** The entire tactical stack —
every detector, event, scouting number, and streaming machine — has been validated only on
fixtures we authored to confirm our own heuristics. The first real tracking export is the
moment the project stops being "plausible on paper" and starts being measurable. Until then,
treat every Phase 4/5 accuracy expectation as untested.

---

## One-shot recipe for the next GPU session

When the quota resets, this is the whole path from cold Colab (GPU runtime) to real tracking
output on a custom clip. `setup.sh` now installs the `video_demo.yaml` config (step 8) and
applies the three custom-video patches automatically (step 9), so there is nothing to
reconstruct by hand.

```bash
# 1. Clone this repo and run the one-shot environment setup.
#    Installs Python 3.9 + venv, uv, sn-gamestate + TrackLab, mmcv, installs
#    research/configs/video_demo.yaml, and auto-applies patch_sn_gamestate.py
#    (calibration .iloc[0], 720p radar, O(n) folder decode).
#    Idempotent — safe to re-run after a VM restart.
git clone https://github.com/fkanellos/football-tactical-engine.git
bash football-tactical-engine/research/setup.sh

# 2. Extract frames ONCE (decode-once path; avoids the O(n^2) vid:// re-seek).
#    Point ffmpeg at your clip; 25 fps here, the analysis grid downsamples later.
mkdir -p /content/frames
ffmpeg -i clip.mp4 -vf fps=25 /content/frames/%06d.jpg

# 3. Run the pipeline on the FRAME FOLDER (treated as ONE video so tracking
#    persists across frames — do NOT pass the .mp4 directly).
cd /content/sn-gamestate
uv run --python .venv/bin/python tracklab -cn video_demo \
  dataset.video_path=/content/frames dataset.nframes=-1 num_cores=0
```

Notebook alternative: open `research/00_setup_sn_gamestate.ipynb`, run the setup cell (it calls
`setup.sh`), then **switch the kernel to "Python 3.9 (sn-gamestate)"** before the demo cell —
the kernel is registered by setup step 2 and the demo hangs on the default Python 3.12 kernel.

**Priority once it runs:** export the per-frame tracking state to disk and feed it through
`pipeline/patterns/tracking.py`'s adapter → `FeatureExtractor`. That is the first real-data
validation of Phase 4 and the thing every "synthetic-only" caveat above is waiting on.
Second priority, same session: export the per-frame **calibration** state (keypoints, lines,
camera params, H) and run `pipeline/calibration/`'s measurement layer over it — the concrete
experiment is scripted in docs/calibration-design.md §9. Note the re-run must use the fixed
`video_demo.yaml` (calibration image-size keys) or every pitch coordinate repeats the
1080p/720p distortion.
~~Third priority, same session (~5 min GPU): run the **ball probe**~~ — **done 2026-09-13,
and it never needed the GPU.** `research/ball_probe.py` runs on CPU in ~10 min per resolution
for 858 frames; the full six-resolution sweep, the labelling, and the analysis were all done
locally while the quota stayed exhausted. Results: the ball section above and
ball-tracking-design.md §11. Worth generalising: **check whether an experiment actually needs
a GPU before queueing it behind one.**

Two things the ball work leaves queued, neither GPU-bound:
- **Label the BvB–PSG control passage** (`bvb_frames/`, extracted and probed) to separate
  "the detector is weak" from "that pitch was covered in confetti".
- ~~**Cut detection**~~ — **built 2026-09-13**, `pipeline/video/`. See the shot-detection
  section below.
