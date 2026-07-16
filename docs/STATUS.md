# STATUS — where the project actually stands

_Snapshot: 2026-07-16. Read this before trusting any "done" claim elsewhere._

The one-line honest version: **the perception front-end (stages 1–4) is proven end-to-end on a
full real broadcast clip — detection, tracking, team ID, pitch mapping, and the 2D minimap all
rendering — while everything downstream of tracking (stages 4b–6) is implemented and green, but
only against synthetic fixtures we wrote ourselves.** No tactical detector, event inference,
scouting, or streaming component has ever consumed real tracking output. The join between "real
perception" and "real tactics" has not happened yet.

Test suite: **155 passed** (`python -m pytest -q`, 1.5 s) as of this snapshot — all synthetic.

---

## ✅ Proven on real data — the OFI vs Olympiacos run

Ran the **full 858-frame clip** of **real broadcast footage** (OFI–Olympiacos, 1280×720, ~34 s
@ 25 fps) through the sn-gamestate/TrackLab stack in Colab, end-to-end with no per-frame
visualizer failures. Run metrics: **7,813 detections carrying a track_id**, **85 total track
IDs** across the clip (down from 825 on the earlier fragmented 150-frame pass), **VIZ_ERR: 0**.
Every stage produced sane output on real pixels:

- **Detection** — players, referees, ball detected per frame.
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

**Caveat, stated plainly:** "worked" here means *ran without crashing and produced
qualitatively reasonable output*. We have **not** measured accuracy against ground truth
(no tracking-quality metrics, no calibration-error numbers, no per-stage precision/recall).
The `boundary_noise_m = 0.5` used throughout the event layer is still an engineering guess,
not a measured value from this footage.

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
