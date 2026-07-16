# STATUS — where the project actually stands

_Snapshot: 2026-07-16. Read this before trusting any "done" claim elsewhere._

The one-line honest version: **the perception front-end (stages 1–4) has run end-to-end on
one real broadcast clip; everything downstream of tracking (stages 4b–6) is implemented and
green, but only against synthetic fixtures we wrote ourselves.** No tactical detector, event
inference, scouting, or streaming component has ever consumed real tracking output. The join
between "real perception" and "real tactics" has not happened yet.

Test suite: **155 passed** (`python -m pytest -q`, 1.5 s) as of this snapshot — all synthetic.

---

## ✅ Proven on real data — the OFI vs Olympiacos run

Run on **real broadcast footage** (OFI–Olympiacos, 1280×720) through the
sn-gamestate/TrackLab stack in Colab. These stages produced sane output on real pixels:

- **Detection** — players, referees, ball detected per frame.
- **Tracking** — identities tracked across frames (StrongSORT).
- **Team classification** — left/right team split worked.
- **Jersey OCR** — jersey numbers read (aids long-term track stitching).
- **Calibration & pitch mapping** — homography estimated, positions projected to the 2D pitch;
  the radar minimap rendered once the 720p fixes were applied.

Getting there required three patches to stock sn-gamestate for custom (non-SoccerNet) video,
now committed as `research/patch_sn_gamestate.py` and auto-applied as step 8 of
`research/setup.sh`: calibration `.iloc[0]` positional indexing, radar minimap sized to the
real frame instead of a hardcoded 1920×1080, and an O(n²)→O(n) folder-of-frames decode path.

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

- **Colab free-tier GPU quota is exhausted.** No further GPU runs until it resets.
- **Full-clip re-run with the patched minimap is pending** that reset — the OFI–Olympiacos run
  that proved stages 1–4 predates the radar/decode patches landing in `setup.sh`, so a clean
  full-length pass (and, critically, the *export of real tracking output* that would let Phase
  4 finally eat real data) has not been done.

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
output on a custom clip. `setup.sh` now applies the three custom-video patches automatically
(step 8), so there is nothing to reconstruct by hand.

```bash
# 1. Clone this repo and run the one-shot environment setup.
#    Installs Python 3.9 + venv, uv, sn-gamestate + TrackLab, mmcv, and
#    auto-applies patch_sn_gamestate.py (calibration .iloc[0], 720p radar,
#    O(n) folder decode). Idempotent — safe to re-run after a VM restart.
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
