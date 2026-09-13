# Ball Tracking: Diagnosis, Measurement, and the Honest Path

**Status:** the §7.1 probe **has been run** on two clips and the §6.4 labels
**have been collected** for both — see **§11 for measured results**, which
supersede the predictions in §4 and correct the prescription in §7.2.

Headline, and read both lines or neither: a stock COCO checkpoint recalls the
ball on **30.4%** of hand-labelled OFI frames at 640, collapsing to 4.9% at
1280 — but on a **clean-pitch control clip** of identical format it reaches
**41.5% at 960** and holds 35.1% at 1280. The optimal inference size is a
property of **the footage, not the detector**, and the thing that moves it is
ground debris. What is genuinely settled, on both clips and independent of
resolution, is §11.9: **~70% recall when the ball is near camera, 0% on the far
side of the pitch.**

The measurement layer, the ball motion model, and the ground-truth evaluation
entry point are **implemented and green** against synthetic trajectories,
synthetic labels, and synthetic detections (63 tests, `pipeline/ball/`, run with
the main suite); the labelling tool (`tools/annotator.html` ball mode) and the
workflow that joins it to the probe are §6.6. The diagnosis (§2) is grounded in
the actual sn-gamestate/TrackLab source (pinned clones read for this pass:
sn-gamestate main, tracklab v1.3.24) and the annotated OFI frames — and it is
**conclusive without any further measurement: the deployed pipeline contains no
ball detector at all** (§2.1). That remains true: §11 measures what a detector
*would* see, not something the pipeline now carries.

**Scope:** the ball channel of the perception front-end — detection, position,
and the per-frame `ball_quality` gate — which the possession state machine,
every pressure signal, the counter-attack detector, and the entire Tier 1/2
event layer stand on. Companion to
[calibration-design.md](calibration-design.md) (same methodology, one layer
up), [phase4-5-design.md](phase4-5-design.md) §2.2/§2.4 (the signals and the
data-quality contract this feeds), and
[event-inference-design.md](event-inference-design.md) (whose Tier 1–2
detectors are the most ball-hungry consumers).

---

## 1. Why the ball is load-bearing

The design docs already assume a ball stream with a validity flag
(phase4-5-design.md §2.2 "carry a validity flag, never interpolate through
long gaps"; §2.4 folds "ball validity" into frame quality). What was never
measured — or, it turns out, produced — is the stream itself. The dependency
chain, from the docs, by consumer:

| Consumer | What it takes from the ball | Without a ball |
|---|---|---|
| `possession.state` (features.py stage 3) | nearest-player-within-2 m inference | UNKNOWN — no possession states at all |
| `possession.time_since_turnover` / turnover anchors | possession flips | no turnovers ⇒ **counter-attack detector cannot run** |
| `press_closing_speed`, `defenders_within_5m/15m`, `nearest_defender_dist` | distances measured **to the ball** | the whole pressure family is undefined |
| high press (§3.2) | ball-in-build-up-zone gate + pressure family | cannot run as designed (see §6.3 for the degraded form) |
| ball progression speed | d/dt of ball x' | counter-attack progression evidence gone |
| flank overload (§3.4) | ball-in-wide-lane gate | gate approximable by player mass, at reduced confidence |
| low block (§3.3) | only "ball in D's half" context | mostly survives — shape signals are player-only |
| offside trap (§3.5) | ball-distance context, through-ball corroboration | line dynamics survive; corroboration gone |
| **Tier 1 restarts** (event doc §4) | trajectory crossing a boundary line | exit evidence gone; only restart morphology remains |
| **Tier 2 passes/shots/crosses** (event doc §5) | launch → flight → reception arcs | **cannot function, period** |
| Tier 3 stoppage (event doc §6) | ball-at-rest-or-lost as half its evidence | survives — ball absence is itself evidence |
| scouting aggregates | everything above | inherit every gap |

Two consumers degrade gracefully; the tactically central ones (possession,
pressing, transitions, passes, shots) do not. The event doc's §2 confidence
hierarchy implicitly prices in a *noisy* ball; it never priced in *no* ball.

## 2. What we actually run (read from source, not memory)

### 2.1 The finding: there is no ball in the pipeline

`research/configs/video_demo.yaml` selects `modules/bbox_detector:
yolo_ultralytics` — TrackLab's `YOLOUltralytics` wrapper
(`tracklab/wrappers/bbox_detector/yolo_ultralytics_api.py`), running a **stock
COCO `yolo11m.pt`**. The wrapper's `process()` keeps exactly one class:

```python
# check for `person` class
if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:   # min_confidence: 0.4
```

COCO class 32 (`sports ball`) is discarded on the line above, along with
everything else that is not a person. All four TrackLab detector wrappers
(yolo_ultralytics, mmdetection, rtmlib, transformers) hardcode the same
person-only filter. **The ball detection rate of every run we have ever done
is exactly 0, by construction.** "The ball isn't reliably tracked in these
frames" — the observation this investigation started from — was an
understatement of a categorical fact. STATUS.md's "ball detected per frame"
claim was wrong and has been corrected.

This is not an sn-gamestate bug we tripped over; it is upstream's documented
design. The GSR paper (Somers et al. 2024, §4.1) removed the ball from the
dataset because its ground-plane projection fails for an airborne ball ("we
remove the ball as it spends significant time in the air"), §4.2 declares ball
detections "ignored" for GSR v1, §6.1 states the baseline "filter[s] the
model's output to retain only the 'person' class detections", and GS-HOTA
scores athletes only. No GSR challenge winner (2024, 2025) carries a ball
component; the sn-gamestate repo has zero issues/PRs mentioning the ball. We
adopted a stack whose benchmark never asked the question we need answered.

### 2.2 Where the "ball" you sometimes see in sn-gamestate output comes from

The role vocabulary does contain `ball` — as a **classification of person
crops**. PRTreid's role head (`sn_gamestate/reid/prtreid_api.py:43`,
`role_mapping = {'ball': 0, 'goalkeeper': 1, ...}`) classifies each detected
*person bbox*; `voting_role_jn` majority-votes it per tracklet. A "ball" in the
output is therefore a person-detection whose crop the ReID model judged
ball-like — on our footage, a misclassification channel and nothing else. And
the radar visualizer **explicitly skips** ball-role detections
(`sn_gamestate/visualization/pitch.py:86`: `if detection.role == "ball":
continue`), so the minimap could never show a ball even if one were found —
consistent with all three annotated frames.

### 2.3 Aggravating configuration facts (for when detection is enabled)

- **Inference resolution.** The wrapper calls `self.model(images,
  verbose=False)` with no `imgsz` — ultralytics defaults to **640**, so our
  1280×720 frames are downscaled 2× before detection. Every ball pixel count
  in §3 halves at inference time. (The same 640-default applies to person
  detection, which survives it; a 5 px ball does not.)
- **If ball detections were simply un-filtered**, they would flow into the
  person machinery: BPBReIDStrongSORT association (appearance embeddings of a
  ≤10 px crop are noise; the Kalman/IoU gating assumes person-like motion),
  team k-means (a ball would get a team), jersey OCR. The architecturally
  correct route is a **separate single-object channel** — there is exactly one
  ball, it needs no ReID and no identity management, it needs a *state
  estimator* (§5.3, §6.5). This matches the one documented SoccerNet precedent
  (MOT4MOT, §5.1).
- **Ground-plane projection.** `get_bbox_pitch` back-projects the bbox
  bottom-centre through the ground-plane homography. For an airborne ball this
  is systematically wrong: the projection lands where the camera ray meets the
  grass, displaced away from the camera by roughly *h·d/(H−h)* (ball height
  *h*, camera height *H*, horizontal camera–ball distance *d*) — for h = 2 m,
  H = 20 m, d = 60 m that is ~6–7 m of error. This is exactly why GSR dropped
  the ball (§2.1). Any pitch-coordinate ball stream we produce carries this
  flight bias; §6 treats positions as *ground-track estimates* and the doc
  says so wherever they are consumed.

## 3. What the OFI frames show (evidence, decoded)

Frames at `~/Downloads/ofi-annotated-frames/` (1280×720, 25 fps run of
2026-07), re-read for this pass.

- **Frame 800 (near-goal framing): the ball is plainly visible** — white,
  ~8–10 px across, near the penalty-area line right of the six-yard box — and
  carries no annotation of any kind. With §2.1 this is expected, not
  mysterious: nothing in the pipeline could have detected it. It also
  calibrates the pixel budget: even in the *favourable* framing the ball is
  ~10 px, i.e. ~5 px at the wrapper's 640 inference size.
- **Frames 060/430 (wide/mid framings): no ball identifiable by eye** at
  reading distance; at these framings the scale is ~0.04–0.07 m/px (players
  25–50 px tall), putting a 0.22 m ball at **3–6 px** — 1.5–3 px after the 640
  downscale.
- **The pitch is strewn with white paper debris** (fan confetti) in all three
  frames — hundreds of ball-coloured, ball-sized blobs on the grass, plus red
  paper piles (frame 430) and the usual white line markings and boots. This
  footage is close to a worst case for a "small white round thing" detector:
  the false-positive supply is enormous and *static*, which is why motion/
  temporal evidence and player-consistency checks (§6) are first-class signals
  here, not nice-to-haves.
- **No minimap ball dot in any frame** — explained structurally by §2.2.

## 4. Diagnosis: the certainty, then ranked failure modes for what comes next

**H0 — No ball detector in the deployed pipeline (certain; architecture
fact).** Established by code reading (§2.1); no measurement needed. Every
downstream "ball" assumption is currently unbacked. Fix classes: §7 items 1–2.

The remaining hypotheses concern **what will fail once detection is enabled** —
they are ranked predictions the §7.1 probe turns into measurements, with the
evidence that would confirm or refute each:

> **Measured 2026-09-13 — read §11 before trusting the rankings below.** H1 is
> **refuted, and inverted**: recall *falls* with resolution above 640. H2 is
> **not confirmed as the dominant failure**: a confidence floor alone reaches
> 93.8% precision, and the false positives that remain concentrate in the
> stands, not on the grass. H5's person-count check found **no frame with zero
> persons** in the OFI clip, and an independent cut check found **no shot
> changes at all** — so neither replays nor camera cuts explain any of the
> 23% of the clip the ball channel goes dark. The unmeasured hypotheses (H3,
> H4, H7) are untouched and stay as written.

**H1 — Too few pixels at wide framing (high; the binding constraint).**
0.22 m at 0.04–0.07 m/px ⇒ 3–6 px wide-shot, ~10 px near-goal (§3), halved at
default 640 inference. Published detectors bottom out well above this:
FootAndBall's ISSIA regime is 8–20 px; the diameter-regression work needs
~14 px; the only verified modern broadcast system (Vorobev 2025) had a 6K
feed — 4.8× our linear resolution. *Confirm:* probe recall at imgsz 1280 ≫
recall at 640 (resolution is the lever); detections concentrate in near
framings / near zones (`zone_counts`). *Refute:* comparable recall at both
sizes (then the constraint is appearance, not pixels).

**H2 — White-debris false positives (high on this footage specifically).**
§3: the confetti supply. *Confirm:* multi-candidate frames common
(`n_candidates > 1`); candidates that are static across frames while play
moves (`isolation_spans`, teleport flags when selection jumps). *Refute:* few
candidates, all moving plausibly.

**H3 — Motion blur on struck balls (medium-high).** At 25 fps broadcast
shutter (~1/50–1/100 s), a 30 m/s ball smears 30–60 cm — several ball
diameters — into a low-contrast streak. *Confirm:* `gap_contexts` showing
gaps whose implied crossing speed is high (ball reappears far away) clustered
right after launches; detection confidence dropping with implied speed.
*Refute:* misses uncorrelated with speed.

**H4 — Occlusion by players (medium).** Feet, bodies, goalmouth scrambles.
*Confirm:* gaps whose bracketing detections are near players
(`nearest_player_distances` small at gap edges), short gap durations.

**H5 — Ball out of frame (medium; structural, not a detector fault).** The
camera follows play imperfectly; replays/close-ups (already segmented by the
cut logic) plus genuine exits. *Confirm:* gaps bracketed by boundary-adjacent
detections (`near_boundary`), and gaps coinciding with frames where the person
count also collapses (probe records `n_persons`).

**H6 — Association/identity errors if the ball is routed through the person
tracker (certain if attempted; avoided by design).** §2.3: ReID embeddings are
meaningless for a ball, person motion gates reject kicks. Not a hypothesis to
test — a route not to take. The separate-channel design (§6.5) plus the §5
literature make this a settled architectural decision.

**H7 — Airborne ground-projection bias (certain mechanism, unmeasured
magnitude).** §2.3: metres of displacement during flight, zero when the ball
is on the turf. *Confirm/measure:* restart geometry — balls at corners/kicks
are on known points (event doc §4.3); flight segments show the bias as
overspeed at launch/landing asymmetry. Mitigation is honest labelling
(ground-track estimate), not a fix; 3D from one camera at our resolution is
out of reach (§5.2).

**What tells them apart in one sentence each:** H1 lives in the 640-vs-1280
recall ratio and the zone distribution; H2 in candidate counts and static
isolated detections; H3 in high-implied-speed gap contexts; H4 in
player-adjacent gap edges; H5 in boundary-adjacent gap edges and person-count
collapse; H7 in restart-point residuals.

## 5. Literature (verified July 2026)

Same standard as the calibration doc: claims verified by fetching the source;
NOT-FOUND stated where the record is empty. One process note: an early
machine-summarization pass **fabricated a results table** for Maksai et al.
2016; every number below survived a direct read of the actual PDF or page.
Full citations in §10.

### 5.1 The SoccerNet record: our stack's lineage

- **GSR (our stack) excludes the ball by design** — dataset, baseline, and
  metric (§2.1 quotes). No 2024/2025 GSR entry adds one.
- **SoccerNet-Tracking (2022) is the one public benchmark with per-frame ball
  ground truth**: 1080p 25 fps main-camera clips, ball as a class — 297 ball
  tracklets / 215,156 ball boxes. The paper calls ball tracking "extremely
  challenging due to its small size (< 100 pixels), occlusions from players,
  fast motion, incurring blurring effects, and shape shifting", and notes
  person-trained detectors "never detect the ball"; fine-tuning on
  humans+ball lifted their FairMOT baseline to 57.9 HOTA. **No SoccerNet
  report publishes a ball-vs-player per-class HOTA/DetA — NOT-FOUND across
  2022–2025.** The MOT task itself was discontinued after 2023.
- **The documented bolt-on precedent is MOT4MOT (2023 challenge, 3rd, 66.27
  HOTA): treat the ball as single-object detection with a fine-tuned YOLOv8l
  (conf floor 0.05; ball AP@0.5 = 0.95), fit a 3rd-order polynomial over a
  51-frame window, reject detections > 100 px from the fit, linearly
  interpolate the survivors.** Separate detector, no ReID, trajectory-fit
  gating, honest interpolation — the architecture §6/§7 adopt. The 2022
  winner's ball handling was a bbox-size prior in association ("only small
  bounding boxes can extend the ball track"); the 2022 challenge itself was
  oracle-detection, so ball *detection* was never tested there.
- Ball Action Spotting is temporal event spotting (timestamps + classes), not
  ball localization — not applicable to per-frame position.

### 5.2 Detecting a 4–10 px ball: what exists and what it needs

- **WASB (BMVC 2023) is the strongest verified soccer result**: HRNet-based
  heatmap detector with stem strides removed (explicitly for tiny balls),
  3-frame MIMO input, on soccer (ISSIA, re-annotated) F1 88.3 / AP 86.2 at a
  4 px tolerance, >30 fps, MIT code with weights, plus re-implementations of
  DeepBall/BallSeg/TrackNetV2/MonoTrack under one roof. Two caveats: its
  soccer benchmark is **static-camera** footage, and its 512×288 input would
  shrink our ball to 2–4 px — using it here means retraining at higher input
  resolution or on tiles. Its tracker is a constant-acceleration prediction
  gate; the authors report Kalman/particle filters added nothing over it.
- **Soccer-specific CNN detectors** (DeepBall AP 0.877; FootAndBall AP 0.909,
  ball 8–20 px, 37 fps at Full HD, MIT code) established the
  full-resolution-heatmap approach but on static elevated cameras.
- **TrackNet family** (tennis/badminton, heatmap MIMO, V3 F1 98.6 on
  shuttlecock) never published soccer-broadcast results — NOT-FOUND.
- **Resolution is the highest-leverage variable, three independent sources**:
  a 2026 UAV study (640→1280 input: +25% relative detection vs +6% for a P2
  small-object head); SAHI slicing (+5–14 AP on small-object benchmarks, at
  ~6–7× inference cost for our geometry — derived, not published); 4×
  super-resolution on low-res football frames (+12% mAP, Seweryn 2024).
  Running 720p footage through a 640 pipeline is fighting ourselves (§2.3).
- **The two-stage global→crop→local cascade is published** (TTNet, CVPRW
  2020: coarse detect on downscaled frame, re-detect on full-res crop, 2 px
  RMSE — table tennis, static camera, 120 fps) and patch-level candidate
  verification exists precisely for debris-like false positives (Van Zandycke
  & De Vleeschouwer 2022). No verified soccer-broadcast replication of the
  cascade — the closest is a 2025 SPIE paper (prior-position-guided YOLOv8)
  whose numbers are paywalled. NOT-FOUND under any "cascade crop ball
  tracking" terminology.
- **Single-camera 3D ball** (diameter-regression: 1.6 px diameter MAE needs
  ~14+ px balls; Vorobev 2025's broadcast system had 6144×3240 input):
  **out of reach at our resolution** — stated as a boundary, not a plan.
- **A 720p-broadcast, 4–10 px soccer-ball benchmark: NOT-FOUND anywhere.**
  Our probe will be, in a small way, first data on this regime.

### 5.3 Ball motion models, gap-bridging, and playing without the ball

- **The published pattern is a mode-switching filter, not a better Kalman
  gain**: flying = gravity parabola (drag uniformly neglected in verified
  soccer systems), rolling = constant velocity, possessed = pinned to a
  player, plus out-of-play. Maksai et al. (CVPR 2016) formalise exactly this
  as a MIP with per-state physics and per-state max-speed transition gates;
  Vorobev et al. (2025) make it recursive/real-time (4 discrete modes, beam
  search, 53 fps on CPU, accuracy@1 m 0.66 vs a Maksai-style baseline's 0.64
  at 10× less compute). Ren et al. (2009) is the classical anchor: CV-Kalman
  + gravity parabola, track-back buffering lifting ball detection 57.6% →
  81.1%, linear interpolation across short occlusions, **player-path
  substitution during possession**.
- **Two Maksai findings matter for our priorities**: on their soccer data the
  ball was present in only ~75% of frames, and "the ball is almost never seen
  flying" — the possessed + rolling states dominate, so *possession
  attribution, not ballistic modelling, is where soccer accuracy lives*. Our
  alpha-beta smoother with COAST/REANCHOR (§6.5) is deliberately the cheap
  version of the mode-switcher: MEASURED≈rolling/flying-ground-track,
  REANCHOR≈kick transition, and possession pinning arrives via the feature
  layer's existing nearest-player machinery rather than a new estimator.
- **Gap semantics have precedent**: linear interpolation for short gaps
  (MOT4MOT, Ren), possession substitution for long ones, and *hybrid beats
  both* — Ball Radar (KDD 2023) infers the ball **from player behaviour
  alone** at 3.66 m mean error, but with just 20% of ball observations
  surviving, error drops to **1.31 m**. Detection and context inference are
  complements, not rivals.
- **Ball-from-players has a real but full-pitch-only literature**: Amirli &
  Alemdar 2022 (MAE 7.6/5.0 m per axis), Gongora 2021 (8.4 m, Set
  Transformer, hosted by TRACAB), Ball Radar (3.66 m, hierarchical
  possession-then-position), PathCRF (2026: skip the ball entirely, infer the
  possession *path* — events without ball coordinates). **All require all 22
  players**; on broadcast that means off-screen player imputation first
  (Omidshafiei 2022 is the published pointer). This is a v2+ path, priced
  honestly: it is where the literature says the ceiling is, and it is not a
  this-quarter deliverable.
- **Industry (SkillCorner) ships broadcast ball x,y,z with "intelligent
  extrapolation"** — flags in their open data (`is_detected`,
  `*_extrapolated.jsonl`) prove the practice, but no methodology or accuracy
  for ball imputation is published anywhere we could find — NOT-FOUND as
  documentation, useful only as an existence proof.

### 5.4 What sn-gamestate already does vs what is genuinely additive

Already there: nothing, for the ball (§2.1) — that is the finding. Genuinely
additive, in effort order: keeping COCO class 32 + native-resolution inference
(config/patch); a separate single-object ball channel with trajectory-gated
selection (MOT4MOT pattern; our `pipeline/ball/` motion layer implements the
gate/bridge already); a fine-tuned small-ball detector (SoccerNet-Tracking's
215k ball boxes are public training data; WASB/FootAndBall are open baselines);
possession-pinning and ball-from-players inference (v2+, needs the §5.3
caveats).

## 6. Measurement methodology (implemented: `pipeline/ball/`)

"The ball isn't reliable" is not a measurement. §6.1–6.3 and §6.5 are
computable from a per-frame ball export alone — no labels, by design; §6.4/§6.6
are the one exception, the ~30 minutes of hand-clicking that buys the one thing
self-consistency cannot. Where the calibration layer leaned on known
pitch geometry, this layer leans on **known ball physics and known player
behaviour**: a real ball moves like a ball and is chased by people.

### 6.1 The signal catalogue

| Signal | Function | Needs | Diagnostic of |
|---|---|---|---|
| Detection rate | `coverage_report` | ball stream | the headline unknown — decides everything (§7) |
| Gap-length distribution (isolated / interpolable / blackout) | `coverage_report` | ball stream | interpolable vs structural absence — different downstream worlds |
| Gap context: implied crossing speed, boundary adjacency | `gap_contexts` | ball stream | H3 (struck-ball misses) vs H5 (exits) vs H4 (traffic) |
| Zone distribution of detections | `zone_counts` | ball stream | H1 (distance/scale failures) |
| Overspeed / teleport / out-of-bounds counts | `plausibility_report` | ball stream | H2 (identity jumps to debris), gross FPs |
| Nearest-player distance, sustained isolation spans | `consistency.py` | ball + player positions | H2 (the unchased "ball") |
| Candidate count per frame | `n_candidates` (ingest) | raw probe dump | H2 (false-positive pressure), threshold sweeps |

### 6.2 The per-frame `ball_quality` gate

`score_ball_series(...) → FrameBallQuality` with `quality ∈ [0,1]`: **0 when
no ball**, else the product of confidence (ramped detector score), kinematics
(implied speed vs the ~40 m/s ceiling; teleports slashed), temporal support
(a lone blip inside a blackout is likelier debris than ball), player
consistency (isolation ramp; missing player data discounts once, mirroring
calibration's MISSING_* idiom), and candidate ambiguity. Interpolated samples
are capped at 0.6 — bridged evidence can never outrank observed evidence.
Components stay visible so a rejected frame is explainable. Thresholds are
engineering priors pinned by the synthetic tests, to be re-fit on the first
real export — the standing STATUS.md caveat applies verbatim.

**Wiring into Phase 4** (the seam §2.4 already budgets for): `TrackingFrame`'s
ball observation gains `ball_quality`; the feature layer multiplies it into
frame data-quality exactly like visibility; possession inference and every
ball-anchored feature gate to None below a floor (proposal: 0.2 — re-fit on
real data before freezing, same deferral as calibration's 0.3). Deliberately
not wired yet — same reason as calibration: the floor should be set by the
first real export, not by fiat.

### 6.3 Who degrades gracefully, and who simply cannot function

Stated honestly, because the design docs currently assume a ball validity flag
without pricing sustained invalidity:

- **Cannot function without the ball** — pass detection, shot detection,
  cross/cutback subtypes, Tier 1 exit evidence, possession states, turnover
  anchors, counter-attack, `press_closing_speed` and the radius counts, ball
  progression speed. Below the quality floor these must emit nothing —
  "discount, don't drop" applies to *frames*, not to multi-second ball
  blackouts.
- **Degrade gracefully** — low block (shape-only; its "ball in D's half" gate
  can fall back to the visible-action centroid at reduced confidence), the
  offside-trap line-management aggregates, Tier 3 stoppage (ball absence is
  half its evidence anyway), team shape/compactness, restart *morphology*
  (players forming a wall / box loading are person-signals; only the exit
  crossing needs the ball — a restart can still be flagged, less typed).
- **Press detection's designed degradation**: with no ball, "pressing" can be
  approximated as collective closing on the *inferred action point* (densest
  opponent-adjacent cluster) — a v2 fallback flagged here for the record, at
  materially lower confidence, never silently substituted.

### 6.4 The one place ground truth is cheap

Ball GT is uniquely cheap to make: one point per frame, no boxes, no
identities. Hand-clicking the ball centre in ~100 uniformly sampled frames of
the OFI clip (~30 min of analyst time) turns the probe's detection rate into
real recall/precision and calibrates every threshold above. Worth doing in the
same session as §7. Tooling exists: `tools/annotator.html` ball mode, and
`pipeline/ball/evaluation.py` scores its export against the probe dump — the
workflow is §6.6.

Three properties of that labelling pass are load-bearing, and all three are
easy to lose by accident:

- **Uniform sampling, not cherry-picking.** Every 8th frame, whatever is in it.
  Label only the frames where you can see the ball and recall measures the
  annotator's eyesight, not the detector.
- **"Ball not visible" is a label, not a skip.** An absent-ball frame is ground
  truth: any detection there is a false positive, and absent frames are the
  only thing that puts a real number on the confetti hypothesis (H2). Skips —
  frames you did not adjudicate — must stay distinguishable from adjudicated
  absences, because they mean the opposite thing. The export keeps skips out of
  the labels array and marks absences `status: "absent"`; the evaluator counts
  skips and never scores them.
- **Native pixel space, resolution stated.** Not pitch coordinates: routing
  ground truth through the homography folds calibration error and the airborne
  ground-plane bias (§2.3) into what is meant to be a *detector* measurement,
  and leaves one number where two failures need telling apart.

### 6.5 The motion layer (implemented: `pipeline/ball/motion.py`)

`BallSmoother`: 2D alpha-beta filter with dt-aware innovation gates
(`gate_base_m + gate_growth_ms·dt`), velocity capped at 40 m/s, quality-
weighted gains, COAST with exponential velocity decay through short misses,
**REANCHOR** when consecutive rejections are mutually consistent (a kick is a
velocity step — snap to the new motion, don't fight it; at 25 Hz a 20 m/s
kick stays inside the gate and is simply tracked, at the 5 Hz analysis grid it
reanchors — both pinned by tests), **LOST** after `max_coast_s` (1.2 s) — the
frame is honestly ball-less rather than silently extrapolated. `reset()` at
segment boundaries; forward+backward averaging in batch mode (same parity
idiom as the calibration smoother). `interpolate_gaps` bridges only gaps
≤ 1.0 s between detections, tags every bridged sample `interpolated=True`
with decayed confidence — enforcing phase4-5-design.md §2.2's "never
interpolate through long gaps" in code. This is the cheap two-mode version of
the literature's mode-switching filter (§5.3); possession pinning stays in
the feature layer where the player machinery already lives.

### 6.6 The ground-truth workflow, end to end

Four steps: **extract → label → probe → evaluate**. The whole thing hangs on
one invariant, so it is stated first.

**The join key is the frame index, and the frame index is sorted position in
the frames folder.** `research/ball_probe.py` computes it as

```python
frames = sorted(p for p in Path(args.frames).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
```

enumerated from 0, and `tools/annotator.html` reproduces that exactly — same
extension set, same non-recursive scope, same plain code-point name ordering.
So index 0 is `000001.jpg`, index 7 is `000008.jpg`, and *anything that changes
the folder's contents changes every index after it*. Keep only the extracted
frames in the folder: one stray `.jpg`, one nested subfolder of crops, one
re-extraction at a different fps, and the labels silently point at the wrong
frames.

The join is therefore **verified, not assumed** — a shifted join reads exactly
like a bad detector, and "recall is 2%" is a conclusion someone will act on.
`evaluate_detections` refuses to score when the label file's `n_frames_total`
disagrees with the probe's line count, *and* when the two disagree about which
filename an index refers to (both files record the stem per frame; two
different 858-frame folders would otherwise join cleanly and produce a
plausible, wrong table). `parse_ground_truth` likewise rejects a label clicked
at a resolution other than the one the document declares.

**1. Extract the frames — this exact command:**

```bash
ffmpeg -i clip.mp4 -vf fps=25 /content/frames/%06d.jpg
```

The same command `research/setup.sh` prints and the same folder
`tracklab -cn video_demo dataset.video_path=/content/frames` consumes, so the
frames you label are the frames the probe sees. Label the *same folder* the
probe runs on (copy it down, or run the identical command on the identical
source file); re-encoding, a different `fps=`, `-q:v`, or a resize produce
different pixels and a different frame count. 858 frames for the OFI clip.

**2. Label — `tools/annotator.html`, "Ball ground truth" mode:**

Double-click the file (no server, no install, no network; the frames never
leave the machine). Choose the frames folder, set the stride (8 → ~107 of 858),
and work through them:

| key | |
|---|---|
| `←` `→` | previous / next sampled frame |
| click | place the ball centre (coarse) |
| click in the Zoom panel | place it precisely — the panel is locked on the point |
| `shift`+`←→↑↓` | nudge the point 1 px |
| `x` | ball not visible — a **label**, and it auto-advances |
| `u` | undo the last label, wherever it was, and jump back to it |
| `j` | jump to the next unlabelled frame |
| `-` `+` | magnifier zoom |

The magnifier and the zoom panel are not polish. At 1280×720 scaled to a laptop
panel one screen pixel is ~1.4 native px and the ball is 4–10 px (§3): measured
on a synthetic 9 px ball, an unaided click lands **~9 px** from the centre — a
miss at any tolerance worth quoting — and the same click corrected in the zoom
panel lands **~0.5 px** out. Without the zoom the labels would not be ground
truth, they would be noise with a decimal point.

Labels autosave to localStorage, keyed by a fingerprint of the frames
themselves (count plus the first and last file's name/size/mtime) rather than
by the folder name — §6.6 tells everyone to extract to `frames/`, so a
name-based key would hand clip B clip A's labels. Import JSON resumes from an
export, and can be imported before the folder is opened. Export writes
`<clip>.ball-gt.json`
(`ball-ground-truth/v1`): `n_frames_total`, `stride`, an explicit
`resolution`, `sampled_indices` (what was offered — so skips are countable),
and one `labels` entry per adjudicated frame, `status` `visible` (with `x`,`y`
in native pixels) or `absent`. Mixed decoded resolutions block the export
outright rather than emitting pixel labels with no stated pixel frame.

**3. Run the probe** — §9.1, unchanged:

```bash
uv run --python .venv/bin/python research/ball_probe.py \
    --frames /content/frames --out /content/ball_probe \
    --weights pretrained_models/yolo/yolo11m.pt --imgsz 640 1280 --conf 0.05
```

**4. Evaluate:**

```bash
python -m pipeline.ball.evaluation ofi.ball-gt.json \
    ball_probe_640.jsonl ball_probe_1280.jsonl --conf 0.1 0.2 0.3 0.4
```

which prints the §9.2 precision/recall table per confidence floor per
inference size — **the 640-vs-1280 recall ratio that decides H1** (§9.5). Or
from Python, `load_ground_truth` + `load_ball_probe` + `sweep`. Two knobs worth
turning rather than accepting:

- `--tolerance-px` (default 8, ~a ball diameter at the favourable framing).
  WASB quotes soccer F1 at 4 px (§5.2); quote both if comparing to it.
- `--selection` — `best` scores the highest-confidence candidate, the policy
  `ingest.probe_to_samples` actually applies downstream. `any` credits a hit if
  *any* candidate is on the ball. The gap between them is **selection** error
  (we found the ball and picked confetti → fix with trajectory-gated selection,
  §7.2) as opposed to **detection** error (never found → fix needs a better
  detector, §7.4). Those are different proposals; do not average them into one
  number.

Comparison is bbox centre vs labelled centre. (ingest.py projects the bbox
*bottom*-centre instead, because there the question is where the ball touches
the ground, not whether it was found.)

## 7. Proposal, ranked by expected information (then improvement) per effort

1. **Measure before changing anything: run `research/ball_probe.py` on the
   OFI frames (committed).** Stock `yolo11m.pt` over the frames folder,
   dumping *every* COCO sports-ball candidate at conf ≥ 0.05, at imgsz 640
   **and** 1280; join offline to pitch coordinates via the existing
   `calibration_dump.jsonl` homographies (`pipeline/ball/ingest.py`) and run
   the §6 measurement layer. Cost: ~5 min GPU + offline analysis. This decides
   everything below and tests H1/H2 directly. *Class: standalone script — no
   pipeline change at all.*
2. **If (and only if) the probe shows usable recall: carry the ball as a
   separate single-object channel.** Patch the YOLO wrapper via
   `patch_sn_gamestate.py` to *dump* class-32 candidates per frame during
   normal runs (same jsonl, zero pipeline-behaviour change — detections still
   person-only), ~~raise inference to `imgsz=1280`~~ **keep inference at the
   stock `imgsz=640`**, and select/gate offline with `BallSmoother` +
   `ball_quality`. Explicitly **do not** route ball detections into
   StrongSORT/ReID/team/OCR (§2.3, H6). *Class: sn-gamestate patch (small,
   sentinel-guarded like the existing four) + this repo's already-implemented
   offline layer.*

   > **Corrected 2026-09-13, twice (§11.1, then §11.9).** The strike-through
   > above records the first correction: on the OFI clip, 640 recalls the ball 6×
   > more often than 1280, so raising resolution would have spent GPU to make the
   > channel *worse*. The control clip then showed that result was confetti, not
   > physics — on a clean pitch the curve climbs to **960 (41.5% recall, 81.2%
   > precision)**, well past 640.
   >
   > So neither "1280" nor "640" is the prescription. **The size is footage-
   > dependent**, and §11.9.2 proposes candidate density as the signal that picks
   > it without labels. Pin nothing in config until that is tested; until then,
   > probe the clip at 640/960 and choose. Both corrections are kept visible
   > because the reasoning behind each is one a reader will re-derive unprompted
   > — "small object ⇒ more pixels" the first time, "measured: more pixels lose"
   > the second.
3. **Wire `ball_quality` into the Phase 4 adapter and the event layer's
   kinematics input** once the first real export exists — the §6.2 seam, the
   §6.3 degradation table, and re-fit of the ramp thresholds. *Class: this
   repo, small, deliberately deferred until real data sets the floor.*
4. **If stock-COCO recall is inadequate (the likely H1 outcome at wide
   framing): fine-tune a dedicated small-ball detector.** Training data
   exists publicly (SoccerNet-Tracking: 215k ball boxes on 1080p main-camera
   footage — the right domain); the MOT4MOT recipe (fine-tuned YOLO, single
   class, low conf floor, trajectory-gated) is the documented precedent, and
   WASB (MIT, weights, tiny-ball-specific architecture) is the stronger but
   retrain-at-higher-res option. Add the TTNet-style crop cascade only if
   full-frame 1280 still misses far-side balls. *Class: new component +
   training run — the first genuinely expensive item, taken only with §7.1
   numbers in hand.*
5. **The honest fallback, stated plainly:** it is entirely possible the
   answer is "at 720p broadcast wide-angle, the ball is detectable in near
   framings and structurally absent in wide ones." If the probe shows that,
   the right move is **not** an ever-bigger tracker: it is (a) gate every
   ball-dependent consumer on `ball_quality` and let the §6.3 table dictate
   what runs; (b) bridge only interpolable gaps; (c) lean on restart
   *morphology* (person-signals) where Tier 1 currently leans on exit
   trajectories; and (d) treat ball-from-players inference (Ball Radar-class,
   1.3–3.7 m, full-pitch-only today) as the researched v2+ ceiling, blocked
   on off-screen player imputation, not as a promise. A measured "wide
   framings have no ball" changes Phase 4/event promises (§8) and that is a
   more valuable outcome than an ambitious tracker that silently hallucinates
   one. *Class: design stance + gating glue, mostly already implemented here.*

Deliberately not proposed: SAHI-style tiling as the first move (~6–7× compute
for gains native-1280 likely captures at ~2–4×; revisit only if §7.1 shows
1280 recall clustering at near framings — H1 confirmed at the far end);
single-camera 3D ball (needs ~14+ px or 6K feeds — out of reach, §5.2);
routing the ball through the person tracker (H6 — wrong in kind, not degree).

## 8. What this changes in the standing promises

- **phase4-5-design.md §2.2's ball row** ("least reliable channel") becomes
  "currently non-existent channel; validity flag generalised to
  `ball_quality` (ball-tracking-design.md §6.2)". The §2.4 data-quality
  contract is unchanged in shape — this doc supplies the missing signal.
- **event-inference-design.md Tier 2 is conditional** on §7.1's detection
  rate: pass/shot recall floors were already flagged (§3, §5.1 of that doc);
  what is new is that *until proposal 2 lands, Tier 2 has literally no
  input*. Tier 1 keeps restart morphology; its exit-evidence factor is
  ball-gated. Tier 3 unaffected.
- **STATUS.md** corrected (the "ball detected per frame" claim). The
  "single most important gap" statement stands — and the ball is now
  measured as part of it rather than assumed into existence.
- **No threshold in any shipped detector changes** — the ball work is
  additive; the possession radius (2.0 m) and it stay as literature-grounded
  defaults pending the same real-data pass as everything else.

## 9. Next GPU session: the concrete experiment

In order, single Colab session, additive to the calibration §9 plan (same
run):

1. **Run the probe** (~5 min GPU): `research/ball_probe.py --frames
   /content/frames --out /content/ball_probe --weights
   pretrained_models/yolo/yolo11m.pt --imgsz 640 1280 --conf 0.05`. Retrieve
   both jsonl files — the session's non-negotiable ball output. (The main
   `tracklab -cn video_demo` run of the calibration plan is unchanged and
   supplies `calibration_dump.jsonl`.)
2. **Offline (no GPU, this repo):** `ingest.probe_to_samples` at conf floors
   {0.1, 0.2, 0.3, 0.4} × {640, 1280} → `coverage_report`,
   `plausibility_report`, `zone_counts`, `gap_contexts`,
   `isolation_spans` (players from the states export), `score_ball_series`.
   Deliverables: detection-rate table (floor × imgsz), gap-length histogram
   with the interpolable/blackout split, per-hypothesis verdict table
   (§4 confirm/refute columns filled in), candidate-count distribution
   (the debris number), and frames 060/430/800's component breakdown.
3. **Label ~107 frames' ball centre** — every 8th frame of the 858, uniformly,
   absences included (§6.4, ~30 min, offline: `tools/annotator.html` ball mode
   → `python -m pipeline.ball.evaluation`, workflow in §6.6) → real
   recall/precision per configuration; re-fit §6.2 ramps.
4. **Decide** by the §7 ladder: rate ≥ ~50% with mostly-interpolable gaps at
   1280 → proposals 2–3 (patch + wire-in); rate materially better at 1280
   but concentrated near-framing → proposal 4 (fine-tune, keep 1280);
   rate < ~20% everywhere → proposal 5 is the plan of record and §8's
   conditionals activate.
5. **First hypothesis to test is H1 via the 640/1280 recall ratio** — it is
   the cheapest decisive number in the whole plan: if 1280 doesn't
   substantially beat 640, more resolution/tiling is not the answer and
   proposal 4's value drops accordingly.

## 10. References (verified July 2026)

Fetched and read unless marked otherwise; [industry]/[thesis] as flagged. An
early summarization pass fabricated numbers for one paper below (caught by
direct PDF read) — treat any second-hand citation of this literature with
suspicion.

**SoccerNet lineage**
- Somers, Magera, Cioppa et al. *SoccerNet Game State Reconstruction: End-to-
  End Athlete Tracking and Identification on a Minimap.* CVPRW 2024,
  arXiv:2404.11335. Ball removed from dataset (§4.1), ignored in v1 (§4.2),
  person-only baseline filter (§6.1), GS-HOTA athletes-only.
- Cioppa, Giancola et al. *SoccerNet-Tracking.* CVPRW 2022, arXiv:2204.06918.
  Ball class: 297 tracklets / 215k boxes; "extremely challenging" (§1);
  person detectors "never detect the ball" (§4); FairMOT-ft 57.9 HOTA.
- Giancola et al. *SoccerNet 2022 Challenges Results.* MMSports '22,
  arXiv:2210.02365. Oracle-box tracking; Kalisteo ball-size prior; "physical
  constraints on ball size or players maximum speed" as winning priors.
- Cioppa et al. *SoccerNet 2023 Challenges Results.* arXiv:2309.06006.
  No oracle boxes; Kalisteo 75.61 HOTA.
- Shitrit, Be'ery, Yerhushalmy. *MOT4MOT* (SoccerNet 2023 tracking, 3rd).
  arXiv:2308.16651. Ball = single-object fine-tuned YOLOv8l (conf 0.05,
  AP@0.5 0.95); 3rd-order polynomial / 51-frame window / 100 px rejection;
  linear interpolation.
- SoccerNet 2024 / 2025 Challenges Results. arXiv:2409.10587,
  arXiv:2508.19182. No ball in GSR either year; MOT task discontinued after
  2023; Ball Action Spotting is temporal-only.

**Small-object / ball detection**
- Tarashima et al. *Widely Applicable Strong Baseline for Sports Ball
  Detection and Tracking (WASB).* BMVC 2023, arXiv:2311.05237; code
  nttcom/WASB-SBDT (MIT). Soccer (re-annotated ISSIA): F1 88.3 / AP 86.2 at
  τ=4 px; HRNet heatmap, 3-frame MIMO, 288×512 input; motion-prediction gate;
  "no improvement from Kalman/particle filters".
- Komorowski et al. *DeepBall.* VISAPP 2019, arXiv:1902.07304 (AP 0.877,
  ISSIA). *FootAndBall.* VISAPP 2020, arXiv:1912.05445 (ball AP 0.909, ball
  8–20 px, 37 fps Full HD; code jac99/FootAndBall).
- Huang et al. *TrackNet.* arXiv:1907.03698 (tennis). Sun et al.
  *TrackNetV2.* ICPAI 2020 (badminton, 512×288). *TrackNetV3.* MMAsia 2023,
  code qaz812345/TrackNetV3. No soccer-broadcast results in the family —
  NOT-FOUND.
- Van Zandycke & De Vleeschouwer. *3D Ball Localization From A Single
  Calibrated Image.* CVPRW 2022, arXiv:2204.00003 (diameter→depth; needs
  ~14+ px). *BallSeg.* MMSports '19, arXiv:2007.11876.
- Vorobev, Prosvetov, Elhadji Daou. *Real-time Localization of a Soccer Ball
  from a Single Camera.* arXiv:2506.07981 (2025). 6144×3240 @ 25 fps single
  camera; RTMDet-t + hybrid-mode recursive filter (flying/possessed/wait/out);
  accuracy@1 m 0.66; kick F1 0.74; 53 fps CPU.
- Akyon et al. *Slicing Aided Hyper Inference (SAHI).* ICIP 2022,
  arXiv:2202.06934; code obss/sahi. +5–14 AP small objects; cost multiplier
  for our geometry ~6–7× (derived, not published).
- Seweryn et al. *Improving Object Detection Quality in Football Through
  Super-Resolution.* arXiv:2402.00163. 4× SR → +12% mAP on SoccerNet frames.
- *Systematic Evaluation of YOLOv8 Variants for UAV-Based Object Detection.*
  Applied Sciences 16(7):3559, 2026 [search-verified — page 403'd]. Input
  640→1280: +25% relative vs +6% for a P2 head on VisDrone.
- Voeikov, Falaleev, Baikulov. *TTNet.* CVPRW 2020. Global→crop→local
  cascade; 2 px RMSE (table tennis, 120 fps static camera).
- Xu et al. *TOTNet: Occlusion-aware temporal tracking.* arXiv:2508.09650
  [abstract-only]. No soccer.
- Kamble, Keskar, Bhurchandi. *Ball tracking in sports: a survey.* Artificial
  Intelligence Review 52:1655–1705 (2019) [citation-level].
- NOT-FOUND: any peer-reviewed benchmark of ball detection on 720p
  moving-broadcast soccer at 4–10 px ball size.

**Motion models, gap-bridging, ball-from-players**
- Maksai, Wang, Fua. *What Players do with the Ball: A Physically Constrained
  Interaction Modeling.* CVPR 2016, arXiv:1511.06181; code
  cvlab-epfl/balltracking. MIP with per-state physics + per-state max-speed
  gates; possession = ball pinned to player; soccer: ball present ~75% of
  frames, "almost never seen flying". Multi-camera, not broadcast.
- Ren, Orwell, Jones, Xu. *Tracking the Soccer Ball using Multiple Fixed
  Cameras.* CVIU 113 (2009) 633–642. CV-Kalman + gravity parabola (drag
  disregarded); detection 57.6→81.1% with track-back; linear interpolation
  through short occlusions; player-path substitution during possession.
- Kim & Kim. *Soccer Ball Tracking Using Dynamic Kalman Filter with Velocity
  Control.* CGIV 2009 [abstract-only]. Occlusion-time velocity adjustment
  from nearby players.
- Kim, Choi, Kim, Yoon, Ko. *Ball Trajectory Inference from Multi-Agent
  Sports Contexts (Ball Radar).* KDD 2023, arXiv:2306.08206; code
  hyunsungkim-ds/ballradar. 3.66 m mean error from players alone; **1.31 m
  with 20% of ball observations** — hybrid beats either; requires full-pitch
  tracking; possession-pinning postprocessing.
- Gongora. *Estimating Football Position from Context.* KTH MSc thesis 2021,
  DiVA kth:diva-305682 [thesis; host ChyronHego/TRACAB]. 8.4 m test error
  from player trajectories; Set Transformer permutation invariance −28%.
- Amirli & Alemdar. *Prediction of the Ball Location on the 2D Plane in
  Football Using Optical Tracking Data.* APJESS 10(1):1–8 (2022). MAE
  7.56 m (x) / 5.01 m (y); full-pitch optical input.
- Kim et al. *PathCRF: Ball-Free Soccer Event Detection.* arXiv:2602.12080
  [abstract-only; repo says KDD 2026]. Possession-path inference without ball
  coordinates.
- Omidshafiei et al. *Multiagent off-screen behavior prediction in football.*
  Scientific Reports 12:8638 (2022) [citation-level]. Off-screen player
  imputation — the missing prerequisite for ball-from-players on broadcast.
- Ponglertnapakorn & Suwajanakorn. *Where Is The Ball: 3D Ball Trajectory
  Estimation From 2D Monocular Tracking.* CVSports 2025, arXiv:2506.05763
  [abstract-only].
- *Common Data Format* (tracking-data standard). arXiv:2505.15820. Ball
  fields standardised; provider imputation methods undocumented.
- SkillCorner open data [industry]. github.com/SkillCorner/opendata.
  `*_tracking_extrapolated.jsonl`, `is_detected` flags; ball imputation
  methodology unpublished — NOT-FOUND as documentation.

---

## 11. Measured results (2026-09-13) — the first real perception numbers

Everything above this section is prediction. This section is measurement, and
where the two disagree the measurement wins. It supersedes §4's hypothesis
rankings, corrects §7.2's prescription, and closes §9's experiment.

**What was run.** `research/ball_probe.py` over all 858 OFI frames at six
inference sizes; 108 frames hand-labelled through `tools/annotator.html` ball
mode at stride 8 (102 visible + 6 adjudicated absent, native 1280×720);
`pipeline/ball/evaluation.py` scoring the two against each other in pixel
space. Reproduce every table below with:

```bash
python research/ball_analysis.py
```

Inputs are tracked: `research/ball_probe_out/*.jsonl` (the dumps) and
`research/ofi_frames.ball-gt.json` (the labels). The probe ran on **CPU**, not
the GPU session §9 assumed — ~10 min per resolution for 858 frames, which
retires "blocked on Colab quota" as a reason this experiment ever waited.

### 11.1 The resolution curve on the OFI clip

> **Scope, added after the control clip landed (§11.9): everything in this
> subsection is true of the OFI footage and does not generalise.** The peak at
> 640 and the collapse at 1280 are caused by confetti, not by the detector. On
> clean footage the curve keeps climbing to 960. The first version of this
> section drew the general conclusion "the more-pixels ladder is closed"; that
> was wrong, and §11.9 has the corrected reading.

Raw recall ignores confidence entirely: did *any* candidate land within 8 px of
the label. It is the ceiling no threshold or selection rule can beat.

| imgsz | raw recall | hits | candidates | labelled frames with no candidate |
|------:|-----------:|-----:|-----------:|----------------------------------:|
|   320 |       1.0% |    1 |         10 |                           101/102 |
|   480 |      16.7% |   17 |        207 |                            74/102 |
| **640** | **30.4%** | **31** |    **588** |                        **46/102** |
|   800 |      22.5% |   23 |      1 500 |                            26/102 |
|   960 |      22.5% |   23 |      1 605 |                            11/102 |
|  1280 |       4.9% |    5 |      1 968 |                            25/102 |

A clean inverted U peaking at 640 — which is the stock ultralytics default and
the size the deployed wrapper already uses. §9.5 nominated the 640-vs-1280
ratio as "the cheapest decisive number in the whole plan". It is decisive
against the plan: **1280 is 6× worse, not better.**

Read the last two columns together for the mechanism. As resolution rises,
frames with *nothing* fall (101 → 11) while candidates explode (10 → 1 968).
Higher resolution makes the detector **more talkative, not more correct**: it
finds steadily more things and steadily fewer of them are the ball. The
hypothesis that fits — untested, offered as a hypothesis — is that downscaling
compresses a motion-blurred ellipse into the compact blob COCO's prior expects,
while native resolution resolves every head, confetti flake and line marking
into something with enough structure to claim the class.

### 11.2 Operating points at 640

| conf | P | R | F1 | TP | FP | FP on absent frames |
|-----:|--:|--:|---:|---:|---:|--------------------:|
| 0.05 | 50.0% | 28.4% | 36.2 | 29 | 29 | 33.3% |
| **0.10** | **70.3%** | **25.5%** | **37.4** | 26 | 11 | 16.7% |
| **0.20** | **93.8%** | **14.7%** | 25.4 | 15 |  1 | **0%** |
| 0.30 | 75.0% |  2.9% |  5.7 |  3 |  1 | 0% |

640 dominates every other size at every floor — best F1, best precision, best
recall. Two defensible operating points: **0.10** maximises F1, **0.20** buys
93.8% precision and perfect silence on the six frames the annotator adjudicated
as ball-absent. For a channel whose consumers are gated on `ball_quality`
(§6.2), 0.20 is the better default: a sparse clean signal degrades gracefully,
a dense dirty one poisons possession.

Contrast 960 at conf 0.05, where **83.3% of the absent-ball frames received a
false detection**. Those six frames are the cheapest labels in the set and the
only ones that price precision honestly — "ball not visible" as a first-class
label (§6.4) earned its keep here.

### 11.3 Detections are bimodal — there are no near misses

Across all 102 visible labels, at both 640 and 1280, the distance from label to
nearest candidate falls into exactly two regimes: **under 4 px, or over 300 px.
Zero detections land between 4 px and 20 px.**

Three consequences. (a) Localisation is not a problem — when the detector finds
the ball it is sub-pixel-accurate (median error 1.6 px at 640). (b) The
tolerance constant is not a lever: 4 px and 8 px produce byte-identical tables,
so quoting recall "at 8 px tolerance" versus WASB's 4 px (§5.2) is a
distinction without a difference *on this footage*. (c) Annotator precision is
not the bottleneck — a hand click anywhere inside the blob is within tolerance,
and missing by enough to matter would mean clicking a different part of the
pitch.

### 11.4 Gap structure at 640 / conf 0.10

Over all 858 frames: detections in 316 (36.8%), 78 gaps.

- median gap **3 frames** (120 ms), mean 6.9, max 49 (2.0 s)
- **65% of gaps ≤ 5 frames** — bridgeable by `BallSmoother` without inventing
  trajectory (§6.5)
- **5 blackouts over 1 s, spanning 199 frames = 23% of the clip** — not
  bridgeable, and the §6.3 degradation table is what runs there

Computed over unlabelled frames, so false positives split real holes: the
figures are optimistic by roughly (1 − precision). Stated rather than corrected,
because correcting it needs labels on every frame.

**Two things that are *not* the explanation for the blackouts**, both checked
rather than assumed:

- **Not camera cuts.** An `ffmpeg` scene-change pass over the 858 frames finds
  **zero** shot changes — the clip is one continuous take, confirmed
  independently by `pipeline/video/shots.py`. Worth knowing that this is luck:
  over the full BvB–PSG broadcast (100 min) ffmpeg finds **340 cuts, one every
  17.7 s**, and our detector finds **525, one every 11.5 s** (93.5% agreement
  on ffmpeg's calls; three sampled extras were all genuine cuts ffmpeg missed).
  So **a randomly chosen 34 s broadcast clip spans two or three cuts.** Every
  future passage must be cut inside a verified continuous shot, or tracking
  breaks for reasons that will be misattributed. This also quantifies red-team
  §1.1/§3.1: the missing cut classifier is a real gap, just not one that
  contaminated *this* measurement.

  > **Correction.** An earlier version of this section, and the commit that
  > introduced it, said "244 cuts, one every 24.7 s". That count was read from
  > an ffmpeg process that was still running — a partial result reported as
  > final. The true figure is 340, and the argument it supports is stronger,
  > not weaker. Recorded rather than quietly overwritten because the mistake
  > was procedural (trusting a background job's output before it exited) and
  > the same trap is available to anyone re-running these experiments.
- **Not the ball leaving frame with the play.** The probe records `n_persons`
  per frame, and **no frame in either dump has zero persons** — there is no
  replay or close-up segment hiding in the clip (H5's structural check).

So the 23% is the detector going blind on footage where the ball is present,
in shot, and hand-findable.

### 11.5 H2 (confetti) — real, but not the dominant failure

The pitch is strewn with white paper (§3), and the confetti *is* visible in the
candidate counts. But it is not what caps recall, and precision does not need a
clever filter to survive it: **a confidence floor alone reaches 93.8%**.

Candidates by image row, conf ≥ 0.05:

| band | imgsz 640 | median width | imgsz 1280 | median width |
|------|----------:|-------------:|-----------:|-------------:|
| y 0–200 (stands, boards) | **31.1%** | 15.7–22.8 px | **22.5%** | 12.3–17.1 px |
| y 200–600 (pitch) | 48.3% | 8.4–14.6 px | 63.2% | ~8.2 px |
| y 600–720 (near touchline) | 20.6% | 16.5 px | 1.6% | 7.9 px |

Roughly a quarter to a third of all candidates sit **above the pitch**, in the
crowd and advertising boards, at about twice a real ball's width. A pitch mask
deletes them for free without touching recall — cheap, but a precision
improvement on a channel whose problem is recall, so: worth doing, not worth
prioritising.

### 11.6 Selection error is small at the operating point

§6 chose trajectory-gated offline selection over confidence ranking. The full
label set says this matters less than a 13-label preview suggested:

| imgsz / conf | `selection=best` TP | `selection=any` TP |
|---|---:|---:|
| 640 / 0.05 | 29 | 31 |
| 640 / 0.10 | 26 | 27 |
| 640 / 0.20 | 15 | 15 |
| 1280 / 0.05 | 1 | 5 |

At 640 the gap is one or two frames — highest-confidence is *nearly* the right
pick, and the ceiling (31) is close. The design choice stands on its own merits
(trajectory continuity, gap bridging), **not** on a claim that confidence
selection is broken; at 640 it very nearly is not. The gap is real only at
1280, where nothing works anyway.

### 11.7 Where this leaves §7's ladder

§9.4 defined the branches: ≥50% with interpolable gaps → proposals 2–3;
materially better at 1280 → proposal 4; <20% everywhere → proposal 5. The
measurement landed **between them at ~26%**, with the 1280 branch dead. The
honest synthesis is a hybrid, and it is cheaper than any single branch:

1. **Proposal 2, at 640 and therefore free of any inference change** — dump
   class-32 candidates during normal runs and select offline.
2. **Proposal 3 with a measured floor** — the §6.2 ramps now have a real number
   to sit on instead of an engineering guess.
3. **Proposal 5's stance for the 23%** — blackouts do not get bridged. Ball
   dependent consumers gate off, per the §6.3 table, and the §8 promise changes
   stay in force.

**Proposal 4 (fine-tuning) is neither confirmed nor refuted by this.** What is
refuted is the *resolution* lever; a fine-tune at 640 on SoccerNet-Tracking ball
boxes remains the strongest available improvement and is now better targeted,
since we know the operating size.

### 11.8 What these numbers do not establish

Stated plainly, because the tables above are quotable and the caveats are not:

- **102 labels, 15–31 true positives.** Roughly ±9 points on any recall figure.
  Order of magnitude, not benchmark.
- **One clip: 34 s, one camera, one match, one confetti-covered pitch.** The
  confetti control (a clean-pitch BvB–PSG passage at identical 1280×720/25 fps)
  is extracted and probed; until it is labelled, "26%" is a property of *this
  footage*, not of the detector.
- **One checkpoint** (`yolo11m.pt`, stock COCO). Nothing here says anything
  about a football-specific detector.
- **Recall is measured against frames where the ball is hand-findable.** It says
  nothing about whether the *resulting positions* are good enough for
  possession, which is a `ball_quality` question the §6 layer answers separately.

---

## 11.9 The control clip — what generalises and what was a property of one pitch

§11 rested on one clip. The obvious objection — that a confetti-covered pitch is
not football footage in general — was answered by running the whole workflow a
second time on a control chosen to differ in exactly one respect.

**The control.** BvB–PSG, a Champions League broadcast: 1280×720 at 25 fps, the
*identical* format to the OFI clip, so nothing needed re-tuning. A 34.32 s
passage (858 frames, `t=860s`) cut from inside a verified continuous shot —
verified because by then we knew broadcast cuts every ~12–18 s (§11.4), so an
arbitrary window would have spanned two or three. Clean pitch, no confetti. 108
frames labelled at stride 8: 94 visible, 14 absent.

### 11.9.1 Recall is depth-limited, and this is the finding that generalises

Recall at 640, split by where the ball sits in the frame — which in a broadcast
framing is depth, and therefore ball size in pixels:

| zone | y range | OFI | BvB |
|---|---|---:|---:|
| far side | 0–250 | **0.0%** (0/30) | **0.0%** (0/11) |
| mid-far | 250–400 | 16.0% | 15.8% |
| mid-near | 400–550 | 11.1% | 52.6% |
| near camera | 550–720 | **68.4%** | **71.4%** |

Two independent clips, two competitions, two productions, two pitch conditions,
and the same shape: **~70% near, 0% far, across 41 labelled far-side frames with
not one detection between them.** That is not "sometimes misses" — it is a floor.

This reframes every aggregate in §11. "26% recall" is not a property of the
detector; it is the clip's *mix of framings* multiplied by a step function. Change
the mix and the number moves without anything real having changed, which is why
the aggregate should not be quoted again without the zone table beside it.

**H1's diagnosis was right and its test was wrong.** The hypothesis said pixels
are the binding constraint (§4); the confirmation it proposed was "recall at 1280
≫ recall at 640". The diagnosis is now confirmed decisively — near is big is
found, far is 3–6 px is never found — while the test it proposed measures
something else entirely. Upscaling at inference does not add information to a
blurred 4 px blob; it only gives every other small bright object more structure
to claim the class. Correct cause, wrong instrument, and a remedy that treats the
instrument.

### 11.9.2 Resolution: the OFI curve was a confetti artifact

| imgsz | 480 | 640 | 800 | 960 | 1280 |
|---|---:|---:|---:|---:|---:|
| OFI raw recall | 16.7% | **30.4%** | 22.5% | 22.5% | **4.9%** |
| BvB raw recall | 14.9% | 25.5% | 29.8% | **41.5%** | 35.1% |

The clean clip does not collapse. It peaks at **960 with 41.5% recall at 81.2%
precision (F1 54.9)** — the best operating point measured anywhere in this work,
and 63% more recall than 640 — and still holds 35.1% at 1280 where OFI managed
4.9%.

So the **§11.1 generalisation was wrong**: more pixels do help the ball. What
also happens is that more pixels resolve every confetti flake into something
ball-shaped, and on a debris-heavy pitch that supply overtakes the signal above
640. Two effects with opposite signs, and which one wins is a property of the
grass.

Candidate volume shows the mechanism directly, since it needs no labels:

| imgsz | OFI candidates | BvB candidates | ratio |
|---|---:|---:|---:|
| 480 | 207 | 210 | 1.0× |
| 640 | 588 | 315 | 1.9× |
| 800 | 1 500 | 359 | **4.2×** |

At 480 the confetti is too small to resolve and the clips behave identically. The
gap opens exactly as resolution makes debris legible.

**A practical consequence, offered as a hypothesis with support rather than a
result:** candidate density is a *self-diagnosing* signal for this. It needs no
ground truth, it is computed by the probe already, and a clip producing ~1.7
candidates per frame at 800 (OFI) versus ~0.4 (BvB) is announcing its own debris
level. An adaptive rule — raise inference size while candidate density stays low,
back off when it spikes — would pick 960 for BvB and 640 for OFI without anyone
labelling anything. Untested; worth testing before any resolution is pinned in a
config.

### 11.9.3 H2 was right about the mechanism and wrong about the victim

Confetti does not touch recall: 25–30% at 640 on both clips. It costs
**precision**, by up to 3.7× (800: OFI 22.5% vs BvB 82.4%). The §11.5 reading
stands with its emphasis corrected — debris is a precision problem, and the
reason it *looks* like a recall problem on OFI is that it forces the inference
size down to 640, which costs the recall that 960 would have delivered.

### 11.9.4 What §7's ladder looks like now

Unchanged: **proposal 2 at whatever size the footage supports**, proposal 3 with
a measured floor, proposal 5's stance for the far side.

Changed: the far-side floor is now measured at **0%**, on two clips, which is
stronger than the "<20% everywhere" branch anticipated and makes §7.5 the plan of
record rather than the fallback — *for the far side specifically*. The near field
at ~70% is a usable channel, and the §6.3 degradation table should be re-read as
a function of ball depth rather than of the clip.

**Proposal 4 (fine-tuning) gains, rather than loses, from this.** §11.1 read as
evidence against it; that reading was resolution-shaped. The real finding is a
detector that is fine at 10–12 px and blind at 3–6 px, which is precisely the gap
a small-ball architecture (WASB, MOT4MOT's recipe — §5.2) exists to close. It
remains the first expensive item and still waits on the foundation.

### 11.9.5 Bounds

Two clips are not a sample. 94 and 102 visible labels, TP counts of 15–39, so
±9 points on any single figure. Two broadcast productions of one format
(1280×720/25 fps) — nothing here transfers to 1080p, to a fixed tactical camera,
or to a Veo-style panoramic feed, where the ball would sit at a roughly uniform
~7 px (estimated from one frame's centre-circle scale, not measured) and the
zone table implies that is the wrong side of the cliff. One stock COCO
checkpoint. And zone boundaries drawn at round pixel values on two clips of the
same framing style — the *shape* replicates, the cut points are not calibrated.
