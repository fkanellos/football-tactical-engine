# Calibration: Diagnosis, Measurement, and Stabilisation

**Status:** the measurement layer and pose-space smoother are **implemented and
green** against synthetic camera trajectories (44 tests, `pipeline/calibration/`,
run with the main suite). The diagnosis (§4) is grounded in the actual
sn-gamestate/nbjw_calib source and the annotated OFI frames, but **no
calibration signal has been computed on real footage yet** — the per-frame
export that enables that is the top item for the next GPU session (§9). The
pixel-frame mismatch fix (§3.4) is committed to `research/configs/video_demo.yaml`.

**Scope:** stage 3 of the six-stage pipeline (camera calibration / pitch
mapping), which every Phase 4+ metre-denominated feature stands on. Companion to
[phase4-5-design.md](phase4-5-design.md) (whose §1.1 input contract this layer
feeds) and [live-architecture-design.md](live-architecture-design.md) (whose
segment machinery the smoother plugs into).

---

## 1. Why this is the weakest load-bearing link

Every Phase 4 feature is computed **in metres from the homography**:
`def_line_height`, `width`/`depth`/`hull_area`, `lane_occupancy`, centroids,
every pressure radius, every possession-inference distance
(phase4-5-design.md §2). The event layer's pass/shot geometry and the scouting
aggregates inherit the same coordinates. A calibration error is therefore not
one bad signal among many — it is a *multiplier on all of them*, and no amount
of detector-threshold tuning downstream can compensate for positions that are
wrong at the source. phase4-5-design.md §1.1 already stated the contract
honestly ("noisy, occasionally wildly wrong when calibration slips"); the OFI
run made the slip visible.

Two distinct failure currencies matter downstream:

- **Bias/skew** (frames 430/800): positions systematically displaced — line
  heights and compactness silently wrong by metres. The worst kind: no variance
  signature, invisible to smoothing.
- **Jitter** (all frames, worst near goal): per-frame independent noise —
  fake velocities. `def_line_velocity` (the offside-trap detector's core
  signal) and `press_closing_speed` are *derivatives*, so at 25 fps even ±0.3 m
  of white positional noise manufactures multi-m/s phantom velocities.

## 2. What we actually run (read from source, not memory)

The OFI run used `modules/pitch: nbjw_calib` + `modules/calibration: nbjw_calib`
(`research/configs/video_demo.yaml`) — the **"No Bells, Just Whistles"**
keypoint pipeline (Gutiérrez-Pérez & Agudo, CVPRW 2024), *not* TVCalib. TVCalib
(segmentation + camera-distribution optimisation) ships in sn-gamestate as an
alternative module set, unused by our config. Source read:
`sn_gamestate/calibration/nbjw_calib.py` and
`plugins/calibration/nbjw_calib/{utils_calib.py, utils_heatmap.py, utils_calib_seq.py}`.

The per-frame chain, with the properties that matter for diagnosis:

1. **Two HRNet heatmap models on a 960×540 resize**: one for **57 named
   keypoints** (line intersections, penalty spots, circle/arc tangencies, goal
   posts), one for **23 line classes** (each reduced to *two endpoint
   estimates*). Keypoint acceptance threshold 0.1449, lines 0.2983.
2. **Integer-grid extraction**: keypoints are argmax cells on the ~**480×270**
   heatmap grid, scaled ×2 — **no sub-pixel refinement**
   (`get_keypoints_from_heatmap_batch_maxpool`). Quantisation step ≈ 2.7 px at
   1280-wide, and it is temporally white: a stationary landmark flips between
   adjacent cells frame to frame.
3. **Keypoint "completion"** (`complete_keypoints`): any of the 30 primary
   intersections not detected directly is synthesised by intersecting two
   detected *lines* — each a `linregress` through **two** endpoint estimates —
   plus 16 auxiliary intersections (58–73). Completed points are accepted up to
   **half a frame outside the image** and are assigned confidence **p = 1.0**,
   i.e. *higher* than any genuinely detected keypoint. Near-parallel line pairs
   make these intersections extremely sensitive to endpoint noise.
4. **Homography**: `FramebyFrameCalib.get_homography_from_ground_plane` runs
   unweighted `cv2.findHomography(world, image, RANSAC, threshold)` over all
   detected+completed ground-plane keypoints. The wrapper calls it with
   **`use_ransac=50`** — a 50 px inlier threshold at 960×540, which rejects
   essentially nothing. The degeneracy check is only "not all x equal / not all
   y equal"; near-collinear or tightly clustered sets pass. Player positions
   (`bbox_pitch`) come from this H⁻¹ applied to bbox bottom points.
5. **Camera parameters** (exported as `parameters`, used by nothing downstream
   of us): a separate `heuristic_voting` loop — up to 18 `cv2.calibrateCamera`
   runs (3 modes × 6 RANSAC thresholds), distortion hard-fixed to zero,
   principal point fixed at frame centre — sorted by RMS reprojection error.
6. **Temporal coupling: none.** The class is named `FramebyFrameCalib` and
   means it. The only stateful behaviour is `use_prev_homography: True`:
   *hold the last H* whenever a frame fails outright — so during failures,
   players are projected through a stale camera. Upstream *ships* a
   `SequentialCalib` (utils_calib_seq.py) that regularises consecutive
   estimates — and nothing in sn-gamestate imports it. Notably it regularises
   **raw 9-entry homography vectors** against pixel residuals with no relative
   weighting — entries spanning ~5 orders of magnitude — which is the
   parameterization §7 argues against.
7. **The visualiser draws detected keypoints, not the fit**: the coloured
   overlay lines are polylines *through the (completed) keypoints themselves*
   (`kp_to_line` → `draw_pitch`), with the centre circle ellipse-fitted to
   detected circle keypoints. So the wobble/skew visible in the annotated
   frames is the **detector's error field directly**, upstream of the
   homography — and the homography consumes exactly those points.

## 3. What the OFI frames show (evidence, decoded)

Frames at `~/Downloads/ofi-annotated-frames/` (1280×720, 25 fps run of
2026-07). Line colours decoded via `SoccerPitch.palette`.

### 3.1 Frame 060 — wide midfield shot (the *easy* regime)
Centre circle (blue) and halfway line (yellow) detected roughly right but the
halfway-line polyline is rotated a few degrees off the real paint. More
important: a **hallucinated "Big rect. left main" (grey) and "Circle left"
(orange) float on empty grass** in the upper-left — the line model fired above
threshold on landmarks that are nowhere near the frame. Any completed keypoint
built from those lines enters the homography at p = 1.0 (§2.3, §2.4).
The minimap shows dots for only a fraction of the visible, tracked players —
frames where `bbox_pitch` is None or off-map are already being dropped
silently.

### 3.2 Frame 430 — mid-zoom toward the right box
Far touchline (red) roughly aligned at its left end, diverging to the right;
the right-box family (teal "Big rect. right top", cyan penalty arc, olive
"Big rect. right main") mutually inconsistent — they do not form a coherent
box. A stray red segment cuts across mid-frame players: `kp_to_line` draws one
polyline per line class in fixed keypoint order, so a **single misplaced
keypoint produces exactly this zigzag** — direct evidence of individual
keypoint outliers, not just soft noise.

### 3.3 Frame 800 — near-goal framing (the failure regime)
The entire projected box geometry is rotated and offset: white "Small rect.
right main" and purple/pink goal frame drawn beside the real goal, olive
penalty-box line cutting across players at a visibly wrong angle, cyan arc
displaced. Every keypoint in the cluster is off in a *correlated* way — the
detector is out of distribution on this framing, and the landmark set it does
produce is confined to one box (degenerate support for a full-pitch
homography).

### 3.4 The pixel-frame mismatch (found during this code read — a real bug)
`modules/calibration/nbjw_calib.yaml` hardcodes `image_width: 1920,
image_height: 1080`. Keypoints are normalised to [0,1] and then denormalised by
`FramebyFrameCalib(1920, 1080)` — so the fitted H maps **world → virtual
1920×1080 pixels**. But `get_bbox_pitch(h)` is applied to detection bboxes in
**real 1280×720 pixels**. A projective map does not commute with that 1.5×
scale: every projected position in the OFI run is systematically distorted —
compressed toward the world region imaged at the frame's top-left corner (the
one fixed point of the scale map). The suspiciously tight clustering of minimap
dots relative to the on-screen player spread in all three frames is consistent
with exactly this.

Consequences: (a) **every `bbox_pitch` of the OFI run carries this systematic
error on top of everything else** — do not treat that export as a calibration
baseline; (b) the fix is two config keys, now committed
(`video_demo.yaml` sets `image_width/height: 1280/720` for the pitch and
calibration modules); (c) already-exported homographies are repairable in place
via `pipeline.calibration.rescale_homography` (H₇₂₀ = diag(1280/1920, 720/1080, 1)·H₁₀₈₀)
— implemented and tested (`test_calibration_quality.py::test_pixel_frame_mismatch_is_caught_by_residuals`).
Note the overlay lines in the annotated frames are *not* affected (the
visualiser scales normalised coordinates by the real frame size), which is why
the frames look plausibly annotated while the metric positions are worse than
they look.

## 4. Diagnosis: ranked failure hypotheses

Ranked by expected contribution to what we observed. Each carries the evidence
that would confirm or refute it on real data — all of it computable from the
§9 export with the implemented measurement layer, none of it needing labels.

**H0 — Pixel-frame mismatch (certain; our-config-specific).** Found by code
reading, §3.4. *Already confirmed statically.* Quantify on real data: residuals
of the exported H against keypoints re-expressed in true frame pixels collapse
after `rescale_homography` (the test demonstrates the signature). Fix shipped.

**H1 — Keypoint/line detector out of distribution (high, and worst exactly on
tight/near-goal framings).** Evidence for: hallucinated left-box lines on
frame 060; coherently-wrong box cluster on frame 800; Magera et al. 2025
document *unbounded* errors for keypoint models (PnLCalib) on OOD central
views; our footage (Greek Super League, this stadium, night lighting, 720p) is
not SoccerNet-Calib distribution. Confirm: per-frame detected-keypoint
residuals vs the *fitted* H stay large even for well-conditioned landmark sets;
hallucination rate = detections at image positions whose back-projection lands
>10 m from the landmark's true pitch location. Refute: if residuals on
detected keypoints are small and errors only appear at extrapolated landmarks
(then it's H2, geometry not detection).

**H2 — Degenerate landmark geometry on tight views (high; multiplies H1).**
All visible landmarks cluster in one box: the DLT is ill-conditioned and
extrapolates garbage while *interpolating* perfectly. Demonstrated
quantitatively in the synthetic harness: a near-goal fit with realistic noise
is **0.8 px accurate at the penalty spot and 159 px wrong at the centre spot**
(`test_calibration_camera.py::test_near_goal_fit_interpolates_but_extrapolates_garbage`).
Confirm on real data: `nullspace_gap` (second-smallest/largest eigenvalue of
the DLT normal matrix) collapses by ~1.5 orders of magnitude on the bad frames
(synthetic: 5e-2 wide vs 2e-3 near-goal); error grows with distance from the
support hull. Refute: bad frames with *wide, well-spread* inlier sets.

**H3 — Per-frame independence + integer-grid quantisation ⇒ jitter (high for
wobble, certain mechanism).** The ~2.7 px temporally-white quantisation floor
(§2.2), amplified through ill-conditioned fits, with zero temporal coupling.
BroadTrack (WACV 2025) independently documents NBJW's per-frame parameter
traces oscillating with the implied camera position wandering up to 20 m
within 30 s. Confirm: on static-camera spans (detected keypoints' pixel motion
≈ 0), `world_jitter_m` of the H sequence ≫ 0 — pure error by construction.
Synthetic reproduction: 0.72–0.99 m median frame-to-frame world jitter under
realistic noise. Refute: real static-span jitter at the few-cm level (it won't be).

**H4 — Completed-keypoint instability (medium-high; the outlier factory).**
Two-point `linregress` lines intersected at up to half-a-frame off-screen,
injected at p = 1.0, into a fit whose 50 px RANSAC threshold rejects nothing
(§2.3–2.4). Near-parallel pairs (box lines × goal line in a foreshortened
view) are precisely the near-goal case. Confirm: recompute completions from
the exported lines; error spikes correlate with frames whose H excursions are
driven by completed (not detected) points; refitting without completed points
removes the spikes. Refute: excursions persist with completions excluded.

**H5 — Unmodelled lens distortion (medium; a bias floor, not the wobble).**
nbjw_calib fixes all distortion to zero (§2.5). Literature: one radial
coefficient is the *largest single accuracy term* in BroadTrack's ablation
(JaC@5 42→54); Magera et al. 2024 show camera-model choice moves reconstructed
player positions by often >1 m and that WC14 homography "ground truth" cannot
even reproduce the imaged markings without k1. But distortion is temporally
*stable* — it cannot explain jitter, and our overlay skew (polylines between
keypoints) wouldn't render it visible. Confirm: on wide frames, systematic
bowed residual pattern of straight-line keypoints vs the fitted H,
strongest near frame edges. Caveat from TVCalib's own appendix: per-frame
joint distortion fitting collapses on low-FoV frames — if we model it, fit on
marking-rich frames and freeze.

**H6 — hold-last-homography across failures (low for this clip, real in
general).** `use_prev_homography: True` projects players through a stale
camera for however long the detector fails; on cut-heavy broadcast this
produces coherent-looking garbage. Confirm: teleport bursts
(`teleport_count`) coinciding with frames whose keypoint export is empty.

**What tells them apart in one sentence each:** H0 is killed by rescaling; H1
lives in residuals-at-detected-points; H2 lives in conditioning + error-vs-
distance-from-support; H3 lives in static-span jitter; H4 disappears when
completed points are excluded; H5 is a stable edge-correlated residual bow;
H6 is teleports aligned with detection gaps.

## 5. Literature (verified July 2026)

Same honesty standard as phase4-5-design.md §8: every claim below was verified
by fetching the source; where the record is thin we say so. Full citations in
§10.

### 5.1 Per-frame calibration: where nbjw_calib sits

*This subsection is being finalised against the SoccerNet-challenge research
pass — see §10 for the sources already verified.*

### 5.2 Temporal approaches — what exists, what it reports

- **The direct match to our symptoms is BroadTrack (Magera et al., WACV
  2025)**: semantic segmentation points + **Lucas-Kanade optical-flow
  correspondences back-projected through the previous camera** + a **tripod
  constraint** (pan/tilt axes meet at a point fixed for the whole match),
  Cauchy-robust LM over {f, k1, pan, tilt, roll, position}, initialised from
  the previous frame. On the sn-gamestate test set it **halves NBJW's mean
  reprojection error (10.28 → 5.02 px) and lifts JaC@5 37.1 → 56.9 at 100%
  completeness**, with visibly smooth parameter traces. Open source
  (evs-broadcast/BroadTrack); 16 fps on 2× RTX 4090.
- **Parameter-space smoothing is the published route, not H-entry smoothing.**
  The SoccerNet 2024 GSR *winner* (Constructor Tech) regresses decomposed
  (x, y, z, pan, tilt, roll, FoV) per frame and applies **Savitzky-Golay
  filtering with ±2°/±2 m clamps** at 0.5 s latency; the 2025 3rd place fixes
  position and smooths (pan, tilt, roll, f). Claasen & de Villiers (2024) is
  the one Kalman formulation over homography entries — but its dynamics are
  carried by an *estimated inter-frame affine transform*, not a naive
  random-walk on entries; it cut mean projection error 0.30 → 0.23 m and
  mostly removed outlier homographies. Citraro et al. (2020) particle-filter
  the decomposed pose (soccer mean translation error 9.77 → 4.90 m). Consistent
  pattern across all three: **temporal coupling improves means (outliers) far
  more than medians** — it is an outlier-killer and gap-bridger, which is
  exactly the failure profile we observed.
- **Keyframe/anchor + flow propagation** is published mainly as a patent
  (Stats LLC US 11,593,581: keyframe STN calibration + optical-flow
  interpolation) and as AuxFlow/Ziegler 2025-26 (anchor frames from NBJW
  keypoints + LK tracking + RANSAC updates; "consistent improvements in
  robustness and smoothness" on SoccerNet-GSR — abstract-level verification
  only, SSRN blocks fetching).
- **The PTZ prior is old, cheap, and effective**: Thomas (BBC R&D, 2007)
  clamps camera position (solved once per match from 10-20 wide frames,
  ±0.3 m) and estimates only (pan, tilt, f) per field — achieving ~0.02° pan
  noise *with no temporal filter at all*. Chen & Little (2019) constrain
  position with a Gaussian prior and fix mount rotation. BroadTrack's ablation
  quantifies the tripod term (+~2 JaC@5 alone, and it stops the focal-point
  wander); their "vertigo effect" caveat — position error absorbed by
  compensating focal length with overlays still looking right — is the sharpest
  known argument that *overlay-looks-fine is not a calibration metric*.
- **Degenerate views**: Nie et al. (WACV 2021) carry the only explicit
  degeneracy gate in this literature (grid-occupancy of inliers at 2×2/4×4/8×8 +
  temporal-consistency IoU; degenerate ⇒ discard inliers and lean on dense
  refinement + tracking loss). General theory (Acuna & Willert 2018) ties DLT
  accuracy to the data-matrix condition number — the formal basis for our
  `nullspace_gap` signal. For circle-dominated central views, Magera et al.
  2025 synthesise correspondences from conic geometry rather than smoothing
  through the gap. **Nothing published addresses behind-goal or
  penalty-box-only broadcast views specifically** — searched and absent;
  BroadTrack explicitly filters such cameras out of its evaluation scope.
  Closest tools: 2-point pan/tilt solvers under a known-position prior
  (Chen, Zhu & Little 2018; BroadTrack's reinitialiser), and goalposts as
  off-plane landmarks (Thomas 2007; SN-Calib annotates them; nbjw detects
  them but excludes them from the ground-plane fit by construction).
- **Cuts**: SoccerNet-GSR sequences are deliberately single-shot 30 s clips,
  so the challenge literature simply never faces cuts; the published pattern
  is a cheap self-diagnosis + re-anchor loop (BroadTrack's template-overlap
  confidence triggering 2-point re-initialisation, ~15-frame outages) or the
  patented trackable-frame classifier (Stats US 11,861,848). Our streaming
  design's segment machinery already provides the cut boundaries; the smoother
  consumes them as `reset()`.
- **Already-in-TVCalib/NBJW vs additive** (so we don't reinvent): TVCalib
  already does per-frame test-time optimisation over a camera-pose
  distribution with a rejection threshold τ, and *optionally* lens distortion
  (appendix-grade, collapses on low-FoV frames). NBJW/PnLCalib already do
  keypoint completion from lines and multi-mode voting. **Genuinely additive
  for our stack**: any temporal coupling at all, conditioning-aware quality
  gating, the fixed-mount prior, k1 distortion in the deployed module, and
  honest per-frame quality output.

### 5.3 What accuracy is achievable

BroadTrack's 5.02 px mean reprojection error on 1080p sn-gamestate footage and
Claasen's 0.23 m mean projection error on (re-annotated) WorldCup-class data
bound what's realistic for broadcast main-camera footage. No paper publishes a
"metres of player-position error at pitch scale" for the near-goal regime we
care about — that mapping is exactly what our measurement layer produces.

## 6. Measurement methodology (implemented: `pipeline/calibration/quality.py`)

There is no calibration ground truth for our footage, so "looks better" is not
a metric (the vertigo effect above makes it actively misleading). Everything
below is computable from a per-frame export of {H, camera params, detected
keypoints/lines} — no labels, no pixels.

### 6.1 The known-geometry principle
The pitch is a metric object we know exactly (105 × 68 m, boxes, spots, a
9.15 m circle — `pipeline/calibration/pitch.py`, names joined to the SoccerNet
vocabulary). Every check is some form of "does the estimate respect geometry
we know" — across space (residuals), conditioning (could anything *else* also
fit), physics (does the implied camera exist), and time (does it move like a
mounted camera).

### 6.2 The signal catalogue

| Signal | Function | Needs | Cheap? | Diagnostic of |
|---|---|---|---|---|
| Reprojection residuals (px and **metres**) | `reprojection_metrics` | H + keypoints | yes | H1 (detected-point errors), H0 (frame mismatch: metre-scale residuals) |
| Support geometry: n, principal spreads, collinearity, hull coverage | `support_metrics` | keypoints | yes | H2 (degenerate support) |
| **Conditioning**: DLT `nullspace_gap` | `fit_homography` diagnostics | keypoints | yes | H2 — the "residuals tiny, extrapolation garbage" detector |
| Physical plausibility: decomposability, camera height/roll/focal, horizon row, chirality, pitch-in-frame fraction, m/px scale | `plausibility_metrics` | **H only** | yes | garbage fits of every kind; works even before keypoint export exists |
| Static-span world jitter | `static_spans` + `world_jitter_m` | H sequence + keypoints | yes | H3 — on static spans, frame-to-frame world motion of fixed pixels is pure error |
| Teleport rate | `teleport_count` | player tracks | yes | H6, gross excursions (>12 m/s is not running) |
| Temporal pose deviation vs prediction | `score_frame(reference_pose_vector=…)` | H sequence | yes | single-frame excursions |

**Vanity signals we deliberately did not build:** overlay IoU eyeballing (not
a metric — vertigo effect); centre-circle ellipse checks as a *standalone*
invariant (any homography maps the circle to *some* ellipse — the constraint
only binds against detected circle points, which the reprojection residuals
already cover); full-pitch-area back-projection of the frame quad (blows up
past the horizon on every wide shot; replaced by the forward-projected
pitch-in-frame fraction).

### 6.3 The per-frame `calibration_quality` gate

`score_frame(...) → FrameCalibrationQuality` with `quality ∈ [0,1]` =
product of component scores (residual · support/conditioning · plausibility ·
temporal), each a named linear ramp in `CalibrationQualityConfig` (dataclass
config, every threshold a named field — same idiom as the detectors). Missing
*inputs* discount once (no keypoints ⇒ ×0.7 cap; no temporal reference ⇒
×0.85), so a bare-homography stream still gets a usable gate. Components stay
visible on the result so a rejected frame is explainable.

Calibrated against the synthetic regimes: median quality **0.85** on wide
shots, **0.79–0.85** during pans, **0.16** on the near-goal degenerate framing
— the OFI failure regime separates by >4× before any real-data tuning.
Thresholds carry the same status as detector defaults pre-validation:
engineering priors pinned by tests, to be re-fit on the first real export.

**Wiring into Phase 4** (the seam the design docs already budget for,
phase4-5-design.md §2.4): `TrackingFrame` grows an optional
`calibration_quality` field populated by the export adapter; the feature layer
multiplies it into the frame's data-quality score exactly like visibility and
ball-validity, and metre-sensitive features (`def_line_height`,
`def_line_velocity`, lane counts) are gated to None below a floor (proposal:
0.3) so detectors "discount, don't drop" per their existing contract. Deliberately
not implemented yet — it touches the Phase 4 contract, and we want the first
real export to set the floor before freezing it.

## 7. Temporal smoothing (implemented: `pipeline/calibration/smoothing.py`)

### 7.1 What is smoothed — the parameterization argument
Raw homography entries mix translation-scale (~10²) and perspective-row
(~10⁻³) magnitudes; isotropic noise models, gates, or gains on them are
dimensionally meaningless. This isn't hypothetical: upstream's own unused
`SequentialCalib` regularises 9-entry vectors against pixel residuals with no
relative weighting. The literature smooths **decomposed camera pose** (§5.2),
and so do we: state = (pan, tilt, roll, **ln f**, position), obtained by
closed-form decomposition of H under square-pixels/centred-principal-point —
the same assumptions nbjw itself bakes into calibrateCamera. ln f makes zoom
gates relative; position gets a near-zero gain because a broadcast main camera
is a fixed mount (the strongest prior we own, and the one nbjw ignores).
Decomposition *failure* is itself a quality signal (garbage H's routinely
don't decompose to a camera above the pitch).

### 7.2 Mechanics
Per-parameter alpha-beta filter with innovation gating, quality-weighted gains
(a 0.2-quality frame nudges; a 0.9-quality frame corrects), COAST on rejection
or missing measurement (velocity decays), **REANCHOR** when `reanchor_run`
consecutive rejections agree with each other (whip pan or filter divergence —
snap, don't fight), **LOST** after `max_coast_frames` (~1 s) — the frame is
honestly uncalibrated rather than silently interpolated, which is the direct
repair for nbjw's hold-last-H-forever. `reset()` at every segment boundary:
nothing bridges a broadcast cut (the segment/cut machinery from the streaming
design provides the boundaries; restart-detection's cut-spanning exemption
does not apply here — a camera pose is continuous evidence).

Batch mode runs forward+backward causal passes and averages (cancels phase lag
to first order); live mode is the forward pass alone — same parity idiom as
the streaming detectors.

### 7.3 Measured on synthetic trajectories (200-frame broadcast script:
static wide → pan+zoom → static near-goal → pan back, realistic noise:
1.5 px Gaussian + 2.7 px quantisation + 15% dropout):

| Metric | Raw per-frame | Smoothed (batch) |
|---|---|---|
| Pan RMSE, common frames | 2.38° | **1.40°** |
| Static near-goal world jitter (median m/frame) | 0.99 m | **0.11 m** |
| Static wide world jitter | 0.72 m | 0.43 m |
| Injected garbage frame excursion | full | gated (COAST), <1° leak |

The jitter numbers are the headline: on static spans the smoother removes
85–90% of the frame-to-frame wobble that manufactures phantom velocities.
These are synthetic-fixture numbers — same epistemic status as every detector
test in this repo (STATUS.md caveat applies verbatim).

## 8. Proposal, ranked by expected improvement / effort

1. **Fix the pixel-frame mismatch — DONE (config change).**
   `video_demo.yaml` now pins `image_width/height` to the clip resolution;
   `rescale_homography` repairs any already-exported H. Effort ~zero; removes
   a systematic, possibly dominant metre-scale distortion from every position.
   *Class: config change (committed).*
2. **Export per-frame calibration state and run the measurement layer on the
   real clip.** Everything in §4 becomes measured instead of hypothesised;
   thresholds in §6.3 get re-fit; this decides everything below. Needs a small
   serialisation patch (§9). Effort: small. *Class: sn-gamestate patch +
   offline analysis (analysis code already implemented here).*
3. **Tighten the per-frame fit via `patch_sn_gamestate.py`:** RANSAC threshold
   50 → 5 px in the wrapper's `get_homography_from_ground_plane` call, and
   stop trusting completed keypoints at p = 1.0 / half-a-frame off-screen
   (either drop off-frame completions or exclude completions from RANSAC
   consensus scoring). Kills the H4 outlier factory feeding H1 hallucinations
   into the fit. Effort: small, isolated, A/B-able in one GPU session.
   *Class: sn-gamestate patch.*
4. **Wire quality gate + smoother onto the exported stream** (offline
   post-process first, the causal path already exists for live parity):
   decompose → `score_frame` → `CameraSmoother` → recomposed H → repaired
   `bbox_pitch` + per-frame `calibration_quality` into the Phase 4 adapter.
   Implemented and synthetic-tested here; remaining work is glue on real
   export format. Effort: small-moderate. Expected: the §7.3 numbers, i.e.
   jitter mostly gone, single-frame excursions gated, near-goal frames
   honestly flagged instead of silently wrong. *Class: new component in this
   repo (done) + thin adapter (pending).*
5. **A/B pnlcalib and k1 distortion on the same clip.** sn-gamestate ships a
   `pnlcalib` module set (NBJW successor; our patch script already covers its
   indexing bug). Literature says one radial coefficient is the largest single
   accuracy term on SoccerNet data; measure, don't assume — and if modelling
   distortion, fit it on marking-rich frames and freeze (TVCalib's low-FoV
   collapse). Effort: config swap + one GPU session. *Class: config change +
   measurement.*
6. **Adopt BroadTrack as the calibration module** if, after 1–5, measured
   accuracy on good frames is still the binding constraint (rather than
   coverage of bad frames). Best published match to our symptoms, open source,
   evaluated against NBJW on sn-gamestate data; integration cost is real
   (TrackLab module wrapper, GPU budget: 16 fps on 2×4090 — offline-viable on
   Colab, not live). *Class: genuinely new component. Decision explicitly
   deferred until the measurement exists.*

**The honest fallback, stated plainly:** for near-goal/tight framings there is
no published method that makes a *good* full-pitch homography out of one box's
landmarks — the geometry doesn't support it (H2), and the field's own answer
is priors + propagation + rejection, not per-frame heroics. If the export
confirms §4, the correct behaviour on those frames is: smoothed/coasted pose
while the camera physics allows it, then **LOST + `calibration_quality` ≈ 0 —
and Phase 4 discounts or skips those frames**, which its §2.4/§3.1 contract
was designed for from the start. Discarding measured-bad frames is a result,
not a failure. What the fixed-mount prior *does* license on those frames is
far better than nothing: pan/tilt/zoom change but position doesn't, so a
coasted position + 2-point-style update from whatever two landmarks exist
(goal posts included) keeps ball-local geometry usable even when the far
half-pitch is unknowable. That is the v2 upgrade path (BroadTrack's
reinitialiser is exactly this), noted but deliberately not built yet.

## 9. Next GPU session: the concrete experiment

Goal: turn §4 into measurements on the OFI clip. One session, in order:

1. **Re-run with the fixed config** (already committed; setup.sh path
   unchanged, STATUS.md recipe applies). Sanity-check the minimap: player
   spread should now match on-screen spread (§3.4 signature gone).
2. **Export per-frame calibration state — already wired.**
   `patch_sn_gamestate.py` step 4 (applied automatically by `setup.sh`) makes
   `NBJW_Calib.process` append one JSON line per frame to
   `calibration_dump.jsonl` in the hydra run directory: frame id, pixel-frame
   size, keypoints in that pixel frame, detected lines, and the image→pitch
   homography (null on failure). Camera pose is re-derived offline by
   `decompose_homography`, so nothing else needs to be serialised. Verified
   against the pinned sn-gamestate clone (applies, compiles, idempotent).
   Retrieving this file is the session's non-negotiable output. Additionally
   set `state: {save_file: states.pklz}` to capture per-detection `bbox_pitch`
   for the teleport metric.
3. **Run the measurement layer offline** (no GPU needed, this repo):
   per-frame `score_frame` with correspondences from the exported keypoints;
   `static_spans` + `world_jitter_m` over the whole clip; `teleport_count`
   per track from `bbox_pitch`. Deliverables: quality time-series aligned to
   the annotated video; component breakdown for frames 060/430/800
   specifically; jitter medians on static spans; per-hypothesis verdict table
   (§4 confirm/refute columns filled in).
4. **A/B the cheap fixes on the same frames** (H4/H3): rerun with RANSAC 5 px
   patch and with completions-untrusted patch; compare quality distributions
   and static-span jitter against baseline. Decide proposal items 3-5 with
   numbers.
5. **First hypothesis to test is H2-vs-H1 on frame 800's segment** (the
   ranked-top unknown): from the export, compute residuals at *detected*
   keypoints vs `nullspace_gap`. If residuals are small while the gap has
   collapsed → geometry (H2) dominates → gating+smoothing is the right
   spend. If residuals at detected points are themselves large → detector OOD
   (H1) dominates → the ceiling argument for BroadTrack/pnlcalib strengthens.

Total GPU time: one video_demo run per config variant (~10 min each on T4 for
858 frames) — well within a single Colab session.

## 10. References (verified July 2026)

Fetched and read unless marked otherwise; [industry]/[patent] as flagged.

- Gutiérrez-Pérez & Agudo. *No Bells, Just Whistles: Sports Field Registration
  by Minimalist Keypoints* (CVPRW 2024) / *PnLCalib* (CVIU 2026).
  arXiv:2404.08401. The deployed method; per-frame, keypoints + completion.
- Magera, Hoyoux, Barnich, Van Droogenbroeck. *BroadTrack: Broadcast Camera
  Tracking for Soccer.* WACV 2025, arXiv:2412.01721. Tripod constraint + LK
  flow + k1; NBJW jitter documented (Fig. 3); JaC@5 56.9 vs NBJW 37.1 on
  sn-gamestate test; open source.
- Golovkin et al. (Constructor Tech). *From Broadcast to Minimap.*
  arXiv:2504.06357. SoccerNet-GSR 2024 winner; Savitzky-Golay over decomposed
  7-param pose, ±2°/±2 m clamps, 0.5 s latency.
- Claasen & de Villiers. *Video-based Sequential Bayesian Homography
  Estimation for Soccer Field Registration.* ESWA 252:124156 (2024),
  arXiv:2311.10361. Two-stage KF; mean projection error 0.30→0.23 m; filters
  are outlier-killers (mean improves ≫ median).
- Nie, Chen, Hamid. *A Robust and Efficient Framework for Sports-Field
  Registration.* WACV 2021. Grid-occupancy degeneracy gate; tracking loss;
  keypoint-only 50.6 m → full 0.22 m mean projection error (American football).
- Citraro et al. *Real-time camera pose estimation for sports fields.* MVA
  31:16 (2020). Particle filter over pose; soccer mean translation 9.77→4.90 m.
- Thomas. *Real-time camera tracking using sports pitch markings.* J.
  Real-Time Image Proc. 2(2-3) 2007 / BBC WHP 168. Fixed-position PTZ prior;
  ±0.3 m position from 10-20 frames; 0.02° pan noise with no filter; goalposts
  for near-goal shots; pitch crown ≈ 3 px systematic warning.
- Theiner & Ewerth. *TVCalib.* WACV 2023, arXiv:2207.11709. Segment-loss
  test-time optimisation over camera distribution; τ rejection; distortion
  optional and fragile on low-FoV (their Appendix D).
- Magera et al. *A Universal Protocol to Benchmark Camera Calibration for
  Sports.* CVPRW 2024. k1 changes reconstructed player positions by often
  >1 m; WC14 homography GT cannot reproduce imaged markings.
- Magera et al. *Can Geometry Save Central Views for Sports Field
  Registration?* arXiv:2504.20052 (2025). Conic-derived correspondences;
  keypoint detectors show unbounded OOD errors on central views.
- Chen & Little. *Sports Camera Calibration via Synthetic Data.* CVPRW 2019.
  Pose engine with Gaussian position prior, PTZ-only freedom.
- Chen, Zhu & Little. *A two-point method for PTZ camera calibration in
  sports.* WACV 2018 [abstract-level]. Two correspondences suffice under a
  known-position prior.
- Lu, Chen & Little. *Pan-tilt-zoom SLAM for sports videos.* BMVC 2019
  [abstract-level + BroadTrack's independent evaluation: fixed-focal-point
  assumption measurably hurts].
- Guo et al. *PTZ-Calib.* arXiv:2502.09075 (2025). Offline PTZ bundle
  adjustment + online relocalisation.
- Sinha & Pollefeys. *Pan-tilt-zoom camera calibration and high-resolution
  mosaic generation.* CVIU 103(3) 2006. PTZ self-calibration incl.
  distortion-vs-zoom.
- Mahony, Hamel, Morin, Malis. *Nonlinear complementary filters on the special
  linear group.* IJC 85(10) 2012. SL(3) filtering exists; **no sports
  application found** — gap, not standard.
- Acuna & Willert. *Insights into the robustness of control point
  configurations for homography and planar pose estimation.* arXiv:1803.03025.
  DLT accuracy governed by data-matrix conditioning.
- Somers et al. *SoccerNet Game State Reconstruction.* CVPRW 2024,
  arXiv:2404.11335. The GSR task/dataset (30 s single-shot clips — no cuts by
  construction); baseline falls back to median camera position when markings
  are insufficient.
- SoccerNet 2024/2025 Challenges Results. arXiv:2409.10587, arXiv:2508.19182.
- Ziegler et al. *AuxFlow / anchor-frame LK propagation over NBJW keypoints.*
  SSRN 5607133 / CVIU 264 (2026) [abstract-level only — SSRN blocks fetching].
- Stats LLC patents [patent]: US 11,593,581 B2 (keyframe calibration +
  optical-flow interpolation); US 11,861,848 B2 (trackable-frame
  classification for broadcast).
- Beetz et al. *ASPOGAMO.* IJCAI 2007 [abstract-level]. Iterated EKF over
  camera parameters with optical flow.
- Hayet, Piater, Verly. *Incremental rectification of sports fields.*
  ACIVS/BMVC 2004 [abstract-level]. Earliest soccer incremental propagation.
