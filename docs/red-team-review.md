# Red-team review

_2026-07-17. Scope: all seven docs in /docs, all of /pipeline (source + fixtures + tests), /research
(patches, config, probe). Method: full read of the source, suite executed (241 passed, 1.8 s).
Standard applied: judge it as inherited code. No praise; findings only._

The organizing fact, restated once: three real-data findings exist, all three were
wiring-class ("it doesn't work" = "it was never connected correctly"), and everything past
stage 4 has been validated only against fixtures authored by the same model that wrote the
detectors. This review hunts for the fourth wiring bug, the fixtures that prove nothing, and
the assumptions that die on first contact.

---

## 1. What breaks first

Ranked by P(wrong) × downstream blast radius.

### 1.1 "Broadcast cuts leave gaps in the tracking stream" — nothing produces those gaps

**The assumption.** The entire batch stack assumes the input frame stream has *holes* at
broadcast cuts: `tracking.py:117-123` ("Frames are NOT guaranteed contiguous — broadcast cuts
leave gaps"), `features.py` module docstring, phase4-5-design.md §1.2 "data reality #3", and
every "episodes never span segments" guarantee. Segmentation is implemented purely as
timestamp-gap splitting: `features.py:466` and `kinematics.py:272` split when
`Δt > max_gap_s` (1 s).

**Why it's wrong.** The deployed sn-gamestate stack emits detections for **every decoded video
frame** — replays, close-ups, crowd shots included. Timestamps are `frame_index / fps`,
continuous by construction (the folder-of-frames patch makes the whole clip ONE video
precisely so tracking never resets). Nothing in the deployed pipeline classifies or drops
non-pitch frames; the unimplemented adapter's responsibility list (`tracking.py:128-144`)
does not mention cut detection; the live design has a cut gate (§5) but it is design-only and
batch has no counterpart. "Data reality #3" is not a reality — it is an unbuilt requirement.
The OFI clip never exposed this because a 34 s GSR-style clip is single-shot (the calibration
doc itself notes SoccerNet-GSR sequences are "deliberately single-shot 30 s clips — no cuts by
construction", §5.2/§10).

**What real data reveals.** On any real multi-shot broadcast: one giant segment; replays
ingested as continuous play (time-travelling positions — the exact "phantom segment" failure
the live doc describes in §5.1 and batch does nothing about); close-ups producing 2-3 giant
bboxes projected to garbage pitch coordinates *inside a live possession window*; the
calibration smoother's `reset()`-at-cuts contract never invoked. Every detector, the
possession machine, counter-attack windows, and calibration smoothing corrupt silently.

**Cheapest experiment.** No GPU needed beyond the next planned run: feed 3-5 minutes of real
multi-shot broadcast (not the single-shot clip) through stages 1-4, export, run
`FeatureExtractor`, and count segments. Prediction: 1. Then plot `teleport_count` around the
known cut timestamps. This is the strongest candidate for structural bug #4 (see §3.1).

### 1.2 Every confidence threshold was calibrated in a quality = 1.0 world

**The assumption.** Confidence bands transfer from fixtures to reality: tests define
"confident" as ≥ 0.5 (`test_detectors.py:50-53`), scouting counts a sighting at
`mean_confidence ≥ 0.30` (`aggregate.py:55`), live confirmation requires raw ≥ 0.40
(`priors.py:52`).

**Why it's wrong.** `quality.score = min(nh,8)/8 × min(na,8)/8, ×0.5 if no ball`
(`features.py:811-814`), and `episode_confidence` multiplies mean quality in directly
(`detectors/base.py:177-180`). Every pattern fixture has all 20 outfielders visible and a
ball in every frame → quality is exactly 1.0 in every test that pinned these bands. Real
broadcast (the docs' own numbers): 12-16 of 22 visible → per-team 6-8 → quality ≈ 0.55-0.75
with a ball, ≈ 0.3 without one (today's reality). The entire confidence distribution shifts
down ~0.6-0.7× *before any detector is wrong about anything*. Consequences: events that
should be confident land under the scouting presence gate → prevalence reads ~0 →
`TeamScoutingProfile` says the opponent "never presses"; live CONFIRMED starves against the
0.40 raw floor. Not noise — a systematic scale error between the synthetic and real worlds.

**Cheapest experiment.** Offline, first real export: histogram `quality.score` per frame;
re-run the synthetic suite with fixture quality forced to the real median and count which
assertions still pass. Zero GPU.

### 1.3 `def_line_height` silently degenerates exactly when the high press needs it

**The assumption.** phase4-5-design.md §2.1: line height "only meaningful when the back line
is on-camera — every detector must gate or discount"; `high_press.py:111` docstring: "missing
back line => mild discount, not a veto".

**Why it's wrong.** There is no gate. `features.py:733` sets `def_line_height = xs_rel[1]`
(second-deepest **visible** outfielder) whenever ≥ 5 players are visible. The value is never
*missing* in the promised sense — it is *wrong*: with the pressing team's back line
off-frame (common when the camera sits on the opponent's box, i.e. during a press), the
deepest visible players are the pressers themselves, so the "line" reads 70-90 m and the
`≥ 40 m` factor — the sole discriminator that keeps this detector from being the bare
proximity rule Bauer & Anzer showed over-detects (72% of transitions) — saturates to ~1.0
precisely when visibility is worst. The MISSING_FEATURE_FACTOR discount path is effectively
dead code for this feature. Same corrupted series feeds `def_line_velocity` (the offside
trap's core signal) via `_post_derivatives`.

**What real data reveals.** High-press false positives on any sustained opponent-box
pressure; offside-trap "step-ups" manufactured by visibility churn (a defender walking into
frame drops `xs_rel[1]` instantly; one walking out raises it — a discontinuity the
derivative reads as line motion).

**Cheapest experiment.** From the first real export: per team, plot `def_line_height`
against `n_visible_outfield`; count discontinuities > 5 m coinciding with visibility count
changes. Pure offline.

### 1.4 `boundary_noise_m = 0.5` is contradicted by this repo's own calibration numbers

**The assumption.** All Tier 1 exit evidence scores excursions against a 1σ = 0.5 m position
noise near lines (`kinematics.py:39`); "definite out" is 1.0 m = 2σ (`restarts.py:43`). The
event doc calls this "measurement debt" but Tier 1's "high precision" tier claim (§2) is
priced on it.

**Why it's wrong.** The calibration work in the same repo measured 0.72-0.99 m **median
frame-to-frame world jitter** on static-camera synthetic spans under realistic noise
(calibration-design.md §7.3), documented metre-scale bias regimes (§3.4 pixel-frame bug;
§4 H2's 0.8 px→159 px interpolate/extrapolate asymmetry), and the far touchline — where
throw-ins are detected — is the *most extrapolated, worst-conditioned* region of every
main-camera homography. A realistic touchline 1σ of 1.5-2.5 m puts "definite out" inside
noise: spurious exits fire on jitter, and the noise-guard (`restarts.py:227-233`) only
suppresses those that return within 2 s. The two sibling docs never confront each other on
this number.

**Cheapest experiment.** Already scripted: ball-at-rest scatter at known restart points
(ball doc §6.4, ~30 min labelling), or static-span `world_jitter_m` evaluated at
boundary-adjacent pixels from the calibration export. Replaces the constant with a measured
value before Tier 1 is trusted at all.

### 1.5 Possession goes CONTESTED exactly when a real press succeeds

**The assumption.** During a press, possession inference keeps reporting the pressed team so
`frame_score` has a side to score against (`high_press.py:58-60` returns None whenever
possession side is None).

**Why it's wrong — and the fixture hides it.** The high-press fixture *scripts pressers to
stop 2.8 m from the holder*, with a comment saying why: "never inside the 2m possession
radius, so possession stays cleanly AWAY" (`patterns/testing/synthetic.py:143-171`). Real
pressing puts a defender inside 2 m constantly; `features.py:598-603` then reports CONTESTED
→ score None → the episode ends unless the contested spell is shorter than `merge_gap_s`
(2 s). A good press — sustained duels, back-and-forth touches — fragments or vanishes. The
detector's recall is anti-correlated with press quality. (DEAD/CONTESTED handling is also
why `counter_attack` windows `break` on the first opponent frame — one misattributed
possession frame mid-counter kills the window, `counter_attack.py:133-135`.)

**Cheapest experiment.** First real export: mark 3-4 pressing spells by eye in the annotated
video, measure the CONTESTED fraction and the score-None run lengths inside them.

### 1.6 Shot outcomes: first restart within 90 s wins

**The assumption.** A kickoff following a shot corroborates a goal (`shots.py:44`
`restart_corroboration_s = 90.0`; `shots.py:199-210`).

**Why it's wrong.** `_resolve_outcome` takes the *first* KICKOFF/GOAL_KICK/CORNER whose
`start_s` falls in `(t, t+90]` — with no requirement that play stopped in between, no
period check, and the kickoff branch evaluated *before* the blocked/saved branches. Real
matches produce a restart roughly every 30-60 s, so: every blocked or saved shot in the
minute before an actual goal is labelled `goal` (0.45); shots followed within 90 s by an
unrelated goal kick become `off_target` (0.6). Fixtures never stressed this because each
scenario contains exactly one restart. Outcome labels on real data will be
restart-proximity noise wearing confidence numbers.

**Cheapest experiment.** Pure synthetic, no GPU: add a fixture with a blocked shot 60 s
before a scoring sequence; assert the blocked shot's outcome. It will fail today.

### 1.7 Launch detection SNR at 5 Hz on real ball noise

**The assumption.** A kick is a speed step ≥ 1.8 m/s per grid step (≈ 9 m/s²,
`kinematics.py:42`), detectable after 0.4 s smoothing.

**Why it's suspect.** With realistic per-frame ball position noise σₚ (0.3-1.0 m once a
detector exists; more via homography error), finite-difference speed noise at 5 Hz is
σₚ·√2/0.2 ≈ 2-7 m/s per step — at or above the trigger. Every Tier 2 event and every Tier 1
resumption is keyed on launches; the doc prices the *recall* floor (soft kicks smeared away)
but not the *false-positive* ceiling (noise steps everywhere). Expect either hundreds of
launches per minute or, after re-tuning upward, missing most passes. This is the
"detection rate decides everything" argument of the ball doc applied one derivative higher.

**Cheapest experiment.** Offline once the probe runs: `detect_launches` over the joined
probe stream; launches/minute against a sane 10-20.

### 1.8 Lower-ranked but real

- **Track traits "last wins"** (`features.py:482`): one late ID switch reassigns a track's
  team for the *whole segment*, retroactively corrupting every shape aggregate. Fixtures
  have no ID switches.
- **Canonical 105 × 68 pitch** assumed everywhere (`tracking.py:20-21`; hardcoded `105.0` in
  `passes.py:329` ignoring `meta.pitch_length_m`). Real pitches vary 100-110 × 64-75; nbjw
  fits a template, so the error mode is a uniform scale bias on all metre thresholds for
  non-standard grounds. Unmeasured for the footage's actual stadiums.
- **Static `hidden_tracks`**: fixture visibility never *changes mid-scenario*, so centroid /
  width / hull discontinuities from players crossing the frame edge — the dominant real
  noise source for shape features — are exercised nowhere in the test suite.

---

## 2. The synthetic-fixture audit

Structural properties shared by all pattern/event fixtures, before per-fixture verdicts:

1. **Generated at exactly `resample_hz`** (5 Hz, `build_match` default) — stage 1 resampling
  is an identity map in every test. Interpolation, nearest-obs snapping, and gap-splitting
  logic have never processed a timestamp they weren't built from.
2. **Zero positional noise, zero ID switches, zero team misclassification, ball present in
   every frame at confidence 1.0** (patterns fixtures; events fixtures add ball-hidden spans
   but still no noise). The calibration and ball fixtures DO inject noise, quantisation,
   dropout, and hallucinations — the discipline exists in the repo; the tactical fixtures
   just don't apply it.
3. **Quality pinned at 1.0** (§1.2) — every confidence assertion in the suite is scale-bound
   to a regime real data cannot produce.
4. **Single segment, no cuts** — the segment-locality machinery and the cross-cut restart
   linking (the one detector that *should* span cuts) are tested only in the trivial case.
5. **Signals placed 10-50% past their thresholds** — the tests pin *ordering* semantics
   (fires on A, silent on B), which is genuinely valuable, but almost none would catch a
   20% threshold regression. They are semantic pins, not sensitivity tests, and should not
   be described as validation.

Most-distrusted fixtures, ranked:

| Fixture | Why it's near-tautological | What a real clip contains that it doesn't |
|---|---|---|
| `line_step_scenario` (`synthetic.py:403-450`) | Back four rises 4.5 m in 1.5 s at exactly 3 m/s (2× threshold), perfectly flat, zero noise, on a series whose derivative the calibration doc says is dominated by phantom velocity from ±0.3 m jitter. This fixture passes with *any* detector that computes a derivative; it cannot distinguish the shipped heuristic from a broken one. | Line noise the same order as the signal; visibility churn in the back four; steps that are 60% of threshold; drops-then-steps. |
| `high_press_scenario` (`synthetic.py:139-200`) | Pressers scripted to *stop outside the possession radius* so the possession machine never enters the CONTESTED state a real press produces (§1.5) — the fixture encodes the detector's input requirements, not a press. Line height fully visible, so §1.3 never triggers either. | Duels inside 2 m; back line off-frame; the ball actually moving under pressure. |
| `counter_attack_scenario` (`synthetic.py:457-526`) | The ball teleports 3 m from the AWAY carrier to the HOME interceptor between waypoints (t=8.0→8.2, `synthetic.py:513-522`) — an interception with zero loose-ball time; runners then sustain 6.7 m/s for 6 s (elite sprint, sustained, unsmoothed). The 2 s persistence hysteresis is fed the cleanest possible signal; the `flicker` negative tests a 1 s flicker but nothing in the 1.5-2.5 s band where the boundary actually lives. | Scrambles, deflections, 50/50s; smoothed velocities that undercount runners at the 5.5 m/s gate (the doc's own §9 caveat (b)). |
| `throw_in_scenario` + Tier 1 fixtures (`events/testing/synthetic.py`) | Excursions are exact multiples of an *assumed* σ (0.5 m), with zero injected boundary noise; the confidence-band assertions test the soft-scoring arithmetic against a noise model no measurement supports (§1.4). Asserting "0.3 m excursion → moderate confidence" is asserting the code implements its own formula. | Jittering coordinates near the worst-calibrated region of the homography; balls lost *before* the line; exits during pans. |
| `low_block` / `mid_block` | Legitimate as semantics pins (the cleanest of the five), but the adaptive hull quantile is computed over a clip that is ~75% deep block — the adaptive path is validated against a distribution no real match produces. | A full match's hull distribution; being pinned back vs choosing to block (acknowledged unknowable, fine). |

The events fixtures deserve one credit and one charge: credit — deliberate negatives and
confidence-band assertions are the right idea. Charge — `flight()` in the ball fixtures
models an airborne ball as a clean constant-velocity ground track, which is the *charitable*
reading of the airborne-projection problem; the real artefact is a position bias of metres
toward the far side of the camera ray (ball doc §2.3's own h·d/(H−h) formula) that no
tactical or event fixture injects anywhere, despite passes and shots being precisely the
airborne cases.

---

## 3. More structural bugs — hunting the fourth

Same class as the three known ones (hardcoded frame, silent class filter, coordinate-frame
mismatch): things that are *wired wrong or not wired*, invisible until measured.

### 3.1 The batch pipeline has no cut classifier and its data model requires one

Full argument in §1.1. Classified here because it is exactly the pattern: a structural
absence (like the ball detector) that every downstream doc treats as a solved input.
Verdict: **confirmed by reading; will manifest on the first multi-shot clip.**

### 3.2 The 1080p fix re-hardcoded 720p

`video_demo.yaml:60-61` pins `image_width: 1280, image_height: 720` — in a config whose own
header says it exists "for running on an arbitrary broadcast clip". The repair for
"upstream hardcodes 1920×1080" is "we hardcode 1280×720". Run any 1080p clip through this
config and the identical silent distortion returns, mirrored, with no runtime check — the
patch could trivially assert decoded-frame size == configured size and does not
(`patch_sn_gamestate.py` step 4 dumps `self.image_width`, i.e. the *configured* size, not
the decoded one). Worse, because the dumped keypoints are denormalised into the same
configured frame as the fitted H, **the exported evidence is self-consistent under a
recurrence of H0 — reprojection residuals cannot catch a repeat of the pixel-frame bug**
(the residual test that "catches" H0 works by re-expressing keypoints in the true frame,
which requires knowing the true frame out-of-band). Verdict: **confirmed latent; the sharp
edge that already cut once is still exposed.** Fix class: one assert + dump the decoded
size. 

### 3.3 Inference at imgsz 640 also degrades the *person* channel

The ball doc (§2.3) documents the ultralytics 640 default for the ball; nobody has asked
what it does to far-side player detection at 720p→640 (distant players are ~15-25 px).
The canonical "12-16 of 22 visible" partial-observability figure is treated as a camera
property, but part of it may be a resolution artefact of an inherited default — the same
"upstream benchmark decision we inherited without noticing" class. Cheap check: the probe
already records `n_persons` at both 640 and 1280 (`ball_probe.py:82-96`); diff the two
columns. Zero extra GPU cost — the data will already exist after the next session.

### 3.4 The probe↔calibration join can misalign silently

`ingest.load_homographies` falls back to *enumeration order* when a dump line's frame id
doesn't parse (`ingest.py:78-84`); the calibration dump writes lines inside
`try/except: pass` (`patch_sn_gamestate.py:188-203`), so a single serialisation hiccup
drops a line and shifts every subsequent order-keyed join by one — the same silent-failure
idiom that hid all three known bugs. The probe side keys by sorted-filename position and
converts to time as `fr.index / fps` with fps defaulted to 25 (`ingest.py:104,117`) while
the analysis grid is 5 Hz — a wrong fps or a dropped line poisons every implied-speed and
teleport metric downstream. Verdict: **plausible, cheap to harden** — assert
line-count == frame-count at load, and parse-don't-fallback.

### 3.5 Smaller confirmed items

- `high_press` "trigger: restart" metadata scans only the current segment for a preceding
  DEAD spell (`high_press.py:209-214`), but the event doc §4.3 states the broadcast almost
  always cuts between out-of-play and restart — on real footage the DEAD spell lives in a
  different segment and the tag will read `open_play` near-always. Doc-vs-code
  contradiction with a silent wrong-answer failure mode.
- `restarts.py:416` identifies outfielders by `tr.role.value == "outfield"` (string
  compare) where every sibling uses the enum — works today, breaks silently if `Role`
  values are ever renamed.
- `passes.py:329` hardcodes `105.0` for the pitch length in through-ball geometry.
- STATUS.md:37-43 says setup applies "the three custom-video patches";
  `patch_sn_gamestate.py` applies four. Whoever debugs a missing `calibration_dump.jsonl`
  from the STATUS description will look for three.

---

## 4. Over-engineering

Judged by one question: what is the minimum that must be true for this to earn its place,
and how far is that from being known?

| Layer | Size (src) | Verdict | Reasoning |
|---|---|---|---|
| Streaming/live (`patterns/streaming/`) | ~1,900 lines + ~1,100 test lines | **Defer hard; stop investing** | Built to broadcast tactical alerts in real time, before a single batch run has produced output anyone has looked at. The shipped "causal core" cannot process one `TrackingFrame` end-to-end: `StreamingFeatureExtractor.push` is `NotImplementedError` (`streaming/features.py:150-163`) — what is tested are lifecycle machines fed hand-computed scores. The parity principle is good *design*; the earn-in conditions are (a) one real batch match profile a human found useful, (b) measured live-profile GPU throughput ≥ 2 Hz (currently an estimate stack, live doc §1). Neither is scheduled. The three remaining detector wrappers, `pipeline/live/`, and the WS contract should not be touched until both exist. |
| Event inference Tier 2 (`events/passes,shots,setpieces`) | ~1,200 lines | **Freeze** | Has literally no input: pass/shot detection consumes a ball stream that does not exist (ball doc §8: "until proposal 2 lands, Tier 2 has literally no input"). The code is written and green against fixtures with a perfect ball; nothing it can do until a probe shows a detection rate. Sunk cost — keep, don't extend, and don't let its green tests count as progress. The §8 possession refactor being unimplemented is correct; keep it that way. |
| Event inference Tier 1 (restarts) | ~500 lines | **Freeze, re-scope after noise measurement** | Exit evidence is ball-gated (gone today) and σ-priced on an unmeasured constant its sibling doc undercuts (§1.4). Restart *morphology* (person-signals) survives; the honest v1 might be morphology-only. |
| Scouting (`scouting/`) | ~800 lines | **Keep the math, defer the priors** | Aggregation/Wilson/n_eff is small, pure, auditable and genuinely footage-independent — the cheapest layer to have built early. The *priors* mechanism is a bounded display nudge for a live system that doesn't run, parameterized by an invented league baseline (p₀ = 0.35 flat, `priors.py:51`) and calibrated thresholds (`presence_min_confidence = 0.30`) that §1.2 shows won't survive the real confidence scale. Structurally safe by design — also structurally pointless until there are profiles from real matches. |
| Recommendations (`recommendations/`) | ~350 lines, skeleton | **Keep frozen** | Correctly deferred (NotImplementedError with pinned signatures). The one risk is documentary: a reader of `pipeline/README.md` could believe Phase 5 is nearer than it is. |
| Calibration + ball measurement layers | ~1,900 lines | **Keep — this is the part that earns its place now** | Both are measure-first, consume the *next* GPU session's export directly, and encode falsifiable predictions (H1-H7 with confirm/refute columns). This is what the rest of the repo should have looked like. |

The aggregate picture: of ~6,500 source lines in /pipeline, roughly two-thirds consume
inputs that do not exist yet (real tracking export, a ball, a live stream, an event-fed
possession machine), and one-third measures whether those inputs *can* exist. Two days of
six parallel sessions produced the two-thirds; the ratio is inverted relative to the risk.

---

## 5. Inter-doc contradictions

Reported, not fixed. The consolidation pass found C1; parallel sessions produced more of the
same class.

1. **Possession hysteresis: prose says 0.5-1 s, everything else says 2 s.**
   phase4-5-design.md §2.2 ("a new holder must be established for ~0.5-1 s before the state
   flips") vs `FeatureExtractorConfig.turnover_persistence_s = 2.0` (`features.py:324`), the
   live doc's latency table ("+2.0 s possession persistence"), event doc §8.3, and the
   streaming default. Exactly C1's failure mode — §9-table-vs-§3-prose drift — in the
   section family the reconciliation pass declared fixed.
2. **STATUS.md:12 "199 passed" vs 241 actual** (verified this review). The same file's 🟡
   section lists the ball layer's 42 tests — the snapshot is internally inconsistent, and
   "the blunt version" undercounts its own suite by 20%.
3. **docs/README.md counts drift twice in one file**: line 14 "The five design docs" heads a
   six-row table; line 45 "I cross-checked all four docs" precedes a reconciliation over six.
4. **"Three patches" vs four**: STATUS.md:37-43 and its recipe describe three auto-applied
   patches; `patch_sn_gamestate.py` ships four (the §9.2 calibration export is step 4).
5. **"Wide lane" means two incompatible things.** The feature layer's five-lane scheme puts
   the wide lane at |y'| > 20.4 (`features.py:74-75`); the flank-overload gate calls
   |y'| > 13.6 "ball in a wide lane" (`flank_overload.py:38`, phase4-5 §3.4). Doc and code
   agree with each other while both contradict the taxonomy they cite.
6. **"Missing back line ⇒ mild discount" is promised, not delivered** (phase4-5 §2.1/§3.2,
   `high_press.py:111`) — the feature layer produces a *wrong* value, not a missing one
   (§1.3). A safeguard documented in two places exists in zero.
7. **The two quality gates are documented as the integration story and connect to nothing.**
   calibration-design §6.3 (floor 0.3) and ball-design §6.2 (floor 0.2) both say wiring into
   `TrackingFrame`/Phase 4 is deliberately deferred — but STATUS's 🟡 headline
   ("implemented, only validated against synthetic fixtures") lets a reader believe the
   gates *gate something*. Today `calibration_quality` and `ball_quality` are computed by
   nothing, consumed by nothing.
8. **Live doc §4.3 promises a per-detector provisional cap** ("their provisional cap is
   lower" for the offside trap); no per-detector cap exists — one default 0.8 for all
   (`StreamingEpisodeConfig`, `detector.py:80`).
9. **Grid alignment by coincidence of defaults.** Event doc §3 says kinematics
   `resample_hz`/`max_gap_s` "default to the feature layer's values" so timestamps align —
   true today because two independent dataclasses happen to carry the same literals
   (`features.py:313-314`, `kinematics.py:35-36`); nothing asserts it, and the §8
   integration depends on it.
10. **The restart-trigger metadata contradiction** (event doc §4.3's cuts-are-normal vs
    `high_press.py`'s same-segment DEAD scan) — see §3.5.

---

## 6. The pre-mortem

It is July 2027 and the repo's last commit is months old. Ranked causes, most probable first.

### 6.1 The perception floor was never proven, and the tower built on it rotted (~40%)

The failure is visible in this two-day window in miniature: six sessions, ~19,000 new lines
of docs+code+tests, zero new bits of information about the two questions that decide
everything — *can the ball be detected at 720p broadcast* (the repo's own literature review
found no precedent at this regime: ball doc §5.2, "NOT-FOUND anywhere") and *what is the
real calibration error at the pitch boundary*. Everything downstream of those answers was
built anyway, at xhigh effort, in parallel, and validated against itself. The pattern that
killed it: producing code was cheap and unblocked; producing evidence required a GPU session
that was quota-blocked, so the project optimized the unblocked axis. Every synthetic-green
test made the stack *feel* more real while the two load-bearing unknowns stayed exactly as
unknown as on day one.

**The decision that caused it:** treating "designed ahead of the data" as a virtue at every
layer simultaneously (each doc individually justifies it; jointly they built phases 4-6 of a
six-phase pipeline on an unmeasured phase 2-3). **Avoidable:** yes, by the project's own
stated discipline — the calibration and ball docs both rank "measure before changing
anything" as step 1; applied at project scope, that rule stops perhaps four of the six
sessions. **Early warning sign:** STATUS.md's "single most important gap" paragraph — the
document names the disease precisely while the commit log keeps producing more of it.
The honest sentence the brief asked for: **a substantial part of this body of work is a
research project mistaking itself for a product.** Stages 1-4 plus the two measurement
layers are research with good instrumentation; events, streaming, scouting, and
recommendations are product scaffolding for a product whose feasibility number has not been
measured.

### 6.2 GPU-quota starvation as a structural condition, not an incident (~20%)

All validation gates on free Colab quota; the quota is exhausted *in STATUS.md right now*
(🔴). Each evidence round-trip costs days of waiting; each waiting period gets filled with
more synthetic-validated code (see 6.1). A project whose entire falsification loop runs on
a free tier someone else can throttle is structurally starved.

**Decision:** no compute budget line — the cost of one Colab Pro month is below the noise
floor of the time already invested. **Avoidable:** trivially. **Warning sign:** "next GPU
session" appearing as the blocking dependency in five separate documents.

### 6.3 The first real export lands, everything underwhelms, and there is no tuning path (~15%)

When the export finally arrives: confidence scale shifted (§1.2), possession fragmenting
(§1.5), line heights corrupted (§1.3), launches noisy (§1.7). None of this is fatal — *if*
there are labels to tune against. The validation plan (phase4-5 §7) requires ~50-100
analyst-labelled clips per motif; the annotator tool has existed throughout and the number
of labelled clips in the repo is zero. Sixty-plus named thresholds, a grid-search harness
design, and no objective function. The likely trajectory is one demoralizing session of
"everything fires wrong or not at all", no principled way to fix it, and drift to
abandonment.

**Decision:** deferring the annotation loop (the doc's own "single highest-leverage
investment") behind more detector-building. **Avoidable:** yes — 30 minutes of labelling
per motif was always cheaper than any session this week. **Warning sign:** already visible —
the tool shipped before this burst; usage is zero.

### 6.4 Maintenance collapse of the parallel-session estate (~10%)

Nine markdown files (~230 KB) and five interlocking cross-layer contracts (parity, tiers,
priors protocol, quality gates, schemas), produced by sessions that couldn't see each other,
maintained by one person. The drift is already measurable *within days*: the §5 list above,
including a C1-class contradiction in the exact place C1 was fixed. Every future change now
pays a reconciliation tax across ~4 docs; the tax compounds; eventually edits stop being
propagated and the docs become adversarial to the code they describe.

**Decision:** optimizing for per-session output volume over integration (six parallel
sessions, no shared context, no post-merge contract check). **Avoidable:** partially — a
single generated source of truth for constants (thresholds in one place, docs citing it)
removes the largest drift class. **Warning sign:** needing a "cross-doc reconciliation"
section at all, and it being stale within 24 hours of being written.

### 6.5 The user was never in the loop (~10%)

The terminal deliverable — YAML counter-strategy rules fired off pattern rates, rendered
into a report — has had zero contact with a working coach or analyst. The scouting doc
concedes the footage library is "aspirational" (§5); the rules table contains illustrative
placeholders; nobody has confirmed that "9 of 12 overloads on their left" clears the bar of
what an analyst already knows from watching the match once. Even with perfect perception,
the product hypothesis is untested.

**Decision:** building the full insight chain before one conversation with its intended
consumer. **Avoidable:** completely; it costs an afternoon. **Warning sign:** the
`SquadProfile`/rules machinery acquiring schema versions before acquiring a user.

### 6.6 Residual (~5%)

Upstream drift breaking the string-match patches (mitigated: `apply()` fails loudly),
Python 3.9-Colab vs local 3.12 divergence, sn-gamestate abandonment upstream, the Kotlin UI
never starting. None of these kills the project on its own.

---

## Appendix: checks that settle the uncertain claims

Claims above marked "plausible" rather than "confirmed", with their deciding check:

| Claim | Check | Cost |
|---|---|---|
| §1.4 real touchline σ ≫ 0.5 m | static-span `world_jitter_m` at boundary pixels from the calibration export | offline, next session's data |
| §1.7 launch false-positive rate | `detect_launches` per minute on joined probe stream | offline |
| §3.3 person recall vs imgsz | diff `n_persons` at 640 vs 1280 in probe output | free (already recorded) |
| §3.4 join misalignment | line-count assert + non-numeric-id scan on first real dump | minutes |
| §1.2 confidence-scale shift magnitude | histogram `quality.score` on real export; re-run fixtures at that quality | offline |
| §1.1 segment count on multi-shot footage | run extractor on a 5-min multi-shot clip's export; count segments | one GPU run, shared with planned session |
