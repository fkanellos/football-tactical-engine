# docs — design index

Architecture notes and research findings: the decisions and results that need to outlive a
single notebook or PR (pipeline architecture, model/dataset evaluations, the `/pipeline`↔`/ui`
data schemas, and design rationale).

For the overall six-stage pipeline (ingestion → detection/tracking → calibration → game-state
→ tactical patterns → recommendations), start with the [top-level README](../README.md). The
design docs below cover **stages 4–6** in depth. For where the project actually stands today —
what runs on real data vs synthetic fixtures vs design-only — see **[STATUS.md](STATUS.md)**.

---

## The four design docs

| Doc | Covers | Status |
|---|---|---|
| [phase4-5-design.md](phase4-5-design.md) | **The core.** Feature layer, the five batch pattern detectors (high press, low block, flank overload, offside trap, counter-attack), the Phase 5 rule-based recommendation engine, threshold provenance. Everything else references this. | Phase 4 **implemented + tested** (synthetic fixtures, 42 tests); Phase 5 **skeleton**. Not validated on real tracking. |
| [event-inference-design.md](event-inference-design.md) | The **layer below Phase 4**: inferring discrete match events (restarts, passes, shots, set-pieces, a coarse stoppage signal) from tracking alone, organised by a three-tier confidence hierarchy. | Detectors **implemented + tested** (synthetic). The §8 Phase-4 possession refactor is **designed and flagged, deliberately not implemented**. |
| [live-architecture-design.md](live-architecture-design.md) | Running the *same* Phase 4 logic frame-by-frame with bounded latency: the parity principle, causal smoothing, episode lifecycle machines, the WebSocket contract, the live-profile GPU story. | Causal core (smoothing, possession, lifecycle machines, WS schema) **implemented + tested**; feature assembly + `pipeline/live/` service layer **skeleton / design-only**. |
| [opponent-scouting-design.md](opponent-scouting-design.md) | Multi-match aggregation into `TeamScoutingProfile`s, the day-before scouting report, and live detection **priors** — the bridge between batch and live. | Aggregation, priors, report schema/adapter **implemented + tested**. Rendered recommendations **blocked on the Phase 5 engine**; footage-source question **open by design**. |

**Nothing in any of these has touched real tracking output yet** — every "tested" above means
against synthetic fixtures we authored. See [STATUS.md](STATUS.md) for the blunt version.

## Intended reading order

1. **[phase4-5-design.md](phase4-5-design.md)** — the foundation. The four "data realities"
   (§1.2), the feature layer (§2), and the detector idiom (§3.1) are prerequisites for the
   other three docs; §8/§9 of the sibling docs assume them.
2. **[event-inference-design.md](event-inference-design.md)** — reads as "the more truthful
   foundation Phase 4 should eventually sit on." Its §8 is a *proposed* seam into Phase 4, not
   a done change.
3. **[live-architecture-design.md](live-architecture-design.md)** — extends #1 to a streaming
   context. Its whole premise (the *parity principle*, §2) is "same numbers as batch, later."
4. **[opponent-scouting-design.md](opponent-scouting-design.md)** — the third consumer of
   Phase 4 output; its §3 priors hook into the live machines from #3, so read it last.

---

## Cross-doc reconciliation

I cross-checked all four docs for conflicting schemas, incompatible assumptions, duplicated
concepts under different names, and one doc assuming what another refuses. The docs are
**substantially coherent** — the parallel sessions cross-referenced each other well. Findings
below. One drift (C1) has been fixed; the rest were checked and confirmed non-issues.

### ✓ Resolved — C1 (prose drift, fixed 2026-07-16)

**C1 — `phase4-5-design.md` §3 heuristic prose was stale relative to its own §9 provenance
table and the shipped code.** The §9 "threshold provenance" pass (July 2026) replaced several
detector defaults, and the code was updated, but the §3 *prose* describing the heuristics was
never back-updated. **Now corrected** — the §3.2/§3.6 prose and the `high_press.py:111`
docstring were edited to the shipped values below. The code was already authoritative and
correct, so this was a zero-behaviour copy edit. The drift that was fixed:

| Parameter | §3 prose says | §9 table + shipped code | Where verified |
|---|---|---|---|
| Counter-attack window | "Within a 10 s window" (§3.6) | **14 s** | `counter_attack.py:32` (`window_after_turnover_s = 14.0`) |
| Counter-attack progression | "≥ ~25 m" (§3.6) | **16 m** | `counter_attack.py:33` (`min_progression_m = 16.0`) |
| Counter-attack runner velocity | "forward velocity ≥ ~4 m/s" (§3.6) | **5.5 m/s** | `features.py:330` (`runner_speed_ms = 5.5`) |
| High-press pressure radius | "nearest_defender_dist ≤ 6 m" (§3.2) | **4.6 m** | `high_press.py:38` (`max_nearest_defender_dist_m = 4.6`) |

The corrected values are the ones that ship, are literature-grounded (Yang et al. 2025 for the
counter values; Gualtieri 2023 for 5.5 m/s; StatsBomb 5-yard radius for 4.6 m), and are echoed
correctly by the *other* docs — e.g. `live-architecture-design.md` §4.3 says "opens a 14 s
window." So §3 prose is the lone outlier, not the code.

The `high_press.py:111` docstring carried the same stale 6 m (a code-comment instance of the
same drift, harmless to behaviour since the dataclass default 4.6 is what runs); it was
corrected in the same pass.

### ✓ Checked and reconciled — no action needed

Listed for transparency, so you know they were examined and are *not* live conflicts:

- **"No event data" vs the event-inference doc.** `phase4-5-design.md` §1.2 reality #4 states
  bluntly "we have no pass/shot/tackle stream." A skim reader could think this contradicts the
  entire `event-inference-design.md`. It doesn't: §1.2 #4 means *no external event feed*, while
  the event doc *infers* events from tracking geometry, and it explicitly flags (§8) that its
  integration back into Phase 4's possession logic is **not yet implemented**. The two are
  compatible; the tension is real enough that it's worth knowing where the seam is.
- **`events` namespace collision.** `pipeline/events/` (inferred *match* events) vs
  `pipeline/patterns/streaming/events.py` (the live *wire* schema for pattern-lifecycle
  messages) share the word "events" but are unrelated namespaces. The author flagged this
  directly (event doc §7.1). No code collision.
- **Segment-locality inversion.** Pattern detectors must *never* span a broadcast cut (an
  episode is continuous evidence); restart detection *deliberately does* span cuts (a restart
  is a discontinuity). This looks contradictory but is an intentional, documented deviation
  (event doc §4.3). Consistent by design.
- **Schema namespaces.** `pattern-events/v1`, `pattern-profile/v1`, `match-events/v1`,
  `scouting-report/v1`, and the WebSocket `v:1` schema are all distinct and non-colliding. The
  annotator tool (`tools/annotator.html`) exports `pattern-events/v1`, which both the patterns
  and events docs reference consistently as the shared label schema.
- **Priors / lifecycle contract.** The scouting doc's `ConfidencePrior` protocol (§3) and the
  live doc's consumption of it (§7) agree on every constant and safeguard (κ = 0.85, n₀ = 4,
  raw-confidence floor 0.40, identity-when-thin). No drift.
