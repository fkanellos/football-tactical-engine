# pipeline

Future home of the Python FastAPI service that wraps the CV pipeline.

Once the computer vision approach has been validated in `/research`, the resulting
detection, tracking, calibration, and event/pattern-recognition logic gets productionized
here as a proper Python service: broadcast video in, structured tracking data and tactical
insights out, exposed over a FastAPI HTTP API for the `/ui` desktop application to consume.

The service itself doesn't exist yet, but the Phase 4/5 skeletons do — designed ahead
of time so the architecture is settled before the tracking data arrives (see
[`/docs/phase4-5-design.md`](../docs/phase4-5-design.md)):

- [`patterns/`](patterns/) — tactical pattern detection: tracking data model, feature
  extraction layer, per-motif detector strategies, and match-profile aggregation.
  **Fully implemented** and validated against synthetic textbook scenarios
  (`patterns/testing/synthetic.py`); run the suite with
  `python3 -m unittest discover -s pipeline/tests`. The TrackLab ingest adapter
  (`patterns/tracking.py:load_tracklab_states`) stays stubbed until `/research`
  produces real output to pin the format against.
- [`events/`](events/) — discrete match-event inference from tracking alone (see
  [`/docs/event-inference-design.md`](../docs/event-inference-design.md)): boundary
  restarts (throw-in/corner/goal kick/kickoff), passes with subtypes, shots with an
  outcome honesty ladder, set-piece organization, and a coarse stoppage signal —
  each event carrying its confidence tier. Sits *below* `patterns/`; the possession
  machinery there is flagged to consume this stream (design doc §8, not yet wired).
  **Fully implemented** against synthetic fixtures; emits `match_events.json`
  (`match-events/v1`).
- [`recommendations/`](recommendations/) — rule-based counter-strategy engine: analyst-
  authored rules (YAML) matched against detected pattern profiles. Still skeleton.
- [`patterns/streaming/`](patterns/streaming/) — live/broadcast-time detection (see
  [`/docs/live-architecture-design.md`](../docs/live-architecture-design.md)): causal
  twins of the batch feature helpers (same numbers, delayed — parity-tested), the
  PROVISIONAL/CONFIRMED/CLOSED episode lifecycle machines, and the WebSocket event
  schema for the `/ui` client. Feature-extractor assembly and the serving layer are
  design-only pending real tracking data.
- [`scouting/`](scouting/) — opponent scouting from historical footage (see
  [`/docs/opponent-scouting-design.md`](../docs/opponent-scouting-design.md)):
  recency-weighted multi-match tendency profiles with explicit uncertainty, the
  pre-match report, and bounded Bayesian priors that feed the live detectors.
  Aggregation/priors/report math implemented and tested; rendered recommendations
  await the Phase 5 engine.
