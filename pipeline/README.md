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
- [`recommendations/`](recommendations/) — rule-based counter-strategy engine: analyst-
  authored rules (YAML) matched against detected pattern profiles. Still skeleton.
