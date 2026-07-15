"""Phase 4: tactical pattern detection.

Layering (see /docs/phase4-5-design.md §4):

    tracking.py   raw per-frame tracking data model + TrackLab adapter
    features.py   normalisation + per-frame tactical features (FeatureSeries)
    detectors/    one Strategy class per tactical motif (PatternDetector ABC)
    runner.py     orchestration: features -> events -> MatchPatternProfile

Detectors never touch raw tracking; the recommendation engine (Phase 5,
pipeline.recommendations) never touches anything below MatchPatternProfile.
"""
