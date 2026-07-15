"""Orchestration: tracking -> features -> detectors -> MatchPatternProfile.

``MatchPatternProfile`` is the ONLY input the Phase 5 recommendation engine sees
(design doc §4.2): the Phase 4/5 boundary is events + aggregates, never frames or
features. It is also the source of the two persisted JSON documents (events.json,
profile.json — schemas in design doc §4.3) that form the pipeline<->UI contract.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .detectors.base import PatternDetector, PatternEvent
from .features import FeatureExtractor, FeatureSeries
from .tracking import MatchTracking, TeamSide

logger = logging.getLogger(__name__)


@dataclass
class PatternAggregate:
    """Match-level summary of one (pattern, team) pair.

    All rates are per OBSERVED 90 minutes (broadcast cuts mean we never see the
    full match; normalising by match minutes would silently deflate every rate —
    design doc §1.2).

    ``breakdowns`` maps a metadata key to counts of its values, e.g.
    ``{"side": {"left": 9, "right": 3}}`` — this is what Phase 5 trigger ``where``
    filters run against. ``phase_counts`` buckets events into 15-minute bins of
    footage time for "when do they do it" narratives.
    """

    pattern_id: str
    team: TeamSide
    events: List[PatternEvent] = field(default_factory=list)
    count: int = 0
    rate_per_90_observed: float = 0.0
    mean_confidence: float = 0.0
    mean_intensity: float = 0.0
    max_intensity: float = 0.0
    total_duration_s: float = 0.0
    breakdowns: Dict[str, Dict[str, int]] = field(default_factory=dict)
    phase_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class MatchPatternProfile:
    """Everything Phase 4 hands to Phase 5 (and the UI) for one match.

    ``tendencies`` holds detector-specific aggregate profiles that are not
    event-shaped (currently the offside-trap line-management tendency,
    design doc §3.5), keyed by pattern_id then team.
    """

    match_id: str
    observed_minutes: float = 0.0
    aggregates: List[PatternAggregate] = field(default_factory=list)
    tendencies: Dict[str, Dict[TeamSide, Dict[str, Any]]] = field(default_factory=dict)

    def get(self, pattern_id: str, team: TeamSide) -> Optional[PatternAggregate]:
        """Aggregate for one (pattern, team), or None if the pattern never fired."""
        for agg in self.aggregates:
            if agg.pattern_id == pattern_id and agg.team is team:
                return agg
        return None


class PatternDetectionPipeline:
    """End-to-end Phase 4 run for one match: features -> all detectors -> profile.

    Detector failures are contained: one broken detector logs an error and is
    skipped rather than sinking the whole match run.
    """

    def __init__(self, detectors: List[PatternDetector], extractor: Optional[FeatureExtractor] = None):
        self.detectors = detectors
        self.extractor = extractor or FeatureExtractor()

    def run(self, match: MatchTracking) -> MatchPatternProfile:
        series = self.extractor.extract(match)
        events: List[PatternEvent] = []
        tendencies: Dict[str, Dict[TeamSide, Dict[str, Any]]] = {}
        for detector in self.detectors:
            try:
                events.extend(detector.detect(series, match.meta))
            except Exception:
                logger.exception("detector %s failed; skipping", detector.pattern_id)
                continue
            tendency_fn = getattr(detector, "line_tendency", None)
            if callable(tendency_fn):
                tendencies[detector.pattern_id] = {
                    team: tendency_fn(series, match.meta, team)
                    for team in (TeamSide.HOME, TeamSide.AWAY)
                }
        profile = self.aggregate(events, series)
        profile.tendencies = tendencies
        return profile

    def aggregate(self, events: List[PatternEvent], series: FeatureSeries) -> MatchPatternProfile:
        """Events -> per-(pattern, team) aggregates; rates per observed 90."""
        observed_s = series.observed_seconds()
        profile = MatchPatternProfile(
            match_id=series.meta.match_id, observed_minutes=round(observed_s / 60.0, 2)
        )
        grouped: Dict[Any, List[PatternEvent]] = defaultdict(list)
        for e in events:
            grouped[(e.pattern_id, e.team)].append(e)

        for (pattern_id, team), evs in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
        ):
            evs = sorted(evs, key=lambda e: e.start_s)
            agg = PatternAggregate(pattern_id=pattern_id, team=team, events=evs, count=len(evs))
            agg.rate_per_90_observed = (
                round(len(evs) * 5400.0 / observed_s, 2) if observed_s > 0 else 0.0
            )
            agg.mean_confidence = round(sum(e.confidence for e in evs) / len(evs), 3)
            agg.mean_intensity = round(sum(e.intensity for e in evs) / len(evs), 3)
            agg.max_intensity = round(max(e.intensity for e in evs), 3)
            agg.total_duration_s = round(sum(e.duration_s for e in evs), 1)

            breakdowns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for e in evs:
                for key, value in e.metadata.items():
                    if isinstance(value, str):
                        breakdowns[key][value] += 1
                    elif isinstance(value, bool):
                        breakdowns[key]["true" if value else "false"] += 1
            agg.breakdowns = {k: dict(v) for k, v in breakdowns.items()}

            phases: Dict[str, int] = defaultdict(int)
            for e in evs:
                bin_start = int(e.start_s // 900) * 15
                phases[f"{bin_start}-{bin_start + 15}min"] += 1
            agg.phase_counts = dict(phases)
            profile.aggregates.append(agg)
        return profile
