"""Scouting priors for live detection: bounded, floored, audit-logged.

Design: /docs/opponent-scouting-design.md §3. The mechanism in one paragraph:
from a pattern's scouting tendency we form a Beta-shrunk estimate of how likely
this opponent is to show the pattern in a match (``PatternPrior``); at match
time a ``PriorAdjuster`` — which satisfies the streaming layer's
``ConfidencePrior`` protocol (pipeline.patterns.streaming.detector) — converts
the gap between that estimate and the league baseline into a log-odds shift on
*reported* confidence, and a scaling of how long the detector waits before its
first provisional alert. The shift is capped, vanishes when history is thin,
and NEVER replaces evidence: per-frame scores and episode membership are
untouched, and confirmation always requires the RAW confidence to clear an
absolute floor.

Why log-odds: it is the only shift that behaves sensibly at both ends of the
[0,1] scale (a +0.85 logit shift moves 0.5 -> 0.70 but 0.05 -> only 0.11), so a
prior can never turn "nothing is happening" into an alert — the failure mode a
naive multiplicative or additive adjustment has.

The confirmation-bias risk is real and is handled structurally, not by hoping
the numbers are small (design doc §3.4): cap, floor, n_eff gate, raw+adjusted
in every event, and a post-match reconciliation that measures the prior's
actual influence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import PatternTendency


@dataclass(frozen=True)
class PriorConfig:
    """Safeguard-first tunables (every one exists to bound the prior's power).

    ``pseudo_matches``: Bayesian stiffness — the scouting evidence is blended
    against this many matches' worth of "the league baseline"; with n_eff
    effective matches the prior earns weight n_eff/(n_eff+pseudo_matches).
    ``max_logit_shift`` (κ): hard cap; 0.85 logits ≈ the difference between
    0.50 and 0.70. ``confirm_floor``: RAW confidence a live instance must reach
    to confirm regardless of any prior. ``min_n_eff``: below this the adjuster
    is identity — one match of history adjusts nothing.
    """

    baseline_prevalence: float = 0.35
    pseudo_matches: float = 4.0
    max_logit_shift: float = 0.85
    min_n_eff: float = 2.0
    confirm_floor: float = 0.40
    max_provisional_scale_delta: float = 0.25  # wait scaled within [0.75, 1.25]


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class PatternPrior:
    """This opponent's estimated per-match probability of showing a pattern.

    ``p_team`` is the Jeffreys-shrunk prevalence: (s + 0.5) / (n_eff + 1) with
    s = prevalence · n_eff — so "2 of 2 matches" gives 0.83, not a reckless
    1.0, and an empty profile gives 0.5 with n_eff 0 (which the min_n_eff gate
    then ignores entirely).
    """

    pattern_id: str
    p_team: float
    n_eff: float

    @classmethod
    def from_tendency(cls, tendency: PatternTendency) -> "PatternPrior":
        s = tendency.prevalence * tendency.n_eff
        return cls(
            pattern_id=tendency.pattern_id,
            p_team=(s + 0.5) / (tendency.n_eff + 1.0),
            n_eff=tendency.n_eff,
        )


class PriorAdjuster:
    """Satisfies ``pipeline.patterns.streaming.detector.ConfidencePrior``.

    shift = clamp(λ · (logit(p_team) − logit(baseline)), ±κ)
    λ      = n_eff / (n_eff + pseudo_matches),   0 if n_eff < min_n_eff
    adjust(c) = σ(logit(c) + shift)

    Positive shift (the opponent does this more than the baseline) raises
    reported confidence and shortens the provisional wait; negative shift does
    the reverse — out-of-character patterns need stronger live evidence before
    the system speaks. ``confirm_allowed`` ignores the shift entirely: it
    checks the raw confidence against the absolute floor.
    """

    def __init__(
        self,
        prior: PatternPrior,
        config: Optional[PriorConfig] = None,
        baseline_prevalence: Optional[float] = None,
    ):
        self.prior = prior
        self.config = config or PriorConfig()
        self.baseline = (
            self.config.baseline_prevalence
            if baseline_prevalence is None
            else baseline_prevalence
        )
        if self.prior.n_eff < self.config.min_n_eff:
            self._lambda = 0.0
        else:
            self._lambda = self.prior.n_eff / (self.prior.n_eff + self.config.pseudo_matches)
        raw_shift = self._lambda * (_logit(self.prior.p_team) - _logit(self.baseline))
        kappa = self.config.max_logit_shift
        self.shift = max(-kappa, min(kappa, raw_shift))

    # -- ConfidencePrior protocol ------------------------------------------------

    def adjust_confidence(self, raw: float) -> float:
        if self.shift == 0.0 or raw <= 0.0:
            return raw
        return _sigmoid(_logit(raw) + self.shift)

    def provisional_after_scale(self) -> float:
        """Scale for the wait before a first provisional alert, in
        [1-Δ, 1+Δ]: expected patterns alert sooner, unexpected ones later."""
        kappa = self.config.max_logit_shift
        if kappa <= 0:
            return 1.0
        return 1.0 - self.config.max_provisional_scale_delta * (self.shift / kappa)

    def confirm_allowed(self, raw_confidence: float) -> bool:
        """The absolute floor: evidence must stand on its own to confirm.
        Deliberately uses RAW confidence — no prior can buy confirmation."""
        return raw_confidence >= self.config.confirm_floor

    def describe(self) -> Dict[str, Any]:
        """Audit block attached to every live event this prior touched."""
        return {
            "applied": self.shift != 0.0,
            "pattern": self.prior.pattern_id,
            "p_team": round(self.prior.p_team, 3),
            "baseline": round(self.baseline, 3),
            "n_eff": round(self.prior.n_eff, 2),
            "lambda": round(self._lambda, 3),
            "shift": round(self.shift, 3),
        }
