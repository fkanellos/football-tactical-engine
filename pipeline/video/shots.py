"""Shot-boundary detection: where the broadcast cut, and therefore where a segment ends.

The batch stack asserts everywhere that "frames are NOT guaranteed contiguous —
broadcast cuts leave gaps" (`patterns/tracking.py`, `patterns/features.py`,
phase4-5-design.md §1.2 "data reality #3"), and segments on timestamp gaps
alone. Nothing produced those gaps. sn-gamestate emits a detection row for
*every decoded frame* — replays, close-ups and crowd shots included — and the
folder-of-frames patch deliberately makes a whole clip one video so tracking
never resets. So "data reality #3" was an unbuilt requirement, and the first
multi-shot clip fed through the stack would have produced one giant segment
with replays ingested as continuous play (red-team review §1.1/§3.1). This
module is the missing producer.

**Scale of the problem, measured.** Over the full BvB-PSG broadcast (100 min,
150 455 frames), ffmpeg's scene filter finds **340 cuts, one every 17.7 s**, and
this module finds **525, one every 11.5 s** — agreeing on 93.5% of ffmpeg's
calls and adding 209 of its own. Three of those extras, sampled and eyeballed,
were all genuine: a close-up-to-wide switch mid-play, a shot change during a
run, and a star-wipe into a replay. So the true rate is at least 340 and
plausibly near 525, and **a 34 s clip picked at random from real broadcast spans
two or three shot changes**. The OFI clip every existing measurement in this
repo rests on happens to be single-shot — luck, and the same luck SoccerNet-GSR
engineers deliberately ("single-shot 30 s clips, no cuts by construction",
calibration-design.md §5.2).

**No pixels in here.** `pipeline/` is pure stdlib and runs its whole suite in
seconds; decoding frames would drag in an imaging stack and make this layer
untestable without fixtures made of JPEGs. So this module consumes
``FrameSignature`` — a coarse per-frame colour histogram someone else computed
(``research/frame_signatures.py`` does it with ffmpeg and no image library at
all) — exactly as ``ball/ingest.py`` consumes the probe's jsonl rather than
running YOLO. The same split lets the live gate (live-architecture-design.md
§5.1 signal 1) feed signatures from a stream instead of a folder.

**What this catches and what it does not.** Hard cuts, reliably: a cut to a
replay, a close-up, or the crowd moves the colour distribution enough that no
threshold subtlety is needed. Not caught, and *not* silently pretended
otherwise:

- **Dissolves and wipes** spread the change over 10-25 frames, so no single
  frame-to-frame distance spikes. ``min_shot_frames`` keeps a gradual
  transition from firing as a burst of adjacent cuts, but the boundary lands
  somewhere inside the dissolve rather than at its start.
- **Wide-angle replays**, the hard case the live doc names (§5.1): a replay of
  open play from a similar camera looks like live play to *any* histogram
  method. Signals 2-4 there (calibration collapse, detection evaporation,
  whole-frame position teleports) are what catch those, and they need the
  tracking stage this module deliberately runs before.
- **Camera pans and zooms** are continuous change, not boundaries. They are
  what forces an adaptive threshold: an absolute distance that is "large" on a
  locked-off shot is ordinary two frames into a fast pan.

A learned shot-boundary model (TransNetV2-class) is the known-better
replacement for all three and slots in behind ``detect_shot_boundaries``'s
signature — the honest-ML note the live design already carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Frames whose distance exceeds median + ``k * MAD`` are candidate boundaries.
#: Median/MAD rather than mean/stdev because the distance series is *made* of
#: outliers at exactly the points of interest: a handful of cuts would inflate a
#: standard deviation enough to hide themselves. 6.0 is deliberately permissive
#: — a missed cut corrupts a segment silently, a spurious one splits a segment
#: that the feature layer then treats as two, which is the cheaper error.
DEFAULT_MAD_K = 6.0

#: A cut must also clear this absolute distance, whatever the adaptive floor
#: says. Without it, a locked-off shot with a near-zero distance series produces
#: a near-zero MAD, and ordinary compression noise clears `median + k*MAD`.
DEFAULT_MIN_DISTANCE = 0.15

#: Shots shorter than this are implausible in football broadcast and usually
#: mean a dissolve fired repeatedly, or a flash (camera strobe, pyro) split one
#: shot in three. Boundaries are merged until every shot clears it.
DEFAULT_MIN_SHOT_FRAMES = 8


@dataclass(frozen=True)
class FrameSignature:
    """One frame reduced to a comparable fingerprint.

    ``histogram`` is a normalised colour histogram (bins sum to 1.0) at whatever
    binning the producer chose; all that matters here is that every signature in
    a sequence shares a binning. Coarse is correct: the question is "did the
    whole image change", and fine bins make a pan look like a cut.
    """

    index: int  # position in the frame sequence, the join key everything else uses
    timestamp_s: float
    histogram: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.histogram:
            raise ValueError(f"frame {self.index}: empty histogram")


@dataclass(frozen=True)
class ShotBoundary:
    """A detected cut: the first frame of the *new* shot."""

    index: int
    timestamp_s: float
    distance: float  # histogram distance across the boundary
    threshold: float  # what it had to clear, for auditing a marginal call

    @property
    def margin(self) -> float:
        """How decisively it cleared. Near 1.0 means "only just" — worth a look."""
        return self.distance / self.threshold if self.threshold > 0 else math.inf


@dataclass(frozen=True)
class Segment:
    """A maximal run of frames between cuts — one continuous camera take.

    This is the unit every downstream guarantee is written against: episodes
    never span segments, the calibration smoother resets at segment starts, and
    possession cannot carry across one.
    """

    start_index: int  # inclusive
    end_index: int  # exclusive
    start_time_s: float
    end_time_s: float

    @property
    def n_frames(self) -> int:
        return self.end_index - self.start_index

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


def histogram_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """L1 distance halved — 0.0 for identical, 1.0 for disjoint distributions.

    Chi-squared is the textbook choice and is more sensitive to small shifts in
    sparse bins; that sensitivity is a liability here, because a scoreboard
    graphic animating in a corner is a small shift in sparse bins and is not a
    cut. L1 on a normalised histogram answers the blunter question this module
    actually asks — what fraction of the image's colour mass moved — and its
    0..1 range makes ``DEFAULT_MIN_DISTANCE`` mean something without reference
    to bin count.
    """
    if len(a) != len(b):
        raise ValueError(f"histogram length mismatch: {len(a)} vs {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b)) / 2.0


def consecutive_distances(
    signatures: Sequence[FrameSignature],
) -> List[float]:
    """Distance between each frame and its predecessor; ``[0]`` is 0.0 by fiat.

    Index i holds the distance *into* frame i, so a boundary reported at index i
    means "frame i starts a new shot" — the convention ``Segment`` boundaries and
    every caller assume.
    """
    if not signatures:
        return []
    out = [0.0]
    for prev, cur in zip(signatures, signatures[1:]):
        out.append(histogram_distance(prev.histogram, cur.histogram))
    return out


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def adaptive_threshold(
    distances: Sequence[float],
    mad_k: float = DEFAULT_MAD_K,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> float:
    """``max(median + k * MAD, min_distance)`` over the whole distance series.

    Global rather than windowed, deliberately. A windowed threshold adapts to a
    burst of rapid cutting (replay montages, goal celebrations) by raising its
    own bar exactly where cuts are densest, which is the opposite of useful. The
    cost is that one long locked-off passage and one long handheld passage in
    the same clip share a threshold; if that ever bites, the fix is per-segment
    re-estimation on a second pass, not a sliding window.
    """
    if len(distances) < 2:
        return min_distance
    body = list(distances[1:])  # index 0 is the synthetic 0.0
    med = _median(body)
    mad = _median([abs(d - med) for d in body])
    return max(med + mad_k * mad, min_distance)


def detect_shot_boundaries(
    signatures: Sequence[FrameSignature],
    mad_k: float = DEFAULT_MAD_K,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> List[ShotBoundary]:
    """Frames that start a new shot, in index order.

    Three passes: threshold the distance series, cluster boundaries closer than
    ``min_shot_frames``, then decide each cluster's fate by asking whether the
    scene *actually changed across it* — see ``_resolve_clusters``.
    """
    if len(signatures) < 2:
        return []
    distances = consecutive_distances(signatures)
    threshold = adaptive_threshold(distances, mad_k, min_distance)

    positions = [i for i in range(1, len(signatures)) if distances[i] > threshold]
    return _resolve_clusters(
        signatures, positions, distances, threshold, min_shot_frames
    )


def _resolve_clusters(
    signatures: Sequence[FrameSignature],
    positions: Sequence[int],
    distances: Sequence[float],
    threshold: float,
    min_shot_frames: int,
) -> List[ShotBoundary]:
    """Turn threshold crossings into boundaries, rejecting transients.

    Two distinct things produce a tight cluster of crossings, and they need
    opposite treatment:

    - **A transient** — a camera flash, pyro, a one-frame graphic wipe. The
      image leaves the scene and comes *back to it*. Two large distances, and
      no shot boundary at all: a detector that keeps one of them invents a cut
      in the middle of continuous play and hands the feature layer two segments
      where there was one possession.
    - **A rapid cut sequence** — genuinely short shots, which broadcast does use
      (a three-shot replay montage). The image leaves the scene and *stays
      away*.

    Frame-to-frame distance cannot tell these apart; distance *across* the
    cluster can. So compare the frame before the cluster to the frame after it:
    if they still look like the same scene, the excursion was a transient and
    the whole cluster is dropped. Otherwise the scene really did change and the
    cluster's strongest crossing is the boundary — strongest rather than first
    because a short ramp's largest step sits at its steepest point, which is
    closer to where the new shot truly begins than the ramp's toe.
    """
    if not positions:
        return []

    clusters: List[List[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if min_shot_frames > 1 and pos - clusters[-1][-1] < min_shot_frames:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    kept: List[ShotBoundary] = []
    for cluster in clusters:
        if len(cluster) > 1 and _is_transient(signatures, cluster, threshold):
            continue
        best = max(cluster, key=lambda p: distances[p])
        kept.append(
            ShotBoundary(
                index=signatures[best].index,
                timestamp_s=signatures[best].timestamp_s,
                distance=distances[best],
                threshold=threshold,
            )
        )
    return kept


def _is_transient(
    signatures: Sequence[FrameSignature],
    cluster: Sequence[int],
    threshold: float,
) -> bool:
    """Did the scene return to what it was before this cluster of crossings?

    Compared against the same ``threshold`` the crossings had to clear, so the
    test reads as "the endpoints are closer together than a cut": if the frame
    before and the frame after are *not* separated by a cut's worth of change,
    nothing was cut.
    """
    before = cluster[0] - 1
    after = cluster[-1]
    if before < 0 or after >= len(signatures):
        return False
    return (
        histogram_distance(
            signatures[before].histogram, signatures[after].histogram
        )
        <= threshold
    )


def segments_from_boundaries(
    signatures: Sequence[FrameSignature],
    boundaries: Sequence[ShotBoundary],
) -> List[Segment]:
    """Split a signature sequence into continuous takes.

    ``end_time_s`` is the timestamp of the segment's last frame, not of the cut
    that follows it: the last frame is the last moment the segment is known to
    be valid, and attributing the cut frame's timestamp to the old segment would
    stretch every span by one frame in the direction of the thing that broke it.
    """
    if not signatures:
        return []
    starts = [0]
    by_index = {sig.index: i for i, sig in enumerate(signatures)}
    for boundary in boundaries:
        pos = by_index.get(boundary.index)
        if pos is not None and pos > 0:
            starts.append(pos)
    starts = sorted(set(starts))

    segments = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(signatures)
        segments.append(
            Segment(
                start_index=signatures[start].index,
                end_index=signatures[stop - 1].index + 1,
                start_time_s=signatures[start].timestamp_s,
                end_time_s=signatures[stop - 1].timestamp_s,
            )
        )
    return segments


def segment_for_frame(
    segments: Sequence[Segment], index: int
) -> Optional[Segment]:
    """Which take a frame belongs to — the lookup the feature layer needs.

    Returns ``None`` for an index outside every segment rather than guessing a
    nearest one: a frame that belongs to no known take is exactly the case a
    caller must handle explicitly, not one to paper over.
    """
    for segment in segments:
        if segment.start_index <= index < segment.end_index:
            return segment
    return None


@dataclass(frozen=True)
class ShotReport:
    """Whole-clip summary — the numbers that say whether a clip is usable at all."""

    n_frames: int
    n_segments: int
    n_boundaries: int
    duration_s: float
    threshold: float
    longest_segment: Optional[Segment]
    median_segment_s: float
    cuts_per_minute: float

    @property
    def single_shot(self) -> bool:
        """True when the whole clip is one take — the OFI case, and the only
        case every existing measurement in this repo was made under."""
        return self.n_segments <= 1


def shot_report(
    signatures: Sequence[FrameSignature],
    mad_k: float = DEFAULT_MAD_K,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> ShotReport:
    """Detect, segment, and summarise in one call."""
    boundaries = detect_shot_boundaries(
        signatures, mad_k, min_distance, min_shot_frames
    )
    segments = segments_from_boundaries(signatures, boundaries)
    distances = consecutive_distances(signatures)
    threshold = adaptive_threshold(distances, mad_k, min_distance)
    duration = (
        signatures[-1].timestamp_s - signatures[0].timestamp_s if signatures else 0.0
    )
    return ShotReport(
        n_frames=len(signatures),
        n_segments=len(segments),
        n_boundaries=len(boundaries),
        duration_s=duration,
        threshold=threshold,
        longest_segment=max(segments, key=lambda s: s.n_frames) if segments else None,
        median_segment_s=_median([s.duration_s for s in segments]),
        cuts_per_minute=(len(boundaries) / (duration / 60.0)) if duration > 0 else 0.0,
    )
