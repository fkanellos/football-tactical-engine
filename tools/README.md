# tools

Small standalone utilities supporting the pipeline's research/validation loop.

## annotator.html — tactical pattern annotation tool

The clip-annotation loop is the highest-leverage Phase 4 investment (design doc §7):
~50 analyst-labelled clips per motif turn threshold tuning from vibes into
measurement. This is that loop's tool — usable **today**, on any match video you
have, independent of the rest of the pipeline.

**Usage:** open `annotator.html` directly in a browser (double-click; no server, no
install, no network — the video never leaves your machine). Load a clip, scrub,
mark a time range (`[` and `]`), pick a pattern label (`1`–`6`, including `none`
for reviewed-negative spans), team, confidence (certain/likely/borderline), add an
optional note, hit Enter.

**Output:** Export JSON produces a `pattern-events/v1` document — the same schema
Phase 4's detectors emit (design doc §4.3) — so labelled clips plug straight into
the future evaluation harness as ground truth. `intensity` is `null` for manual
annotations; confidence encodes the certain/likely/borderline choice.

Annotations autosave to browser localStorage per match id; Import JSON resumes a
previous session from an exported file.
