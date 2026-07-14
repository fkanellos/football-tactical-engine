# football-tactical-engine

Turn broadcast football video into strategic insight: track players and the ball, map them
onto a 2D pitch, recognize tactical patterns, and surface recommendations a coach or analyst
can actually use.

## Goal

Given raw broadcast footage of a match, the engine should be able to answer questions like
"what shape was the defense in during this passage of play?" and "what patterns lead to this
team's chances?" — automatically, without manual tagging.

## Pipeline

The engine is organized as a six-stage pipeline, going from raw video to strategic output:

1. **Ingestion** — load and preprocess broadcast video (frame extraction, shot/scene handling).
2. **Detection & tracking** — detect players, referees, and the ball per frame and track their
   identities across frames.
3. **Calibration & pitch mapping** — estimate camera pose/homography per frame and project
   tracked positions onto a canonical 2D pitch (bird's-eye minimap).
4. **Game state reconstruction** — combine tracking, team/jersey classification, and pitch
   mapping into a structured, per-frame game state (who is where, in possession of what).
5. **Tactical pattern recognition** — analyze sequences of game state to detect formations,
   pressing triggers, build-up patterns, and other tactical structure.
6. **Strategic recommendations** — turn recognized patterns into actionable insight: strengths,
   weaknesses, and suggestions for a coaching staff.

## Repository layout

- [`research/`](research/) — Colab-ready Jupyter notebooks for CV/ML experimentation
  (stages 1-4, primarily).
- [`pipeline/`](pipeline/) — future Python FastAPI service that productionizes the validated
  CV/ML approach into an HTTP API.
- [`ui/`](ui/) — future Kotlin + Compose Desktop application for coaches/analysts to consume
  pipeline output.
- [`docs/`](docs/) — architecture notes and research findings.

## Tech stack

- **CV/ML backend**: Python, built on top of existing game-state-reconstruction research
  (e.g. [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) and its TrackLab
  dependency) as a starting point, evolving toward a FastAPI service in `/pipeline`.
- **Desktop UI**: Kotlin + Compose Desktop, consuming the pipeline over HTTP.

## Status

Early stage. The current focus is validating game-state reconstruction on SoccerNet data in
`/research` before productionizing anything in `/pipeline` or building `/ui`.
