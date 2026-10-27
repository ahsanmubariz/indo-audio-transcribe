# The web app owns the transcription pipeline

The web app drives `sea_asr` directly rather than invoking `transcribe.sh` as a
subprocess, and the script becomes a thin CLI over a shared module. Shelling out
would have kept a single source of truth, but the script reports only exit codes:
no per-piece progress, no cancellation between pieces, and no partial output —
exactly the things the web UI exists to provide. Measured on a 13-minute
recording, driving the per-piece loop was also ~13% faster than the library's
default batching (100.6s vs 115.4s on a 551s subset), because batching pads every
piece to the longest in its batch and so computes audio that does not exist.

## Considered Options

- **Shell out to `transcribe.sh`** — one source of truth, but exit-code-only
  feedback, unclean cancellation, and slower.
- **Duplicate the pipeline in a second script** — rejected; two copies of the
  ffmpeg and model-loading logic drift apart immediately.

## Consequences

`transcribe.sh` must not grow behaviour the web app lacks, or the two front ends
diverge. Both call the same module; the script keeps argument parsing and file
I/O only.

Progress and cancellation are only possible because the app owns the loop —
`SeaASR.transcribe()` exposes no callback.