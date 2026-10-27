## Problem Statement

Transcribing an Indonesian recording currently means opening a terminal and
running `./transcribe.sh input.mp3 hasil.txt`. That works, but it is unusable
from a browser and gives no feedback while it runs: a 13-minute recording takes
about 2.5 minutes of CPU time with no output until the very end, no way to see
how far along it is, and no way to stop it. The output format has to be
remembered as a command-line flag, and a refused input (a video, a corrupt file)
is reported as a terminal error that has to be read and interpreted.

The user wants to drop an audio file into a page, watch it being worked on, and
end up with the transcript as a downloadable file.

## Solution

A local web app, bound to `127.0.0.1`, that accepts an audio upload, shows it as
a queued or running **Job** with live per-**Piece** progress and an **Estimate**,
and lets it be downloaded in the chosen format when finished. One **Worker**
processes Jobs strictly one at a time, because the model's 7.7 GB peak RSS leaves
no room for a second. Only the language-model **Mode** is offered, since it
measures the same speed as `greedy` and is more accurate. Nothing is stored: a
finished **Transcript** is held in memory for 60 minutes or until downloaded.

`transcribe.sh` keeps working exactly as it does today, reimplemented as a thin
CLI over the same pipeline module the web app uses.

## User Stories

1. As a user, I want to upload an audio file from the browser, so that I do not have to use a terminal.
2. As a user, I want to select the output format before the Job starts, so that I get a `.txt`, `.srt`, `.vtt` or `.json` file without re-running.
3. As a user, I want `.txt` to be preselected, so that the common case needs no decision.
4. As a user, I want to see the uploaded file's duration and format before committing, so that I know I picked the right file.
5. As a user, I want a rejected upload to explain why in plain language, so that I can fix it without reading a stack trace.
6. As a user, I want to upload a video file and have its audio track used, so that a `.mp4` from a meeting recording just works.
7. As a user, I want to see my Job enter a Queue and learn my position, so that I know it was accepted and roughly when it will start.
8. As a user, I want an Estimate of how long the Job will take, so that I can decide whether to wait or do something else.
9. As a user, I want live progress as each Piece finishes, so that the wait is visibly moving rather than silent.
10. As a user, I want progress to survive a page refresh, so that an accidental reload does not lose sight of a running Job.
11. As a user, I want to cancel a Job that is still queued, so that I can change my mind instantly.
12. As a user, I want to cancel a running Job, so that I can stop work I no longer need.
13. As a user, I want cancellation to be prompt, so that I am not waiting for a long Job to finish after asking it to stop.
14. As a user, I want to see partial text as Pieces complete, so that I can tell the transcription is working on real content.
15. As a user, I want to download the finished Transcript, so that I can use it elsewhere.
16. As a user, I want the download to be obvious, so that I do not lose a 2.5-minute result by closing the page.
17. As a user, I want the finished Transcript to remain available for 60 minutes, so that a refresh or a moment's distraction does not destroy it.
18. As a user, I want a Job that has been downloaded to remain downloadable again briefly, so that a second download does not require re-running.
19. As a user, I want to know when a Transcript has expired, so that I understand why a download no longer works.
20. As a user, I want the audio kept in a temporary location for the life of its Job, so that the Transcript can be derived without holding the whole file in memory.
21. As a user, I want clear errors when the Worker is unavailable, so that I can tell the difference between "broken" and "busy".
22. As a user, I want a failure during transcription to be reported against the Job, so that I do not have to guess what happened.
23. As a user, I want to see the Job's Mode and format in the UI, so that I know what produced the result.
24. As a user, I want only useful options exposed, so that I cannot pick a slower, worse configuration by accident.
25. As a user, I want the page to work without a build step or a running dev server, so that starting it is one command.
26. As a user, I want the app to bind to localhost only, so that I am not exposing an unauthenticated service to my network.
27. As a user, I want the port configurable, so that it does not collide with the other services on this machine.
28. As a user, I want the app to keep working when the model has already been loaded, so that a second Job does not pay the load cost again.
29. As a user, I want the model to load once at startup with a clear indication, so that the first upload is not mysteriously slow.
30. As a user, I want unchanged `transcribe.sh` behaviour, so that my existing command lines and habits keep working.
31. As a user, I want `transcribe.sh` to still support its current flags and output extensions, so that nothing I script breaks.
32. As a user, I want the web app and the script to produce the same transcript for the same input, so that I can trust both.
33. As a user, I want the web app to use all available cores, so that a Job finishes as fast as the machine allows.
34. As a user, I want the thread count configurable, so that I can leave headroom when the machine is busy.
35. As a user, I want an oversized upload to warn me about the wait without being rejected, so that I can make an informed choice.
36. As a user, I want to queue more than one file, so that I can hand over a batch and walk away.
37. As a user, I want queued Jobs to run in the order they were submitted, so that the queue is predictable.
38. As a user, I want to see my position move up the Queue, so that a long wait feels accounted for.
39. As a user, I want progress to be accurate rather than decorative, so that I can trust a displayed percentage.
40. As a user, I want the app never to claim a Mode it is not actually using, so that the UI is not lying about how the text was produced.
41. As a user, I want the Indonesian-only nature of the model reflected in the UI, so that I am not offered a meaningless language picker.
42. As a user, I want temp audio cleaned up when a Job ends, so that stray files do not accumulate on disk.
43. As a user, I want the app to start as a managed background process, so that it does not occupy a terminal.
44. As a user, I want a health endpoint, so that I can check the Worker is alive and the model is loaded.
45. As a user, I want the app to refuse to start if the model cannot load, so that I learn immediately rather than at the first upload.

## Implementation Decisions

### Modules

- **Pipeline module** — extracts the logic currently embedded in `transcribe.sh`
  into an importable module. Responsibilities: audio probing and optimisation via
  ffmpeg, Piece splitting, batched transcription through `SeaASR`, writing each
  output format, and reporting progress. It exposes a single entry point that
  takes a source audio path, an output format, and an optional progress callback
  and cancellation check, and returns the finished Transcript.
- **CLI front door** — `transcribe.sh`'s argument parsing, venv management and
  messages are retained; the transcription body delegates to the pipeline module.
  External behaviour, flags and output formats are unchanged (ADR-0001).
- **Worker** — owns the loaded `SeaASR` instance and the Queue. Processes one Job
  at a time, driving the pipeline module Piece by Piece so that progress and
  cancellation are observable (ADR-0002).
- **Job store** — in-memory record of each Job: identity, Original audio path,
  format, Mode, state, progress, partial text, finished Transcript, and creation
  and completion timestamps. No database (ADR-0003).
- **HTTP surface** — upload, enqueue, status snapshot, SSE progress stream,
  cancellation, and download.

### Interfaces

- The pipeline entry point is the only thing both front ends call. It must not
  take FastAPI types, and must not know about Jobs or Queues — it knows about
  audio, formats, Pieces, progress and cancellation only.
- Progress is reported as Pieces completed out of Pieces total, plus the partial
  text accumulated so far. Percentages are derived for display, never stored as
  the source of truth.
- Cancellation is cooperative: checked between Pieces, so the worst-case lag is
  one Piece (at most 30 seconds). See "Further Notes" for a decision still open.

### Job lifecycle

A Job moves through: queued → running → finished, with failed and cancelled as
terminal states reachable from queued or running. Cancelling a queued Job is
immediate; cancelling a running Job takes effect at the next Piece boundary.
Downloading does not end a Job; the Transcript remains downloadable until its
60-minute retention expires or the Worker restarts.

### API contracts

- Upload accepts one audio file plus a format; returns a Job identity and queue
  position. Validation happens before enqueue: no audio stream, or an unreadable
  file, is a rejected upload with a human-readable reason, not a failed Job.
- A status snapshot returns state, Pieces completed and total, queue position,
  Estimate, format, Mode, and any error message. This endpoint exists so that a
  reloaded page can reconstruct state before subscribing to the stream.
- The progress stream is Server-Sent Events, emitting the same payload shape as
  the snapshot. It terminates when the Job reaches a terminal state.
- Cancellation is idempotent: cancelling a finished, failed or already-cancelled
  Job is not an error.
- Download serves the Transcript in the format chosen at upload, with a filename
  derived from the Original audio's name.

### Technical clarifications

- The Worker holds the model for the process lifetime; the model loads once at
  startup, not per Job. Model load was measured at ~5.9–10.7 s, substantial
  relative to a short Job.
- The Worker defaults to **all** cores rather than the script's half-cores
  default, because measurement showed 8.3x realtime at 12 threads against 6.5x at
  6 threads. The script's conservative default stays as it is.
- Only the language-model Mode is wired up. `greedy` and `rescore` are not
  exposed; `rescore` in particular is unmeasured and would add roughly 4 GB of
  float32 weights on top of a 7.7 GB peak.
- Word timings are decided at upload because requesting them changes the decode
  path, not just the serialisation. The `.txt` default avoids them entirely.
- **Word timings are relative to their Piece and must be rebased.** Verified
  empirically: Piece 1 of the reference recording starts at 28.7 s and reports its
  first word at `start: 0.56`. A pipeline that forgets to add the Piece's offset
  produces subtitles whose timestamps reset to zero every 30 seconds — plausible
  looking files that are silently wrong. Every word's start and end must have its
  Piece offset added before any timestamped format is written.
- There is no upload size cap. Instead, an upload whose audio is long enough to
  imply a substantial wait shows an Estimate before and after enqueue.
- The service binds to `127.0.0.1` and takes its port from configuration.
- `python-multipart`, FastAPI, uvicorn and pytest are not currently installed in
  the existing virtual environment. A dry-run install of all four was checked and
  is **purely additive**: it adds fastapi, starlette, pydantic, uvicorn, pytest,
  python-multipart and a few small transitive packages, and upgrades nothing in
  the working model stack (torch, transformers, sea_asr untouched). Installing
  into the existing environment is therefore safe, with a recorded dependency lock
  so the working model versions cannot drift underneath it.

## Testing Decisions

A good test here asserts externally observable behaviour and is indifferent to
how the code is arranged internally. It should fail when behaviour changes, not
when a function is renamed or moved. Concretely: tests drive the public entry
points (the pipeline entry point and the HTTP surface), assert on returned
Transcripts, emitted progress and HTTP responses, and never reach into the Job
store's internals or assert on private helpers.

Two seams were agreed.

**Seam 1 — the pipeline module.** Tested with a fake ASR so the Piece loop,
progress reporting, cancellation and format writing can be exercised quickly and
deterministically. Plus slow tests using the real model, asserting the pipeline
produces output matching what `transcribe.sh` produces for the same input. Those
are the standing proof of ADR-0001's parity claim and are marked slow so the
default suite stays fast.

> **Amended during the build.** This originally asked for *exactly one* slow test
> on a few seconds of audio. That number became untenable once the acceptance
> criteria were written: byte equality, the Piece count, the 5 ms tolerance across
> every cue and the under-200 s budget can only be checked against a real
> long recording. Kept honest instead of quietly exceeded: the one-shot comparison
> the spec describes runs on a 60 s slice (a one-shot pass over a long file needs
> the whole decode resident and was being OOM-killed beside the per-Piece runs),
> while the full recording goes through the per-Piece path that ships. Every
> expensive run is shared across tests.
>
> **No recording ships with the repo.** A real recording is somebody's speech and
> its transcript is their words, so the parity suite takes them from the
> environment (`MYTRANSCRIBE_REFERENCE_AUDIO`, `MYTRANSCRIBE_REFERENCE_TRANSCRIPT`)
> and skips with instructions when they are absent.

The parity assertion has **two different tolerances**, because the two output
classes behave differently, both measured against the one-shot path on a real
13-minute Indonesian recording:

- **Text formats** (`.txt`) must match the one-shot output **byte for byte**.
  Measured: identical, character for character.
- **Timestamped formats** (`.srt`, `.vtt`, `.json`) must match within **5 ms** per
  timestamp, not byte for byte. Measured: maximum drift 1 ms, mean 0.03 ms, with a
  handful of lines differing by a single millisecond of float rounding. Demanding
  byte equality here would produce a test that fails on differences no human can
  perceive.

A third assertion is worth adding beyond parity: for a multi-Piece input, the
timestamps in the resulting `.srt` must be **monotonically increasing** and the
last cue must end near the audio's duration. That is what catches the
missing-offset bug directly, rather than relying on a byte comparison to notice
it.

**Seam 2 — the HTTP surface**, driven through the framework's test client with
the pipeline injected as a fake. Covers upload validation, enqueue ordering,
queue position, snapshot payload, the SSE stream's framing and termination,
idempotent cancellation, download content and filename, retention expiry, and
rejection of a file with no audio stream.

**Prior art:** none — this is the first test suite in the repo. The conventions
established here become the prior art: tests live alongside the code they cover,
the fast suite runs by default, and slow model tests are opted into explicitly.

The refactor in step one is the riskiest part of the build, since it touches the
only thing that currently works. Parity tests come before the web app, not after.

## Out of Scope

- Speaker diarisation, punctuation restoration beyond what the model produces,
  translation, and summarisation.
- Languages other than Indonesian; `PixelML/sea-asr-id` is fixed.
- `greedy` and `rescore` Modes in the UI.
- Persistence of any kind: no database, no history, no browsing past
  transcripts, no accounts.
- Authentication, authorisation, quotas and multi-user support.
- Deployment beyond a local managed process: no container, no TLS, no reverse
  proxy.
- Transcribing multiple files as one Job.
- Editing, correcting or re-formatting a Transcript after it is produced.
- Job resumption across a Worker restart.
- Re-encoding or trimming the Original audio in the UI.

## Further Notes

**Open decision — cancellation latency.** Cooperative checks between Pieces mean
up to 30 seconds before a cancel takes effect. With 29 Pieces on a 13-minute file
that was measured as acceptable, but it is a real limitation and should be stated
in the UI rather than discovered. If it proves annoying, the fix is to check
inside the batch at a finer boundary, which changes the pipeline module's
contract and should be a deliberate follow-up rather than a quiet one.

**The compounding failure to watch.** No persistence and no upload cap combine
badly: a long upload blocks the Queue for a long time, and the result is lost if
the page closes before download. The mitigations are a visible Estimate and an
unmissable download, not a hidden safety net. If this bites in practice, revisit
ADR-0003 rather than patching around it.

**Threads.** The 27% gain from 12 versus 6 threads is measured, but scaling is
flattening: 6→8 gained 19 s, 8→12 gained only 6.7 s. On a busier machine the
conservative choice may be better, which is why the count is configurable.

**Numbers worth re-measuring, not assuming:** 7.7 GB peak RSS, 6.2x–8.3x
realtime, and ~5–10 s model load. Any change to the model, the library version or
the hardware invalidates the concurrency decision that rests on them.

## Acceptance Criteria

Every criterion is checkable by running something, not by reading code.

**Parity (the refactor is correct)**

1. On the reference recording, the pipeline's `.txt` output is byte-identical to
   the one-shot script's output for the same recording, byte for byte.
2. `./transcribe.sh -i <recording>.mp3 -o out.txt --mode lm` still runs and
   produces that same output after the refactor.
3. `transcribe.sh` still accepts its documented flags (`-i/-o/-m/-t`, `--raw`,
   `--keep-wav`, `--allow-video`) and all four output extensions.
4. `.srt` from the pipeline matches `.srt` from the one-shot path within 5 ms per
   timestamp (measured ceiling: 1 ms).
5. In a multi-Piece `.srt`, timestamps are monotonically increasing and the final
   cue ends within one Piece of the audio's duration (measured: within 6 s on a
   13-minute recording   file).

**Speed (no regression)**

6. Transcribing the reference recording via the pipeline completes in under
   200 s at 12 threads. (Measured: 101.4 s per-Piece at 12 threads; 147.9 s
   one-shot.)
7. Peak RSS for one Job stays under 9 GB. (Measured: 7.7 GB.) Exceeding this
   invalidates ADR-0002 and must be surfaced, not ignored.

**Job lifecycle**

8. Uploading two files while nothing is running yields exactly one `running` Job
   and one `queued` Job, in submission order.
9. Cancelling a queued Job removes it from the Queue immediately, and its status
   endpoint reports `cancelled`.
10. Cancelling a running Job reports `cancelled` within one Piece, and no further
    Pieces are transcribed afterwards.
11. Cancelling an already-finished Job returns success and does not change its
    stored Transcript.

**Progress**

12. A status snapshot returns state, pieces completed, pieces total, queue
    position, Estimate, format, Mode and any error — enough to redraw the page
    with no prior state.
13. The SSE stream emits at least one event per completed Piece and terminates
    after the terminal event.
14. Reloading the page mid-Job reconstructs the same progress as the live stream
    showed.

**Upload validation**

15. A file with no audio stream is rejected before enqueue with a human-readable
    reason, and no Job is created.
16. A `.mp4` containing audio is accepted and transcribed. (Verified extractable:
    a 3.0 s test video yielded 3.02 s of audio.)
17. An unreadable or corrupt file is rejected without creating a Job.

**Retention**

18. A finished Transcript downloads successfully, and can be downloaded again
    immediately afterwards.
19. A Transcript is still downloadable at 59 minutes and gone at 61 minutes.
20. Requesting an expired Transcript returns a clear "expired" response, not a
    500.

**Operational**

21. `GET /health` reports model-loaded status.
22. The server refuses to start, with a clear message, if the model cannot load.
23. The server binds only to `127.0.0.1`.
24. The port is configurable and the app does not bind `3080`.

## Doability Notes

What verification confirmed, so implementers are not re-litigating settled ground:

- **The whole approach works.** Per-Piece transcription through the public API
  reproduces the one-shot text exactly, so nothing depends on private internals
  beyond the audio helpers already used (`split_long`, `make_batches`, `Piece`).
- **The speed claim is real but modest.** 101.4 s versus 147.9 s is a genuine
  improvement, but both are minutes on a 13-minute file; the UI's value is
  feedback during the wait, not speed.
- **One known trap, now documented:** word timings are Piece-relative and must be
  rebased. This is the single most likely way to ship a plausible but wrong
  transcript.
- **Do not take byte-equality of SRT as the parity bar** — it fails on
  millisecond float rounding. Use the 5 ms tolerance.
