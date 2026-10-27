# indo-audio-transcribe

Turn Indonesian recordings into text — from a browser, or from the terminal.

A local web app and a command-line script share one transcription pipeline, so the
two cannot drift apart. Drop in an audio or video file, watch it work, download the
transcript as `.txt`, `.srt`, `.vtt` or `.json`.

Runs entirely on your own machine: CPU only, no API keys, no uploads to a third
party, no accounts.

---

## The model

This project does not train or tune anything. It wraps an existing open model:

| | |
| --- | --- |
| **Model** | [`PixelML/sea-asr-id`](https://huggingface.co/PixelML/sea-asr-id) |
| **Library** | [`sea-asr`](https://github.com/PixelML/sea-asr) 0.2.1 |
| **Architecture** | CTC speech recognition on the EmbeddingGemma 2 audio tower |
| **Language** | **Indonesian only** — the model has no other language |
| **Runs on** | CPU, PyTorch 2.14.1 (CPU wheel) |

Two consequences follow, and both are design constraints rather than choices:

**There is no language picker.** The model is Indonesian-only and fixed, so
offering a language dropdown would be a lie. If you need another language, you need
a different model — this one will produce confident nonsense in Javanese, English
or anything else.

**There is no model picker.** Same reason. The model id lives in one place
([`mytranscribe/pipeline.py`](mytranscribe/pipeline.py)) and every Job records which
model produced its Transcript.

`sea-asr` offers three decoders, exposed here as `Mode`:

| Mode | What it adds | Speed |
| --- | --- | --- |
| `greedy` | nothing — raw acoustic output | fastest |
| `lm` | an Indonesian n-gram language model (**default**) | same as greedy |
| `rescore` | rescoring with a 1B parameter model | several times slower |

`lm` is the default and costs nothing in speed, because the acoustic encoder
dominates: measured 127.5 s versus 127.4 s, with visibly better output. `rescore`
is accurate and slow on CPU; reach for it only when the text matters more than the
wait.

The first run downloads the model (a few hundred MB) and takes ~12–25 s to load.
Hugging Face may ask you to accept Google's Gemma terms for the base encoder; if
so, the app prints the exact steps and the URL rather than failing obscurely.

---

## Requirements

- **Python 3.12+**
- **ffmpeg** and **ffprobe** (`sudo apt install ffmpeg`)
- **git** (the CLI installs `sea-asr` from a Git tag)
- **RAM:** one transcription peaks around 7.7 GB. On a 16 GB machine that means
  running one Job at a time, which is exactly what the Worker does. See
  [ADR-0002](docs/adr/0002-single-worker-serial-jobs.md).

No GPU needed. No CUDA. The CPU-only PyTorch wheel is ~200 MB instead of ~3 GB.

---

## Quick start — the web app

```bash
./serve.sh start            # http://127.0.0.1:8765
```

Open the page, drop in a recording, pick a format, press the button. The first
start creates `.venv`, installs dependencies and downloads the model, so give it a
minute and watch the log:

```bash
./serve.sh status
./serve.sh logs        # follow the log
./serve.sh stop
./serve.sh restart 9100
```

`./serve.sh` runs detached, so closing the terminal does not kill a transcription
in progress.

**It binds `127.0.0.1` only, and that is deliberate.** There is no authentication,
no user accounts and no TLS, so exposing it to a network would hand anyone your
transcription service. Verified: a request from another machine on the LAN is
refused.

### What the page shows you before you commit

Choosing a file probes it and shows its duration, codec, sample rate, channel count
and a predicted wait — before anything is queued. The wait is a range, not a single
number, and it accounts for whatever is already ahead of you in the queue.

### What it deliberately does not do

- **No persistence.** Uploaded audio is deleted when the Job ends. The transcript
  lives in memory for 60 minutes, then expires and the words are gone. Download it
  before then. There is no database, no history, and nothing survives a restart.
- **No upload cap.** A long recording warns you about the wait instead of being
  refused. The Estimate is what makes that safe.
- **No speaker labels, punctuation restoration, translation or summarisation** —
  you get what the model produces.
- **No editing.** The Transcript is final; re-run if you want a different format.
- **Cancelling a running Job takes up to 30 seconds.** Cancellation is cooperative
  and checked between Pieces, so the worst case is one Piece. The page says this
  rather than letting you discover it.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MYTRANSCRIBE_PORT` / `PORT` | `8765` | port to listen on |
| `MYTRANSCRIBE_HOST` | `127.0.0.1` | address to bind |
| `MYTRANSCRIBE_THREADS` | all cores | CPU threads for the model |
| `MYTRANSCRIBE_RETENTION_SECONDS` | `3600` | how long a finished transcript stays downloadable |
| `MYTRANSCRIBE_WORK_ROOT` | `.jobs` | where per-Job temp audio lives |

### HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | the page |
| `GET` | `/health` | `200` with the model loaded, `503` without |
| `POST` | `/api/inspect` | duration, format and predicted wait **before** it becomes a Job |
| `POST` | `/api/jobs` | upload `file` + `format`, returns the Job |
| `GET` | `/api/jobs` | every Job this session, in submission order |
| `GET` | `/api/jobs/{id}` | one status snapshot — enough to redraw the page |
| `GET` | `/api/jobs/{id}/events` | Server-Sent Events: replays history, then live updates |
| `POST` | `/api/jobs/{id}/cancel` | idempotent; returns a snapshot |
| `GET` | `/api/jobs/{id}/download` | the transcript, with the right extension |

---

## Quick start — the command line

```bash
./transcribe.sh rekaman.mp3 hasil.txt
./transcribe.sh -i rapat.m4a -o rapat.srt --mode lm -t 12
```

Output format follows the extension: `.txt` (default), `.srt`, `.vtt`, `.json`.

```bash
./transcribe.sh --help
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `-m, --mode` | `greedy`, `lm` (default) or `rescore` |
| `-t, --threads` | CPU threads; defaults to half the cores |
| `--raw` | convert to 16 kHz mono without the clean-up filters |
| `--keep-wav` | keep the optimised WAV next to the output |
| `--allow-video` | accept a video file and use its audio track |
| `--reinstall` | reinstall the Python packages |

The script creates and manages its own `.venv`, so it works from a clean checkout
with nothing but Python, ffmpeg and git installed. It reports the input's format,
codec, sample rate and duration, then transcribes.

Both front doors call the same audio preparation and the same pipeline; the script
owns argument parsing, the venv and the messages, and nothing else.

---

## How it works

Long recordings are cut into **Pieces** of at most 30 seconds. Each Piece is
transcribed in turn with the Piece's audio offset applied to its word timings, so a
13-minute file becomes 29 sequential Pieces rather than one 13-minute allocation.
That is what keeps memory flat and makes progress reporting and cancellation
possible — you get "14 of 29 Pieces", not a spinner.

The consequence to know about: word timings come back relative to the Piece, so
they must be rebased onto the whole recording. Getting that wrong produces
subtitles that restart every 30 seconds. It is the single most likely bug in this
design, which is why the slow suite asserts that cue timestamps are monotonically
increasing and that the last cue ends near the end of the audio.

Design decisions and their reasoning are in [`docs/adr/`](docs/adr/):

- [ADR-0001](docs/adr/0001-web-app-owns-the-pipeline.md) — one pipeline, two front doors
- [ADR-0002](docs/adr/0002-single-worker-serial-jobs.md) — a single Worker, Jobs run serially
- [ADR-0003](docs/adr/0003-no-persistence-download-only.md) — no persistence, download only

Domain vocabulary (Job, Queue, Worker, Piece, Transcript, Estimate) is in
[`GLOSSARY.md`](GLOSSARY.md). The measured numbers behind the design are in
[`docs/measured-constraints.md`](docs/measured-constraints.md).

---

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[dev]'

.venv/bin/python -m pytest       # fast suite, ~4 min
```

The `--extra-index-url` is not decoration. Without it pip resolves `torch` from
PyPI and downloads the CUDA build — 1.1 GB instead of 770 MB, plus its `nvidia-*`
dependencies — for a program that never touches a GPU. `transcribe.sh` pins the same
index.

`.venv` here has no GPU build and needs no model weights, but it does need the
`sea-asr` library, because the audio loader and subtitle writers live there. It is
not on PyPI, so `pyproject.toml` points at the upstream git tag. `./transcribe.sh`
installs it the same way.

The fast suite drives a fake transcriber, so it never loads the 2.8 GB of weights
and never downloads a model.

### The parity suite

The slow tests prove the standing claim that driving the model Piece by Piece
reproduces what the one-shot path produces. They need a real recording — and none
ships with this repo, because a recording is somebody's speech and its transcript
is their words.

Point them at your own:

First install the extra the parity suite needs — `lm` mode decodes with a language
model, which is a separate C++ build:

```bash
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[slow]'
```

Then point the suite at your own recording:

```bash
export MYTRANSCRIBE_REFERENCE_AUDIO=/path/to/recording.mp3
export MYTRANSCRIBE_REFERENCE_TRANSCRIPT=/path/to/one-shot-output.txt

MYTRANSCRIBE_RUN_SLOW=1 .venv/bin/python -m pytest tests/slow -m slow -o addopts=""
```

The transcript should be what `./transcribe.sh` produces for that same recording.
Without these, the slow tests skip with an explanation rather than fail.

They assert three things:

- `.txt` is **byte-identical** to the one-shot output;
- `.srt`, `.vtt` and `.json` match within **5 ms** per timestamp — not byte for
  byte, because formatting milliseconds from floats rounds differently;
- timestamps are **monotonically increasing** and the last cue ends within a Piece
  of the audio's duration, which is what catches a missing Piece offset.

Expect ~8–10 minutes and real memory use: run it with the server stopped, and only
one model process at a time.

### Layout

```
mytranscribe/
  audio.py     probe and prepare input: reject, decode, filter, resample
  prepare.py   the same preparation, as a command the bash script runs
  pipeline.py  cut into Pieces, transcribe, rebase word timings, render
  jobs.py      Job state, the Queue, and the signals observers wait on
  worker.py    one thread, one Job at a time, holding the model
  estimate.py  the wait, predicted from duration and queue depth
  config.py    what the app needs to know to start
  app.py       the HTTP surface
  server.py    startup: load the model, then serve
  cli.py       the command-line entry point transcribe.sh calls
  static/      the page, as plain files — no build step
transcribe.sh  the original CLI, now a thin wrapper over the same pipeline
serve.sh       start/stop/status/logs for the managed background process
tests/         fake-transcriber suite; tests/slow/ drives the real model
docs/          ADRs, spec, measurements
```

---

## Troubleshooting

**`Hugging Face refused the download`** — the base encoder is gated. Open the URL
the error prints, accept the terms, then `hf auth login` (or `export HF_TOKEN=...`).
The message names the exact page.

**The first run takes a long time.** It is installing PyTorch (~200 MB) and
downloading the model. Only the first run.

**`Address already in use`.** Port 8765 is taken. `./serve.sh start 9100`, or set
`MYTRANSCRIBE_PORT`.

**Transcription is slow.** Check the thread count: `-t` for the CLI,
`MYTRANSCRIBE_THREADS` for the app. Measured throughput on a 12-core CPU is
roughly 6–8x realtime, so a 13-minute recording takes about 100–130 s.

**Out of memory.** One transcription peaks near 7.7 GB. Stop anything else using
memory; do not run two model processes at once.

**`kenlm` build fails.** The `lm` mode compiles a C++ extension:
`sudo apt install build-essential cmake python3-dev zlib1g-dev libbz2-dev liblzma-dev`,
or use `--mode greedy`.

---

## Licence and credit

The model is [`PixelML/sea-asr-id`](https://huggingface.co/PixelML/sea-asr-id),
built on Google's EmbeddingGemma 2 audio tower; its terms and the Gemma terms apply
to the model weights. The `sea-asr` library is
[PixelML/sea-asr](https://github.com/PixelML/sea-asr).