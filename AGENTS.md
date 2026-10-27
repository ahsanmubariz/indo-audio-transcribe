# Working on this repo

Notes for anyone — human or agent — changing this code.

## Read these first

- [`GLOSSARY.md`](GLOSSARY.md) — the domain vocabulary. Use these words in code,
  comments, commit messages and UI copy. **Job**, **Queue**, **Worker**, **Piece**,
  **Transcript**, **Word timings**, **Mode**, **Original audio**, **Estimate**.
  Avoid the listed alternatives: "task", "request", "chunk", "segment", "result",
  "output", "ETA".
- [`docs/adr/`](docs/adr/) — why the architecture is what it is. Read the ADR before
  proposing a change that contradicts one; if the change is right, supersede the
  ADR rather than quietly diverging.
- [`docs/spec/web-transcription-app.md`](docs/spec/web-transcription-app.md) — the
  behaviour the app is meant to have.

## The one invariant

**Both front doors call the same pipeline.** `transcribe.sh` and the web app differ
in argument parsing, process management and wording — never in how audio is
prepared or transcribed. `mytranscribe/audio.py` and `mytranscribe/pipeline.py` are
shared; `prepare.py` exists so the bash script can reach `audio.py` before the venv
exists.

If you find yourself writing ffmpeg flags or a Piece loop in `transcribe.sh`, stop:
that is the duplication the split exists to prevent.

## The one bug to expect

Word timings from the model are **relative to the Piece**, not the recording.
Anything that writes timestamps must add the Piece offset, or subtitles restart
every 30 seconds. The slow suite asserts monotonicity and an end-time near the
audio's duration specifically to catch this.

## Tests

```bash
.venv/bin/python -m pytest          # fast, fake transcriber, offsets never load, ~4 min
```

Everything in the default suite injects a fake transcriber, so it never loads the
weights. It still needs the `sea-asr` library — the audio loader and subtitle
writers live there — but not the model download. Keep it that way: new tests should
use `tests/fakes.py` unless the real model is genuinely what is under test.

Real-model tests live in `tests/slow/`, are marked `slow`, and are skipped unless
`MYTRANSCRIBE_RUN_SLOW=1` and a reference recording are both supplied. Write new
ones the same way: guard with `require_reference_audio()` so a missing recording
skips with instructions instead of failing.

Measured behaviour beats assumed behaviour. If you change the model, the library
version or the hardware, the numbers in `docs/measured-constraints.md` are void —
re-measure them rather than editing the prose to match a guess.

## Commit messages

Conventional prefixes, imperative mood, explaining *why*:

```
web app: keep the download visible after a Job finishes
pipeline: rebase word timings onto the whole recording
tests: cover cancellation between Pieces, not during one
docs: record the memory ceiling the slow suite hit
```

## Style

No formatter is enforced. Match the file you are in:

- Python: standard library types, `from __future__ import annotations`, docstrings
  that explain reasoning rather than restating the signature.
- Comments explain *why*. The code already says what.
- UI copy is Indonesian, and written for a person who has never used the app.