"""The preparation front door: facts out, a 16 kHz mono WAV, or a refusal.

`transcribe.sh` runs this **before** it creates the venv, so this module and
`audio.py` must stay importable with the standard library alone — no torch, no
sea_asr, no third-party anything. That is also why the output is a flat
`key=value` stream on stdout rather than JSON or Python objects: bash reads it
with a `while IFS='=' read` loop, which is what the script already did.

The contract:

    stdout   key=value lines describing the Original audio, then `wav=<path>`
    exit 0   audio is ready at the `wav=` path — empty when --keep-wav was not
             given, because the audio is removed on exit
    exit 1   refused, with a human-readable reason on stderr

The scratch directory is always removed before this process exits, so a caller
that wants the audio must ask for it with `--keep-wav`. That is why `wav=` names
the kept copy rather than the scratch path: a path that no longer exists is worse
than no path.

The refusal sentences live with the reasons, in `audio.py`, so there is one
wording per reason rather than one per front door.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .audio import (
    NO_AUDIO_STREAM,
    UNREADABLE,
    VIDEO_NOT_ALLOWED,
    AudioRejected,
    prepare_audio,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mytranscribe.prepare",
        description="Probe and optimise audio for transcription.",
    )
    parser.add_argument("source", help="the audio file to prepare")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="only convert to 16 kHz mono, skip the clean-up filters",
    )
    parser.add_argument(
        "--allow-video",
        action="store_true",
        help="accept a video container and use its audio track",
    )
    parser.add_argument(
        "--keep-wav",
        metavar="PATH",
        help="copy the prepared WAV to this path",
    )
    parser.add_argument("--threads", type=int, default=None, help="ffmpeg thread count")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        prepared = prepare_audio(
            args.source,
            raw=args.raw,
            allow_video=args.allow_video,
            threads=args.threads,
        )
    except AudioRejected as rejected:
        raise SystemExit(refusal_sentence(rejected)) from rejected

    facts = prepared.facts
    # The WAV is printed before cleanup so the caller can copy it; whoever owns
    # the audio must take it now, because this process removes it on the way out.
    try:
        print(f"duration={facts.duration_s:.3f}")
        print(f"format_name={facts.format_name}")
        print(f"codec={facts.codec}")
        print(f"sample_rate={facts.sample_rate}")
        print(f"channels={facts.channels}")
        print(f"audio_tracks={facts.audio_tracks}")
        # `wav` names a file the caller can use after this process exits, so it is
        # the kept copy when one was asked for. Without --keep-wav the audio is
        # gone on exit and the field says so rather than naming a deleted path.
        if args.keep_wav:
            keep = Path(args.keep_wav)
            # The caller names where it wants the audio; making the parent is ours
            # because this process writes there and the caller may not have.
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared.path, keep)
            print(f"wav={args.keep_wav}")
        else:
            print("wav=")
    finally:
        # Standalone use must not leak a scratch directory per run. The script
        # also cleans up after itself, but neither caller should have to.
        prepared.cleanup()
    return 0


def refusal_sentence(rejected: AudioRejected) -> str:
    """The refusal, in the words this front door's callers expect."""
    if rejected.reason == VIDEO_NOT_ALLOWED:
        # audio.py already carries this sentence; no second copy here.
        return str(rejected)
    if rejected.reason == NO_AUDIO_STREAM:
        return f"Not an audio file (ffprobe found no audio stream): {rejected.detail}"
    if rejected.reason == UNREADABLE:
        # The script's wording for a file ffprobe cannot open at all.
        return f"Could not read this file: {rejected.detail}"
    return str(rejected)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())