"""The CLI front door: what `transcribe.sh` runs after it has prepared the venv.

Argument parsing, venv management and the human-facing messages stay in
`transcribe.sh`, because that is the interface its users script against
(ADR-0001). Everything below the messages — probing, Piece cutting, the model
loop, format writing — lives in the modules this front door calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3:
        print("[x] usage: python -m mytranscribe.cli WAV OUTPUT FORMAT MODE THREADS", file=sys.stderr)
        return 2

    source, output, output_format, mode, threads = args[:5]
    extra = args[5:]

    import torch

    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(max(1, int(threads) // 2))
    except RuntimeError:
        pass

    from .pipeline import MODEL_ID, TranscribeRequest, transcribe_audio

    import time

    loaded_at = time.monotonic()
    asr = load_asr(MODEL_ID, mode)
    print(f"[+] Model loaded in {time.monotonic() - loaded_at:.1f}s", file=sys.stderr)

    request = TranscribeRequest(
        source=Path(source),
        output_format=output_format,
        destination=Path(output),
        model=MODEL_ID,
        mode=mode,
    )
    duration = float(extra[0]) if extra and extra[0] not in ("", "N/A") else 0.0

    started = time.monotonic()
    transcript = transcribe_audio(request, transcriber=asr)
    elapsed = time.monotonic() - started

    speed = f", {duration / elapsed:.1f}x real time" if duration and elapsed else ""
    print(f"[+] Transcribed in {elapsed:.1f}s{speed}", file=sys.stderr)
    return 0


def load_asr(model_id: str, mode: str):
    """Load the model the way the script always has, with its download advice."""
    from sea_asr import SeaASR

    keyword_args: dict[str, object] = {}
    if mode in ("lm", "rescore"):
        keyword_args["lm"] = model_id
    if mode == "rescore":
        keyword_args["rescore"] = True

    try:
        try:
            return SeaASR.from_pretrained(model_id, device="cpu", **keyword_args)
        except TypeError:  # a release without the device argument
            return SeaASR.from_pretrained(model_id, **keyword_args)
    except Exception as error:  # noqa: BLE001 - the message is the point
        message = str(error)
        gated = ("401", "403", "gated", "Gated", "restricted", "authenticat")
        if any(marker in message for marker in gated):
            raise SystemExit(
                "\n[x] Hugging Face refused the download. The base encoder "
                "(google/embeddinggemma-2)\n"
                "    may require accepting Google's Gemma terms:\n"
                "      1. open https://huggingface.co/google/embeddinggemma-2 "
                "and accept the terms\n"
                "      2. source .venv/bin/activate && hf auth login   "
                "(or export HF_TOKEN=hf_...)\n"
                f"    Original error: {message}"
            ) from error
        raise


if __name__ == "__main__":  # pragma: no cover - process entry point
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())