"""Starting the app: load the model once, then serve on loopback.

The model loads at startup, not per Job — a Job pays no load cost after the first
(measured 5.9–10.7 s). If it cannot load, the server refuses to start with a clear
message rather than failing mysteriously at the first upload.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import AppConfig


def load_transcriber(config: AppConfig):
    """Load the model, or raise with a message worth reading."""
    import torch

    torch.set_num_threads(int(config.threads))
    try:
        torch.set_num_interop_threads(max(1, int(config.threads) // 2))
    except RuntimeError:
        pass  # already initialised by an earlier call

    from sea_asr import SeaASR

    try:
        try:
            return SeaASR.from_pretrained(config.model_id, device="cpu", lm=config.model_id)
        except TypeError:  # a release without the device argument
            return SeaASR.from_pretrained(config.model_id, lm=config.model_id)
    except Exception as error:  # noqa: BLE001 - the message is the whole point
        raise ModelUnavailable(_model_advice(config.model_id, error)) from error


class ModelUnavailable(RuntimeError):
    """The model could not be loaded, so the app has nothing to transcribe with."""


def _model_advice(model_id: str, error: Exception) -> str:
    message = str(error)
    gated = ("401", "403", "gated", "Gated", "restricted", "authenticat")
    if any(marker in message for marker in gated):
        return (
            f"The model {model_id} could not be downloaded: Hugging Face refused it.\n"
            "  The base encoder (google/embeddinggemma-2) may require accepting "
            "Google's Gemma terms:\n"
            "    1. open https://huggingface.co/google/embeddinggemma-2 and accept the terms\n"
            "    2. run: hf auth login   (or export HF_TOKEN=hf_...)\n"
            f"  Original error: {message}"
        )
    return (
        f"The model {model_id} could not be loaded, so the app will not start.\n"
        "  Check that the model files are available and that this machine has "
        "enough memory.\n"
        f"  Original error: {message}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mytranscribe",
        description="Transcribe Indonesian recordings in the browser, on this machine only.",
    )
    parser.add_argument("--port", type=int, help="port to listen on (default from config)")
    parser.add_argument("--host", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--threads", type=int, help="CPU threads to use (default: all cores)")
    parser.add_argument(
        "--retention-minutes",
        type=float,
        help="how long a finished transcript stays downloadable (default 60)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="start without loading the model (for UI work only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env()
    if args.port is not None:
        config.port = args.port
    if args.host is not None:
        config.host = args.host
    if args.threads is not None:
        config.threads = args.threads
    if args.retention_minutes is not None:
        config.retention_seconds = args.retention_minutes * 60

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    transcriber = None
    if args.no_model:
        _log("Starting WITHOUT a model: uploads will be refused. For interface work only.")
    else:
        _log(f"Loading model {config.model_id} on {config.threads} threads\u2026")
        try:
            import time

            started = time.monotonic()
            transcriber = load_transcriber(config)
            _log(f"Model loaded in {time.monotonic() - started:.1f}s")
        except ModelUnavailable as unavailable:
            print(f"\n[x] {unavailable}\n", file=sys.stderr)
            return 1

    from .app import create_app

    application = create_app(
        config,
        transcriber=transcriber,
        model_loaded=transcriber is not None,
    )

    import uvicorn

    _log(
        f"Listening on http://{config.host}:{config.port} "
        f"(threads={config.threads}, model={'loaded' if transcriber else 'missing'}, "
        f"retention={int(config.retention_seconds // 60)} min)"
    )
    uvicorn.run(application, host=config.host, port=config.port, log_level="warning")
    return 0


def _log(message: str) -> None:
    print(f"[*] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())