"""The Worker: one thread, one Job at a time, holding the model.

A single Worker is a memory constraint, not a simplification — one Job peaks at
7.7 GB on a 15 GB machine (ADR-0002). Serialising also makes progress honest,
since there is only one Job whose progress is moving.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from .jobs import JobStore
from .pipeline import Cancelled, PipelineError, TranscribeRequest, render, transcribe_audio


class Worker:
    """Runs Jobs off the Queue, one at a time, until stopped."""

    def __init__(self, store: JobStore, transcriber, *, poll_seconds: float = 0.05) -> None:
        self._store = store
        self._transcriber = transcriber
        self._poll_seconds = poll_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="transcribe-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._store.claim_next()
            if job is None:
                self._stop.wait(self._poll_seconds)
                continue
            self._work(job)

    def _work(self, job) -> None:
        prepared = None
        try:
            from .audio import optimise_audio, probe_audio

            facts = probe_audio(job.original_path)
            prepared = optimise_audio(
                job.original_path, facts=facts, work_dir=job.work_dir, threads=None
            )
            transcript = transcribe_audio(
                TranscribeRequest(
                    source=prepared.path,
                    output_format=job.output_format,
                    model=job.model,
                    mode=job.mode,
                ),
                transcriber=self._transcriber,
                on_progress=lambda progress: self._store.report_progress(job.id, progress),
                should_cancel=lambda: self._store.should_cancel(job.id),
            )
            self._store.finish(job.id, transcript, transcript.rendered())
        except Cancelled as cancelled:
            self._store.cancel_pending(job.id)
            _ = cancelled
        except PipelineError as error:
            self._store.fail(job.id, str(error))
        except Exception as error:  # noqa: BLE001 - a Job must never take the Worker down
            self._store.fail(job.id, _plain_reason(error))
        finally:
            if prepared is not None:
                prepared.cleanup()
            shutil.rmtree(job.work_dir, ignore_errors=True)


def _plain_reason(error: Exception) -> str:
    """Turn an unexpected failure into something a person can read."""
    message = str(error).strip() or error.__class__.__name__
    if "memory" in message.lower():
        return (
            "There was not enough memory to finish this file. "
            "Try a shorter recording, or close other programs and try again."
        )
    return f"Transcription failed: {message}"