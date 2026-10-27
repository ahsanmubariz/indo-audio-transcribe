"""The HTTP surface: upload, status, stream, cancel, download, health, and the page.

The page is served by the same process that transcribes, as plain files — no
build step and no dev server (user story 25).
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from .audio import AudioRejected, probe_audio
from .config import AppConfig
from .jobs import CANCELLED, FINISHED, QUEUED, RUNNING, TERMINAL_STATES, JobStore
from .estimate import estimate_for
from .pipeline import OUTPUT_FORMATS
from .worker import Worker

STATIC_DIR = Path(__file__).parent / "static"

# The formats a Job may be started in. `.txt` is the default because it is the
# common case and the only one that needs no word timings.
DEFAULT_FORMAT = "txt"

MODE_LABEL = "lm"


def create_app(
    config: AppConfig,
    *,
    transcriber=None,
    model_loaded: bool = False,
    store: JobStore | None = None,
) -> FastAPI:
    """Build the app. Both `transcriber` and `model_loaded` are injected so tests
    can drive the surface without loading 2.8 GB of weights."""
    job_store = store or JobStore(retention_seconds=config.retention_seconds)
    worker = Worker(job_store, transcriber) if transcriber is not None else None

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    app = FastAPI(
        title="mytranscribe",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.store = job_store
    app.state.worker = worker
    app.state.model_loaded = model_loaded

    # ---------------------------------------------------------------- health

    @app.get("/health")
    def health() -> Response:
        payload = {
            "status": "ok" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "model": config.model_id,
            "mode": MODE_LABEL,
            "format_default": DEFAULT_FORMAT,
            "formats": list(OUTPUT_FORMATS),
            "retention_minutes": int(config.retention_seconds // 60),
        }
        return Response(
            content=json.dumps(payload),
            status_code=200 if model_loaded else 503,
            media_type="application/json",
        )

    # ---------------------------------------------------------------- upload

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: UploadFile = File(...),
        format: str = Form(DEFAULT_FORMAT),  # noqa: A002 - the wire name is `format`
    ) -> Response:
        # Without a Worker nothing would ever run this Job: it would sit queued
        # forever, holding its audio. "Broken" is a different answer from "busy".
        if worker is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The transcription model is not loaded, so nothing can be "
                    "transcribed right now. Check /health, restart the app, and "
                    "try again."
                ),
            )

        output_format = (format or DEFAULT_FORMAT).strip().lower().lstrip(".")
        if output_format == "":
            output_format = DEFAULT_FORMAT
        if output_format not in OUTPUT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"That output format is not available. Choose one of: "
                    f"{', '.join(OUTPUT_FORMATS)}."
                ),
            )

        work_dir = Path(tempfile.mkdtemp(prefix="job-", dir=_ensure(config.work_root)))
        original_name = _safe_name(file.filename or "rekaman")
        original_path = work_dir / f"original{_suffix(original_name)}"
        try:
            await _save_upload(file, original_path)
            facts = probe_audio(original_path, allow_video=True)
        except AudioRejected as rejected:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(rejected)) from rejected
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        job = job_store.create(
            original_name=original_name,
            original_path=original_path,
            work_dir=work_dir,
            output_format=output_format,
            mode=MODE_LABEL,
            model=config.model_id,
            facts=facts,
        )
        snapshot = job_store.snapshot(job.id) or {}
        return Response(
            content=json.dumps(snapshot),
            status_code=201,
            media_type="application/json",
            headers={"Location": f"/api/jobs/{job.id}"},
        )

    @app.post("/api/inspect")
    async def inspect(file: UploadFile = File(...)) -> dict:
        """Read a file's duration and kind before it becomes a Job.

        User story 4: know you picked the right file before committing. This
        writes the upload to a scratch file, probes it, and deletes it — no Job,
        no Queue, nothing kept.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="inspect-", dir=_ensure(config.work_root)))
        try:
            target = work_dir / f"probe{_suffix(_safe_name(file.filename or 'x'))}"
            await _save_upload(file, target)
            try:
                facts = probe_audio(target, allow_video=True)
            except AudioRejected as rejected:
                raise HTTPException(status_code=422, detail=str(rejected)) from rejected
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        # The glossary defines an Estimate from the Original audio's length *and*
        # the current Queue depth. Omit the depth and this number contradicts the
        # one the Job itself will show the moment it is queued.
        estimate = estimate_for(
            duration_s=facts.duration_s,
            work_ahead_s=job_store.work_ahead_seconds(),
        )
        return {
            "original_name": _safe_name(file.filename or "rekaman"),
            "duration_seconds": facts.duration_s,
            "duration_label": facts.duration_label,
            "original_audio": facts.original_audio(),
            "audio_tracks": facts.audio_tracks,
            "estimate": estimate.as_dict(),
        }

    # ---------------------------------------------------------------- status

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        snapshot = job_store.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")
        return snapshot

    # ----------------------------------------------------------------- queue

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        """Everything the page needs to redraw itself after a reload, in submission order."""
        return {"jobs": job_store.snapshots()}

    # ---------------------------------------------------------------- stream

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, request: Request) -> StreamingResponse:
        snapshot = job_store.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")

        def stream():
            sent = 0
            # A late subscriber replays history, so a reload reconstructs progress
            # rather than starting from whatever happens next.
            while True:
                history, closed = job_store.history(job_id)
                while sent < len(history):
                    yield _event(history[sent])
                    sent += 1
                if closed:
                    return
                if _client_gone(request):
                    return
                job_store.wait_for_change(job_id, timeout=1.0)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ---------------------------------------------------------------- cancel

    @app.delete("/api/jobs/{job_id}")
    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        """Idempotent: cancelling a finished, failed or already-cancelled Job is not an error.

        The response is a full status snapshot, because the page stores it as the
        Job's state — a body of a different shape would blank the card it is meant
        to update (found in the usability walkthrough).
        """
        state = job_store.request_cancel(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")
        snapshot = job_store.snapshot(job_id) or {}
        snapshot["cancel_requested"] = job_store.should_cancel(job_id)
        # Stated by the API, not just the page: a running Job stops at the next
        # Piece boundary, which is at most 30 seconds. The note is present only
        # while stopping is still ahead, so it is never stale advice.
        snapshot["cancellation_note"] = (
            "A running transcription stops at the end of the 30-second Piece it "
            "is working on, so it can take up to 30 seconds."
            if snapshot["cancel_requested"]
            else None
        )
        return snapshot

    # -------------------------------------------------------------- download

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> Response:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")
        job_store.sweep_expired()
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")

        if job.state not in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="This transcription has not finished yet.")
        if job.state == CANCELLED:
            raise HTTPException(status_code=409, detail="This transcription was cancelled, so there is nothing to download.")
        if job.state != FINISHED:
            raise HTTPException(
                status_code=409,
                detail=job.error or "This transcription failed, so there is nothing to download.",
            )
        if job.rendered is None:
            raise HTTPException(
                status_code=410,
                detail=(
                    "This transcript has expired. Transcripts are kept for "
                    f"{int(config.retention_seconds // 60)} minutes after they finish. "
                    "Upload the recording again to get a new one."
                ),
            )

        filename = f"{Path(job.original_name).stem}.{job.output_format}"
        return Response(
            content=job.rendered,
            media_type=_media_type(job.output_format),
            headers={"Content-Disposition": _disposition(filename)},
        )

    # ------------------------------------------------------------------ page

    @app.get("/", response_class=PlainTextResponse)
    def index() -> Response:
        page = STATIC_DIR / "index.html"
        return Response(content=page.read_text(encoding="utf-8"), media_type="text/html")

    @app.get("/app.css")
    def stylesheet() -> Response:
        return Response(
            content=(STATIC_DIR / "app.css").read_text(encoding="utf-8"),
            media_type="text/css",
        )

    @app.get("/app.js")
    def script() -> Response:
        return Response(
            content=(STATIC_DIR / "app.js").read_text(encoding="utf-8"),
            media_type="text/javascript",
        )

    return app


# --------------------------------------------------------------------- helpers

def _ensure(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


async def _save_upload(upload: UploadFile, target: Path) -> None:
    with target.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            handle.write(chunk)


def _safe_name(name: str) -> str:
    """Keep the user's name for display and for the download filename."""
    cleaned = Path(name.replace("\\", "/")).name.strip()
    return cleaned or "rekaman"


def _suffix(name: str) -> str:
    suffix = Path(name).suffix
    return suffix if suffix else ".bin"


def _media_type(output_format: str) -> str:
    return {
        "txt": "text/plain; charset=utf-8",
        "srt": "application/x-subrip; charset=utf-8",
        "vtt": "text/vtt; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }.get(output_format) or mimetypes.guess_type(f"x.{output_format}")[0] or "application/octet-stream"


def _disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "")
    return f'attachment; filename="{ascii_name}"'


def _event(snapshot: dict) -> str:
    return f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"


def _client_gone(request: Request) -> bool:
    try:
        return bool(request and request.client is None)
    except Exception:  # noqa: BLE001 - a closed transport is not an error worth raising
        return True
