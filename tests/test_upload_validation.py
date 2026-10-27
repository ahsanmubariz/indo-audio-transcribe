"""t05 — bad input is refused before it becomes a Job.

Upload validation is a different event from transcription failure, with a
different fix. A rejected upload creates no Job and leaves no temp files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber
from media import corrupt_audio, video_with_audio, video_without_audio

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def context(work_dir: Path):
    transcriber = FakeTranscriber()
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, config, transcriber


def upload(client, path: Path, name: str | None = None, output_format: str = "txt"):
    with path.open("rb") as handle:
        return client.post(
            "/api/jobs",
            files={"file": (name or path.name, handle, "application/octet-stream")},
            data={"format": output_format},
        )


def files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


def test_video_without_audio_is_refused_and_creates_no_job(context, work_dir: Path) -> None:
    client, config, _ = context
    source = video_without_audio(work_dir / "muted.mp4")

    response = upload(client, source)

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "no audio" in detail.lower()
    # Plain language: no stack trace, no internal jargon.
    assert "Traceback" not in detail
    assert "ffprobe" not in detail
    assert "ffmpeg" not in detail
    # No Job exists.
    assert client.get("/api/jobs").json()["jobs"] == []


def test_corrupt_file_is_refused_and_creates_no_job(context, work_dir: Path) -> None:
    client, _, _ = context
    source = corrupt_audio(work_dir / "broken.mp3")

    response = upload(client, source)

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "could not be read" in detail.lower()
    assert "Traceback" not in detail
    assert client.get("/api/jobs").json()["jobs"] == []


def test_video_with_audio_is_accepted(context, work_dir: Path) -> None:
    client, _, transcriber = context
    source = video_with_audio(work_dir / "rapat.mp4", seconds=3.0)

    response = upload(client, source, name="rapat.mp4")

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["original_audio"]["channels"] == 1
    assert payload["duration_seconds"] == pytest.approx(3.0, abs=0.5)

    _await_state(client, payload["id"], "finished")
    assert transcriber.calls > 0


def test_rejected_upload_leaves_no_temp_files(context, work_dir: Path) -> None:
    client, config, _ = context
    source = video_without_audio(work_dir / "muted.mp4")

    upload(client, source)

    assert files_under(config.work_root) == []


def test_rejected_upload_does_not_create_a_job_id_on_the_status_endpoint(context, work_dir: Path) -> None:
    client, _, _ = context
    source = corrupt_audio(work_dir / "broken.wav")

    response = upload(client, source)

    assert response.status_code == 422
    assert "Location" not in response.headers


def test_unsupported_format_is_refused_with_a_readable_reason(context, work_dir: Path) -> None:
    client, config, _ = context
    from media import tone_wav

    source = tone_wav(work_dir / "note.wav", seconds=2.0)

    response = upload(client, source, output_format="doc")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "txt" in detail
    assert "not available" in detail.lower()
    assert files_under(config.work_root) == []


def test_an_upload_with_no_file_is_refused_cleanly(context) -> None:
    client, _, _ = context

    response = client.post("/api/jobs", data={"format": "txt"})

    assert response.status_code == 422


def test_a_rejected_job_is_not_reported_as_a_failed_job(context, work_dir: Path) -> None:
    """'Upload failed' and 'transcription failed' must not be conflated."""
    client, _, _ = context
    source = video_without_audio(work_dir / "muted.mp4")

    upload(client, source)

    jobs = client.get("/api/jobs").json()["jobs"]
    assert [job for job in jobs if job["state"] == "failed"] == []


def _await_state(client, job_id: str, state: str, *, timeout: float = 20.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    snapshot: dict = {}
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["state"] == state:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {state}; last was {snapshot}")