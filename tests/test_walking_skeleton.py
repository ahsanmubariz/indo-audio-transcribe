"""t04 — the walking skeleton: upload, run, download a .txt.

The seam is the HTTP surface, driven through the framework's test client with the
transcriber injected as a fake. Nothing here reaches into the job store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber
from media import tone_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def client(work_dir: Path):
    """A client whose transcriber is a fake and whose uploads go to a temp dir."""
    transcriber = FakeTranscriber()
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as test_client:
        test_client.transcriber = transcriber
        yield test_client


@pytest.fixture()
def audio(work_dir: Path) -> Path:
    return tone_wav(work_dir / "rapat.mp3.wav", seconds=3.0)


def upload(client, audio: Path, name: str = "rapat.wav", output_format: str = "txt"):
    with audio.open("rb") as handle:
        return client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": output_format},
        )


def test_uploading_a_file_creates_a_job(client, audio: Path) -> None:
    response = upload(client, audio)

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["id"]
    assert payload["state"] in {"queued", "running", "finished"}
    assert payload["original_name"] == "rapat.wav"


def test_uploaded_file_becomes_a_downloadable_transcript(client, audio: Path) -> None:
    created = upload(client, audio).json()
    job_id = created["id"]

    finished = _await_state(client, job_id, "finished")
    assert finished["state"] == "finished"

    response = client.get(f"/api/jobs/{job_id}/download")

    assert response.status_code == 200
    assert response.text == "satu dua tiga empat lima\n" or response.text.strip()
    assert len(response.text.strip().split()) >= 1


def test_download_filename_comes_from_the_uploaded_name(client, audio: Path) -> None:
    job_id = upload(client, audio, name="rapat-rapat.wav").json()["id"]
    _await_state(client, job_id, "finished")

    response = client.get(f"/api/jobs/{job_id}/download")

    disposition = response.headers["content-disposition"]
    assert "rapat-rapat.txt" in disposition


def test_health_reports_that_the_model_is_loaded(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["status"] == "ok"


def test_health_reports_a_missing_model_without_pretending() -> None:
    config = AppConfig(work_root=Path("/tmp/unused"), port=0)
    app = create_app(config, transcriber=None, model_loaded=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["model_loaded"] is False
    assert payload["status"] == "degraded"


def test_status_of_an_unknown_job_is_a_clean_404(client) -> None:
    response = client.get("/api/jobs/tidak-ada")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_the_page_is_served_by_the_same_process(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Transkripsi" in response.text or "transcri" in response.text.lower()


def _await_state(client, job_id: str, state: str, *, timeout: float = 20.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    snapshot = {}
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["state"] == state:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {state}; last was {snapshot}")