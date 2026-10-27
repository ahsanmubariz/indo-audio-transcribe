"""Regression tests for defects found by using the app, not by reading it.

Each of these shipped passing tests and was still wrong, because the tests checked
that something existed rather than that it worked: a cancel that blanked the card
it was meant to update, a page promising a duration it never showed, a pre-upload
estimate that ignored the queue, and an inspection that could quietly leak a file.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber
from media import piece_flavoured_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def context(work_dir: Path):
    transcriber = FakeTranscriber(per_piece_seconds=0.12)
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client


def upload(client, path: Path, name: str = "rapat.wav") -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": "txt"},
        )
    assert response.status_code == 201, response.text
    return response.json()


def until(client, job_id: str, predicate, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = client.get(f"/api/jobs/{job_id}").json()
        if predicate(seen):
            return seen
        time.sleep(0.01)
    raise AssertionError(f"never matched; last was {seen}")


# ------------------------------------------------- the cancel response's shape

def test_the_cancel_response_has_the_same_shape_as_a_snapshot(context, work_dir: Path) -> None:
    """The page stores this body as the Job's state, so it must be a state."""
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(context, audio)["id"]
    running = until(context, job_id, lambda s: s["state"] == "running")
    body = context.post(f"/api/jobs/{job_id}/cancel").json()

    for field in running:
        assert field in body, f"the cancel response lost {field}, which the page renders"

    # And the values agree with a fresh fetch apart from the state that is still moving.
    fetched = context.get(f"/api/jobs/{job_id}").json()
    for field in ("id", "original_name", "format", "mode", "pieces_total", "duration_label"):
        assert body[field] == fetched[field], field

    until(context, job_id, lambda s: s["state"] == "cancelled")


def test_the_cancel_response_does_not_claim_a_running_job_already_stopped(context, work_dir: Path) -> None:
    """It says the request was accepted, not that the work has already stopped."""
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(context, audio)["id"]
    # Wait until the Worker has reported the Piece count, so the snapshot has
    # something to carry.
    until(context, job_id, lambda s: s["state"] == "running" and s["pieces_total"] > 0)

    body = context.post(f"/api/jobs/{job_id}/cancel").json()

    assert body["cancel_requested"] is True
    # A running Job cannot be stopped instantly, and the response must not pretend.
    assert body["state"] == "running"
    assert body["queue_position"] is None
    assert body["pieces_total"] > 0
    assert body["cancellation_note"]
    until(context, job_id, lambda s: s["state"] == "cancelled")


def test_a_queued_cancel_reports_the_job_is_already_cancelled(context, work_dir: Path) -> None:
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=4)

    running = upload(context, audio, "satu.wav")["id"]
    until(context, running, lambda s: s["state"] == "running")
    waiting = upload(context, audio, "dua.wav")["id"]

    body = context.post(f"/api/jobs/{waiting}/cancel").json()

    assert body["state"] == "cancelled"
    assert body["cancel_requested"] is False


# ------------------------------------------ every claim in the page copy is true

def test_the_page_shows_duration_and_estimate_before_committing(context, work_dir: Path) -> None:
    """User story 4, reopened by the Spec review.

    The first fix softened the promise instead of keeping it. The honest fix is to
    keep the promise: probe the file, show its facts and the wait, then let the
    user decide.
    """
    from media import tone_wav

    page = context.get("/").text
    assert "sebelum Anda memulai" in page
    # The panel the promise refers to.
    assert 'id="inspect"' in page
    assert 'id="inspect-duration"' in page
    assert 'id="inspect-estimate"' in page

    audio = tone_wav(work_dir / "calon.wav", seconds=40.0)
    with audio.open("rb") as handle:
        response = context.post("/api/inspect", files={"file": ("calon.wav", handle, "audio/wav")})

    assert response.status_code == 200, response.text
    facts = response.json()
    assert facts["duration_seconds"] == pytest.approx(40.0, abs=0.3)
    assert facts["duration_label"] == "0:40"
    assert facts["estimate"]["range_label"]
    assert facts["original_audio"]["channels"] == 1


def test_inspecting_creates_no_job_and_keeps_no_file(context, work_dir: Path) -> None:
    """Looking at a file must not queue it, nor leave a copy behind."""
    from media import tone_wav

    audio = tone_wav(work_dir / "calon.wav", seconds=5.0)
    before = context.get("/api/jobs").json()["jobs"]

    with audio.open("rb") as handle:
        context.post("/api/inspect", files={"file": ("calon.wav", handle, "audio/wav")})

    assert context.get("/api/jobs").json()["jobs"] == before
    leftovers = [p for p in Path(context.app.state.config.work_root).rglob("*") if p.is_file()]
    assert leftovers == [], f"inspection left: {leftovers}"


def test_inspecting_a_file_with_no_audio_is_refused_readably(context, work_dir: Path) -> None:
    from media import video_without_audio

    muted = video_without_audio(work_dir / "bisu.mp4")
    with muted.open("rb") as handle:
        response = context.post("/api/inspect", files={"file": ("bisu.mp4", handle, "video/mp4")})

    assert response.status_code == 422
    assert "no audio" in response.json()["detail"].lower()

def test_the_estimate_before_upload_includes_work_already_queued(context, work_dir: Path) -> None:
    """Found in both review axes: the pre-upload Estimate ignored Queue depth.

    GLOSSARY.md defines an Estimate from the Original audio's length *and* current
    Queue depth. Passing only the duration made the number the page shows before
    upload disagree with the number the Job shows a moment later.
    """
    from media import tone_wav

    small = tone_wav(work_dir / "kecil.wav", seconds=20.0)
    big = tone_wav(work_dir / "besar.wav", seconds=200.0)

    with small.open("rb") as handle:
        idle = context.post(
            "/api/inspect", files={"file": ("kecil.wav", handle, "audio/wav")}
        ).json()
    idle_seconds = idle["estimate"]["seconds"]

    # Put a much longer Job in the Queue.
    with big.open("rb") as handle:
        context.post(
            "/api/jobs",
            files={"file": ("besar.wav", handle, "audio/wav")},
            data={"format": "txt"},
        )
    context.get("/api/jobs")  # let the Worker pick it up

    with small.open("rb") as handle:
        busy = context.post(
            "/api/inspect", files={"file": ("kecil.wav", handle, "audio/wav")}
        ).json()

    assert busy["estimate"]["seconds"] > idle_seconds, (
        "the wait shown before upload ignored the Job already queued: "
        f"{idle_seconds}s idle vs {busy['estimate']['seconds']}s busy"
    )
