"""t09 — cancelling a queued or running Job.

A queued Job stops instantly. A running Job stops at the next Piece boundary,
which is at most 30 seconds — a deliberate, accepted limit that the UI states
rather than lets the user discover.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber, GatedTranscriber
from media import piece_flavoured_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def paced(work_dir: Path):
    """A Worker whose Pieces are slow enough to be cancelled between."""
    transcriber = FakeTranscriber(per_piece_seconds=0.12)
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber, config


@pytest.fixture()
def gated(work_dir: Path):
    transcriber = GatedTranscriber()
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber, config


def upload(client, path: Path, name: str = "rapat.wav") -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": "txt"},
        )
    assert response.status_code == 201, response.text
    return response.json()


def snapshot(client, job_id: str) -> dict:
    return client.get(f"/api/jobs/{job_id}").json()


def wait_for(client, job_id: str, predicate, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = snapshot(client, job_id)
        if predicate(seen):
            return seen
        time.sleep(0.01)
    raise AssertionError(f"never matched; last was {seen}")


# ------------------------------------------------------------------- queued

def test_cancelling_a_queued_job_removes_it_immediately(gated, work_dir: Path) -> None:
    client, transcriber, _ = gated
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    running = upload(client, audio, "satu.wav")["id"]
    transcriber.entered.wait(10)
    waiting = upload(client, audio, "dua.wav")["id"]
    assert snapshot(client, waiting)["state"] == "queued"

    response = client.post(f"/api/jobs/{waiting}/cancel")

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    after = snapshot(client, waiting)
    assert after["state"] == "cancelled"
    assert after["queue_position"] is None
    transcriber.gate.set()
    wait_for(client, running, lambda s: s["state"] == "finished")


def test_a_cancelled_queued_job_never_runs(gated, work_dir: Path) -> None:
    client, transcriber, _ = gated
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    running = upload(client, audio, "satu.wav")["id"]
    transcriber.entered.wait(10)
    waiting = upload(client, audio, "dua.wav")["id"]

    client.post(f"/api/jobs/{waiting}/cancel")
    transcriber.gate.set()
    wait_for(client, running, lambda s: s["state"] == "finished")
    time.sleep(0.3)

    assert snapshot(client, waiting)["state"] == "cancelled"
    # The cancelled Job transcribed nothing.
    assert snapshot(client, waiting)["pieces_completed"] == 0


# ------------------------------------------------------------------ running

def test_cancelling_a_running_job_stops_before_the_next_piece(paced, work_dir: Path) -> None:
    client, transcriber, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=8)

    job_id = upload(client, audio)["id"]
    running = wait_for(client, job_id, lambda s: s["state"] == "running" and s["pieces_completed"] >= 1)

    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200

    cancelled = wait_for(client, job_id, lambda s: s["state"] == "cancelled")
    calls_at_cancel = transcriber.calls

    assert cancelled["state"] == "cancelled"
    # It stopped where it was, having transcribed strictly fewer Pieces than the total.
    assert cancelled["pieces_completed"] < cancelled["pieces_total"]
    assert cancelled["pieces_completed"] >= running["pieces_completed"]

    time.sleep(0.5)
    assert transcriber.calls == calls_at_cancel, "the Worker kept transcribing after cancellation"


def test_a_running_cancel_says_how_long_it_may_take(paced, work_dir: Path) -> None:
    """The 30-second limit is stated, not discovered."""
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")

    payload = client.post(f"/api/jobs/{job_id}/cancel").json()

    assert "30" in payload["cancellation_note"]
    wait_for(client, job_id, lambda s: s["state"] == "cancelled")


def test_cancelling_before_the_first_piece_finishes_still_cancels(paced, work_dir: Path) -> None:
    client, transcriber, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")

    client.post(f"/api/jobs/{job_id}/cancel")

    cancelled = wait_for(client, job_id, lambda s: s["state"] == "cancelled")
    assert cancelled["pieces_completed"] <= 1


# --------------------------------------------------------------- idempotency

def test_cancelling_a_finished_job_succeeds_and_keeps_the_transcript(paced, work_dir: Path) -> None:
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "finished")
    before = client.get(f"/api/jobs/{job_id}/download").text

    response = client.post(f"/api/jobs/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["state"] == "finished"
    assert snapshot(client, job_id)["state"] == "finished"
    assert client.get(f"/api/jobs/{job_id}/download").text == before


def test_cancelling_twice_is_not_an_error(paced, work_dir: Path) -> None:
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")

    first = client.post(f"/api/jobs/{job_id}/cancel")
    second = client.post(f"/api/jobs/{job_id}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    wait_for(client, job_id, lambda s: s["state"] == "cancelled")


def test_cancelling_a_job_that_failed_is_not_an_error(work_dir: Path) -> None:
    class Exploding(FakeTranscriber):
        def transcribe(self, audio, return_words: bool = False):
            raise RuntimeError("the model fell over")

    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=Exploding(), model_loaded=True)
    with TestClient(app) as client:
        audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)
        job_id = upload(client, audio)["id"]
        wait_for(client, job_id, lambda s: s["state"] == "failed")

        response = client.post(f"/api/jobs/{job_id}/cancel")

        assert response.status_code == 200
        assert response.json()["state"] == "failed"


def test_cancelling_an_unknown_job_is_a_clean_404(paced) -> None:
    client, _, _ = paced

    response = client.post("/api/jobs/tidak-ada/cancel")

    assert response.status_code == 404


# ------------------------------------------------------------------- cleanup

def test_a_cancelled_job_leaves_no_audio_behind(paced, work_dir: Path) -> None:
    client, _, config = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id, lambda s: s["state"] == "cancelled")

    # Give the Worker a moment to finish its `finally` block.
    deadline = time.monotonic() + 10
    remaining = None
    while time.monotonic() < deadline:
        remaining = sorted(p for p in config.work_root.rglob("*") if p.is_file())
        if not remaining:
            break
        time.sleep(0.05)
    assert remaining == [], f"left behind: {remaining}"


def test_a_cancelled_job_cannot_be_downloaded(paced, work_dir: Path) -> None:
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=6)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id, lambda s: s["state"] == "cancelled")

    response = client.get(f"/api/jobs/{job_id}/download")

    assert response.status_code == 409
    assert "cancel" in response.json()["detail"].lower()


def test_the_snapshot_marks_a_cancelled_job_as_terminal(paced, work_dir: Path) -> None:
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=5)

    job_id = upload(client, audio)["id"]
    # Let at least one Piece land, so there is partial text to preserve.
    wait_for(client, job_id, lambda s: s["state"] == "running" and s["pieces_completed"] >= 1)
    client.post(f"/api/jobs/{job_id}/cancel")

    cancelled = wait_for(client, job_id, lambda s: s["state"] == "cancelled")
    assert cancelled["queue_position"] is None
    assert cancelled["error"] is None
    # Work that was already done is kept: cancelling stops the Job, it does not undo it.
    assert cancelled["partial_text"]
    assert cancelled["pieces_completed"] >= 1


def test_cancelling_before_any_piece_completes_leaves_no_text(paced, work_dir: Path) -> None:
    """The complement: a Job stopped before its first Piece honestly reports nothing."""
    client, _, _ = paced
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=5)

    job_id = upload(client, audio)["id"]
    wait_for(client, job_id, lambda s: s["state"] == "running")
    client.post(f"/api/jobs/{job_id}/cancel")

    cancelled = wait_for(client, job_id, lambda s: s["state"] == "cancelled")
    assert cancelled["pieces_completed"] <= 1
    if cancelled["pieces_completed"] == 0:
        assert cancelled["partial_text"] == ""