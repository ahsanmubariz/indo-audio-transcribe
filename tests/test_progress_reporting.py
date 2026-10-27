"""t07 — Piece-level progress and partial text in the status snapshot.

A snapshot alone must be enough to redraw the page: no client-side history, no
arithmetic the server could have done. Pieces done out of total is the source of
truth; percentages are derived from it.
"""

from __future__ import annotations

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
    transcriber = FakeTranscriber(
        texts=["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan"],
        # Paced so polling can actually observe the states in between.
        per_piece_seconds=0.12,
    )
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber


def upload(client, path: Path, name: str = "rapat.wav") -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": "txt"},
        )
    assert response.status_code == 201, response.text
    return response.json()


def wait_until(client, job_id: str, predicate, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    snapshot: dict = {}
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"never matched; last was {snapshot}")


def test_a_queued_job_reports_zero_progress(context, work_dir: Path) -> None:
    """Zero, not a guess — a Job in the Queue has not done any work yet."""
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job = upload(client, audio)

    assert job["state"] == "queued"
    assert job["pieces_completed"] == 0
    assert job["pieces_total"] == 0
    assert job["progress"] == 0.0
    assert job["partial_text"] == ""


def test_a_running_job_reports_pieces_completed_and_total(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=4)

    job = upload(client, audio, "rapat.wav")
    job_id = job["id"]

    running = wait_until(client, job_id, lambda s: s["state"] == "running" and s["pieces_total"] > 0)

    assert running["pieces_total"] >= 3
    assert running["pieces_completed"] == 0 or running["pieces_completed"] < running["pieces_total"]
    wait_until(client, job_id, lambda s: s["state"] == "finished")


def test_partial_text_grows_as_pieces_complete(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=5)

    job_id = upload(client, audio, "rapat.wav")["id"]

    observed: list[str] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["partial_text"] and snapshot["partial_text"] not in observed:
            observed.append(snapshot["partial_text"])
        if snapshot["state"] == "finished":
            break
        time.sleep(0.01)

    # Each observation must be an extension of the last: text only ever grows.
    for earlier, later in zip(observed, observed[1:]):
        assert later.startswith(earlier), f"{later!r} does not extend {earlier!r}"
    assert len(observed) >= 2, f"partial text never grew: {observed}"


def test_partial_text_equals_the_final_text_at_100_percent(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio, "rapat.wav")["id"]
    finished = wait_until(client, job_id, lambda s: s["state"] == "finished")

    downloaded = client.get(f"/api/jobs/{job_id}/download").text
    assert finished["pieces_completed"] == finished["pieces_total"]
    assert finished["progress"] == 1.0
    assert finished["partial_text"] == downloaded.strip()


def test_a_snapshot_carries_everything_needed_to_draw_the_card(context, work_dir: Path) -> None:
    """No client-side history: one snapshot is enough."""
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job = upload(client, audio, "rapat-pagi.wav")

    for field in (
        "id",
        "state",
        "original_name",
        "format",
        "mode",
        "pieces_completed",
        "pieces_total",
        "progress",
        "partial_text",
        "duration_seconds",
        "duration_label",
        "original_audio",
        "queue_position",
        "error",
        "created_at",
    ):
        assert field in job, f"snapshot is missing {field}"

    assert job["original_name"] == "rapat-pagi.wav"
    assert job["format"] == "txt"
    assert job["mode"] == "lm"
    assert job["duration_label"]
    assert job["original_audio"]["channels"] == 1


def test_progress_fraction_never_exceeds_one(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio, "rapat.wav")["id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        assert 0.0 <= snapshot["progress"] <= 1.0
        assert snapshot["pieces_completed"] <= snapshot["pieces_total"] or snapshot["pieces_total"] == 0
        if snapshot["state"] == "finished":
            break
        time.sleep(0.01)