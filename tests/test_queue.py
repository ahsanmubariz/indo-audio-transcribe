"""t06 — queue Jobs and process them one at a time, in submission order.

Two Jobs at once is not an option: one Job peaks at 7.7 GB on a 15 GB machine
(ADR-0002). The honest behaviour is a visible queue.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import GatedTranscriber
from media import piece_flavoured_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig


@pytest.fixture()
def context(work_dir: Path):
    transcriber = GatedTranscriber()
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber


def upload(client, path: Path, name: str):
    with path.open("rb") as handle:
        return client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": "txt"},
        )


def snapshot(client, job_id: str) -> dict:
    return client.get(f"/api/jobs/{job_id}").json()


def wait_state(client, job_id: str, state: str, timeout: float = 25.0) -> dict:
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = snapshot(client, job_id)
        if seen["state"] == state:
            return seen
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never reached {state}; last {seen}")


def test_two_uploads_while_idle_leave_one_running_and_one_queued(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    first = upload(client, audio, "satu.wav").json()
    transcriber.entered.wait(5)

    second = upload(client, audio, "dua.wav").json()

    states = [snapshot(client, first["id"])["state"], snapshot(client, second["id"])["state"]]
    assert states == ["running", "queued"]
    transcriber.gate.set()


def test_queue_position_is_reported_for_waiting_jobs(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    running = upload(client, audio, "satu.wav").json()
    transcriber.entered.wait(5)
    second = upload(client, audio, "dua.wav").json()["id"]
    third = upload(client, audio, "tiga.wav").json()["id"]

    assert snapshot(client, running["id"])["queue_position"] is None
    assert snapshot(client, second)["queue_position"] == 1
    assert snapshot(client, third)["queue_position"] == 2
    transcriber.gate.set()


def test_queue_position_moves_up_as_the_queue_drains(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    first = upload(client, audio, "satu.wav").json()["id"]
    transcriber.entered.wait(5)
    second = upload(client, audio, "dua.wav").json()["id"]
    third = upload(client, audio, "tiga.wav").json()["id"]

    assert snapshot(client, third)["queue_position"] == 2
    transcriber.gate.set()
    wait_state(client, first, "finished")
    # With the first Job done and the second running, the third is next in line.
    final = wait_state(client, third, "finished")
    assert final["queue_position"] is None


def test_completion_order_matches_submission_order(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    ids = [upload(client, audio, f"berkas{i}.wav").json()["id"] for i in range(3)]
    transcriber.gate.set()

    completed = []
    deadline = time.monotonic() + 40
    while len(completed) < 3 and time.monotonic() < deadline:
        for job_id in ids:
            if job_id in completed:
                continue
            if snapshot(client, job_id)["state"] == "finished":
                completed.append(job_id)
        time.sleep(0.02)

    assert completed == ids


def test_only_one_job_runs_at_a_time_even_with_five_uploaded(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    ids = [upload(client, audio, f"berkas{i}.wav").json()["id"] for i in range(5)]
    running = [job_id for job_id in ids if snapshot(client, job_id)["state"] == "running"]
    assert len(running) == 1

    transcriber.gate.set()
    for job_id in ids:
        wait_state(client, job_id, "finished", timeout=60)
    assert transcriber.max_concurrent == 1


def test_a_finished_job_leaves_the_queue(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio, "satu.wav").json()["id"]
    transcriber.gate.set()

    finished = wait_state(client, job_id, "finished")
    assert finished["state"] == "finished"
    assert finished["queue_position"] is None


def test_listing_jobs_keeps_the_submission_order(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    ids = [upload(client, audio, f"berkas{i}.wav").json()["id"] for i in range(3)]
    listed = [job["id"] for job in client.get("/api/jobs").json()["jobs"]]

    queued = [job_id for job_id in ids if job_id != ids[0]]
    assert listed[: len(ids)] == ids, f"expected submission order, got {listed}"
    assert set(queued).issubset(set(listed))
    transcriber.gate.set()