"""t12 — an Estimate from audio duration and Queue depth.

The estimate exists so that accepting any size upload is safe: an hour-long file
announces its cost instead of quietly blocking everything behind it. It is a
prediction, so it is a range, and it must not report more certainty than the work
justifies.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber
from media import tone_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig
from mytranscribe.estimate import LONG_JOB_SECONDS, estimate_for


@pytest.fixture()
def context(work_dir: Path):
    transcriber = FakeTranscriber(per_piece_seconds=0.05)
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


def until(client, job_id: str, predicate, timeout: float = 40.0) -> dict:
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = client.get(f"/api/jobs/{job_id}").json()
        if predicate(seen):
            return seen
        time.sleep(0.01)
    raise AssertionError(f"never matched; last was {seen}")


# ------------------------------------------------------------------ present

def test_a_queued_job_reports_an_estimate(context, work_dir: Path) -> None:
    client, _ = context
    audio = tone_wav(work_dir / "note.wav", seconds=40.0)

    job = upload(client, audio)

    assert job["estimate"] is not None
    assert job["estimate"]["seconds"] > 0
    assert job["estimate"]["upper_seconds"] >= job["estimate"]["lower_seconds"]
    assert job["estimate"]["label"]


def test_a_running_job_reports_an_estimate(context, work_dir: Path) -> None:
    client, _ = context
    audio = tone_wav(work_dir / "note.wav", seconds=70.0)

    job_id = upload(client, audio)["id"]
    running = until(client, job_id, lambda s: s["state"] == "running")

    assert running["estimate"] is not None
    assert running["estimate"]["seconds"] > 0
    until(client, job_id, lambda s: s["state"] == "finished")


def test_a_finished_job_reports_no_remaining_estimate(context, work_dir: Path) -> None:
    client, _ = context
    audio = tone_wav(work_dir / "note.wav", seconds=35.0)

    job_id = upload(client, audio)["id"]
    finished = until(client, job_id, lambda s: s["state"] == "finished")

    assert finished["estimate"]["seconds"] == 0


# ------------------------------------------------------------ queue awareness

def test_a_queued_job_estimates_including_the_work_ahead_of_it(context, work_dir: Path) -> None:
    """A Job behind two others must not promise the speed of one that starts now."""
    client, _ = context
    audio = tone_wav(work_dir / "note.wav", seconds=65.0)

    first = upload(client, audio, "satu.wav")["id"]
    until(client, first, lambda s: s["state"] == "running")
    second = upload(client, audio, "dua.wav")["id"]
    third = upload(client, audio, "tiga.wav")["id"]

    alone = estimate_for(duration_s=65.0)
    behind_two = until(client, third, lambda s: s["state"] == "queued")["estimate"]

    assert behind_two["seconds"] > alone.seconds, (
        "a Job with two Jobs ahead of it must estimate longer than one starting now"
    )
    assert behind_two["basis"] == "duration"


# ---------------------------------------------------------------------- basis

def test_an_estimate_follows_audio_duration_not_upload_size(context, work_dir: Path) -> None:
    """Same duration, four times the bytes: the wait must be the same.

    Two Jobs are compared one at a time so neither sees the other in the Queue.
    """
    client, _ = context
    # 8 kHz mono and 16 kHz stereo of the same 90 s carry very different byte counts.
    lean = tone_wav(work_dir / "ramping.wav", seconds=90.0, rate=8000, channels=1)
    fat = tone_wav(work_dir / "gemuk.wav", seconds=90.0, rate=16000, channels=2)
    assert fat.stat().st_size > lean.stat().st_size * 2

    lean_job = upload(client, lean, "ramping.wav")
    until(client, lean_job["id"], lambda s: s["state"] == "finished")
    fat_job = upload(client, fat, "gemuk.wav")

    assert lean_job["duration_seconds"] == pytest.approx(90.0, abs=0.2)
    assert fat_job["duration_seconds"] == pytest.approx(90.0, abs=0.2)
    # The estimate follows duration; the byte difference is not a cost.
    assert fat_job["estimate"]["seconds"] == pytest.approx(
        lean_job["estimate"]["seconds"], rel=0.05
    )


def test_the_estimate_reflects_measured_throughput(context, work_dir: Path) -> None:
    """796 s of audio at 6.2x-8.3x realtime is roughly 96-128 s, plus preparation."""
    estimate = estimate_for(duration_s=795.984)

    # The measured wall clock for the reference recording was 96-127 s.
    assert 96 <= estimate.seconds <= 160, estimate
    assert estimate.lower_seconds <= estimate.seconds <= estimate.upper_seconds


def test_the_reported_range_brackets_the_measured_speed(context, work_dir: Path) -> None:
    estimate = estimate_for(duration_s=795.984)

    fastest_possible = 795.984 / 8.3
    slowest_possible = 795.984 / 6.2
    assert estimate.lower_seconds <= slowest_possible
    assert estimate.upper_seconds >= fastest_possible


# --------------------------------------------------------------- degradation

def test_an_estimate_is_a_range_not_a_single_certain_number(context, work_dir: Path) -> None:
    estimate = estimate_for(duration_s=300.0)

    assert estimate.upper_seconds > estimate.lower_seconds
    assert " to " in estimate.as_dict()["range_label"] or estimate.upper_seconds < 1


def test_a_wrong_estimate_never_reports_more_than_the_remaining_work(context) -> None:
    """A Job cannot have more left than its audio, however the guess went."""
    # A wildly optimistic observed speed must not inflate the remaining estimate
    # beyond what the audio could possibly take.
    estimate = estimate_for(duration_s=60.0, pieces_total=2, pieces_done=0, elapsed_s=0.0)

    assert estimate.seconds <= 60.0 / 6.2 + 30, estimate
    assert estimate.lower_seconds >= 0.0


def test_an_estimate_corrects_itself_from_measured_pieces(context) -> None:
    """Real evidence beats the prior: a slow start lengthens the prediction."""
    fast = estimate_for(duration_s=600.0, pieces_total=20, pieces_done=5, elapsed_s=30.0)
    slow = estimate_for(duration_s=600.0, pieces_total=20, pieces_done=5, elapsed_s=150.0)

    assert slow.seconds > fast.seconds
    assert "measured" in slow.basis


def test_completed_work_is_not_counted_as_remaining(context) -> None:
    nearly_done = estimate_for(duration_s=600.0, pieces_total=20, pieces_done=19, elapsed_s=90.0)
    not_started = estimate_for(duration_s=600.0, pieces_total=20, pieces_done=0, elapsed_s=0.0)

    assert nearly_done.seconds < not_started.seconds


# ------------------------------------------------------------------ long jobs

def test_a_very_long_upload_is_warned_about_not_rejected(context, work_dir: Path) -> None:
    client, _ = context
    long_audio = tone_wav(work_dir / "lama.wav", seconds=LONG_JOB_SECONDS + 120.0)

    job = upload(client, long_audio, "lama.wav")

    assert job["state"] in {"queued", "running", "finished"}
    assert job["estimate"]["warning"]
    assert "wait" in job["estimate"]["warning"].lower()


def test_a_short_upload_gets_no_warning(context, work_dir: Path) -> None:
    client, _ = context
    audio = tone_wav(work_dir / "pendek.wav", seconds=20.0)

    job = upload(client, audio)

    assert job["estimate"]["warning"] is None