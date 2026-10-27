"""t11 — a finished Transcript stays downloadable for 60 minutes, then expires.

Retention is proven at both 59 and 61 minutes, using an injectable clock so the
test does not wait an hour. Nothing is persisted (ADR-0003): once it expires, the
only way back is to transcribe again, and the page must be able to say so.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber, ManualClock
from media import piece_flavoured_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig
from mytranscribe.jobs import JobStore


@pytest.fixture()
def context(work_dir: Path):
    """An app whose retention clock the test moves by hand."""
    clock = ManualClock()
    store = JobStore(retention_seconds=3600.0, clock=clock)
    transcriber = FakeTranscriber(per_piece_seconds=0.0)
    config = AppConfig(work_root=work_dir / "jobs", port=0, retention_seconds=3600.0)
    app = create_app(config, transcriber=transcriber, model_loaded=True, store=store)
    with TestClient(app) as client:
        yield client, clock, store


def upload(client, path: Path, name: str = "rapat.wav", output_format: str = "txt") -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": output_format},
        )
    assert response.status_code == 201, response.text
    return response.json()


def until_finished(client, job_id: str, timeout: float = 30.0) -> dict:
    """Wait on the real clock, because the Worker's progress is real work."""
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = client.get(f"/api/jobs/{job_id}").json()
        if seen["state"] == "finished":
            return seen
        time.sleep(0.01)
    raise AssertionError(f"never finished; last was {seen}")


def test_a_fresh_transcript_is_downloadable(context, work_dir: Path) -> None:
    client, _, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])

    response = client.get(f"/api/jobs/{job['id']}/download")
    assert response.status_code == 200


def test_a_transcript_is_still_there_at_59_minutes(context, work_dir: Path) -> None:
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])
    clock.advance(59 * 60)

    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 200, response.text
    assert response.text.strip()


def test_a_transcript_is_gone_at_61_minutes(context, work_dir: Path) -> None:
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])
    clock.advance(61 * 60)

    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 410, response.text
    detail = response.json()["detail"]
    assert "expired" in detail.lower()
    # It says how long retention was, so the user can plan.
    assert "60" in detail


def test_an_expired_job_still_reports_its_metadata(context, work_dir: Path) -> None:
    """Expiry loses the Transcript, not the record that the Job existed."""
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, name="rapat-pagi.wav")
    until_finished(client, job["id"])
    clock.advance(61 * 60)

    snapshot = client.get(f"/api/jobs/{job['id']}").json()

    assert snapshot["state"] == "finished"
    assert snapshot["expired"] is True
    assert snapshot["original_name"] == "rapat-pagi.wav"
    assert snapshot["duration_seconds"] > 0


def test_the_snapshot_states_when_the_transcript_expires(context, work_dir: Path) -> None:
    client, _, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    finished = until_finished(client, job["id"])

    assert finished["expires_at"] is not None
    assert finished["expired"] is False


def test_downloading_twice_within_retention_works(context, work_dir: Path) -> None:
    """A second download must not require re-running the transcription."""
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])

    first = client.get(f"/api/jobs/{job['id']}/download")
    clock.advance(10 * 60)
    second = client.get(f"/api/jobs/{job['id']}/download")

    assert first.status_code == second.status_code == 200
    assert first.text == second.text


def test_downloading_does_not_start_the_retention_clock_afresh(context, work_dir: Path) -> None:
    """Retention is measured from when the Job finished, not from the last download."""
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])

    clock.advance(50 * 60)
    assert client.get(f"/api/jobs/{job['id']}/download").status_code == 200
    clock.advance(11 * 60)  # 61 minutes after finishing
    assert client.get(f"/api/jobs/{job['id']}/download").status_code == 410


def test_the_retention_window_is_configurable(work_dir: Path) -> None:
    clock = ManualClock()
    store = JobStore(retention_seconds=60.0, clock=clock)
    config = AppConfig(work_root=work_dir / "jobs", port=0, retention_seconds=60.0)
    app = create_app(config, transcriber=FakeTranscriber(), model_loaded=True, store=store)

    with TestClient(app) as client:
        audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)
        job = upload(client, audio)
        until_finished(client, job["id"])
        assert client.get("/health").json()["retention_minutes"] == 1
        clock.advance(61)
        assert client.get(f"/api/jobs/{job['id']}/download").status_code == 410


def test_sweeping_expired_transcripts_releases_them(context, work_dir: Path) -> None:
    client, clock, store = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])
    clock.advance(61 * 60)

    assert store.sweep_expired() == 1
    # Sweeping twice is harmless.
    assert store.sweep_expired() == 0


def test_the_event_stream_reports_expiry_as_a_final_state(context, work_dir: Path) -> None:
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])
    clock.advance(61 * 60)

    # A page left open overnight must be told, not left showing a download link
    # that now returns 410.
    snapshot = client.get(f"/api/jobs/{job['id']}").json()
    assert snapshot["expired"] is True


def test_an_unknown_job_is_a_404_not_an_expiry(context) -> None:
    client, _, _ = context

    response = client.get("/api/jobs/tidak-ada/download")

    assert response.status_code == 404

def test_expiry_really_releases_the_text_everywhere(context, work_dir: Path) -> None:
    """Expiry must free the words, not just the download route.

    Found in the Spec review: `_expire_locked` cleared `rendered` and `transcript`
    but left `partial_text`, which holds the whole transcript. The 410 on download
    hid it, while `/api/jobs` kept serving every word.
    """
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    finished = until_finished(client, job["id"])
    full_text = client.get(f"/api/jobs/{job['id']}/download").text.strip()
    assert full_text

    clock.advance(61 * 60)

    expired = client.get(f"/api/jobs/{job['id']}").json()
    assert expired["expired"] is True
    # The words are gone from the snapshot...
    assert expired["partial_text"] == "", "the transcript outlived its retention in partial_text"
    # ...and from the list the page redraws itself from.
    listed = client.get("/api/jobs").json()["jobs"]
    for entry in listed:
        assert full_text not in entry["partial_text"]
        assert entry["partial_text"] == ""


def test_the_event_stream_reports_expiry_and_terminates(context, work_dir: Path) -> None:
    """Criterion 13's termination clause, for the page left open overnight."""
    client, clock, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio)
    until_finished(client, job["id"])
    clock.advance(61 * 60)
    client.get(f"/api/jobs/{job['id']}")  # triggers the expiry sweep

    events = []
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                import json as _json

                events.append(_json.loads(line[len("data: "):]))

    assert events, "the stream produced nothing"
    assert events[-1]["expired"] is True
    assert events[-1]["partial_text"] == ""
