"""t08 — live progress over SSE, and surviving a reload.

The event stream is a convenience over the status endpoint, not a second source
of truth: every event has the same shape as a status snapshot, and a reload that
fetches a snapshot first must land on the same view the stream would have shown.
"""

from __future__ import annotations

import json
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
    transcriber = FakeTranscriber(per_piece_seconds=0.05)
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber


def upload(client, path: Path, name: str = "rapat.wav", output_format: str = "txt") -> dict:
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data={"format": output_format},
        )
    assert response.status_code == 201, response.text
    return response.json()


def read_events(client, job_id: str, limit: int = 200) -> list[dict]:
    """Collect SSE frames from a Job's stream until it closes."""
    events: list[dict] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line:
                continue
            assert line.startswith("data: "), f"malformed frame: {line!r}"
            events.append(json.loads(line[len("data: "):]))
            if len(events) >= limit:
                break
    return events


def test_the_stream_emits_an_event_per_completed_piece(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=4)

    job_id = upload(client, audio)["id"]
    events = read_events(client, job_id)

    completes = [event["pieces_completed"] for event in events]
    # Each Piece produces a distinct completion count, monotonically increasing.
    assert completes == sorted(completes)
    total = events[-1]["pieces_total"]
    assert total >= 3
    assert set(completes) >= set(range(1, total + 1))


def test_the_stream_terminates_after_the_final_event(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio)["id"]
    started = time.monotonic()
    events = read_events(client, job_id)
    elapsed = time.monotonic() - started

    assert events[-1]["state"] == "finished"
    assert elapsed < 20, "the stream did not close after the Job finished"


def test_stream_payload_has_the_same_shape_as_a_status_snapshot(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio)["id"]
    events = read_events(client, job_id)
    snapshot = client.get(f"/api/jobs/{job_id}").json()

    assert events[-1] == snapshot


def test_a_snapshot_fetched_mid_job_reconstructs_the_streamed_progress(context, work_dir: Path) -> None:
    """This is what a reload does: fetch first, subscribe second."""
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=5)

    job_id = upload(client, audio)["id"]
    time.sleep(0.08)
    snapshot = client.get(f"/api/jobs/{job_id}").json()

    events = read_events(client, job_id)

    # The snapshot is one of the states the stream would have shown: the pieces
    # done before it must include everything up to the snapshot's count.
    earlier = [event for event in events if event["pieces_completed"] <= snapshot["pieces_completed"]]
    if snapshot["pieces_total"]:
        assert earlier, "the snapshot reported progress the stream never had"
        assert earlier[-1]["pieces_total"] == snapshot["pieces_total"]
        # And the stream picks up from there rather than restarting at zero.
        assert events[-1]["pieces_completed"] >= snapshot["pieces_completed"]


def test_subscribing_after_the_job_finished_still_reports_the_final_state(context, work_dir: Path) -> None:
    """A late subscriber must not hang waiting for events that already happened."""
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio)["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["state"] == "finished":
            break
        time.sleep(0.01)

    events = read_events(client, job_id)

    assert events[-1]["state"] == "finished"
    assert events[-1]["pieces_completed"] == events[-1]["pieces_total"]


def test_events_showing_a_queued_job_are_followed_by_running(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio)["id"]
    events = read_events(client, job_id)

    states = [event["state"] for event in events]
    assert states[0] == "queued"
    assert "running" in states
    assert states[-1] == "finished"
    assert states.index("queued") < states.index("running")


def test_unknown_job_stream_is_a_clean_404(context) -> None:
    client, _ = context

    response = client.get("/api/jobs/tidak-ada/events")

    assert response.status_code == 404


def test_stream_sets_headers_that_disable_buffering(context, work_dir: Path) -> None:
    """A proxy that buffers would defeat the point of a stream."""
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job_id = upload(client, audio)["id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert "no-cache" in response.headers["cache-control"]
        assert response.headers.get("x-accel-buffering") == "no"