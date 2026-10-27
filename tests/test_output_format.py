"""t10 — choosing the output format at upload.

The choice must happen at upload, not download: `.srt`, `.vtt` and `.json` need
word timings, and asking for them changes how the audio is decoded during
transcription. A format picked later would arrive with nothing to write from.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber
from media import piece_flavoured_wav

from mytranscribe.app import create_app
from mytranscribe.config import AppConfig

EXPECTED_MEDIA_TYPE = {
    "txt": "text/plain",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "json": "application/json",
}


@pytest.fixture()
def context(work_dir: Path):
    transcriber = FakeTranscriber(per_piece_seconds=0.01)
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=transcriber, model_loaded=True)
    with TestClient(app) as client:
        yield client, transcriber


def upload(client, path: Path, output_format: str | None = None, name: str = "rapat.wav"):
    data = {} if output_format is None else {"format": output_format}
    with path.open("rb") as handle:
        return client.post(
            "/api/jobs",
            files={"file": (name, handle, "audio/wav")},
            data=data,
        )


def finished(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = client.get(f"/api/jobs/{job_id}").json()
        if seen["state"] == "finished":
            return seen
        time.sleep(0.01)
    raise AssertionError(f"never finished; last was {seen}")


def test_txt_is_preselected_when_no_format_is_given(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio).json()

    assert job["format"] == "txt"


def test_txt_job_downloads_plain_text(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, "txt").json()
    finished(client, job["id"])
    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(EXPECTED_MEDIA_TYPE["txt"])
    assert " --> " not in response.text
    assert response.text.strip()
    # No word timings were collected, so the decode path stayed on the cheap one.
    assert "words" not in response.text


def test_a_txt_job_never_asks_the_model_for_word_timings(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, "txt").json()
    finished(client, job["id"])

    # Asking for words changes how the audio is decoded, so .txt must not ask.
    assert transcriber.asked_for_words
    assert not any(transcriber.asked_for_words)


def test_srt_job_downloads_a_valid_srt(context, work_dir: Path) -> None:
    client, transcriber = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=3)

    job = upload(client, audio, "srt", name="sidang.wav").json()
    finished(client, job["id"])
    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(EXPECTED_MEDIA_TYPE["srt"])
    body = response.text
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", body)
    assert body.startswith("1\n")
    assert "sidang.srt" in response.headers["content-disposition"]
    # Word timings were requested, because the format needs them.
    assert transcriber.asked_for_words and all(transcriber.asked_for_words)


def test_vtt_job_downloads_a_valid_vtt(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, "vtt", name="rapat.wav").json()
    finished(client, job["id"])
    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(EXPECTED_MEDIA_TYPE["vtt"])
    assert response.text.startswith("WEBVTT")
    assert "rapat.vtt" in response.headers["content-disposition"]


def test_json_job_contains_model_mode_text_and_words(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, "json", name="catatan.wav").json()
    finished(client, job["id"])
    response = client.get(f"/api/jobs/{job['id']}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(EXPECTED_MEDIA_TYPE["json"])
    payload = json.loads(response.text)
    assert payload["model"] == "PixelML/sea-asr-id"
    assert payload["mode"] == "lm"
    assert payload["text"]
    assert payload["words"]
    assert set(payload["words"][0]) == {"word", "start", "end"}
    assert "catatan.json" in response.headers["content-disposition"]


def test_every_format_downloads_with_its_own_extension(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    for output_format in ("txt", "srt", "vtt", "json"):
        job = upload(client, audio, output_format, name="rapat.wav").json()
        finished(client, job["id"])
        response = client.get(f"/api/jobs/{job['id']}/download")
        disposition = response.headers["content-disposition"]
        assert f"rapat.{output_format}" in disposition, disposition


def test_the_chosen_format_is_reported_in_the_snapshot(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    job = upload(client, audio, "srt").json()

    assert job["format"] == "srt"
    assert client.get(f"/api/jobs/{job['id']}").json()["format"] == "srt"


def test_an_unsupported_extension_is_refused_readably(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    response = upload(client, audio, "docx")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "txt" in detail and "json" in detail
    assert "Traceback" not in detail


def test_a_leading_dot_in_the_format_is_tolerated(context, work_dir: Path) -> None:
    client, _ = context
    audio = piece_flavoured_wav(work_dir / "long.wav", pieces=2)

    response = upload(client, audio, ".srt")

    assert response.status_code == 201, response.text
    assert response.json()["format"] == "srt"