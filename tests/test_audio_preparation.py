"""t01 — audio preparation: probe, optimise, reject.

The seam under test is `mytranscribe.audio.prepare_audio`. Tests assert on the
facts it reports and the file it yields, checked independently with ffprobe.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from media import (
    corrupt_audio,
    probe_streams,
    stereo_mp3,
    tone_wav,
    video_with_audio,
    video_without_audio,
)

from mytranscribe.audio import AudioRejected, prepare_audio


def test_video_file_yields_its_audio_track_with_duration(work_dir: Path) -> None:
    source = video_with_audio(work_dir / "meeting.mp4", seconds=3.0)

    prepared = prepare_audio(source, work_dir=work_dir / "prep")

    # The optimised file is 16 kHz mono 16-bit PCM, whatever the input was.
    facts = probe_streams(prepared.path)
    assert facts == {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": "1"}

    # The facts describe the Original audio, which is what the UI shows.
    assert prepared.facts.duration_s == pytest.approx(3.0, abs=0.5)
    assert prepared.facts.channels == 1
    assert prepared.facts.sample_rate == 44100
    prepared.cleanup()


def test_stereo_mp3_is_downmixed_and_resampled(work_dir: Path) -> None:
    source = stereo_mp3(work_dir / "call.mp3", seconds=4.0)

    prepared = prepare_audio(source, work_dir=work_dir / "prep")

    trained = probe_streams(prepared.path)
    assert trained["sample_rate"] == "16000"
    assert trained["channels"] == "1"
    assert prepared.facts.duration_s == pytest.approx(4.0, abs=0.5)
    prepared.cleanup()


def test_file_without_audio_is_refused_with_a_readable_reason(work_dir: Path) -> None:
    source = video_without_audio(work_dir / "muted.mp4")

    with pytest.raises(AudioRejected) as caught:
        prepare_audio(source, work_dir=work_dir / "prep")

    reason = str(caught.value)
    assert "no audio" in reason.lower()
    # A person reads this, so it must not read like a stack trace.
    assert "Traceback" not in reason
    assert "ffprobe" not in reason


def test_corrupt_file_is_refused_with_a_readable_reason(work_dir: Path) -> None:
    source = corrupt_audio(work_dir / "broken.mp3")

    with pytest.raises(AudioRejected) as caught:
        prepare_audio(source, work_dir=work_dir / "prep")

    reason = str(caught.value)
    assert "read" in reason.lower() or "unreadable" in reason.lower()
    assert "Traceback" not in reason


def test_missing_file_is_refused(work_dir: Path) -> None:
    with pytest.raises(AudioRejected):
        prepare_audio(work_dir / "not-there.mp3", work_dir=work_dir / "prep")


def test_temp_audio_is_cleaned_up_when_preparation_fails_partway(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source that probes as audio but cannot be converted must leave nothing behind."""
    # A WAV whose format tag has no decoder: ffprobe sees an audio stream, ffmpeg refuses it.
    source = work_dir / "undecodable.wav"
    body = b"\x00\x01" * 16_000
    header = (
        b"RIFF" + (36 + len(body)).to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little") + (0x9999).to_bytes(2, "little")
        + (1).to_bytes(2, "little") + (16_000).to_bytes(4, "little")
        + (32_000).to_bytes(4, "little") + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little") + b"data" + len(body).to_bytes(4, "little")
    )
    source.write_bytes(header + body)

    prep_dir = work_dir / "prep"
    with pytest.raises(AudioRejected):
        prepare_audio(source, work_dir=prep_dir)

    assert _files_under(prep_dir) == []


def test_rejection_leaves_no_temp_dir_at_all(work_dir: Path) -> None:
    source = video_without_audio(work_dir / "muted.mp4")
    prep_dir = work_dir / "prep"

    with pytest.raises(AudioRejected):
        prepare_audio(source, work_dir=prep_dir)

    assert not prep_dir.exists() or _files_under(prep_dir) == []


def test_module_does_not_import_the_web_framework() -> None:
    """The pipeline must stay usable from the CLI; FastAPI is a web-front-door concern."""
    code = (
        "import sys, mytranscribe.audio; "
        "leaked = [m for m in ('fastapi', 'starlette', 'uvicorn') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""


def _files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())

# ------------------------------------------------------- the reference recording

@pytest.mark.slow
def test_a_long_recording_prepares_to_16k_mono(reference_audio, work_dir: Path) -> None:
    """Preparation of a real recording, using whatever the suite was pointed at."""
    prepared = prepare_audio(reference_audio, work_dir=work_dir)

    assert prepared.facts.duration_s > 0
    facts = probe_streams(prepared.path)
    assert facts == {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": "1"}
    prepared.cleanup()
