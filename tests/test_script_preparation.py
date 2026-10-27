"""`transcribe.sh` and the web app must prepare audio the same way.

The script used to keep its own `ffprobe` and `ffmpeg` filtergraph in bash while
`mytranscribe/audio.py` held the same filters in Python. Two copies of the thing
the whole project's parity bar rests on. The script's preparation now goes through
the module, and these tests are what stop the copy from growing back.

Found in the Spec review: probing and optimisation were assigned to the pipeline
module by the spec, but only the transcription loop had actually moved.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from media import (
    corrupt_audio,
    tone_wav,
    two_track_audio,
    video_with_audio,
    video_without_audio,
)

from mytranscribe.audio import prepare_audio

SCRIPT = REPO_ROOT / "transcribe.sh"


def run_prepare(source: Path, work_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the preparation front door the way the script runs it.

    `--keep-wav` is passed because the module removes its scratch directory on
    the way out; a test that kept nothing would be asserting against a deleted
    file. That the cleanup happens is its own test.
    """
    if "--keep-wav" not in args:
        args = (*args, "--keep-wav", str(work_dir / "kept.wav"))
    return subprocess.run(
        [sys.executable, "-m", "mytranscribe.prepare", str(source), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )


def parse_facts(stdout: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            facts[key.strip()] = value.strip()
    return facts


# ------------------------------------------------------------------- facts

def test_prepare_emits_the_facts_the_script_prints(work_dir: Path) -> None:
    audio = tone_wav(work_dir / "rapat.wav", seconds=3.0)

    result = run_prepare(audio, work_dir / "work")

    assert result.returncode == 0, result.stderr
    facts = parse_facts(result.stdout)
    assert float(facts["duration"]) == pytest.approx(3.0, abs=0.2)
    assert facts["codec"] == "pcm_s16le"
    assert facts["sample_rate"] == "16000"
    assert facts["channels"] == "1"
    assert facts["audio_tracks"] == "1"
    assert Path(facts["wav"]).is_file()


def test_the_prepared_wav_is_what_the_module_produces(work_dir: Path) -> None:
    """Byte identity: the point of removing the duplicate was that they matched."""
    audio = tone_wav(work_dir / "rapat.wav", seconds=2.0)

    result = run_prepare(audio, work_dir / "work")

    assert result.returncode == 0, result.stderr
    produced = Path(parse_facts(result.stdout)["wav"]).read_bytes()

    ours = prepare_audio(audio, work_dir=work_dir / "module")
    try:
        assert produced == ours.path.read_bytes()
    finally:
        ours.cleanup()


def test_raw_skips_the_clean_up_filters(work_dir: Path) -> None:
    audio = tone_wav(work_dir / "rapat.wav", seconds=2.0)

    filtered = run_prepare(audio, work_dir / "filtered")
    raw = run_prepare(audio, work_dir / "raw", "--raw")

    assert filtered.returncode == 0, filtered.stderr
    assert raw.returncode == 0, raw.stderr
    assert Path(parse_facts(raw.stdout)["wav"]).read_bytes() != Path(
        parse_facts(filtered.stdout)["wav"]
    ).read_bytes()


def test_the_prepared_wav_is_16khz_mono_16bit(work_dir: Path) -> None:
    from media import probe_streams

    audio = tone_wav(work_dir / "stereo.wav", seconds=1.0)
    result = run_prepare(audio, work_dir / "work")

    assert result.returncode == 0, result.stderr
    streams = probe_streams(Path(parse_facts(result.stdout)["wav"]))
    assert streams["sample_rate"] == "16000"
    assert streams["channels"] == "1"


# --------------------------------------------------------------- refusals

def test_a_video_is_refused_with_the_scripts_own_wording(work_dir: Path) -> None:
    """The module supplies the reason; the script supplies the sentence with its flag."""
    video = video_with_audio(work_dir / "rapat.mp4", seconds=1.0)

    result = run_prepare(video, work_dir / "work")

    assert result.returncode != 0
    assert "--allow-video" in result.stderr
    assert "video" in result.stderr.lower()


def test_a_video_is_accepted_when_allowed(work_dir: Path) -> None:
    video = video_with_audio(work_dir / "rapat.mp4", seconds=1.0)

    result = run_prepare(video, work_dir / "work", "--allow-video")

    assert result.returncode == 0, result.stderr
    assert Path(parse_facts(result.stdout)["wav"]).is_file()


def test_a_file_with_no_audio_is_refused_readably(work_dir: Path) -> None:
    muted = video_without_audio(work_dir / "bisu.mp4")

    result = run_prepare(muted, work_dir / "work", "--allow-video")

    assert result.returncode != 0
    assert "no audio" in result.stderr.lower()


def test_a_damaged_file_is_refused_readably(work_dir: Path) -> None:
    broken = corrupt_audio(work_dir / "rusak.mp3")

    result = run_prepare(broken, work_dir / "work", "--allow-video")

    assert result.returncode != 0
    assert result.stderr.strip()


# ------------------------------------------------- the duplicate cannot return

def test_the_script_no_longer_probes_or_filters_audio_itself() -> None:
    """The structural guard. If this fails, the copy is growing back."""
    body = SCRIPT.read_text(encoding="utf-8")

    assert "ffprobe -v error -select_streams" not in body, (
        "transcribe.sh is probing audio itself again; call mytranscribe.prepare"
    )
    assert "highpass=f=80" not in body, (
        "transcribe.sh holds its own ffmpeg filtergraph again; audio.py owns the filters"
    )
    assert "afftdn=nr=10" not in body


def test_the_script_delegates_preparation_to_the_module() -> None:
    assert "mytranscribe.prepare" in SCRIPT.read_text(encoding="utf-8")


def test_the_script_preparation_produces_the_modules_wav(work_dir: Path) -> None:
    """End to end through bash: the script's stage 1-2 equals prepare_audio."""
    audio = tone_wav(work_dir / "rapat.wav", seconds=2.0)
    # The script names the kept WAV after the output, not after the input.
    kept = work_dir / "hasil.16k.wav"

    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "-i", str(audio),
            "-o", str(work_dir / "hasil.txt"),
            "--keep-wav",
            "--threads", "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MYTRANSCRIBE_STOP_AFTER_PREPARE": "1",
            "HOME": str(work_dir),
            "VENV_DIR": str(work_dir / "venv"),
        },
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert kept.is_file(), f"no kept WAV; stderr:\n{result.stderr}"

    ours = prepare_audio(audio, work_dir=work_dir / "module", threads=2)
    try:
        assert kept.read_bytes() == ours.path.read_bytes()
    finally:
        ours.cleanup()

def refusals_from_the_script(source: Path, work_dir: Path, *args: str):
    """Run the script's preparation stage only, so no model is loaded."""
    return subprocess.run(
        [
            "bash", str(SCRIPT),
            "-i", str(source),
            "-o", str(work_dir / "hasil.txt"),
            *args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MYTRANSCRIBE_STOP_AFTER_PREPARE": "1",
            "HOME": str(work_dir),
            "VENV_DIR": str(work_dir / "venv"),
        },
    )


def test_the_script_reports_a_refusal_in_its_own_voice(work_dir: Path) -> None:
    """A bare exit code is not a message. Regression: `die "$?"` printed `[x] 1`."""
    video = video_with_audio(work_dir / "rapat.mp4", seconds=1.0)

    result = refusals_from_the_script(video, work_dir)

    assert result.returncode != 0
    assert "[x]" in result.stderr, result.stderr
    assert "--allow-video" in result.stderr
    # And not the exit code dressed up as a message.
    assert "[x] 1" not in result.stderr


def test_the_script_reports_a_missing_audio_stream_in_its_own_voice(work_dir: Path) -> None:
    muted = video_without_audio(work_dir / "bisu.mp4")

    result = refusals_from_the_script(muted, work_dir, "--allow-video")

    assert result.returncode != 0
    assert "[x]" in result.stderr, result.stderr
    assert "no audio" in result.stderr.lower()


def test_the_script_leaves_no_scratch_files_behind(work_dir: Path) -> None:
    audio = tone_wav(work_dir / "rapat.wav", seconds=2.0)
    before = set(Path("/tmp").glob("*prep*"))

    result = subprocess.run(
        ["bash", str(SCRIPT), "-i", str(audio), "-o", str(work_dir / "hasil.txt"),
         "--keep-wav", "--threads", "2"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MYTRANSCRIBE_STOP_AFTER_PREPARE": "1",
            "HOME": str(work_dir),
            "VENV_DIR": str(work_dir / "venv"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert set(Path("/tmp").glob("*prep*")) == before, "preparation left scratch files in /tmp"
    assert Path(work_dir / "hasil.16k.wav").is_file()


def test_prepare_cleans_up_after_itself(work_dir: Path) -> None:
    """Found in both review axes: the module leaked a scratch dir per run.

    1.9 MB each, and it contradicted the README's "Nothing is stored." The script
    escaped only because its own trap swept up afterwards, so standalone use — the
    contract the docstring advertises — was the case that leaked.
    """
    audio = tone_wav(work_dir / "rapat.wav", seconds=1.0)
    before = set(Path(tempfile.gettempdir()).glob("prepare-*"))

    result = subprocess.run(
        [sys.executable, "-m", "mytranscribe.prepare", str(audio)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )

    assert result.returncode == 0, result.stderr
    after = set(Path(tempfile.gettempdir()).glob("prepare-*"))
    assert after == before, f"left behind: {after - before}"


def test_audio_tracks_is_measured_not_assumed(work_dir: Path) -> None:
    """A hardcoded 1 would be a lie for a file with two audio tracks."""
    double = two_track_audio(work_dir / "dua.m4a")

    result = run_prepare(double, work_dir / "work")

    assert result.returncode == 0, result.stderr
    assert parse_facts(result.stdout)["audio_tracks"] == "2"


def test_the_script_warns_when_a_file_has_several_audio_tracks(work_dir: Path) -> None:
    """The old bash warned about this; the delegation must not lose it."""
    import subprocess as sp

    double = two_track_audio(work_dir / "dua.m4a")

    result = sp.run(
        ["bash", str(SCRIPT), "-i", str(double), "-o", str(work_dir / "hasil.txt"),
         "--keep-wav", "--threads", "2"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MYTRANSCRIBE_STOP_AFTER_PREPARE": "1",
            "HOME": str(work_dir),
            "VENV_DIR": str(work_dir / "venv"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "2 audio tracks" in result.stderr, result.stderr
