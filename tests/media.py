"""Media fixtures built with ffmpeg, so tests depend on real files, not mocks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def _run(args: list[str]) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def tone_wav(path: Path, *, seconds: float = 3.0, rate: int = 16000, channels: int = 1) -> Path:
    """A plain sine tone as WAV — the simplest thing that is definitely audio."""
    _run(
        [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds}",
            "-ac", str(channels),
            "-ar", str(rate),
            "-c:a", "pcm_s16le",
            str(path),
        ]
    )
    return path


def piece_flavoured_wav(path: Path, *, pieces: int, piece_s: float = 30.0, gap_s: float = 5.0) -> Path:
    """Audio whose silent stretches land where Piece cutting looks for them.

    `sea_asr.audio.split_long` cuts at the quietest 50 ms window in the last 5 s
    before each 30 s mark. Each block below is `piece_s` long and ends with
    `gap_s` of silence, so the cut is unambiguous and the file yields exactly
    `pieces` Pieces at 30 s intervals — which lets a test assert on Piece counts
    and offsets without guessing.
    """
    expression = "+".join(
        f"if(between(t,{i * piece_s},{i * piece_s + piece_s - gap_s}),sin(2*PI*440*t),0)"
        for i in range(pieces)
    )
    _run(
        [
            "-f", "lavfi",
            "-i", f"aevalsrc='{expression}':s=16000:d={pieces * piece_s}",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(path),
        ]
    )
    return path


def stereo_mp3(path: Path, *, seconds: float = 3.0) -> Path:
    """A compressed stereo file at a non-16 kHz rate, to prove optimisation happens."""
    _run(
        [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds}",
            "-ac", "2",
            "-ar", "44100",
            "-c:a", "libmp3lame",
            str(path),
        ]
    )
    return path


def video_with_audio(path: Path, *, seconds: float = 3.0) -> Path:
    """An .mp4 with a real video stream and an audio track."""
    _run(
        [
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
            "-map", "0:a", "-map", "1:v",
            "-c:a", "aac", "-c:v", "mpeg4", "-shortest",
            str(path),
        ]
    )
    return path


def video_without_audio(path: Path, *, seconds: float = 2.0) -> Path:
    """An .mp4 that has a video stream and no audio stream at all."""
    _run(
        [
            "-f", "lavfi",
            "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
            "-c:v", "mpeg4",
            str(path),
        ]
    )
    return path


def corrupt_audio(path: Path, *, size: int = 4096) -> Path:
    """A file named like audio whose bytes are not audio."""
    path.write_bytes(bytes(range(256)) * (size // 256))
    return path


def probe_streams(path: Path) -> dict[str, str]:
    """ffprobe the first audio stream's codec/rate/channels — an independent oracle."""
    out = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "default=nw=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    facts = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        facts[key] = value
    return facts

def two_track_audio(path: Path, seconds: float = 1.0) -> Path:
    """An audio file carrying two audio tracks, for the "which one?" warning."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}",
            "-map", "0:a", "-map", "1:a", "-c:a", "aac", str(path),
        ],
        check=True,
    )
    return path
