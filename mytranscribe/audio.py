"""Audio preparation: probe a source, optimise it to 16 kHz mono, report its facts.

This module knows about audio files and nothing else. It has no web-framework
types, no notion of a Job, and no idea a Queue exists — both front ends
(`transcribe.sh` and the Worker) call it and get the same answer.

`prepare_audio` is the seam. It either returns a `PreparedAudio` whose `path` is
16 kHz mono 16-bit PCM and whose `facts` describe the Original audio, or it
raises `AudioRejected` carrying a reason a person can act on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

SAMPLE_RATE = 16_000

# One label per rejection reason, so a front door can choose its own wording
# without re-deriving why the file was refused.
NO_AUDIO_STREAM = "no-audio-stream"
UNREADABLE = "unreadable"
VIDEO_NOT_ALLOWED = "video-not-allowed"
EMPTY_AUDIO = "empty-audio"

_READABLE_REASONS = {
    NO_AUDIO_STREAM: "This file has no audio in it. Choose a recording with sound.",
    UNREADABLE: "This file could not be read. It may be damaged or in a format "
                "that cannot be opened. Try a different file.",
    VIDEO_NOT_ALLOWED: "This is a video file. Audio only, or re-run with --allow-video "
                       "to use its audio track.",
    EMPTY_AUDIO: "This file contains no audio to transcribe.",
}


class AudioRejected(Exception):
    """A source that cannot be turned into audio worth transcribing.

    `reason` is a stable label; the message is what a person should read.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(_READABLE_REASONS.get(reason, reason))


@dataclass(frozen=True)
class AudioFacts:
    """What the UI needs to show about the Original audio."""

    duration_s: float
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    audio_tracks: int = 1

    @property
    def duration_label(self) -> str:
        return format_duration(self.duration_s)

    def original_audio(self) -> dict[str, object]:
        """The facts of the Original audio, shaped for a snapshot or a response.

        One definition, because `app.py` and `jobs.py` both publish it and a new
        field should not need editing in two places.
        """
        return {
            "format": self.format_name,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass
class PreparedAudio:
    """Optimised 16 kHz mono audio plus the facts of the file it came from.

    The directory holding `path` belongs to this object: call `cleanup()` when
    the audio is no longer needed.
    """

    path: Path
    facts: AudioFacts
    work_dir: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)


def format_duration(seconds: float) -> str:
    """795.9 -> '13:15', 3725 -> '1:02:05'."""
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def prepare_audio(
    source: str | os.PathLike[str],
    *,
    work_dir: str | os.PathLike[str] | None = None,
    raw: bool = False,
    allow_video: bool = True,
    threads: int | None = None,
) -> PreparedAudio:
    """Turn `source` into 16 kHz mono audio, or raise `AudioRejected`.

    `work_dir` is the directory the optimised file is created under; a unique
    subdirectory is made inside it and is removed by `cleanup()`. Pass `raw` to
    skip the clean-up filters. `allow_video=False` refuses video containers,
    which is what `transcribe.sh` does unless given `--allow-video`.

    Front doors that want to report progress between the two halves call
    `probe_audio` and `optimise_audio` instead; this is them, composed.
    """
    facts = probe_audio(source, allow_video=allow_video)
    return optimise_audio(
        source, facts=facts, work_dir=work_dir, raw=raw, threads=threads
    )


def probe_audio(
    source: str | os.PathLike[str], *, allow_video: bool = True
) -> AudioFacts:
    """Read a source's facts, or raise `AudioRejected` if it is not usable audio.

    Cheap: this only reads headers, so a front door can report the duration and
    format before committing to the expensive conversion.
    """
    path = Path(source)
    if not path.is_file():
        raise AudioRejected(UNREADABLE, f"no such file: {path}")

    tracks = _count_streams(path, "a")
    if tracks == 0:
        # A container that cannot be opened at all is a different complaint from
        # one that opens fine but carries no sound.
        error = _probe_error(path)
        if error:
            raise AudioRejected(UNREADABLE, error)
        raise AudioRejected(NO_AUDIO_STREAM, f"{path.name} has no audio stream")

    if not allow_video and _count_streams(path, "V") > 0:
        raise AudioRejected(VIDEO_NOT_ALLOWED, f"{path.name} is a video container")

    facts = _probe_facts(path)
    # More than one audio track means the first is used and the rest ignored, which
    # a person should hear about. Measured here so every front door can say it.
    return replace(facts, audio_tracks=tracks)


def optimise_audio(
    source: str | os.PathLike[str],
    *,
    facts: AudioFacts | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    raw: bool = False,
    threads: int | None = None,
) -> PreparedAudio:
    """Convert a source to 16 kHz mono 16-bit PCM, or raise `AudioRejected`."""
    path = Path(source)
    if facts is None:
        facts = probe_audio(path)

    base = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="prepare-", dir=base))
    target = scratch / "input_16k_mono.wav"

    try:
        _optimise(path, target, raw=raw, threads=threads)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise

    if _audio_length_s(target) <= 0.0:
        shutil.rmtree(scratch, ignore_errors=True)
        raise AudioRejected(EMPTY_AUDIO, f"{path.name} decoded to no audio")

    return PreparedAudio(path=target, facts=facts, work_dir=scratch)


# ------------------------------------------------------------------ internals

_FILTERS = (
    # mono -> cut rumble < 80 Hz -> cut > 7.8 kHz (useless at 16 kHz)
    # -> light FFT denoise -> EBU R128 loudness normalisation
    "aformat=channel_layouts=mono,highpass=f=80,lowpass=f=7800,"
    "afftdn=nr=10:nf=-40,loudnorm=I=-18:TP=-2:LRA=11"
)
_RAW_FILTERS = "aformat=channel_layouts=mono"


def _ffprobe() -> str:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError("ffprobe is not installed (sudo apt install ffmpeg)")
    return probe


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is not installed (sudo apt install ffmpeg)")
    return binary


def _count_streams(path: Path, selector: str) -> int:
    """How many streams of a kind ffprobe can see. Capital V means real video only."""
    result = subprocess.run(
        [
            _ffprobe(), "-v", "error", "-select_streams", selector,
            "-show_entries", "stream=index", "-of", "csv=p=0", "--", str(path),
        ],
        capture_output=True,
        text=True,
    )
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def _probe_error(path: Path) -> str:
    result = subprocess.run(
        [_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "--", str(path)],
        capture_output=True,
        text=True,
    )
    return result.stderr.strip() if result.returncode != 0 else ""


def _probe_facts(path: Path) -> AudioFacts:
    result = subprocess.run(
        [
            _ffprobe(), "-v", "error", "-select_streams", "a:0",
            "-show_entries",
            "format=format_long_name,duration:stream=codec_name,sample_rate,channels",
            "-of", "default=nw=1", "--", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioRejected(UNREADABLE, result.stderr.strip())

    raw: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            raw[key] = value

    return AudioFacts(
        duration_s=_number(raw.get("duration"), 0.0),
        format_name=raw.get("format_long_name") or "unknown format",
        codec=raw.get("codec_name") or "unknown",
        sample_rate=int(_number(raw.get("sample_rate"), 0)),
        channels=int(_number(raw.get("channels"), 0)),
    )


def _audio_length_s(path: Path) -> float:
    result = subprocess.run(
        [
            _ffprobe(), "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration", "-of", "csv=p=0", "--", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0.0
    return _number(result.stdout.strip(), 0.0)


def _number(value: str | None, default: float) -> float:
    if not value or value == "N/A":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _optimise(path: Path, target: Path, *, raw: bool, threads: int | None) -> None:
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if threads:
        command += ["-threads", str(threads), "-filter_threads", str(threads)]
    command += [
        "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", _RAW_FILTERS if raw else _FILTERS,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.is_file():
        detail = result.stderr.strip().splitlines()
        raise AudioRejected(
            UNREADABLE,
            detail[-1] if detail else "ffmpeg could not convert the file",
        )