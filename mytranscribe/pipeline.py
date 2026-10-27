"""The transcription pipeline: Pieces, progress, cancellation and output writing.

Both front ends call this module — `transcribe.sh` and the Worker. It knows about
audio, formats, Pieces, progress and cancellation, and nothing about Jobs,
Queues or HTTP.

The seam is `transcribe_audio(request, transcriber, ...)`: prepared 16 kHz mono
audio in, a `Transcript` out, with progress reported through a callback and
cancellation observed through a predicate checked between Pieces.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

# sea_asr cuts long audio into Pieces of at most 30 s, on the quietest moment
# near each 30 s mark. The app mirrors that so progress and cancellation have
# the same granularity as the model.
PIECE_SECONDS = 30.0

MODEL_ID = "PixelML/sea-asr-id"

# Word timings are only collected for formats that need them; asking changes
# how the audio is decoded.
FORMATS_WITH_WORDS = ("srt", "vtt", "json")
OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")


class PipelineError(Exception):
    """Something went wrong that a person should be told about plainly."""


class EmptyAudio(PipelineError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "This audio contains nothing to transcribe. It may be silent or too short."
        )
        self.detail = detail


class Cancelled(PipelineError):
    """The work was stopped on request, not because it failed."""

    def __init__(self, pieces_done: int, text: str = "") -> None:
        self.pieces_done = pieces_done
        self.text = text
        super().__init__("Transcription was cancelled.")


@dataclass(frozen=True)
class Progress:
    """A snapshot of where a transcription has got to.

    `pieces_done` out of `pieces_total` is the source of truth; any percentage a
    UI shows is derived from it.
    """

    pieces_done: int
    pieces_total: int
    partial_text: str

    @property
    def fraction(self) -> float:
        return self.pieces_done / self.pieces_total if self.pieces_total else 0.0


@dataclass(frozen=True)
class TranscribeRequest:
    """What to transcribe, into which format, and where to write it."""

    source: Path
    output_format: str = "txt"
    destination: Path | None = None
    model: str = MODEL_ID
    mode: str = "lm"

    def __post_init__(self) -> None:
        if self.output_format not in OUTPUT_FORMATS:
            raise PipelineError(
                f"Output format must be one of {', '.join(OUTPUT_FORMATS)} "
                f"(got {self.output_format!r})."
            )

    @property
    def wants_words(self) -> bool:
        return self.output_format in FORMATS_WITH_WORDS


@dataclass
class Transcript:
    """The finished text, plus what produced it."""

    text: str
    output_format: str
    model: str
    mode: str
    pieces_total: int
    duration_s: float = 0.0
    words: list[dict] = field(default_factory=list)
    destination: Path | None = None
    elapsed_s: float = 0.0

    def rendered(self) -> str:
        """The Transcript as the bytes that would be downloaded."""
        return render(self)


def transcribe_audio(
    request: TranscribeRequest,
    *,
    transcriber,
    on_progress: Callable[[Progress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe prepared audio one Piece at a time.

    Reports progress as each Piece completes, starting with a report of the
    total, and checks `should_cancel()` before every Piece. Raises `Cancelled`
    when it is asked to stop, `EmptyAudio` when there is nothing to transcribe.
    """
    from sea_asr.audio import load_audio, split_long

    source = Path(request.source)
    try:
        wav = load_audio(str(source))
    except Exception as error:  # noqa: BLE001 - the reason matters, the type does not
        raise PipelineError(f"This audio could not be opened: {error}") from error

    pieces = split_long(wav)
    total = len(pieces)
    if total == 0:
        raise EmptyAudio(f"{source.name} yielded no Pieces")

    duration_s = len(wav) / 16_000
    texts: list[str] = []
    words: list[dict] = []
    report = on_progress or (lambda _progress: None)
    report(Progress(pieces_done=0, pieces_total=total, partial_text=""))

    for done, (start_sample, piece_audio) in enumerate(pieces, start=1):
        if should_cancel is not None and should_cancel():
            raise Cancelled(pieces_done=done - 1, text=" ".join(texts))

        spoken = transcriber.transcribe(piece_audio, return_words=request.wants_words)
        text, piece_words = _unpack(spoken)
        if text:
            texts.append(text)
        words.extend(_rebase(piece_words, start_sample / 16_000))

        report(
            Progress(
                pieces_done=done,
                pieces_total=total,
                partial_text=" ".join(texts),
            )
        )

    transcript = Transcript(
        text=" ".join(texts).strip(),
        output_format=request.output_format,
        model=request.model,
        mode=request.mode,
        pieces_total=total,
        duration_s=duration_s,
        words=words,
        destination=request.destination,
    )
    if request.destination is not None:
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_text(transcript.rendered(), encoding="utf-8")
    return transcript


def render(transcript: Transcript) -> str:
    """Serialise a Transcript in its format.

    `.txt` is the whole text. `.srt`, `.vtt` and `.json` all carry timestamps,
    which by this point have been rebased onto the whole recording — see
    `_rebase` for why that cannot be skipped.
    """
    output_format = transcript.output_format
    if output_format == "txt":
        return transcript.text.strip() + "\n"
    if output_format == "srt":
        return _to_srt(transcript.words)
    if output_format == "vtt":
        return _to_vtt(transcript.words)
    if output_format == "json":
        return json.dumps(
            {
                "model": transcript.model,
                "mode": transcript.mode,
                "text": transcript.text,
                "words": transcript.words,
            },
            ensure_ascii=False,
            indent=2,
        )
    raise PipelineError(f"Output format must be one of {', '.join(OUTPUT_FORMATS)}.")


def _to_srt(words: list[dict]) -> str:
    if not words:
        return ""
    from sea_asr import to_srt

    return to_srt(words)


def _to_vtt(words: list[dict]) -> str:
    if not words:
        return ""
    try:
        from sea_asr import to_vtt
    except ImportError:  # a release that ships SRT only
        from sea_asr import to_srt

        return "WEBVTT\n\n" + to_srt(words).replace(",", ".")
    return to_vtt(words)


def _unpack(spoken) -> tuple[str, list[dict]]:
    """Accept the shapes a transcriber may return."""
    if isinstance(spoken, dict):
        return str(spoken.get("text", "")), list(spoken.get("words") or [])
    return str(spoken), []


def _rebase(words: Iterable[dict], offset_s: float) -> list[dict]:
    """Add a Piece's offset to every word's timings.

    Word timings come back relative to the Piece they were decoded in, not to
    the file. Forgetting this produces subtitles that restart at zero every 30
    seconds — plausible-looking files that are silently wrong.
    """
    rebased = []
    for word in words:
        moved = dict(word)
        moved["start"] = round(float(word.get("start", 0.0)) + offset_s, 3)
        moved["end"] = round(float(word.get("end", 0.0)) + offset_s, 3)
        rebased.append(moved)
    return rebased