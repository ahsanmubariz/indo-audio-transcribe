"""t03 — timestamped formats, and the Piece-offset rebasing they depend on.

The trap this ticket exists for: word timings come back relative to the Piece
they were decoded in. A writer that forgets to add the Piece offset produces
subtitles that restart at zero every 30 seconds — plausible-looking and wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from media import piece_flavoured_wav

from mytranscribe.pipeline import (
    Progress,
    TranscribeRequest,
    Transcript,
    render,
    transcribe_audio,
)

TIMESTAMP = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


@dataclass
class WordFakeASR:
    """One word per Piece, timed relative to the Piece — like the real model.

    The word is placed at 1.0–2.0 s into whatever audio it was handed, which
    makes an unrebased timestamp obvious: it would read 00:00:01 every time.
    """

    words_per_piece: int = 2
    calls: int = 0
    seen_durations: list[float] = field(default_factory=list)

    def transcribe(self, audio, return_words: bool = False):
        import numpy as np

        wav = np.asarray(audio, dtype=np.float32)
        self.seen_durations.append(len(wav) / 16_000)
        index = self.calls
        self.calls += 1
        words = [
            {"word": f"kata{index}a", "start": 1.0, "end": 1.4},
            {"word": f"kata{index}b", "start": 2.0, "end": 2.6},
        ][: self.words_per_piece]
        text = " ".join(word["word"] for word in words)
        return {"text": text, "words": words} if return_words else text


def stamps(text: str) -> list[float]:
    found = []
    for hours, minutes, seconds, millis in TIMESTAMP.findall(text):
        found.append(int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000)
    return found


def audio_with_pieces(path: Path, pieces: int) -> Path:
    return piece_flavoured_wav(path, pieces=pieces)


# ------------------------------------------------------------- rebasing itself

def test_word_times_are_rebased_by_their_piece_offset(work_dir: Path) -> None:
    """The first word of Piece 2 must not report a time inside Piece 1."""
    audio = audio_with_pieces(work_dir / "long.wav", pieces=4)

    transcript = transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt"),
        transcriber=WordFakeASR(),
    )

    starts = [word["start"] for word in transcript.words]
    # Every Piece contributes 2 words at 1.0 s and 2.0 s into itself, so the
    # rebased times must be strictly increasing across Piece boundaries.
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), f"duplicate timestamps: {starts}"
    # With no rebasing, the first word of every Piece would be 1.0.
    assert starts.count(1.0) == 1
    assert transcript.words[0]["start"] == pytest.approx(1.0, abs=0.01)
    assert transcript.words[2]["start"] > 25.0, "Piece 2's words were not rebased"


def test_a_single_piece_is_unaffected_by_rebasing(work_dir: Path) -> None:
    """Offset zero must be a no-op, not a reason to shift anything."""
    audio = audio_with_pieces(work_dir / "one.wav", pieces=1)

    transcript = transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt"),
        transcriber=WordFakeASR(),
    )

    assert [word["start"] for word in transcript.words] == [1.0, 2.0]
    assert [word["end"] for word in transcript.words] == [1.4, 2.6]


def test_rebased_words_are_rounded_to_milliseconds(work_dir: Path) -> None:
    """Subtitle stamps carry milliseconds; do not accumulate float noise."""
    audio = audio_with_pieces(work_dir / "long.wav", pieces=3)

    transcript = transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt"),
        transcriber=WordFakeASR(),
    )

    for word in transcript.words:
        assert word["start"] == round(word["start"], 3)
        assert word["end"] == round(word["end"], 3)


# ---------------------------------------------------------------- the formats

def test_srt_output_is_a_valid_srt_document(work_dir: Path) -> None:
    audio = audio_with_pieces(work_dir / "long.wav", pieces=3)
    out = work_dir / "out.srt"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt", destination=out),
        transcriber=WordFakeASR(),
    )
    written = out.read_text(encoding="utf-8")

    assert written.startswith("1\n")
    assert " --> " in written
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", written)
    # Numbered consecutively from 1.
    numbers = [int(line) for line in written.splitlines() if line.isdigit()]
    assert numbers == list(range(1, len(numbers) + 1))


def test_srt_timestamps_increase_across_pieces(work_dir: Path) -> None:
    audio = audio_with_pieces(work_dir / "long.wav", pieces=5)
    out = work_dir / "out.srt"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt", destination=out),
        transcriber=WordFakeASR(),
    )

    found = stamps(out.read_text(encoding="utf-8"))
    assert found == sorted(found), "cues restart at zero: a Piece offset was missed"


def test_vtt_output_is_a_valid_vtt_document(work_dir: Path) -> None:
    audio = audio_with_pieces(work_dir / "short.wav", pieces=2)
    out = work_dir / "out.vtt"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="vtt", destination=out),
        transcriber=WordFakeASR(),
    )
    written = out.read_text(encoding="utf-8")

    assert written.startswith("WEBVTT")
    # VTT uses a full stop before the milliseconds, never a comma.
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", written)
    assert "," not in written.replace("kata", "")


def test_vtt_timestamps_increase_across_pieces(work_dir: Path) -> None:
    audio = audio_with_pieces(work_dir / "long.wav", pieces=4)
    out = work_dir / "out.vtt"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="vtt", destination=out),
        transcriber=WordFakeASR(),
    )

    found = stamps(out.read_text(encoding="utf-8"))
    assert found == sorted(found)


def test_json_output_reports_model_mode_text_and_words(work_dir: Path) -> None:
    import json

    audio = audio_with_pieces(work_dir / "short.wav", pieces=2)
    out = work_dir / "out.json"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="json", destination=out),
        transcriber=WordFakeASR(),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["model"] == "PixelML/sea-asr-id"
    assert payload["mode"] == "lm"
    assert payload["text"].startswith("kata0a")
    assert payload["words"][0] == {"word": "kata0a", "start": 1.0, "end": 1.4}
    # Words in JSON are rebased too, not Piece-relative.
    assert payload["words"][2]["start"] > 25.0


def test_timestamped_formats_ask_for_word_timings_and_txt_does_not(work_dir: Path) -> None:
    audio = audio_with_pieces(work_dir / "short.wav", pieces=2)

    asked: dict[str, bool] = {}
    for output_format in ("txt", "srt", "vtt", "json"):
        requested = TranscribeRequest(source=audio, output_format=output_format)

        class Spy(WordFakeASR):
            def transcribe(self, audio, return_words: bool = False):
                asked[output_format] = return_words
                return super().transcribe(audio, return_words=return_words)

        transcribe_audio(requested, transcriber=Spy())

    assert asked == {"txt": False, "srt": True, "vtt": True, "json": True}


def test_rendered_transcript_matches_what_was_written(work_dir: Path) -> None:
    """A download must serve exactly what the writer produced."""
    audio = audio_with_pieces(work_dir / "short.wav", pieces=2)
    out = work_dir / "out.srt"

    transcript = transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt", destination=out),
        transcriber=WordFakeASR(),
    )

    assert transcript.rendered() == out.read_text(encoding="utf-8")


def test_rendering_needs_no_destination(work_dir: Path) -> None:
    """The Worker keeps transcripts in memory, so rendering cannot depend on a file."""
    audio = audio_with_pieces(work_dir / "short.wav", pieces=2)

    transcript = transcribe_audio(
        TranscribeRequest(source=audio, output_format="srt"),
        transcriber=WordFakeASR(),
    )

    assert " --> " in transcript.rendered()


def test_empty_words_render_an_empty_document_rather_than_crashing() -> None:
    transcript = Transcript(
        text="",
        output_format="srt",
        model="PixelML/sea-asr-id",
        mode="lm",
        pieces_total=0,
    )

    assert render(transcript) == ""