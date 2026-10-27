"""t02 — the Piece loop: progress, cancellation, text output.

The seam is `mytranscribe.pipeline.transcribe_audio`: prepared audio in, a
Transcript out, progress and cancellation observed through callbacks. The
transcriber is injected so the loop can be driven deterministically with a fake;
exactly one slow test uses the real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from media import piece_flavoured_wav

from mytranscribe.pipeline import (
    Cancelled,
    EmptyAudio,
    Progress,
    TranscribeRequest,
    transcribe_audio,
)


# ------------------------------------------------------------------- a fake ASR

@dataclass
class FakeASR:
    """Stands in for the real model at the only level the loop cares about.

    It returns one word per call, offset-relative (start 0.5 s into the Piece) —
    exactly the behaviour that motivated the rebasing rule.
    """

    text_per_piece: list[str] | None = None
    calls: int = 0
    words_requests: list[bool] = field(default_factory=list)

    def transcribe(self, audio, return_words: bool = False):
        self.words_requests.append(return_words)
        index = self.calls
        self.calls += 1
        text = self._text(index)
        if return_words:
            return {"text": text, "words": [{"word": text.split()[0], "start": 0.5, "end": 0.9}]}
        return text

    def _text(self, index: int) -> str:
        if self.text_per_piece is not None:
            return self.text_per_piece[index % len(self.text_per_piece)]
        return f"piece{index + 1}"


def multi_piece_audio(path: Path, *, pieces: int) -> Path:
    """Audio that certainly contains several Pieces, so the loop is exercised."""
    return piece_flavoured_wav(path, pieces=pieces)


def read_wav(path: Path):
    import soundfile as sf

    wav, _ = sf.read(str(path), dtype="float32")
    return wav


# ------------------------------------------------------------------ Piece loop

def test_progress_is_reported_once_before_and_once_per_piece(work_dir: Path) -> None:
    audio = multi_piece_audio(work_dir / "long.wav", pieces=6)
    seen: list[Progress] = []

    result = transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt"),
        transcriber=FakeASR(),
        on_progress=seen.append,
    )

    total = result.pieces_total
    assert total >= 3, "fixture should span several Pieces"
    # The total is announced before the first Piece, then one report per Piece.
    assert [p.pieces_done for p in seen] == list(range(total + 1))
    assert {p.pieces_total for p in seen} == {total}


def test_pieces_match_the_fraction_reported(work_dir: Path) -> None:
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)
    seen: list[Progress] = []

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt"),
        transcriber=FakeASR(),
        on_progress=seen.append,
    )

    assert seen[0].fraction == 0.0
    assert seen[-1].fraction == 1.0
    total = seen[-1].pieces_total
    assert [p.fraction for p in seen] == [p.pieces_done / total for p in seen]


def test_progress_carries_the_text_accumulated_so_far(work_dir: Path) -> None:
    fake = FakeASR(text_per_piece=["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh"])
    audio = multi_piece_audio(work_dir / "long.wav", pieces=5)
    seen: list[Progress] = []

    result = transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt"),
        transcriber=fake,
        on_progress=seen.append,
    )

    words = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh"][: result.pieces_total]
    assert seen[0].partial_text == ""
    assert [p.partial_text for p in seen[1:]] == [
        " ".join(words[: i + 1]) for i in range(result.pieces_total)
    ]
    # At 100 percent the partial text is the whole text.
    assert seen[-1].partial_text == result.text


def test_progress_is_a_snapshot_not_a_shared_mutable(work_dir: Path) -> None:
    """A consumer that keeps the Progress objects must see distinct values."""
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)
    seen: list[Progress] = []

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt"),
        transcriber=FakeASR(),
        on_progress=seen.append,
    )

    assert len({id(p) for p in seen}) == len(seen)
    assert [p.pieces_done for p in seen] == list(range(len(seen)))


def test_cancelling_after_n_pieces_stops_before_n_plus_one(work_dir: Path) -> None:
    audio = multi_piece_audio(work_dir / "long.wav", pieces=6)
    fake = FakeASR()
    seen: list[Progress] = []

    def should_stop() -> bool:
        return len(seen) >= 3  # the initial report plus two Pieces

    with pytest.raises(Cancelled) as caught:
        transcribe_audio(
            TranscribeRequest(source=audio, output_format="txt"),
            transcriber=fake,
            on_progress=seen.append,
            should_cancel=should_stop,
        )

    assert caught.value.pieces_done == 2
    assert [p.pieces_done for p in seen] == [0, 1, 2]
    # Two Pieces were transcribed and no more.
    assert fake.calls == 2


def test_cancelling_before_any_piece_stops_immediately(work_dir: Path) -> None:
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)
    fake = FakeASR()
    seen: list[Progress] = []

    with pytest.raises(Cancelled):
        transcribe_audio(
            TranscribeRequest(source=audio, output_format="txt"),
            transcriber=fake,
            on_progress=seen.append,
            should_cancel=lambda: True,
        )

    assert [p.pieces_done for p in seen] == [0]
    assert fake.calls == 0


def test_cancelled_carries_the_text_produced_so_far(work_dir: Path) -> None:
    fake = FakeASR(text_per_piece=["satu", "dua"])
    audio = multi_piece_audio(work_dir / "long.wav", pieces=6)
    seen: list[Progress] = []

    with pytest.raises(Cancelled) as caught:
        transcribe_audio(
            TranscribeRequest(source=audio, output_format="txt"),
            transcriber=fake,
            on_progress=seen.append,
            should_cancel=lambda: len(seen) >= 3,
        )

    assert caught.value.text == "satu dua"


def test_txt_output_is_written_and_equals_the_returned_text(work_dir: Path) -> None:
    fake = FakeASR(text_per_piece=["halo", "dunia", "kita", "semua", "selesai", "akhir", "tuntas"])
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)
    out = work_dir / "out.txt"

    result = transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt", destination=out),
        transcriber=fake,
    )

    expected = " ".join(fake.text_per_piece[: result.pieces_total])
    assert result.text == expected
    assert out.read_text(encoding="utf-8") == expected + "\n"


def test_txt_output_has_no_trailing_space_and_one_final_newline(work_dir: Path) -> None:
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)
    out = work_dir / "out.txt"

    transcribe_audio(
        TranscribeRequest(source=audio, output_format="txt", destination=out),
        transcriber=FakeASR(),
    )

    written = out.read_text(encoding="utf-8")
    assert written.endswith("\n") and not written.endswith("\n\n")
    assert " \n" not in written


def test_txt_does_not_ask_for_word_timings(work_dir: Path) -> None:
    """Asking for words changes the decode path, so .txt must not ask."""
    fake = FakeASR()
    audio = multi_piece_audio(work_dir / "long.wav", pieces=4)

    transcribe_audio(TranscribeRequest(source=audio, output_format="txt"), transcriber=fake)

    assert fake.words_requests and not any(fake.words_requests)


def test_audio_with_nothing_in_it_is_reported_plainly(work_dir: Path) -> None:
    audio = work_dir / "silence.wav"
    import soundfile as sf

    sf.write(str(audio), [0.0] * 16, 16_000)

    with pytest.raises(EmptyAudio) as caught:
        transcribe_audio(TranscribeRequest(source=audio, output_format="txt"), transcriber=FakeASR())

    assert "nothing to transcribe" in str(caught.value).lower()


def test_unreadable_audio_is_reported_plainly(work_dir: Path) -> None:
    audio = work_dir / "broken.wav"
    audio.write_bytes(b"not a wav at all")

    with pytest.raises(Exception) as caught:
        transcribe_audio(TranscribeRequest(source=audio, output_format="txt"), transcriber=FakeASR())

    assert "could not be opened" in str(caught.value).lower()


def test_unknown_format_is_refused(work_dir: Path) -> None:
    with pytest.raises(Exception) as caught:
        TranscribeRequest(source=work_dir / "x.wav", output_format="doc")

    assert "txt" in str(caught.value).lower()