"""Parity with the one-shot path, driven by the real model.

These are the standing proof of ADR-0001's claim: driving the model Piece by
Piece reproduces what `transcribe.sh` produces for the same input.

These need a reference recording, and none ships with the repo: a real recording is
somebody's speech and its transcript is their words. Set
`MYTRANSCRIBE_REFERENCE_AUDIO` (and `MYTRANSCRIBE_REFERENCE_TRANSCRIPT`) to your
own pair and the suite runs; without them it skips with instructions.

**Two kinds of run, for two different reasons.** The spec's Testing Decisions ask
for a slow test "on a few seconds of audio" comparing against the one-shot path;
its acceptance criteria need the whole recording. Both are here, and the split
matters for a practical reason.

- The **one-shot comparison** (the 5 ms timestamp tolerance) runs on a short
  slice. A one-shot transcription of a long recording holds the whole decode in
  memory at once — measured at 7.7 GB for 13 minutes — and on a 15 GB machine, run
  alongside the per-Piece passes, it gets OOM-killed. A slice is also the harder
  test of the trap this project cares about, because 60 s still spans more than one
  Piece, so a missing offset shows up.
- The **byte-equality, Piece-count, monotonicity and timing** assertions need the
  real recording, so they use it — with the per-Piece path only, which is the path
  that ships.

Every expensive run happens **once** per module and is shared.

Marked slow and skipped unless `MYTRANSCRIBE_RUN_SLOW=1` is set, so the default
suite stays fast.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest

from conftest import require_reference_audio, require_reference_transcript

from mytranscribe.audio import prepare_audio
from mytranscribe.pipeline import PIECE_SECONDS, TranscribeRequest, transcribe_audio

pytestmark = pytest.mark.slow

MODEL_ID = "PixelML/sea-asr-id"
TIMESTAMP_TOLERANCE_MS = 5


# --------------------------------------------------------------- shared work

@pytest.fixture(scope="module", autouse=True)
def _reference_or_skip():
    """Skip before any model loads.

    Without this the `asr` fixture can be resolved first, so a missing reference
    recording surfaces as a 2.8 GB download and an ImportError rather than a skip.
    """
    require_reference_audio()


@pytest.fixture(scope="module")
def asr():
    """The real model, loaded once — as the Worker does at startup."""
    import torch

    torch.set_num_threads(12)
    from sea_asr import SeaASR

    return SeaASR.from_pretrained(MODEL_ID, lm=MODEL_ID, device="cpu")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    work = tmp_path_factory.mktemp("parity")
    audio = prepare_audio(require_reference_audio(), work_dir=work, threads=12)
    yield audio
    audio.cleanup()


@pytest.fixture(scope="module")
def runs(asr, prepared):
    """One pipeline run per format, plus one one-shot run. Shared by every test.

    `elapsed` is the .txt run's wall time, which is what criterion 6 measures.
    """
    import tempfile

    out_dir = Path(tempfile.mkdtemp(prefix="parity-out-"))

    timed = time.monotonic()
    txt = transcribe_audio(
        TranscribeRequest(
            source=prepared.path, output_format="txt", destination=out_dir / "out.txt"
        ),
        transcriber=asr,
        on_progress=lambda progress: progress,
    )
    elapsed = time.monotonic() - timed

    bodies = {"txt": (out_dir / "out.txt").read_text(encoding="utf-8")}
    words = {"txt": txt.words}
    for output_format in ("srt", "vtt", "json"):
        destination = out_dir / f"out.{output_format}"
        transcript = transcribe_audio(
            TranscribeRequest(
                source=prepared.path, output_format=output_format, destination=destination
            ),
            transcriber=asr,
        )
        bodies[output_format] = destination.read_text(encoding="utf-8")
        words[output_format] = transcript.words

    return {
        "txt": txt,
        "bodies": bodies,
        "words": words,
        "elapsed": elapsed,
    }


@pytest.fixture(scope="module")
def slice_comparison(asr, tmp_path_factory):
    """Both paths on the same 60 s slice: the one-shot-vs-per-Piece comparison.

    A slice, not the full recording, because a one-shot pass over a long file needs
    the whole thing resident and is OOM-killed beside the per-Piece runs. 60 s
    still spans more than one Piece, so a missing offset would still fail here.
    """
    import subprocess

    work = tmp_path_factory.mktemp("slice")
    clip = work / "slice_16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", "120", "-t", "60",
            "-i", str(require_reference_audio()),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clip),
        ],
        check=True,
    )

    one_shot = asr.transcribe(str(clip), return_words=True)

    ours = {}
    for output_format in ("srt", "vtt", "json"):
        destination = work / f"slice.{output_format}"
        transcript = transcribe_audio(
            TranscribeRequest(
                source=clip, output_format=output_format, destination=destination
            ),
            transcriber=asr,
        )
        ours[output_format] = {
            "body": destination.read_text(encoding="utf-8"),
            "words": transcript.words,
            "pieces": transcript.pieces_total,
        }

    txt_destination = work / "slice.txt"
    txt = transcribe_audio(
        TranscribeRequest(source=clip, output_format="txt", destination=txt_destination),
        transcriber=asr,
    )
    ours["txt"] = {"body": txt_destination.read_text(encoding="utf-8"), "words": txt.words,
                   "pieces": txt.pieces_total}

    return {"one_shot": one_shot, "ours": ours}


# ----------------------------------------------------------------- criterion 1

def test_pipeline_txt_is_byte_identical_to_the_one_shot_output(runs) -> None:
    """ADR-0001's claim: Piece-by-Piece matches one-shot, byte for byte.

    Compares against the transcript supplied with MYTRANSCRIBE_REFERENCE_TRANSCRIPT,
    which should be what `./transcribe.sh` produced for the same recording.
    """
    reference = require_reference_transcript().read_text(encoding="utf-8")
    transcript = runs["txt"]

    assert runs["bodies"]["txt"] == reference
    assert transcript.text == reference.strip()


def test_no_piece_is_longer_than_the_limit(runs) -> None:
    """Pieces are cut at the quietest moment *before* each 30 s mark, not at it.

    So each Piece is at most 30 s, which makes the count a lower bound rather than
    an exact quotient: a 796 s recording came out as 29 Pieces, not the 27 the
    arithmetic suggests, because the cut backtracks up to 5 s to find a pause.
    """
    transcript = runs["txt"]
    at_least = math.ceil(transcript.duration_s / PIECE_SECONDS)

    assert transcript.pieces_total >= at_least
    # Backtracking is bounded by the search window, so the count cannot run away.
    assert transcript.pieces_total <= math.ceil(transcript.duration_s / (PIECE_SECONDS - 5.0))


def test_transcription_completes_under_200_seconds(runs) -> None:
    """Acceptance criterion 6, measured on the shared .txt run."""
    elapsed = runs["elapsed"]
    assert elapsed < 200, f"pipeline took {elapsed:.1f}s"


# --------------------------------------------------------------- criteria 4 & 5

def test_the_slice_spans_more_than_one_piece(slice_comparison) -> None:
    """The comparison is only meaningful if rebasing is actually exercised."""
    assert slice_comparison["ours"]["srt"]["pieces"] >= 2


def test_srt_matches_the_one_shot_path_within_5ms(slice_comparison) -> None:
    """Acceptance criterion 4: parity for timestamped output, tolerant of rounding."""
    from sea_asr import to_srt

    expected = to_srt(slice_comparison["one_shot"]["words"])
    drift_ms = _compare_timestamps(expected, slice_comparison["ours"]["srt"]["body"])

    assert drift_ms <= TIMESTAMP_TOLERANCE_MS, f"timestamps drifted {drift_ms} ms"
    # Word count parity — the drift is rounding, not missing words.
    assert len(slice_comparison["ours"]["srt"]["words"]) == len(
        slice_comparison["one_shot"]["words"]
    )


def test_the_text_of_a_slice_is_identical_to_the_one_shot_path(slice_comparison) -> None:
    """ADR-0001's claim, on audio short enough to run both ways."""
    ours = slice_comparison["ours"]["txt"]["body"].strip()
    theirs = str(slice_comparison["one_shot"]["text"]).strip()

    assert ours == theirs


def test_vtt_matches_the_one_shot_timestamps_within_5ms(slice_comparison) -> None:
    """.vtt is in the same tolerance class as .srt, and wears the same stamps."""
    from sea_asr import to_srt

    expected = to_srt(slice_comparison["one_shot"]["words"])
    drift_ms = _compare_timestamps(expected, slice_comparison["ours"]["vtt"]["body"])

    assert drift_ms <= TIMESTAMP_TOLERANCE_MS, f"vtt timestamps drifted {drift_ms} ms"
    assert len(slice_comparison["ours"]["vtt"]["words"]) == len(slice_comparison["one_shot"]["words"])


def test_json_word_timings_match_the_one_shot_timestamps_within_5ms(slice_comparison) -> None:
    """.json carries the timings as numbers, so compare the numbers.

    The other writers format them as HH:MM:SS.mmm; JSON keeps seconds, which is
    the form the words themselves came in. Comparing the two requires reading the
    JSON values rather than scanning for stamps.
    """
    import json

    payload = json.loads(slice_comparison["ours"]["json"]["body"])
    theirs = slice_comparison["one_shot"]["words"]

    assert len(payload["words"]) == len(theirs)
    worst_ms = 0.0
    for ours_word, their_word in zip(payload["words"], theirs):
        assert ours_word["word"] == their_word["word"]
        for edge in ("start", "end"):
            drift = abs(float(ours_word[edge]) - float(their_word[edge]))
            worst_ms = max(worst_ms, drift * 1000)

    assert worst_ms <= TIMESTAMP_TOLERANCE_MS, f"json timings drifted {worst_ms} ms"


def test_srt_timestamps_are_monotonic_and_reach_the_end(runs) -> None:
    """Acceptance criterion 5 — this is what catches a missing Piece offset."""
    cues = _parse_srt(runs["bodies"]["srt"])

    assert cues, "no cues were written"
    starts = [start for start, _ in cues]
    assert starts == sorted(starts), "cues restart: a Piece offset was not applied"
    assert all(start <= end for start, end in cues)
    for earlier, later in zip(starts, starts[1:]):
        assert later >= earlier, "a cue began before the previous one"

    # The last cue should end near the end of the audio. A missing Piece offset
    # leaves it short by a whole Piece; a stray offset pushes it past the audio.
    duration = runs["txt"].duration_s
    assert 0 < cues[-1][1] <= duration + 1
    assert duration - cues[-1][1] < PIECE_SECONDS


def test_json_names_the_model_and_mode_it_used(runs) -> None:
    import json

    payload = json.loads(runs["bodies"]["json"])

    assert payload["model"] == MODEL_ID
    assert payload["mode"] == "lm"
    assert payload["text"]
    assert payload["words"]
    assert set(payload["words"][0]) == {"word", "start", "end"}


# ------------------------------------------------------------------- helpers

def _compare_timestamps(expected: str, produced: str) -> float:
    """Largest per-timestamp drift in seconds, in milliseconds."""
    pattern = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}")
    left = pattern.findall(expected)
    right = pattern.findall(produced)

    assert len(left) == len(right), (
        f"different number of timestamps: {len(left)} vs {len(right)}"
    )

    worst = 0.0
    for first, second in zip(left, right):
        drift = abs(_to_seconds(first) - _to_seconds(second))
        worst = max(worst, drift)
    return worst * 1000


def _to_seconds(stamp: str) -> float:
    hours, minutes, rest = stamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _parse_srt(text: str) -> list[tuple[float, float]]:
    cues = []
    for line in text.splitlines():
        match = re.match(r"(\S+) --> (\S+)", line.strip())
        if match:
            cues.append((_to_seconds(match.group(1)), _to_seconds(match.group(2))))
    return cues