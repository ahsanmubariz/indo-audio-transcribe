"""Fakes for the seams a test cannot afford to reach through: the model, and time."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class FakeTranscriber:
    """Stands in for the loaded model.

    `per_piece_seconds` paces each call so a test can observe a Job mid-flight
    through the HTTP surface; without it a Piece completes faster than any poll.
    `on_call` runs before each Piece returns, for tests that need a latch.
    """

    texts: list[str] = field(default_factory=lambda: ["satu", "dua", "tiga", "empat", "lima"])
    per_piece_seconds: float = 0.0
    calls: int = 0
    durations: list[float] = field(default_factory=list)
    on_call: object | None = None
    # Records whether word timings were asked for, which drives the decode path.
    asked_for_words: list[bool] = field(default_factory=list)

    def transcribe(self, audio, return_words: bool = False):
        import numpy as np

        wav = np.asarray(audio, dtype=np.float32)
        self.durations.append(len(wav) / 16_000)
        self.asked_for_words.append(return_words)
        index = self.calls
        self.calls += 1
        if self.on_call is not None:
            self.on_call(index)
        if self.per_piece_seconds:
            time.sleep(self.per_piece_seconds)
        text = self.texts[index % len(self.texts)]
        if return_words:
            return {
                "text": text,
                "words": [{"word": text, "start": 0.5, "end": 1.0}],
            }
        return text


class GatedTranscriber(FakeTranscriber):
    """Holds every Piece on a latch so a test can inspect the Queue mid-flight."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0

    def transcribe(self, audio, return_words: bool = False):
        with self.lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.entered.set()
            self.gate.wait(20)
            return super().transcribe(audio, return_words=return_words)
        finally:
            with self.lock:
                self.concurrent -= 1


class ManualClock:
    """A clock a test moves by hand, so retention needs no waiting."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds