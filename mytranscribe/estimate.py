"""Estimates: how long a Job will take, said before the wait, not after.

An estimate is a prediction, never a deadline. Two rules keep it honest:

* it is a range, not a single certain number, because throughput varies;
* once a Job is running it is derived from the Pieces actually done, so a bad
  initial guess corrects itself instead of marching confidently to 0.

The planning speeds come from this machine's measurements (see
the spec): 6.2x to 8.3x realtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import format_duration
from .pipeline import PIECE_SECONDS

# The planning range the spec states: "Numbers worth re-measuring, not assuming:
# 7.7 GB peak RSS, 6.2x-8.3x realtime, and ~5-10 s model load". The measurements
# behind it (docs/measured-constraints.md) are 6.5x at 6 threads through 8.3x at
# 12; 6.2 is the spec's conservative floor, used as the upper bound on time so a
# slow machine is not promised fast-machine speed.
SLOWEST_SPEEDUP = 6.2
FASTEST_SPEEDUP = 8.3

# Preparation (decode, filter, resample) is a fixed cost per Job. Measured at
# 26.3 s for the 796 s reference recording, but most of that is per-minute work,
# so a short file is much quicker; this is the floor to add to a full-length job.
PREPARE_SECONDS_PER_JOB = 12.0

# A Job past this length gets a plain warning. There is no size cap, so the
# warning is what makes an hour-long upload safe to accept.
LONG_JOB_SECONDS = 30 * 60


@dataclass(frozen=True)
class Estimate:
    """What the page needs to say how long the wait will be, and how sure it is."""

    seconds: float
    lower_seconds: float
    upper_seconds: float
    basis: str
    warning: str | None = None

    def as_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 1),
            "lower_seconds": round(self.lower_seconds, 1),
            "upper_seconds": round(self.upper_seconds, 1),
            "label": format_duration(self.seconds),
            "lower_label": format_duration(self.lower_seconds),
            "upper_label": format_duration(self.upper_seconds),
            "range_label": _range_label(self.lower_seconds, self.upper_seconds),
            "basis": self.basis,
            "warning": self.warning,
        }


def estimate_for(
    *,
    duration_s: float,
    pieces_total: int = 0,
    pieces_done: int = 0,
    elapsed_s: float = 0.0,
    work_ahead_s: float = 0.0,
) -> Estimate:
    """Estimate the remaining wait.

    `work_ahead_s` is the audio duration queued before this Job, so a Job behind
    three others does not promise the speed of a Job that starts now.
    """
    audio_left = _audio_remaining(duration_s, pieces_total, pieces_done)
    total_audio = audio_left + max(0.0, work_ahead_s)

    observed = _observed_speedup(pieces_done, elapsed_s)
    slowest, fastest = SLOWEST_SPEEDUP, FASTEST_SPEEDUP
    if observed is not None:
        # Once there is real evidence, trust it — but never faster than the
        # machine has ever managed, so a lucky first Piece cannot promise too much.
        slowest = min(observed, SLOWEST_SPEEDUP)
        fastest = min(max(observed, FASTEST_SPEEDUP), FASTEST_SPEEDUP * 1.2)
        slowest = max(min(slowest, fastest), FASTEST_SPEEDUP * 0.5)

    jobs = _jobs_ahead(total_audio)
    upper = total_audio / max(slowest, 0.1) + PREPARE_SECONDS_PER_JOB * jobs
    lower = total_audio / max(fastest, 0.1) + PREPARE_SECONDS_PER_JOB * jobs
    point = (upper + lower) / 2

    if total_audio <= 0.0:
        return Estimate(seconds=0.0, lower_seconds=0.0, upper_seconds=0.0, basis="done")

    basis = "duration"
    if observed is not None:
        basis = "duration+measured"

    return Estimate(
        seconds=point,
        lower_seconds=min(lower, upper),
        upper_seconds=max(lower, upper),
        basis=basis,
        warning=warning_for(duration_s),
    )


def warning_for(duration_s: float) -> str | None:
    """A long recording announces its cost instead of being rejected."""
    if duration_s < LONG_JOB_SECONDS:
        return None
    return (
        f"This is a long recording ({format_duration(duration_s)}). It will be "
        "accepted, but everything queued behind it waits. One Job runs at a time "
        "because a single transcription uses most of this machine's memory."
    )


# --------------------------------------------------------------------- helpers

def _audio_remaining(duration_s: float, pieces_total: int, pieces_done: int) -> float:
    """How much audio is genuinely left, never negative and never above the total."""
    if duration_s <= 0.0:
        return 0.0
    if pieces_total <= 0:
        return duration_s
    done = max(0, min(pieces_done, pieces_total))
    # The last Piece may be shorter than a full one, so clamp to the real total.
    return max(0.0, min(duration_s, (pieces_total - done) * PIECE_SECONDS))


def _observed_speedup(pieces_done: int, elapsed_s: float) -> float | None:
    """Realtime speedup implied by the Pieces finished so far, if that means anything."""
    if pieces_done < 1 or elapsed_s <= 0.0:
        return None
    audio_done = pieces_done * PIECE_SECONDS
    speedup = audio_done / elapsed_s
    # A wildly impossible figure is noise, not evidence.
    return speedup if 0.2 <= speedup <= 100.0 else None


def _jobs_ahead(total_audio_s: float) -> int:
    """At least one Job's preparation cost whenever there is audio to work through."""
    if total_audio_s <= 0.0:
        return 0
    return max(1, int(total_audio_s // 1800) + 1)


def _range_label(lower: float, upper: float) -> str:
    if upper < 1.0:
        return "under a second"
    if abs(upper - lower) < 1.0:
        return format_duration(upper)
    return f"{format_duration(lower)} to {format_duration(upper)}"