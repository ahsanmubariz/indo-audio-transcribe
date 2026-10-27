"""The Job store: every Job's state, in memory, for the life of the Worker.

Nothing is persisted (ADR-0003). A Job outlives the HTTP request that created it,
which is why it is something you observe rather than a response you wait on.

Thread-safety is by one lock. The only writer of progress is the Worker thread;
HTTP request threads read snapshots, enqueue, cancel and download.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import AudioFacts
from .estimate import estimate_for
from .pipeline import Progress, Transcript

QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({FINISHED, FAILED, CANCELLED})


@dataclass
class Job:
    """One Job: what was uploaded, and everything that happened to it."""

    id: str
    original_name: str
    original_path: Path
    work_dir: Path
    output_format: str
    mode: str
    model: str
    facts: AudioFacts
    created_at: float = 0.0

    state: str = QUEUED
    pieces_done: int = 0
    pieces_total: int = 0
    partial_text: str = ""
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    expires_at: float | None = None
    cancel_requested: bool = False
    transcript: Transcript | None = None
    # Rendered download body, kept in memory so a second download costs nothing.
    rendered: str | None = None


@dataclass
class _Signal:
    """A per-Job mailbox: an append-only history of snapshots plus a condition.

    History is kept so a subscriber that starts late — a reloaded page, a slow
    client — still sees every state the Job passed through, rather than only
    the ones that happened while it was watching.
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
    closed: bool = False


class JobStore:
    """Holds Jobs, the Queue, and the signals their observers wait on."""

    def __init__(self, *, retention_seconds: float = 3600.0, clock=None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        # Submission order, kept separately from the Queue so the page can list
        # finished Jobs in the order they were handed over.
        self._order: list[str] = []
        self._queue: deque[str] = deque()
        self._signals: dict[str, _Signal] = {}
        self._retention_seconds = retention_seconds
        # Injectable so retention can be tested without waiting an hour.
        self._clock = clock or time.time

    # ------------------------------------------------------------- creation

    def create(
        self,
        *,
        original_name: str,
        original_path: Path,
        work_dir: Path,
        output_format: str,
        mode: str,
        model: str,
        facts: AudioFacts,
    ) -> Job:
        job = Job(
            created_at=self._clock(),
            id=uuid.uuid4().hex[:12],
            original_name=original_name,
            original_path=original_path,
            work_dir=work_dir,
            output_format=output_format,
            mode=mode,
            model=model,
            facts=facts,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._signals[job.id] = _Signal()
            self._order.append(job.id)
            self._queue.append(job.id)
            self._publish_locked(job)
        return job

    # ------------------------------------------------------------- reading

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            self._expire_locked(job)
            return self._snapshot_locked(job)

    def snapshots(self) -> list[dict[str, Any]]:
        """Every known Job's snapshot, oldest submission first."""
        with self._lock:
            out = []
            for job_id in self._order:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                self._expire_locked(job)
                out.append(self._snapshot_locked(job))
            return out

    def queue_position(self, job_id: str) -> int | None:
        """1 for the Job that runs next, 2 for the one behind it, None if not waiting."""
        with self._lock:
            try:
                return self._queue.index(job_id) + 1
            except ValueError:
                return None

    def running_job(self) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.state == RUNNING:
                    return job
            return None

    def history(self, job_id: str) -> tuple[list[dict[str, Any]], bool]:
        """Every snapshot so far, and whether the Job has stopped moving."""
        with self._lock:
            signal = self._signals.get(job_id)
            if signal is None:
                return [], True
            return list(signal.history), signal.closed

    def wait_for_change(self, job_id: str, timeout: float = 1.0) -> None:
        with self._lock:
            signal = self._signals.get(job_id)
        if signal is None:
            return
        with signal.condition:
            signal.condition.wait(timeout)

    # ---------------------------------------------------------- transitions

    def claim_next(self) -> Job | None:
        """Take the Job at the head of the Queue and mark it running."""
        with self._lock:
            while self._queue:
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.cancel_requested:
                    # Cancelled while waiting: it never runs, and never reports running.
                    self._finish_locked(job, CANCELLED)
                    continue
                job.state = RUNNING
                job.started_at = self._clock()
                self._publish_locked(job)
                return job
            return None

    def report_progress(self, job_id: str, progress: Progress) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.pieces_done = progress.pieces_done
            job.pieces_total = progress.pieces_total
            job.partial_text = progress.partial_text
            self._publish_locked(job)

    def finish(self, job_id: str, transcript: Transcript, rendered: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.transcript = transcript
            job.rendered = rendered
            job.partial_text = transcript.text
            self._finish_locked(job, FINISHED)

    def fail(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.error = message
            self._finish_locked(job, FAILED)

    def request_cancel(self, job_id: str) -> str | None:
        """Cancel a Job. Idempotent: a finished Job is left exactly as it was."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state in TERMINAL_STATES:
                return job.state
            if job.state == QUEUED:
                self._dequeue_locked(job_id)
                self._finish_locked(job, CANCELLED)
            else:
                job.cancel_requested = True
            return job.state

    def should_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def cancel_pending(self, job_id: str) -> None:
        """Called by the Worker when the pipeline reports it stopped on request."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in TERMINAL_STATES:
                return
            self._finish_locked(job, CANCELLED)

    # -------------------------------------------------------------- expiry

    def sweep_expired(self) -> int:
        """Drop Transcripts past their retention. Returns how many were released."""
        with self._lock:
            expired = 0
            for job in self._jobs.values():
                if self._expire_locked(job):
                    expired += 1
            return expired

    def _expire_locked(self, job: Job) -> bool:
        if job.state != FINISHED or job.rendered is None or job.expires_at is None:
            return False
        if self._clock() < job.expires_at:
            return False
        # Expiry means the words are gone, not merely unreachable. partial_text
        # holds the whole transcript by the time a Job finishes, so clearing only
        # `rendered` would leave every word readable through the snapshot while the
        # download route said otherwise.
        job.rendered = None
        job.transcript = None
        job.partial_text = ""
        self._publish_locked(job)
        return True

    # ------------------------------------------------------------- internals

    def _dequeue_locked(self, job_id: str) -> None:
        """Remove a Job from the Queue, wherever it is. A no-op if it is not waiting."""
        self._queue = deque(item for item in self._queue if item != job_id)

    def _finish_locked(self, job: Job, state: str) -> None:
        job.state = state
        job.finished_at = self._clock()
        if state == FINISHED:
            job.expires_at = job.finished_at + self._retention_seconds
        self._dequeue_locked(job.id)
        self._publish_locked(job)

    def _publish_locked(self, job: Job) -> None:
        signal = self._signals.get(job.id)
        if signal is None:
            return
        signal.history.append(self._snapshot_locked(job))
        with signal.condition:
            signal.closed = job.state in TERMINAL_STATES
            signal.condition.notify_all()

    def _snapshot_locked(self, job: Job) -> dict[str, Any]:
        expiry = None
        if job.expires_at is not None:
            expiry = _isoformat(job.expires_at)
        return {
            "id": job.id,
            "state": job.state,
            "original_name": job.original_name,
            "format": job.output_format,
            "mode": job.mode,
            "model": job.model,
            "pieces_completed": job.pieces_done,
            "pieces_total": job.pieces_total,
            "progress": round(job.pieces_done / job.pieces_total, 4) if job.pieces_total else 0.0,
            "partial_text": job.partial_text,
            "queue_position": self._queue_position_locked(job),
            "duration_seconds": job.facts.duration_s,
            "duration_label": job.facts.duration_label,
            "original_audio": job.facts.original_audio(),
            "error": job.error,
            "estimate": self._estimate_locked(job),
            "expired": job.state == FINISHED and job.rendered is None,
            "expires_at": expiry,
            "created_at": _isoformat(job.created_at),
            "started_at": _isoformat(job.started_at),
            "finished_at": _isoformat(job.finished_at),
        }

    def _estimate_locked(self, job: Job) -> dict[str, Any]:
        """How much longer, given what is left of this Job and what is ahead of it."""
        if job.state in TERMINAL_STATES:
            return estimate_for(duration_s=0.0).as_dict()

        elapsed = 0.0
        if job.state == RUNNING and job.started_at is not None:
            elapsed = max(0.0, self._clock() - job.started_at)

        return estimate_for(
            duration_s=job.facts.duration_s,
            pieces_total=job.pieces_total,
            pieces_done=job.pieces_done,
            elapsed_s=elapsed,
            work_ahead_s=self._work_ahead_locked(job),
        ).as_dict()

    def _work_ahead_locked(self, job: Job) -> float:
        """Audio duration of the waiting Jobs in front of this one, plus the running one."""
        ahead = 0.0
        if job.state == RUNNING:
            return 0.0
        for other_id in self._queue:
            if other_id == job.id:
                break
            other = self._jobs.get(other_id)
            if other is not None:
                ahead += other.facts.duration_s
        running = next((j for j in self._jobs.values() if j.state == RUNNING), None)
        if running is not None and job.state == QUEUED:
            # Whatever the running Job has left will finish first.
            remaining_pieces = max(0, running.pieces_total - running.pieces_done)
            if running.pieces_total:
                ahead += remaining_pieces * (running.facts.duration_s / running.pieces_total)
            else:
                ahead += running.facts.duration_s
        return ahead

    def work_ahead_seconds(self) -> float:
        """Audio duration of everything not yet done, for a Job that does not exist.

        `POST /api/inspect` describes a file before it is a Job, so it has no place
        in the Queue to measure from. What it can say is what is already waiting;
        that keeps its Estimate consistent with the one the Job will show.
        """
        with self._lock:
            total = 0.0
            for job in self._jobs.values():
                if job.state in TERMINAL_STATES:
                    continue
                if job.state == RUNNING and job.pieces_total:
                    remaining = max(0, job.pieces_total - job.pieces_done)
                    total += remaining * (job.facts.duration_s / job.pieces_total)
                else:
                    total += job.facts.duration_s
            return total

    def _queue_position_locked(self, job: Job) -> int | None:
        if job.state != QUEUED:
            return None
        try:
            return self._queue.index(job.id) + 1
        except ValueError:
            return None


def _isoformat(when: float | None) -> str | None:
    if when is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))