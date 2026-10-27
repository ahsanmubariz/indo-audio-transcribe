# One Worker, one Job at a time

The app runs a single Worker that holds the model in memory and processes the
Queue strictly serially. This is a memory constraint, not a simplification:
transcribing a 13-minute recording with the language model peaks at **7.7 GB**
RSS on a machine with 15 GB total, so a second concurrent Job would swap or be
OOM-killed. Serialising also makes progress reporting honest, since there is only
one Job whose progress is moving.

## Consequences

Uploads must be queued rather than rejected or run concurrently, so the UI needs a
queue position. A Job that is waiting can be cancelled instantly; a Job that is
running can only be cancelled at a Piece boundary (at most 30 seconds of lag).

This decision is tied to the current hardware and model size. On a machine with
enough memory, a Worker pool would be a reasonable change — but only alongside a
re-measurement of peak RSS, not by assuming it scales.