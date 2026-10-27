# No persistence: download-only transcripts

Transcripts are never stored. A finished Transcript lives in the Worker's memory
and is handed to the browser on download; the Job is then forgotten. There is no
database, so there is nothing to reopen later.

This was chosen deliberately over SQLite persistence, which is the obvious
default. The trade-off is stated plainly because it is a real cost: **closing the
page before downloading loses the transcript**, and re-running the Job costs
another ~2.5 minutes of CPU. The UI must make the download hard to miss rather
than assume the result is retrievable.

## Considered Options

- **SQLite persistence** — reversible, and the recommended option, but rejected
  in favour of keeping the app stateless.
- **Write transcripts to a directory on disk** — the same loss-aversion problem in
  a different shape: without an index there is nothing to browse, so it is storage
  with none of the benefit.

## Consequences

Job state lives in memory, so restarting the Worker discards the Queue and any
undownloaded Transcripts. Since this is a single-user app on localhost, that is
acceptable; it would not be for a shared instance.

If persistence is added later, the Transcript becomes a stored artefact and this
ADR should be superseded rather than quietly contradicted.