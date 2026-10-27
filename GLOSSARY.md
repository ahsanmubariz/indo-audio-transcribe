# Transcription

A local web app for turning Indonesian speech recordings into text, built on the
same `sea_asr` pipeline as `transcribe.sh`. It exists because transcription of a
real recording takes minutes, and a browser needs something to show during those
minutes.

## Language

### Work

**Job**:
One transcription request: a single uploaded audio file, watched from upload to
download. A Job outlives the HTTP request that created it, which is why it is a
thing you observe rather than a response you wait on.
_Avoid_: Task, request, transcription, run

**Queue**:
The ordered list of Jobs that have not yet started. Jobs wait here because only
one can run at a time.
_Avoid_: Backlog, buffer, pool

**Worker**:
The single long-lived process that holds the loaded model and runs Jobs one after
another. "Single" is a constraint, not a simplification.
_Avoid_: Runner, executor, engine

**Piece**:
A slice of the recording, at most 30 seconds, cut on the quietest moment near
each 30-second mark. The unit of progress reporting and of cancellation.
_Avoid_: Chunk, segment, block

### Output

**Transcript**:
The finished text produced by a Job, in the output format the Job was started
with. It exists only in the Worker's memory for 60 minutes after it finishes, or
until it is downloaded.
_Avoid_: Result, output, file, dump

**Word timings**:
The start and end time of every word, produced only for formats that need them.
Asking for them changes how the audio is decoded, so they are decided when the
Job starts, not when its Transcript is downloaded.
_Avoid_: Alignment, timestamps, subtitles

**Mode**:
Which decoder reads the acoustic pass. Only the language-model Mode is offered;
the others exist for comparison and debugging.
_Avoid_: Model, quality, accuracy, engine

**Original audio**:
The file exactly as uploaded, kept unchanged for as long as its Job is known. It
is *not* the optimised 16 kHz mono audio the Worker actually transcribes.
_Avoid_: Source, input, media

**Estimate**:
A prediction of how long a Job will wait and run, derived from the length of its
Original audio and current Queue depth. It is a prediction, never a deadline.
_Avoid_: ETA, progress, deadline