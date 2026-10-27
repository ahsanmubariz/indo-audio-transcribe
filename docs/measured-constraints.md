# Measured constraints on this machine

The app's design is shaped by measurements taken in this repo, not by assumptions.
Recording them here so a future reader can tell which numbers are load-bearing and
how to re-derive them.

Hardware as measured: 12 cores, 15 GB RAM, **no GPU**. Model
`PixelML/sea-asr-id` with its language model, CPU-only PyTorch 2.14.1.

## Transcription speed

13:15 (796 s) recording, `lm` mode, via `./transcribe.sh`:

| Threads | Encoder time | Speedup | Peak RSS |
| ------- | ------------ | ------- | -------- |
| 6 (script default) | 122.5 s | 6.5x | — |
| 8 | 102.9 s | 7.7x | 7.7 GB |
| 12 (all cores) | 96.2 s | 8.3x | — |

`greedy` and `lm` measure the same wall time (127.5 s vs 127.4 s) because the
acoustic encoder dominates the cost; `lm` output was byte-identical to the
one-shot script's transcript for the same recording, while greedy's was noticeably
worse — it dropped recognisable Indonesian words in favour of near-misses. There is
therefore no speed reason to prefer greedy.

**Peak RSS of 7.7 GB is the binding constraint.** It forces one Job at a time on a
15 GB machine (see ADR-0002).

## Batching

The library cuts audio into 30-second Pieces and batches them to a 600 s budget,
padding each Piece to the longest in its batch. Driving one Piece at a time was
**13% faster** (100.6 s vs 115.4 s on a 551 s subset) as well as giving progress
and cancellation. This is why the app owns the loop (ADR-0001).

## Re-deriving

```bash
./transcribe.sh -i <file> -o out.txt --mode lm -t 12
```

Peak RSS via `/usr/bin/time -f '%es %MkB'`. Thread scaling requires re-running the
per-Piece loop at each thread count; the one-shot script cannot express it.