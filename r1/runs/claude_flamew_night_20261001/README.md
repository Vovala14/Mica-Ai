# Overnight Flame-W run (2026-10-01)

**Question:** does removing TinyStories from the training mix, and continuing
full40 at a lower rule learning rate, make Flame-W's sentences more useful?
Every output so far drifts into story phrases ("a little girl named Lily went
to the park with his mom"), and TinyStories is 15% of the mix.

## Run it (PowerShell, in the `mica` folder)

```powershell
cd C:\Users\<you>\PycharmProjects\mica
& ..\.venv-rocm\Scripts\python.exe r1\runs\claude_flamew_night_20261001\night.py --hours 8
```

- One GPU job at a time. Make sure no other agent's training is running, and
  mark the GPU as taken in `COORDINATION.md`.
- **Stop at any time:** create an empty file `PycharmProjects\STOP_MICA`. The state is saved,
  and running the same command again continues from there.
- **Evaluation only** (for example after a stop): add `--skip-train`.

## What it does

1. **Corpus.** Rebuilds the word corpus from `r1/data/mix/v02a` without TinyStories
   (source label 4) into `r1/data/word/v02a_nostories/`. It first proves that the
   rebuild matches `r1/data/word/v02a/train.jsonl` on 5,000 records.
2. **Training.** Two arms continue full40's `resume.pt`, with the best-so-far reset and
   128 validation records:
   - **A:** `--round-lr 0.03`
   - **B:** `--round-lr 0.01`

   The time before the evaluation reserve is split evenly between them.
3. **Evaluation** (CPU) of full40, the 2026-09-30 lr 0.03 run, A and B:
   - next-word top-1/3/10 on chat dev1000 and everyday dev_fresh1000, using the same first
     1,000 positions for every model;
   - bits/word on 500 no-TinyStories validation records;
   - sentences for the 20 holdout and 30 dev prompts, with `w-sent-mmi.3`
     (current) and `w-sent-bos.5f` (`decode_bos.py`, experimental).
4. **Report.** `night_report.zip` in this folder contains logs, progress, scores and
   sentences, and no model files. Send it to Claude, who judges the sentences blind.

## Limits

| Guard | Setting |
|---|---|
| VRAM hard cap | `--gpu-mem-fraction 0.75` (11.9 GiB of 15.9) |
| Training process RAM | `--ram-cap-gb 12`: clean stop above it |
| Free system RAM | `--min-free-ram-gb 4`: clean stop below it |
| CPU | below-normal priority, at most 8 threads |
| Games | training pauses while another program's GPU use jumps |

## Tested in the cloud (CPU, small data)

- The built `train_soft.py` command runs a real fit round.
- Resuming with the best-so-far reset writes a new `best.mica`.
- The deadline stops training cleanly and removes `STOP_MICA`.
- The corpus rebuild is exact, and a mismatch aborts before any training.
- The evaluation writes scores and sentences for both decoders.

Not tested: the GPU itself and the real 4.5 M-record corpus.
