# Overnight Flame-W run 2026-10-01: results

Ran on the owner's PC (RX 9070 XT, VRAM capped at 11.9 GiB) from 23:50 to 05:50. Two arms each
continued full40's saved state for about 2 h 14 min (about 240 rule rounds at 33 s per round)
on the word corpus **without TinyStories**:

- 4,246,235 training records kept
- 261,656 TinyStories records dropped (6% of records, ~15% of text)

## Automatic scores

All scores use the exact integer word engine:
- next-word accuracy on the first 1,000 fixed positions of chat dev1000 and everyday
  dev_fresh1000, the same positions for every model;
- bits per word on 500 no-TinyStories validation records.

| Model | SHA-256 | Bits/word ↓ | Next-word top-1 chat / everyday | top-10 chat / everyday |
|---|---|---|---|---|
| full40 (baseline) | `ef969c0e…` | 6.2154 | 23.1% / 20.0% | 47.8% / 45.1% |
| 2026-09-30 lr 0.03, *with* TinyStories | `e84acfc4…` | 5.4408 | 27.4% / 23.8% | 56.5% / 51.0% |
| A: lr 0.03, no TinyStories | `cae9766e…` | 5.3968 | 25.9% / 24.1% | 57.2% / 51.4% |
| **B: lr 0.01, no TinyStories** | `4d8cab66…` | **5.3514** | **27.5%** / 23.8% | **58.2% / 51.5%** |

Most of the gain comes from continuing the rule fit at a working learning rate. The default
`--round-lr 0.3` diverges for words. The 2026-09-30 run kept TinyStories and still improved
almost as much. Removing TinyStories adds a little bits gain on no-TinyStories text, and
removes the story phrases (below).

B's trainer validation was still falling when its time ran out (5.786 → best 5.179 over
240 rounds), so B has not converged.

## Sentences

**Story phrases** ("named …", "little girl/boy", "Lily/Tim/Sam", "went to the park",
"his mom"), counted over 50 prompts × 2 decoders:

| Model | Count |
|---|---|
| full40 | 4–5 / 50 |
| 2026-09-30 run | 0–2 / 50 |
| A and B | **0 / 50** |

**Blind usefulness** on the 20 holdout prompts. One judge scored all 80 outputs shuffled,
without knowing the system; the share is continuations at least partly useful:

| Model | w-sent-mmi.3 (current) | w-sent-bos.5f (new) |
|---|---|---|
| full40 | 15% | 20% |
| B | 15% | **30%** |

Every judged item is in `results/holdout_judge_items.json`. With 20 prompts, each prompt is
5 points, so this is a direction, not a precise number.

**New failure mode.** The story clichés are replaced by chat-support clichés from the
conversation data: "I don't know if I can ever trust you again", "let me know if you need any
help with anything" and "help you through this tough time". The model falls back on the most
common phrases of whatever data it has once it has lost the prompt, which takes about 4 words.
The fix for that is context (RESEARCH.md §5), not data.

## Guard note

The RAM watchdog logged "peak RAM 0.0 GB". On Windows, a venv's python.exe is a launcher, and
the watchdog measured only the launcher, not its child. `night.py` now measures the whole
process tree; this was tested with a launcher-plus-child process (0.010 GB for the parent alone,
0.60 GB for the tree). The trainer's own free-RAM guard (4 GB) was active all night. Free RAM at
the start was 8.4 GB.

## Next

- Promote B (with `w-sent-bos.5f`) as Flame-W after its `best.mica` is uploaded.
- Continue B (`--arms B_lr010 --skip-train` is evaluation only; `--arms B_lr010` continues
  training), since it had not converged.
- Then the topic-register context experiment.
