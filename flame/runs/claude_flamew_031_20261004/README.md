# MICA Flame-W 0.3.1 (2026-10-04)

0.3.1 keeps the B-740 automaton and changes two things on top of it:

1. **The memory readout is refitted on 7 times more text.** It is the same readout as v0.3
   (recall plus a 16-channel association over the last 64 words), warm-started from v0.3.
2. **Answer mode** (optional) is added for multiple-choice question answering. It adds a fixed
   bonus to content words already seen in the context.

| | |
|---|---|
| Automaton | `flame/runs/claude_flamew_b740_20261001/train/best.mica`, SHA `1f4d5503…`, unchanged |
| Memory | `memory.npz`, SHA `5fda5a6c…`, 673,650 bytes, no pickle; read by `memory.py` |
| Answer mode | `answer_mode.npz`: a mask of 15,516 content words and a gain of 3,328 units (3.25 nats) |

`memory.py` replaces v0.3's. It reads the same tables and takes an optional `answer_mode` file.
The website uses the memory without answer mode.

## Results (exact integer engine)

| | B-740 | + v0.3 memory | + 0.3.1 memory |
|---|---:|---:|---:|
| Bits/word, val500 (no-TinyStories validation, 7,320 targets) | 5.309 | 5.268 | **5.208** |
| Bits/word, dev (chat + everyday dev2, long600; 27,632) | 5.677 | 5.577 | **5.504** |
| Bits/word, held-out long records (≥ 24 words, 40,002), float fit | 6.830 | 6.210 | **6.076** |
| Bits/word, no-TinyStories validation 500–1,499 (14,779), float fit | 5.430 | 5.379 | **5.321** |
| Context use use_w(8), chat / everyday long600 | 0.0006 / 0.0004 | 0.0209 / 0.0407 | 0.0183 / 0.0483 |
| Context use use_w(4), chat / everyday long600 | 0.0068 / 0.0069 | 0.0312 / 0.0695 | 0.0289 / 0.0800 |

val500 and dev were not used for any choice. The two float-fit rows are the selection sets.

### Tiny Theory-of-Mind (AxiomicLabs, 2,000 rows, chance 25%)

| Configuration | per word token: full | held-out half | per character: full | held-out half |
|---|---:|---:|---:|---:|
| B-740 alone | 28.15% | 27.30% | 26.15% | 26.50% |
| + v0.3 memory | 30.95% | 29.70% | 29.90% | 30.00% |
| + v0.3 memory + answer mode (2.5) | 32.20% | 31.70% | 31.40% | 32.30% |
| + 0.3.1 memory | 31.15% | 30.50% | 30.15% | 29.80% |
| **+ 0.3.1 memory + answer mode (3.25) = 0.3.1** | **31.25%** | **31.80%** | **31.45%** | **32.50%** |

- **ToM did not move.** 0.3.1 vs v0.3 + answer mode, per word: +77 / −96 items, McNemar
  p = 0.17. Per character: +85 / −84. The 95% interval on each number is about ±2 points.
- **Against the automaton alone:** +256 / −194, p = 0.004.
- **Leaderboard (36 models):** 0.3.1 is above 6 of them per word token (31.25%) and 7 per
  character (31.45%). v0.3 + answer mode was above 9 per word on the full set, but on the
  held-out half the two are equal.
- **Topics:** strongest are recursive belief 50%, participant role and common ground 46%,
  indirect request 44%. Weakest are bluffing and communication failure 16%, diverse beliefs and
  white lie 18%.
- Held-out half: the odd rows of each topic (`tom_split` in v0.3's answer-mode run). Nothing was
  tuned on them.

## How it was fitted

- **Features.** The automaton is frozen. `mem_pass.py` (in the v0.3 folder) stored the 240 probe
  reads of 3.85 million new word positions: 44,000 more long mix-A training records (shards 31–34,
  ≥ 24 words) and 100,000 fresh no-TinyStories training records. This took 2.6 h on 2 CPU cores.
- **Training text.** 4,506,623 positions: these plus v0.3's 661,000. 75 records identical to
  held-out ones were dropped.
- **Fit.** `fit_mem2.py`: windows are built per batch. Adam at lr 0.002, L2 1e-5, batch 1,024.
  It ran 13,483 steps (about 3 epochs) in 1.3 h.
- **Selection.** The rule is the same as v0.3: the mean bits gain on held-out long records and on
  no-TinyStories validation 500–1,499. The best is step 12,000, at −0.432 (v0.3 −0.336).
- **Integers.** `quant031.py` rounds the fit with v0.3's `final_mem.quantize`, with the association
  mask applied. Run on the v0.3 fit, it reproduces v0.3's `memory.npz` byte for byte.
- **Answer-mode gain.** It was re-picked for the new memory on the same 2,000 synthetic practice
  items, with the same rule (highest accuracy, ties to the smaller gain). The pick is 3.25 nats
  (v0.3: 2.5). The curve is flat from 1.75 to 4.0 (33.95–34.15%). Tiny ToM was scored once,
  after both choices.

## Reproduce

```bash
python flame/runs/claude_flamew_031_20261004/run_tom.py                     # 0.3.1, prints both normalizations
python flame/runs/claude_flamew_031_20261004/run_tom.py --config memory     # without answer mode
python flame/runs/claude_flamew_031_20261004/generate.py "I was thinking about"
```

`results/`: per-item ToM scores for the automaton, + memory and 0.3.1 (`tom_0.3.1_rows.json`), the summary,
bits, context use, the gain pick, the fit log and 8 website-style sentences from both memories
(6 of the 8 are identical).
