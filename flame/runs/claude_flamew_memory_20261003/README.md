# Flame-W memory readout (v0.3, 2026-10-03)

Flame-W's rules reach only a few words back. Before this release, the automaton used almost
nothing beyond the last 4 words: use_w(8) was about 0.0005 bits. This folder adds a memory
readout over the last 64 words. The automaton is unchanged. The whole thing is still integer
arithmetic: int8 tables with int64 sums.

| | |
|---|---|
| Automaton | `flame/runs/claude_flamew_b740_20261001/train/best.mica`, SHA `1f4d5503…`, 34,111,616 bytes ("B-740": checkpoint B continued to round 740 at `--round-lr 0.01`, no TinyStories) |
| Memory | `memory.npz`, SHA `b708f511…`, 673,650 bytes, no pickle; `memory.py` reads it |
| Total | 34.8 MB |

## How it works

At every step, each word that appeared in the last 64 tokens adds to the scores. Everything is
in the engine's 1/1024 logit units, before the softmax:

- **Recall.** Each earlier occurrence of word w adds `R[class(w)][lag bucket]` to w's own score.
  - Classes: names/unknown, punctuation, top-100, rank 100–2,000, rarer.
  - Lag buckets: 1, 2, 3, 4–6, 7–12, 13–24, 25–64.
  - The word just said is pushed down (−0.9 to −2.9). Content words seen 4–64 words ago are
    pushed up (up to +3.0).
- **Association.** The window is summed into a 16-channel memory vector:
  `M = Σ lam[lag bucket] · A[u]`, with lam = 127, 62, 52, 44, 34, 17, 3.
  - Every word w then gets `(Bm[w] · M) · MUL[w] >> 30`.
  - So words related to what was said earlier become more likely ("wallet … missing" →
    "worried"), not only the same words.
  - A and Bm are int8 rows for the 6,000 most frequent words and the hashed names.

`MemoryMicaWord` wraps the engine with the same `start / feed / fork / logp` interface. The
decoders, `run_tom.py`, `generate.py` and the website use it unchanged.

## How it was fitted

- **Automaton frozen.** `mem_pass.py` stores the 240 probe reads of every position.
  B-740's integer scores (`bias + co · f`) can then be recomputed exactly, max difference 0.
- **Training text.** 661 k word positions: 9,000 long mix-A training records (≥ 24 words) and
  14,000 no-TinyStories training records.
- **Method.** Adam, 3 epochs. `fit_mem.py` selects on held-out text only: 40 k positions of
  other long training records, plus 1,000 no-TinyStories validation records (500–1,499).
- **Rounding.** The fit is rounded to integers (`final_mem.py`). Integer and float bits agree
  to 0.0001.
- **Single ToM look.** Tiny Theory-of-Mind was scored once, after selection.
- **Rejected arm.** A long-records-only arm improved long text but made short chat worse
  (val500 5.36 vs 5.31). It was rejected for that, before any ToM run.

## Results (all exact integer engine)

| | B-740 | B-740 + memory |
|---|---:|---:|
| Context use: use_w(8), chat / everyday long600 | 0.0006 / 0.0004 | **0.0209 / 0.0407** |
| Context use: use_w(4) | 0.0068 / 0.0069 | 0.0312 / 0.0695 |
| Bits/word, val500 (no-TinyStories validation, 7,320 targets) | 5.309 | **5.268** |
| Bits/word, dev (chat + everyday dev2, long600; 27,632) | 5.677 | 5.577 |
| Bits/word, held-out long records (≥ 24 words) | 6.83 | 6.21 |
| Next-word top-1, chat / everyday dev (2,000 positions each) | 24.7% / 22.9% | 25.6% / 22.7% |
| **Tiny Theory-of-Mind (2,000 rows, chance 25%)** | 28.15% | **30.95%** |

- **The memory bottleneck is gone in the measured sense.** use_w(8) passes the project's
  0.010 gate on both sets, for the first time for Flame-W. Spark's longer-reach rules had
  reached 0.0016 / 0.0030.
- **ToM gain is significant.** Paired against B-740: +2.80 points, 95% interval [+1.00, +4.55];
  182 items gained and 126 lost; McNemar p = 0.0017.
- **v0.1 recall-only readout** (40 weights, 2026-10-03 morning): ToM 30.60%. The memory
  readout is +0.35 points over it, which is not significant (p = 0.70). Its gain is in bits and
  context use.
- **ToM leaderboard** (dataset README, 36 models): 30.95% is above Syn-2.6M (30.30), Photon-2.0-1M
  (30.05), GPT-S-1.4M (29.95), BananaMind-2.1-Pico-Preview (29.20), Quark-50m-v2 (29.10) and
  dillion-1.2M (28.30). The next model up is CMA-1M-Mini (31.40).
  - Our length normalization is per word token, theirs per subword.
- **Sentences:** blind, 50 prompts, w-sent-bos.5f for both. The previous site model, B
  `4d8cab66`, is compared with B-740 + memory:

  | | relevance | grammar | useful | at least partly useful |
  |---|---:|---:|---:|---:|
  | B (previous site model) | 0.98 | 1.42 | 0.60 | 42% |
  | B-740 + memory | 0.98 | 1.48 | 0.54 | 34% |

  - Paired useful −0.06 [−0.22, +0.12]: no significant difference either way at this sample size.
  - Files: `results/sentences_blind_v3.json`.

## Reproduce

```bash
python flame/runs/claude_flamew_memory_20261003/run_tom.py              # 30.95%
python flame/runs/claude_flamew_memory_20261003/run_tom.py --no-memory  # 28.15%
python flame/runs/claude_flamew_memory_20261003/generate.py "I was thinking about"
```

`mem_pass.py`, `fit_mem.py` and `final_mem.py` are the fitting scripts exactly as run. Their
paths point at the cloud workspace where they ran; change the paths at the top to rerun them.
