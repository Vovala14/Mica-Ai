# MICA Flame-W (word model)

Flame-W is the word-level MICA model: the same integer cellular automaton, with
one symbol per word (16,384-symbol vocabulary, `r1/data/word/vocab.json`).
It is not the byte-level Flame. The byte/letter-level model is [MICA Ember](../ember/).

## Current release: 0.3.5 (2026-10-11)

Weight-averaged automaton. The B-740 geometry with its learned integer rule
immediates and readout averaged over 40 training rounds (stochastic weight
averaging); everything else is byte-identical to 0.3.4. Exact integer bits per
word: short text 5.189 → 5.162, long text 8.086 → 8.074 (paired intervals exclude
zero). Tiny ToM 638 → 645/2,000; HellaSwag 27.30 → 27.70; ARC-Easy 25.80 → 26.18;
ARC-Challenge 19.80 → 20.73; PIQA 51.41 → 51.31; ArithMark-3 26.9 → 27.0.
[Release notes and evidence](runs/claude_flamew_035_20261011/README.md) ·
[Download 0.3.5](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.5) ·
[Hugging Face](https://huggingface.co/vynly/mica-flame-w-0.3.5)

## Previous release: 0.3.4 (2026-10-10)

An experimental cellular belief-memory extension: original B-740 plus unchanged
0.3.3 language memories and sentence decoder, with a separate 32×256-site
radius-one integer COPY/SKIP memory and external English binding for four-choice
QA. This is not a rewritten native B-740 checkpoint. Sentence generation and
loss are unchanged.

Controlled supported development: previous private prototype 899→965/1,000,
66 gains/zero losses, paired 95% CI [+5.1,+8.2] percentage points. Public Tiny
ToM: 638/2,000 (31.90%) word and 635/2,000 (31.75%) character. Public answers
are unchanged versus that private prototype; the +2 word answers against
published 0.3.3 are uncertain (95% CI [−0.10,+0.30] points). No new sentence
or leaderboard quality claim. The controlled grammar is not unrestricted English.

[Download 0.3.4](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.4) ·
[Package and full protocol](runs/codex_flamew_034_20261010/README.md) ·
[Live belief-memory demo](https://mica-ai-ten.vercel.app/#belief).

## Earlier release: 0.3.3 (2026-10-09)

The [public 0.3.3 package](https://huggingface.co/vynly/mica-flame-w-0.3.3) keeps the B-740 learned integer-rule automaton (SHA-256 `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`) and the 0.3.2 decoder. For the first 64 fed word symbols, its memory bonus is `floor((3 * original_short + fitted_short + 2) / 4)`; after 64 symbols it uses the unchanged long bank. The three packed memory files and checkpoint total 36,132,566 bytes. The fitted bank adds 673,650 bytes over 0.3.2. Implementation, exact hashes and frozen gates are in [`runs/codex_flamew_033_20261009/EVALUATION.md`](runs/codex_flamew_033_20261009/EVALUATION.md).

| Exact integer score | 0.3.2 | 0.3.3 | Paired change, 95% interval |
|---|---:|---:|---:|
| Clean short500 bits/word symbol + EOS | 5.208428 | 5.189430 | −0.018998 [−0.021376,−0.016611] |
| Clean long500 bits/word symbol + EOS | 8.082371 | 8.086188 | +0.003818 [+0.002786,+0.004814] |
| Public Tiny ToM, word-normalized answer mode | 632/2,000 (31.60%) | 636/2,000 (31.80%) | +0.20 percentage points [−0.25,+0.65] |

The short-record loss gain is measured; the long-record loss regression is also measured. The ToM interval includes zero. A blinded single-rater check of 40 previously unused sentence prompts found 1 win, 1 loss and 38 ties, so there is no demonstrated sentence-quality gain. The five-task leaderboard was not rerun. No sealed test was used.

## Historical official model: 0.3.1 (2026-10-04)

- **Automaton:** `runs/claude_flamew_b740_20261001/train/best.mica`, SHA-256
  `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`, 34,111,616 bytes.
  This is checkpoint B continued to round 740 at `--round-lr 0.01`.
- **Memory readout:** `runs/claude_flamew_031_20261004/memory.npz`, 674 KB.
  It is an integer recall-and-association readout over the last 64 words. 0.3.1 refits v0.3's
  readout on 4.5 million word positions, 7 times more.
- **Answer mode** (for multiple-choice questions only): `runs/claude_flamew_031_20261004/answer_mode.npz`.
  It adds +3.25 nats to content words already in the context.

| | B-740 | + v0.3 memory | 0.3.1 |
|---|---:|---:|---:|
| Bits/word, no-TinyStories validation (val500) | 5.309 | 5.268 | **5.208** |
| Context use use_w(8), chat / everyday | 0.0006 / 0.0004 | 0.0209 / 0.0407 | 0.0183 / 0.0483 |
| Tiny Theory-of-Mind, per word token / per character | 28.15% / 26.15% | 30.95% / 29.90% | **31.25% / 31.45%** |

A five-prompt byte replay of the playground sentence decoder (memory on, answer mode off) is in [`replay/`](replay/): `python flame/replay/replay_flamew_031.py --check`.

The refit improves text prediction. On Tiny Theory-of-Mind, 0.3.1 is not significantly different
from v0.3 with answer mode (32.20% / 31.40%; p = 0.17), and on the held-out half the two are
within 0.2 points. Details and every file are in [`runs/claude_flamew_031_20261004/`](runs/claude_flamew_031_20261004/).

## Previous official model: B-740 + memory (v0.3, 2026-10-03)

The same automaton with the first memory readout, `runs/claude_flamew_memory_20261003/memory.npz`.

The automaton's rules see about 4 words. The memory gives every word seen in the last 64 tokens,
and the words associated with them, an integer bonus. Everything stays integer arithmetic.

| | B-740 | B-740 + memory |
|---|---:|---:|
| Context use use_w(8), chat / everyday | 0.0006 / 0.0004 | **0.0209 / 0.0407** |
| Bits/word, no-TinyStories validation (val500) | 5.309 | **5.268** |
| Tiny Theory-of-Mind, 2,000 rows (chance 25%) | 28.15% | **30.95%** |

30.95% is above 6 of the 36 models on the benchmark's leaderboard, up to Syn-2.6M (30.30%). In a
blind sentence check, sentences were no different from B's: useful −0.06 [−0.22, +0.12]. Details
and every file are in [`runs/claude_flamew_memory_20261003/`](runs/claude_flamew_memory_20261003/).

## Earlier official model: B (2026-10-01)

`runs/claude_flamew_b_20261001/train/best.mica`
SHA-256 `4d8cab66b9b78bad75243fcbbfce7a4c3aa42d2713c17846f8030a913bf7213d`, 34,111,616 bytes.

B continues full40 for 240 rule rounds at `--round-lr 0.01` on the word corpus without
TinyStories. On the same evaluation sets as full40:

| | full40 | B |
|---|---:|---:|
| Next-word top-1, chat / everyday | 23.1% / 20.0% | **27.5% / 23.8%** |
| Bits/word, no-TinyStories validation | 6.22 | **5.35** |
| Story phrases in 50 sentences | 4–5 | **0** |
| At least partly useful, blind holdout (20 prompts) | 15% | **30%** (with `w-sent-bos.5f`) |

The site used B until v0.3, with `w-sent-bos.5f` for sentences and `w6-mmi.5` for next words. Full details
are in [`r1/runs/claude_flamew_night_20261001/REPORT.md`](../r1/runs/claude_flamew_night_20261001/REPORT.md).

## Earlier official model: full40

`runs/codex_flame_word_full40_20260928/train/best.mica`
SHA-256 `ef969c0e96173fc11ecc3b5cf0d1d28d04e21fe846a0752297d5549e90f7e6b7`, 34,111,616 bytes.

40 native integer-rule rounds on the mix-A word corpus. Scores come from the
exact integer word engine (full details in `REPORT.md`):

| Set | Next-word top-1 | Completion top-1 | Bits/token |
|---|---:|---:|---:|
| Chat dev1000 | 20.70% | 43.64% | 6.9287 |
| Everyday dev_fresh1000 | 18.45% | 40.84% | 7.0222 |
| Chat clean val1000 | | | 6.8414 |
| Everyday clean val1000 | | | 6.9972 |

Every later experiment (PPMI code start, freeze, tape readout, day-chain
continuation) was measured against full40 and not promoted.

Sample output (decoder `w-sent-mmi.3`, reproduced from this repo):

- "I was thinking about" → "what you said."
- "The best part of the weekend was" → "an accident and I just wanted to let you know that I need to make some changes."

This is a research model. Its sentences are not reliably sensible yet.

## Other checkpoints

`runs/claude_flame_word_20260930/a_full40_lr030/`: an experiment from a
learning-rate screen on 2026-09-30. It was not promoted. Its trainer validation
used 128 records, so it can't be compared with the exact-engine scores above.

Each run's `package.json` lets `word_decode.py`/`word_eval.py` load it with
`--model mica:<train folder>`.
