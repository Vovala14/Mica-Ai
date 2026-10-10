# MICA Flame-W 0.3.5 — weight-averaged automaton

0.3.5 keeps everything from 0.3.4 — the same 34 MB integer cellular automaton geometry, the same
language-memory banks, answer mode, cellular belief memory, tokenizer and sentence decoders — and
changes one file: the automaton itself. Its learned integer rule values and readout were
**averaged over 40 training rounds** (stochastic weight averaging) instead of taking a single
round's values. Same size, same speed, better predictions.

MICA is a learned integer cellular automaton, not a transformer: no attention, no neural network
layers, integer state updates.

## Results (exact integer engine, paired against 0.3.4 item by item)

| Test | 0.3.4 | **0.3.5** | Paired difference |
|---|---:|---:|---|
| Short text, bits per word (500 held-out records) | 5.1894 | **5.1622** | −0.027 [−0.039, −0.015] |
| Long text, bits per word (500 held-out long records) | 8.0862 | **8.0738** | −0.012 [−0.016, −0.009] |
| Tiny Theory of Mind, per word (2,000) | 638 (31.90%) | **645 (32.25%)** | +54 / −47, p = 0.55 |
| Tiny Theory of Mind, per character | 635 (31.75%) | **637 (31.85%)** | +58 / −56, p = 0.93 |
| HellaSwag (acc_norm, 10,042) | 27.30 | **27.70** | +188 / −147, p = 0.03 |
| ARC-Easy (acc_norm, 2,376) | 25.80 | **26.18** | +54 / −45, p = 0.42 |
| ARC-Challenge (acc_norm, 1,172) | 19.80 | **20.73** | +30 / −19, p = 0.15 |
| PIQA (acc_norm, 1,838) | **51.41** | 51.31 | +48 / −50, p = 0.92 |
| ArithMark-3 (per token, 1,000) | 26.90 | **27.00** | +37 / −36, p = 1.0 |

Lower bits are better. Both text-prediction gains are significant (95% bootstrap intervals exclude
zero) and HellaSwag improves significantly; the other tests are within noise. Tiny ToM stays at
about 32%: the 33% goal is not reached. Nothing was tuned on any of these tests; the averaging used
only training text.

## What did not change
Memories, answer mode, belief memory and decoders are byte-identical to 0.3.4, so every
0.3.4 feature works the same way. Sentence generation uses the same decoder; a side-by-side check of
8 prompts showed similar or slightly better sentences (not a blind evaluation).

## Identity

- 0.3.5 automaton `best.mica`: `eca9faee6f44e3b2e4c10399ec99d202fa941a10ef78e64fb2094b0de70a4616` (34,111,616 bytes, same geometry as B-740)
- Source automaton B-740: `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`
- Language memories (unchanged): short `5fda5a6c…`, fitted short `5564ba55…`, long `f2abb791…` in
  [`../codex_flamew_033_20261009`](../codex_flamew_033_20261009); answer mode and belief memory as in
  [`../codex_flamew_034_20261010`](../codex_flamew_034_20261010).

## How 0.3.5 was made

Fit-mode training refits the readout and the rule immediates every round on a fresh sample, so
consecutive rounds scatter around the best values. B-740 was continued for 40 rounds with nothing
frozen (40,000 training records per round, round-lr 0.01, its own training data), and the readout and
rule immediates were averaged over those rounds; the average is rounded back to the integer file
format. Codes, wiring and file size are unchanged. Averaging only helps while rounds still move: a
readout-only averaging run started from B-740 changed nothing (byte-identical output), so the rules
had to be free.

## Evidence files

- `results/bits_vs_034.json` — exact bits per word symbol + EOS, 500 short and 500 long held-out records, paired bootstrap intervals
- `results/benchmarks_vs_034.json` — Tiny ToM and the five leaderboard tasks, paired McNemar against 0.3.4 item by item
- `results/five_task_summary.json` — all configurations and normalizations of the five tasks (frozen 0.3.4 protocol)
- `results/tom_results_035.json` — every Tiny ToM item: ending log-likelihoods, predictions, belief-memory route
- `results/sentences_side_by_side.json` — 8 fixed prompts, 0.3.4 and 0.3.5 (not a blind evaluation)

The website backend loads this folder. The standalone download is attached to the
[GitHub release](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.5) and mirrored at
[vynly/mica-flame-w-0.3.5](https://huggingface.co/vynly/mica-flame-w-0.3.5).

Personal and research use only — commercial use is not permitted (see the repository LICENSE).
