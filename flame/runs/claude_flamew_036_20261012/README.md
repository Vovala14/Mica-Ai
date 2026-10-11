# MICA Flame-W 0.3.6 — previous-word bank and phrase copy

0.3.6 keeps 0.3.5's integer cellular automaton, its three language-memory banks, answer mode, cellular
belief memory, tokenizer and sentence decoders, and adds two small integer readouts of the recent words:

1. **Previous-word bank** (`bank.npz`, 0.69 MB). The previous word gives every word an integer bonus
   through 16 int8 values per word, built like the released memory.
2. **Phrase copy** (`ind.npz`, 210 integers). If the last word appeared earlier in the text (up to 256
   words back), the word that followed it there gets an integer bonus that depends on both words' classes
   and the distance, with an extra bonus when the two previous words match.

MICA is a learned integer cellular automaton, not a transformer: no attention, no neural network
layers, integer state updates. `context_banks.py` holds both readouts and the 0.3.6 model class.

## Results (exact integer engine, paired against 0.3.5 item by item)

| Test | 0.3.5 | **0.3.6** | Paired difference |
|---|---:|---:|---|
| Tiny Theory of Mind, per word token (2,000) | 645 (32.25%) | **664 (33.20%)** | +97 / −78, p = 0.17 |
| Tiny Theory of Mind, per character | 637 (31.85%) | **654 (32.70%)** | +94 / −77, p = 0.22 |
| Long text, bits per word (500 held-out long records) | 8.0738 | **7.7236** | −0.350 [−0.370, −0.331] |
| Short text, bits per word (500 held-out records) | 5.1622 | **5.1207** | −0.042 [−0.051, −0.031] |
| Short text, 1,000 more held-out records | 5.2809 | **5.2416** | −0.039 [−0.048, −0.031] |
| HellaSwag (acc_norm, 10,042) | **27.70** | 27.64 | +307 / −313, p = 0.84 |
| ARC-Easy (acc_norm, 2,376) | 26.18 | **27.27** | +64 / −38, p = 0.013 |
| ARC-Challenge (acc_norm, 1,172) | **20.73** | 20.22 | +22 / −28, p = 0.48 |
| PIQA (acc_norm, 1,838) | 51.31 | **52.18** | +58 / −42, p = 0.13 |
| ArithMark-3 (per token, 1,000) | 27.00 | **27.10** | +19 / −18, p = 1.0 |

Lower bits are better. The text gains are large and significant and ARC-Easy improves significantly.
Tiny ToM reaches 33.20%, above the project's 33% goal, but the gain over 0.3.5 is within noise. No task
dropped significantly.

## How 0.3.6 was made

Both readouts were trained on 0.3.5's automaton features of short and long training text (half of every
batch each), with early stopping on held-out selection text, then exported to integers; the automaton,
memories, answer mode and belief memory were not changed. Each step had a pass mark written before it ran
(`results/PASS_RULE_bank.txt`, `results/PASS_RULE_phrase_copy.txt`). The bank alone passed its text rule
but scored 642 on Tiny ToM, below 0.3.5's 645. The phrase-copy table was then trained on text only, once,
with the pre-declared requirement that Tiny ToM per word token beat 645 and no text measure get worse.
No benchmark items or labels were used for training, and nothing was selected on Tiny ToM. Every number was
re-checked independently (`results/verify_*.txt`, 0 mismatches).

## Identity

- Automaton: `../claude_flamew_035_20261011/best.mica`, unchanged from 0.3.5,
  `eca9faee6f44e3b2e4c10399ec99d202fa941a10ef78e64fb2094b0de70a4616`
- Previous-word bank `bank.npz`: `8f9d3655ccafb7091cf929b473afe1fb338468eca327372d986c8c1f36c95585`
- Phrase-copy table `ind.npz`: `a72434c715d1ad57ada2a74e9658b61960f978c1b09ed025fa6c6a0d31c2bb7b`
- Language memories (unchanged): short `5fda5a6c…`, fitted short `5564ba55…`, long `f2abb791…` in
  [`../codex_flamew_033_20261009`](../codex_flamew_033_20261009); answer mode and belief memory as in 0.3.5.

## Evidence files

- `results/induct_report.json`, `results/prevword_report.json` — every paired comparison and the verdicts
- `results/PASS_RULE_bank.txt`, `results/PASS_RULE_phrase_copy.txt` — the pass marks, written before each run
- `results/tom_rows_036.json` — every Tiny ToM item: ending log-likelihoods and predictions
- `results/five_task_summary.json` — the five leaderboard tasks (frozen 0.3.4 protocol)
- `results/sentences_side_by_side.json` — 8 fixed prompts, 0.3.5 and 0.3.6 (not a blind evaluation)
- `results/fit_bank.json`, `results/fit_phrase_copy.json`, `results/verify_bank.txt`, `results/verify_phrase_copy.txt`

The website backend loads this folder. The standalone download is attached to the
[GitHub release](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.6) and mirrored at
[vynly/mica-flame-w-0.3.6](https://huggingface.co/vynly/mica-flame-w-0.3.6).

Personal and research use only — commercial use is not permitted (see the repository LICENSE).
