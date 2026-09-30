# MICA Flame-W (word model)

Flame-W is the word-level MICA model: the same integer cellular automaton, with
one symbol per word (16,384-symbol vocabulary, `r1/data/word/vocab.json`).
It is not the byte-level Flame. The byte/letter-level model is [MICA Ember](../ember/).

## Official model: full40

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
