# MICA Flame-W (word model)

Flame-W is the word-level MICA model: the same cellular automaton, with one
symbol per word (16,384-symbol vocabulary). It is not the byte-level Flame;
the byte/letter-level model is [MICA Ember](../ember/).

## Checkpoints

The official Flame-W is **full40**,
`r1/runs/codex_flame_word_full40_20260928/train/best.mica`
(SHA-256 `ef969c0e96173fc11ecc3b5cf0d1d28d04e21fe846a0752297d5549e90f7e6b7`).
None of the later runs, the PPMI, freeze, tape-readout and day-chain
experiments, was promoted over it. It is not in this repo yet.

The checkpoint below is an experiment and was not promoted. It comes from a
learning-rate screen run on 2026-09-30. Its trainer validation uses 128
records, so its numbers are not comparable with the exact-engine scores of
full40.

### `runs/claude_flame_word_20260930/a_full40_lr030/`

Word-level Flame run that finished at step 225 on 2026-09-30.

| File | What it is |
|------|------------|
| `best.mica` | Best checkpoint (34 MB, `MICAR001` format). SHA-256 `46a26cef51728da789c0b43493a5cade1678577ea11029aed099b23cae4e42e6` |
| `run_info.json` | Training settings: `MICA_*` environment and trainer arguments |
| `progress.json` | Validation history per step |
| `package.json` | Tells `word_eval.py`/`word_decode.py` where the code, model and geometry are (`--model mica:<this folder>`) |
| `FINISHED`, `heartbeat` | Run status markers |

Best validation from `progress.json`: **5.2542 bits** at step 195 (integer readout).
Final step 225: 5.3342 val bits.

Vocabulary: `r1/data/word/vocab.json`. Trained on `r1/data/word/v02a` (train/val JSONL), which is not included here.
See the root README for how to generate text.
