# MICA Flame

## Checkpoints

### `runs/claude_flame_word_20260930/a_full40_lr030/`

Word-level Flame run that finished at step 225 on 2026-09-30.

| File | What it is |
|------|------------|
| `best.mica` | Best checkpoint (34 MB, `MICAR001` format). SHA-256 `46a26cef51728da789c0b43493a5cade1678577ea11029aed099b23cae4e42e6` |
| `run_info.json` | Training settings: `MICA_*` environment and trainer arguments |
| `progress.json` | Validation history per step |
| `FINISHED`, `heartbeat` | Run status markers |

Best validation from `progress.json`: **5.2542 bits** at step 195 (integer readout).
Final step 225: 5.3342 val bits.

Trained on `r1/data/word/v02a` (train/val JSONL), which is not included here.
