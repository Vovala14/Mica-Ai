# MICA AI

MICA is a learned integer-rule cellular automaton language model. Its reference
implementation follows the *MICA exact mechanism and learning specification*,
revision R1.

| Part | Where | Status |
|------|-------|--------|
| Engine and training code (R1) | [`r1/`](r1/) | added |
| MICA Flame-W (word model) | [`flame/`](flame/) | official full40 checkpoint, vocabulary and decoder |
| MICA Ember (byte/letter model) | [`ember/`](ember/) | official v0.2A checkpoint |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest r1/tests -q          # 139 tests, including the §7 worked example
```

Generate text with Flame-W:

```bash
echo '{"id":"1","prompt":"I went to the"}' > prompts.jsonl
python r1/runs/claude_flame_word_20260928/word_decode.py sentence \
  --model mica:flame/runs/codex_flame_word_full40_20260928/train \
  --vocab r1/data/word/vocab.json --prompts prompts.jsonl --tag demo --out out.jsonl
```

`sentence` writes a full sentence ("I went to the" → " movies with you.").
`suggest` gives a 2-3 word next-words suggestion ("Can you help me" → " find my").
Each output line in `out.jsonl` has the prompt and its `continuation`.

Generate text with Ember:

```bash
python r1/generate_bytes.py ember/runs/codex_ember_balanced_v02a_20260928/train "I don't know"
```

Both are research models. Neither produces reliably sensible sentences yet.

## What is where

| Path | Contents |
|------|----------|
| `r1/mica_r1/` | The engine: spec, exact integer engine, model file loader, decoders |
| `r1/data/` | Corpus builders; `r1/data/word/vocab.json` is the Flame-W vocabulary |
| `r1/runs/claude_flame_word_20260928/` | Flame-W word decoder (`word_decode.py`) and word metrics (`word_eval.py`) |
| `r1/run_train.py`, `r1/train_soft.py` | Training |
| `r1/checks/`, `r1/tests/` | Experiments, evaluations and tests |
| `docs/` | Findings, plans and experiment write-ups (`r1-findings.md`, `spec-gaps.md`, ...) |

`r1/mica_r1/score.dll` is a Windows build of `r1/mica_r1/score_kernel.c`.
See [`r1/README.md`](r1/README.md) for the engine layout and training commands.

## License

Free for **personal use** and **research use**.
**Commercial use is forbidden.** See [LICENSE](LICENSE) for the full terms.
For commercial licensing, contact the repository owner.
