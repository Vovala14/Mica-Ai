<p align="center"><img src="assets/mica-flame.svg" alt="MICA flame logo" width="120"></p>

<h1 align="center">MICA AI</h1>

<p align="center"><b>A one-of-a-kind language model: an integer cellular automaton that writes language.</b><br>
Not a transformer and not a neural network: no attention, just learned integer rules on a ring of cells.</p>

<p align="center">
  <a href="https://mica-ai-ten.vercel.app"><b>Try it in your browser</b></a> ·
  <a href="RESEARCH.md">Research guide</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="LICENSE">License (non-commercial)</a>
</p>

<p align="center"><img src="assets/playground.gif" alt="The MICA playground: typing a prompt, the automaton thinking, and Flame-W answering" width="640"><br>
<sub>The live playground with Flame-W. The waiting time is sped up about 2×.</sub></p>

MICA is its own kind of language model, designed from scratch rather than adapted from a
transformer. It is a learned integer-rule cellular automaton. Its reference
implementation follows the *MICA exact mechanism and learning specification*,
revision R1.

- **How it works:** each word (Flame-W) or byte (Ember) is written onto a ring of 768 cells ×
  112 integer channels. 16 phases of learned local rules run, and 240 probes read the result
  to score the next symbol. The same prompt always gives the same output, on any machine.
- **Where it stands:** Flame-W's rules see about the last 4 words.
  - Since v0.3, an integer **memory readout** lets it use the last 64. 0.3.1 refits it on 7
    times more text.
  - It scores **31.25% on Tiny Theory-of-Mind** (31.45% per character), above 6–7 of the 36
    models on that leaderboard, and 5.21 bits per word on validation text.
  - These are research models, not assistants.

| Part | Where | Status |
|------|-------|--------|
| Engine and training code (R1) | [`r1/`](r1/) | added |
| MICA Flame-W (sentence model) | [`flame/`](flame/) | B-740 + memory (0.3.1); sentence continuation and generation |
| MICA Ember (word assistance on a byte engine) | [`ember/`](ember/) | v0.3a integer word readouts on the v0.2A cellular checkpoint |

**Want to experiment or do research with MICA?** Start with [RESEARCH.md](RESEARCH.md): how it works, how to measure it, the open problems, and what has already been tried.

## Quick start

```bash
pip install -r requirements-dev.txt   # requirements.txt is numpy only, for the website
python -m pytest r1/tests -q          # 139 tests, including the §7 worked example
```

Generate text with Flame-W:

```bash
echo '{"id":"1","prompt":"I went to the"}' > prompts.jsonl
python r1/runs/claude_flame_word_20260928/word_decode.py sentence \
  --model mica:flame/runs/claude_flamew_b_20261001/train \
  --vocab r1/data/word/vocab.json --prompts prompts.jsonl --tag demo --out out.jsonl
```

The command above runs the automaton alone. With the 0.3.1 memory readout, as on the website:

```bash
python flame/runs/claude_flamew_031_20261004/generate.py "I went to the"
python flame/runs/claude_flamew_031_20261004/run_tom.py     # Tiny Theory-of-Mind, 31.25% / 31.45%
```

`sentence` writes a full sentence ("I went to the" → " movies with you.").
`suggest` gives a 2-3 word next-words suggestion ("Can you help me" → " find my").
Each output line in `out.jsonl` has the prompt and its `continuation`.

Try Ember v0.3a word suggestions:

```bash
python ember/runs/ember_v03a_20261004/test_word_model.py
python ember/runs/ember_v03a_20261004/word_model.py
```

The original v0.2A byte generator is also available:

```bash
python r1/generate_bytes.py ember/runs/codex_ember_balanced_v02a_20260928/train "I don't know"
```

These are research models. Ember's measured role is next-word suggestion and
typed-word completion; Flame-W handles sentences. The website offers both
Ember v0.3a word actions alongside Flame-W's sentence actions.

## Test website

`public/` and `api/` are a Vercel site where people try both models and rate the output:

| Path | What it does |
|------|--------------|
| `public/index.html` | The page: Flame-W sentence / next words, Ember v0.3a next word / finish word, feedback and corrections |
| `api/flame.py`, `api/ember.py` | Run the official checkpoints with the exact integer engine (`webapp/`) and sign each output |
| `api/log.js` | Saves signed generations and feedback to a private Vercel Blob store |
| `api/export.js` | Owner download of all logs: `curl -H "Authorization: Bearer ADMIN_KEY" <site>/api/export -o logs.jsonl` (`?summary=1` for counts, `?check=1` for a storage check) |
| `r1/data/ingest_site_logs.py` | Turns that export into Ember byte records and Flame-W word records for training |

The site needs three environment variables: `LOG_SECRET`, `ADMIN_KEY` and `BLOB_READ_WRITE_TOKEN`.
The last one is set automatically when the Blob store is connected.

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
