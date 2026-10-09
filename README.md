<p align="center"><img src="assets/mica-flame.svg" alt="MICA flame logo" width="120"></p>

<h1 align="center">MICA AI</h1>

<p align="center"><b>A novel language model built from integer cellular automata.</b><br>
Not a transformer, not a neural network: just learned integer rules on a ring of cells.</p>

<p align="center">
  <a href="https://mica-ai-ten.vercel.app"><b>Try the live demo</b></a> ·
  <a href="https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton"><b>Hugging Face collection</b></a> ·
  <a href="https://github.com/Vovala14/Mica-Ai"><b>Clone</b></a> ·
  <a href="RESEARCH.md">Research guide</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="LICENSE">License (non-commercial)</a>
</p>

<p align="center"><img src="assets/playground.gif" alt="The MICA playground: typing a prompt, the automaton thinking, and Flame-W answering" width="640"><br>
<sub>The live playground with Flame-W. The waiting time is sped up about 2×.</sub></p>

## Start here

**Not a small transformer.** Learned integer rules on a ring of cells. Same prompt → same output on any CPU. Research only.

| | |
|---|---|
| Try it | [Live demo](https://mica-ai-ten.vercel.app) |
| Download | [Hugging Face collection](https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton) · [Flame-W 0.3.3](https://huggingface.co/vynly/mica-flame-w-0.3.3) · [Ember v0.3.1 package](https://github.com/Vovala14/Mica-Ai/tree/main/ember/runs/ember_v031_20261008) |
| Clone | `git clone https://github.com/Vovala14/Mica-Ai.git` |

It is an integer cellular automaton: no attention, no floating point, no neural network. It is not a production assistant. Commercial use is forbidden.

On public Tiny Theory-of-Mind (2,000 items, chance 25%), Flame-W 0.3.3 scores **636/2,000 (31.80%)** with word-normalized answer mode. Version 0.3.2 scored 632/2,000 (31.60%); the paired +0.20 percentage-point difference has a 95% interval of −0.25 to +0.65 points, so a ToM improvement is not established. The demo and historical replay below do not use answer mode.

## What is MICA?

MICA is a research language model built from scratch rather than adapted from a transformer.
It uses a learned integer-rule cellular automaton to generate and score language.

This project explores an alternative path to language modeling:

- No floating point arithmetic
- No attention mechanism
- No neural network training stack as usually defined
- Integer-only local rules on a ring of cells
- Same prompt always gives the same output on any machine

## Why it matters

MICA is designed to ask a simple research question:

Can a compact, fully integer, local-rule system model language well enough to be useful?

The answer appears to be: yes, in a research setting.

- Flame-W 0.3.3 reaches 31.80% on public Tiny Theory-of-Mind with answer mode; its small change from 0.3.2 is uncertain
- Its integer short-memory blend lowers short-record word loss, with a measured small long-record regression
- Ember v0.3.1 improves byte-level loss; its word-suggestion scores remain unchanged from v0.3a
- The model is small, inspectable, and intentionally different from mainstream transformer stacks

## Published model cards

MICA models are also published on Hugging Face in the [vynly collection](https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton):

- [vynly/mica-flame-w-0.3.3](https://huggingface.co/vynly/mica-flame-w-0.3.3) — the current sentence model
- [vynly/mica-ember-0.3a](https://huggingface.co/vynly/mica-ember-0.3a) — the previous word-assistance card
- [Ember v0.3.1 GitHub package](https://github.com/Vovala14/Mica-Ai/tree/main/ember/runs/ember_v031_20261008) — updated byte checkpoint and integer memory

These model cards mirror the research work in this repository and provide a public point of access alongside the GitHub project.

## Current performance snapshot

| Model | Snapshot |
|---|---|
| Flame-W 0.3.3 | Same B-740 cellular rules; short-memory blend improves short word loss, with a small long-record cost; research-only sentence generation |
| Flame-W v0.3 | Improved context use with a learned memory readout |
| Ember v0.3.1 | Better exact integer byte loss; same next-word and two-letter completion heads as v0.3a |

Highlights from the repo:

- Flame-W 0.3.3: 31.80% word-normalized and 31.65% character-normalized on public Tiny Theory-of-Mind, with unchanged answer mode; no established ToM gain over 0.3.2
- Exact integer short500 loss: 5.208428 → 5.189430 bits/word symbol plus EOS; long500: 8.082371 → 8.086188 (0.3.2 → 0.3.3). Blind sentence check: 1 win, 1 loss, 38 ties
- Ember v0.3.1: clean equal-domain byte loss 1.831770 bits/target vs 1.868322 for v0.2A (paired change −0.036552; 95% CI −0.038721 to −0.034442); word top-1 remains 14.25% and two-letter completion remains 48.00%

This is not a production assistant. It is a research model that explores a different architecture and a different training setup.

## Project structure

| Part | Where | Status |
|------|-------|--------|
| Engine and training code (R1) | [`r1/`](r1/) | added |
| MICA Flame-W (sentence model) | [`flame/`](flame/) | B-740 + three integer memory banks (0.3.3); sentence continuation and generation |
| MICA Ember (word assistance on a byte engine) | [`ember/`](ember/) | v0.3.1 byte/readout checkpoint with v0.3a integer word heads |

## Quick start

```bash
git clone https://github.com/Vovala14/Mica-Ai.git
cd Mica-Ai
pip install -r requirements-dev.txt   # requirements.txt is numpy only, for the website
python -m pytest r1/tests -q          # includes the §7 worked example
```

### Historical bit-identical replay (Flame-W 0.3.1)

`flame/replay/prompts.jsonl` is five fixed prompts. `flame/replay/out.jsonl` records the **0.3.1** website decoder's continuation bytes: B-740 plus the 0.3.1 memory, answer mode off. This is a historical replay and does not verify 0.3.3 output. The playground's Flame-W rating chips use these prompt IDs (`s1`–`s5`).

| Piece | Identity |
|---|---|
| Model | `flame-w-0.3.1` |
| Automaton | `flame/runs/claude_flamew_b740_20261001/train/best.mica` |
| Automaton SHA-256 | `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d` |
| Memory | `flame/runs/claude_flamew_031_20261004/memory.npz` |
| Memory SHA-256 | `5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7` |
| Decoder | `w-sent-bos.5f` |
| Committed output SHA-256 | `3b1cac571629157d99c7438066cbc786590e7325d3c838c500561b7cbc899ec3` |
| Hugging Face | [vynly/mica-flame-w-0.3.1](https://huggingface.co/vynly/mica-flame-w-0.3.1) |

```bash
python flame/replay/replay_flamew_031.py --check
```

`--check` reruns every prompt and exits 0 only when the UTF-8 file matches `flame/replay/out.jsonl` byte for byte. Same prompt, same continuation bytes. The weights are in this repository. The Hugging Face model card is the public download of the same checkpoint.

One recorded miss in that file: "I lost my keys and" → " I don't know how to fix it." One recorded continuation that stays on the prompt: "I was thinking about" → " how it would affect you."

The older automaton-only sentence command, without the 0.3.1 memory, is:

```bash
echo '{"id":"1","prompt":"I went to the"}' > prompts.jsonl
python r1/runs/claude_flame_word_20260928/word_decode.py sentence \
  --model mica:flame/runs/claude_flamew_b_20261001/train \
  --vocab r1/data/word/vocab.json --prompts prompts.jsonl --tag demo --out out.jsonl
```

To run the current 0.3.3 release, download the [Hugging Face package](https://huggingface.co/vynly/mica-flame-w-0.3.3) and run its scripts from that directory:

```bash
pip install numpy huggingface_hub
python generate.py "I went to the"
python run_tom.py --out tom_result.json  # downloads the public 2,000-item dataset if needed
```

For the historical 0.3.1 memory readout and answer-mode result:

```bash
python flame/runs/claude_flamew_031_20261004/generate.py "I went to the"
python flame/runs/claude_flamew_031_20261004/run_tom.py     # Tiny Theory-of-Mind, 31.25% / 31.45%
```

On that older checkpoint, `sentence` writes a full sentence ("I went to the" → " movies with you.").
`suggest` gives a 2–3 word next-words suggestion ("Can you help me" → " find my").
Each line of that command's output file has the prompt and its `continuation`. It is separate from `flame/replay/out.jsonl`.

Try Ember v0.3.1 word suggestions:

```bash
python ember/runs/ember_v031_20261008/test_word_model.py
python ember/runs/ember_v031_20261008/test_memory.py
python ember/runs/ember_v031_20261008/word_model.py
```

The original v0.2A byte generator is also available:

```bash
python r1/generate_bytes.py ember/runs/codex_ember_balanced_v02a_20260928/train "I don't know"
```

These are research models. Ember's measured role is next-word suggestion and typed-word completion; Flame-W handles sentences. The website offers Ember v0.3.1 word actions alongside Flame-W's sentence actions. The browser uses the integer word readouts; the separate integer memory sidecar is used by the byte-scoring runtime.

## How it works

MICA is its own kind of language model, designed from scratch rather than adapted from a transformer.
It is a learned integer-rule cellular automaton. Its reference implementation follows the MICA exact mechanism and learning specification, revision R1.

- How it works: each word (Flame-W) or byte (Ember) is written onto a ring of 768 cells × 112 integer channels. 16 phases of learned local rules run, and 240 probes read the result to score the next symbol. The same prompt always gives the same output, on any machine.
- Where it stands: Flame-W's rules see about the last 4 words.
  - Since v0.3, an integer memory readout lets it use the last 64 words. Version 0.3.3 blends two short integer readouts through word 64 and retains the 0.3.2 long bank afterward.
  - It scores 31.80% word-normalized on public Tiny Theory-of-Mind with answer mode; the paired difference from 0.3.2 is uncertain. Its sentences are not reliably coherent.
  - These are research models, not assistants.

## Test website

`public/` and `api/` are a Vercel site where people try both models and rate the output:

| Path | What it does |
|------|--------------|
| `public/index.html` | The page: Flame-W sentence / next words, Ember v0.3.1 next word / finish word, fixed rating prompts, replay link and corrections |
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

## Research and documentation

- [`RESEARCH.md`](RESEARCH.md): overview of the model, evaluation setup, open problems, and experiments
- [`CHANGELOG.md`](CHANGELOG.md): release history and milestones
- [`flame/README.md`](flame/README.md): Flame-W model details and checkpoints
- [`ember/README.md`](ember/README.md): Ember byte/word model details and scores

## License

Free for personal use and research use.
Commercial use is forbidden. See [LICENSE](LICENSE) for the full terms.
For commercial licensing, contact the repository owner.
