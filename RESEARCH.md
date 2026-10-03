# Research with MICA: start here

MICA is a language model whose inference is an **integer cellular automaton**.
There is no neural network, no floating point and no weight matrix at inference.
This page is for anyone who wants to measure it, break it or improve it.
Everything below runs from a fresh clone on a CPU.

Free for personal and research use; commercial use is forbidden ([LICENSE](LICENSE)).

## 1. How it works, in one screen

- **The field.** A ring of 768 cells × 112 integer channels (int8).
- **Input.** Each input symbol, a byte for Ember or a word for Flame-W, writes its learned
  integer code into the cell under a moving write head. The field therefore carries a "tape"
  of recent symbols.
- **Rules.** Between symbols, 16 phases run. In each phase, every cell picks one rule
  candidate from a rule page:
  - the page is chosen by the symbol in the cell;
  - the candidate is chosen by a hash of a few nearby cells ("pairdiff" routing);
  - the chosen rule SETs a work channel to a learned integer.
  - Flame-W has 512 pages × 1,024 candidates; Ember has 256 candidates per page.
- **Readout.** 240 probes read fixed cells and channels. The next-symbol scores are an integer
  bias plus int8 coefficients × probe values, divided by 1,024.
- **Training.** Training ("fit mode") alternates between running the automaton over tens of
  thousands of records and refitting all rule immediates and readout coefficients. Rule
  *wiring* is fixed at initialization; only the integers are learned.

The precise definition is in [`r1/mica_r1/spec.py`](r1/mica_r1/spec.py) and
[`r1/mica_r1/engine.py`](r1/mica_r1/engine.py). The design history and measurements are in
[`docs/r1-findings.md`](docs/r1-findings.md) and [`docs/spec-gaps.md`](docs/spec-gaps.md).

## 2. Set up and check the engine

```bash
pip install -r requirements-dev.txt     # numpy, torch (training only), pyarrow, pytest
python -m pytest r1/tests -q            # 139 tests, including the spec's worked example
```

The integer engine is deterministic. The same model file and the same prompt always
give the same output, on any machine.

## 3. Measure a model

The original evaluation sets are not in the repo, because they were built from training
corpora that are too large to include. Use your own held-out text, one sentence or
message per line:

```bash
# text -> evaluation records (word tokens for bits; bytes for next-word)
python r1/data/text_to_records.py my_text.txt --words my_words.jsonl --bytes my_bytes.jsonl

# bits per word (lower is better)
python r1/runs/claude_flame_word_20260928/word_eval.py bits \
  --model mica:flame/runs/codex_flame_word_full40_20260928/train \
  --vocab r1/data/word/vocab.json --tokens my_words.jsonl --out bits.json

# next-word top-1/3/10 at 2,000 word starts (needs ~300+ lines of normal text)
python r1/runs/claude_flame_word_20260928/word_eval.py nextword \
  --model mica:flame/runs/codex_flame_word_full40_20260928/train \
  --vocab r1/data/word/vocab.json --set my_bytes.jsonl --out nextword.json

# sentences (current decoder: w-sent-mmi.3)
python r1/runs/claude_flame_word_20260928/word_decode.py sentence \
  --model mica:flame/runs/codex_flame_word_full40_20260928/train \
  --vocab r1/data/word/vocab.json --prompts prompts.jsonl --tag test --out out.jsonl

# the byte model (Ember), R1 reference decoder
python r1/generate_bytes.py ember/runs/codex_ember_balanced_v02a_20260928/train "I don't know"
```

`prompts.jsonl` has one `{"id": "...", "prompt": "..."}` per line. For an
experimental decoder that is slightly more useful, see
[`r1/runs/claude_flamew_night_20261001/decode_bos.py`](r1/runs/claude_flamew_night_20261001/decode_bos.py).

## 4. Numbers to beat

All numbers come from the exact integer engine. "Dev" sets were used for selecting models;
"clean" sets were used only for reporting. A sealed test set exists and has never been opened.

| Model | File (SHA-256 prefix) | Metric | Result |
|---|---|---|---|
| Flame-W full40 (words) | `ef969c0e…` | next-word top-1, chat dev / everyday dev | 20.70% / 18.45% |
| | | word completion after 2 letters | 43.64% / 40.84% |
| | | bits per word, clean chat / everyday | 6.8414 / 6.9972 |
| Ember v0.2A (bytes) | `3b2a94e2…` | bits per byte, clean chat / everyday | 1.8609 / 1.8757 |

**Official since v0.3 (2026-10-03): B-740 + memory.** The automaton is checkpoint B continued
to round 740 (SHA `1f4d5503…`). An integer memory readout runs over the last 64 words.

| Metric | B-740 | + memory |
|---|---:|---:|
| use_w(8), chat / everyday long600 | 0.0006 / 0.0004 | 0.0209 / 0.0407 |
| bits per word, no-TinyStories val500 | 5.309 | 5.268 |
| Tiny Theory-of-Mind, 2,000 rows | 28.15% | 30.95% |

See [`flame/runs/claude_flamew_memory_20261003/`](flame/runs/claude_flamew_memory_20261003/).

**Earlier candidate (2026-10-01):** "B" continues full40 for 240 more rounds at
`--round-lr 0.01`, without TinyStories (SHA `4d8cab66…`). On the same 1,000 next-word positions,
it gets 27.5% / 23.8% top-1 (full40: 23.1% / 20.0%) and 5.35 bits per word on no-TinyStories
validation text (full40: 6.22). Details are in
[`r1/runs/claude_flamew_night_20261001/REPORT.md`](r1/runs/claude_flamew_night_20261001/REPORT.md).

Sentence usefulness is judged blind by one judge on 20 holdout prompts, as the share of
continuations that are at least partly useful:

| Model + decoder | At least partly useful |
|---|---|
| full40 + w-sent-mmi.3 | 15% |
| full40 + w-sent-bos.5f | 20% |
| B + w-sent-mmi.3 | 15% |
| **B + w-sent-bos.5f** | **30%** |

20 prompts is a small sample. Each prompt is 5 points.

## 5. Open problems (where help is most useful)

1. **Context.** MICA's rules use only about the last 4–8 bytes, or about 4 words for Flame-W.
   - Flame-W's v0.3 memory readout adds the last 64 words as a readout term.
     use_w(8) went from 0.0005 to 0.02–0.04.
   - The rules themselves still do not carry state. Who-knows-what reasoning is still missing.
   - *How it was measured:* replace the start of a record with another record's start.
     Past about 8 bytes, the bits barely change (use(8) ≈ 0).
   - *The comparison:* a 1M-parameter Transformer on the same data keeps about 0.07 bits
     from 16 bytes back.
   - *What the numbers suggest:* adding long-range features to the linear readout recovers
     at most about 0.003 bits, so long-range information must change **which rule fires**,
     not just feed the readout.
   - *Current hypothesis:* a slowly changing "topic" track, updated by content words, that
     some phases' rule-selection hashes can read (see
     [`docs/2026-09-28-mica-s1-pilot.md`](docs/2026-09-28-mica-s1-pilot.md)).
     *Being tested:* the topic register (`MICA_TOPIC`), a fading sum of recent content
     words' codes that half the rule phases read when choosing a rule; see
     [`r1/runs/claude_flamew_topic_20261002/`](r1/runs/claude_flamew_topic_20261002/).
2. **Stock phrases and loops.** Ember loops on phrases like "the street", and Flame-W drifts
   into story clichés. Data mix, decoding and rules all play a part.
3. **Measuring sentence quality.** Bits and next-word accuracy don't track usefulness well.
   A reliable, cheap, blind sentence judge would speed up every experiment.
4. **Speed.** The Python engine is the reference. A C or SIMD engine that matches it bit for
   bit would make evaluation and search much faster (see `r1/mica_r1/score_kernel.c`).

## 6. Already tried: please don't repeat these without a new idea

| Idea | Result |
|---|---|
| Dilated reads: tape probes at lags up to 64, rules up to 16 cells back | Context use stayed about 0 past 8 bytes; small bits gains, no sentence gain |
| Word or lexical state tracks taking several phases | The state survives, but prediction got worse (it cost 4 of 16 phases) |
| Reversible and stream rule states | Not promoted ([`docs/2026-09-27-mica-rule-selection.md`](docs/2026-09-27-mica-rule-selection.md)) |
| Scaling the Flame geometry (F2–F4) | Rejected; only more rule candidates (F1, 1,024) helped |
| PPMI word codes as the tape start | Better everyday bits; failed the sentence gate |
| Freezing PPMI routing / tape channels | Worse bits, or neutral and unjudged |
| Refitting only the tape readout | Lost on dev bits and next-word accuracy |
| Continuing B for 4 more hours at `--round-lr 0.01` | Bits 5.35 → 5.31, next-word top-1 flat, holdout sentences 30% → 20% partly useful: plateaued |
| Dropping SODA as well as TinyStories (arm C, 1.9 h from B) | Holdout sentences unchanged (6/20 both), dev worse (16/30 vs 20/30); stock phrases change kind, not amount |
| Topic register, first run (2.1 h from B, 8 channels, phases 8-15) | Beats its zero-code control (dev sentences 18 vs 12 of 30 partly useful, val bits 5.31 vs 5.34) but not yet the start model (bits 5.49 vs 5.31); continuing |
| Word-level fit at `--round-lr 0.3` | Diverges (6.75 → 16.3 bits per word); the early stop hides it |
| Training on a TinyStories-heavy mix | Stock story phrases and loops; balanced mixes are better |
| Decoding with a "topic memory" bonus | No systematic change |

What worked (2026-10-01): continuing full40 at `--round-lr 0.01`–`0.03`, which gives large bits
and next-word gains. Removing TinyStories removed the story clichés, but chat-support clichés
replaced them. See the [night report](r1/runs/claude_flamew_night_20261001/REPORT.md).

## 7. Rules that keep results comparable

- Every score names the **model file and its SHA-256**, the **record set**, and the
  **protocol**: exact integer engine or trainer, and bits per byte or per word.
- **Select on dev data; report on clean data.** Never train or tune on evaluation sets.
- **Judge sentences blind**, without knowing which system wrote which output, and
  report the prompt set and the number of prompts.
- **Report negative results too.** Most of section 6 exists because of this rule.

## 8. Share what you find

- **Results, ideas, questions:** open a GitHub issue with the numbers, the model SHA and
  the set.
- **Code:** open a pull request. Keep `pytest r1/tests` passing, and put new experiments in
  their own `r1/runs/<name>/` folder.
- **Try the models** in the browser at https://mica-ai-ten.vercel.app. Ratings and corrections
  there become training data.
