# Flame-W topic register (2026-10-02)

Flame-W's rules read only the last 4–8 words, so in a longer continuation the model loses the
subject: "The museum displayed a collection of" → "... you and I want to help you through this
tough time." Training longer and changing the data mix have both plateaued at about 30%
useful sentences ([night report](../claude_flamew_night_20261001/REPORT.md)). This experiment
changes the machine instead.

## The topic register

A few extra channels in every cell (`MICA_TOPIC=8`). When a word arrives, the new cell's register
is the previous cell's, faded by 1/8 toward zero, plus the word's **topic code**:

    d = prev - sign(prev) * (|prev| >> 3)
    topic = clamp(d + code[word], -127, 127)

- **Codes** (`topic_codes.py`) come from word co-occurrence in the training records (PPMI, then
  SVD). Content words that appear together get similar codes; the 150 most frequent words
  (the, a, I, you, ...) get zero. So the register is a fading sum of the recent content words:
  what the text is about, from further back than the rules can see.
- **Use.** In phases 8–15, two of each rule's six scoring terms read the cell's own register
  instead of a word on the tape. Which rule fires, and so which learned integers the
  readout sees, then depends on the topic *and* the local words. Earlier tests showed that
  long-range information has to change which rule fires; feeding it to the linear readout
  adds almost nothing.
- **Integer and exact,** like the rest of MICA. No rule can write the register. The engine
  (`engine.py`) and the fitter's simulator (`fit._simulate`) run the same arithmetic,
  checked by `r1/tests/test_topic_register.py`.
- **Off by default.** With `MICA_TOPIC` unset, every existing model loads, runs and re-saves
  byte-identically. A topic model's file header records the register, so it cannot be
  loaded by a decoder without it, and vice versa.

The engine before this change is saved on the branch `engine-stable-2026-10-01`.

## The experiment (`topic.py`)

Both arms start from the same model (the night launcher's `B_lr010` best), train on the
same no-TinyStories data with the same rounds at `--round-lr 0.01`, and are evaluated the
same way:

| Arm | What changes |
|---|---|
| `T_topic` | the register with real codes |
| `T_zero` | the same rule change with all-zero codes: the two terms read 0 (control) |

`T_topic` minus `T_zero` is the effect of the topic itself, separate from the cost of
changing the rules. Both are also compared with the start model.

```powershell
& ..\.venv-rocm\Scripts\python.exe r1\runs\claude_flamew_topic_20261002\topic.py --hours 5
```

The run builds the codes (once, CPU), trains both arms in turn, evaluates every model in its
own process, and writes `topic_report.zip`. It uses the same guards as the night launcher.

**Decision rule, written before the run:** promote `T_topic` only if, against `T_zero`, it
is at least as good on bits and next-word accuracy, *and* it is more useful on the blind
sentence comparison (holdout and dev), *and* it is at least as useful as the current
official B.

## Result of the first run (2026-10-01, 18:06–22:32)

Both arms started from `B_lr010` round 740 (SHA `1f4d5503…`), trained 224 rounds (2.1 h each)
at `--round-lr 0.01` on no-TinyStories data, and were evaluated identically
([results/](results/)).

| Model | Bits/word val500 | Next-word top-1 chat / everyday | Best training-val bits |
|---|---|---|---|
| start (B continued) | **5.309** | **27.3% / 24.0%** | — |
| T_topic | 5.493 | 25.6% / 22.2% | **5.309** |
| T_zero (control) | 5.502 | 25.5% / 22.6% | 5.342 |

Sentences with `w-sent-bos.5f`, 50 prompts, all three models shown in random order (A/B/C)
and scored before unblinding:

| Model | Holdout 20: partly / fully useful | Dev 30: partly / fully useful | All 50 partly |
|---|---|---|---|
| start | 4 / 0 | 18 / 10 | 22 |
| **T_topic** | 3 / 1 | **18 / 10** | **21** |
| T_zero | 5 / 0 | 12 / 6 | 17 |

**Reading.**
- **The topic signal is real.** Against its matched control, T_topic is ahead on training
  validation (5.309 vs 5.342 bits) and on dev sentences (18 vs 12 partly useful). On the
  20 holdout prompts it is 3 vs 5, which is within one judge's noise at this size.
- **The rewiring costs more than 224 rounds can repay.** Taking two of six terms away
  from the local words in half the phases reset about 0.2 bits. Both arms were still
  improving when time ran out, and neither has caught up with the start model.
- **The codes capture genre more than subject.** The strongest channels separate
  encyclopedic text ("established, former, January"), story text ("wagged, Amy's, Ben's")
  and captions ("sits, wooden, white"). Mean |code| is 7, so the register moves
  slowly.

**Verdict (by the rule above): not promoted.** T_topic beats T_zero, but it is not yet as good
as B. Next: continue T_topic, which resumes where it stopped, until it passes the start
model on bits, then judge again. The same command does this with `--arms T_topic`.

## Second run: topical data (v03) and better codes

The first run's records were one or two turns (about 20 words), so the register had little to
carry, and its codes mostly told writing styles apart. `T2_topic` changes the data, not the
machine:

- **Data** ([`r1/data/build_topical_corpus.py`](../../data/build_topical_corpus.py)): windows of
  consecutive turns from one dialogue, up to 64 words, added to the no-TinyStories corpus
  (about 12% of records, about 28% of words):
  - Topical-Chat, Taskmaster, SGD and SODA (already used), now as whole dialogues;
  - UltraChat 200k (MIT), OpenAssistant oasst1 (Apache 2.0) and Synthetic-Persona-Chat
    (CC BY 4.0), new.
  - Assistant answers are cut to their first two or three sentences. Turns with code,
    lists or tables are dropped and end the window. Training splits only.
- **Codes**: learned from those windows only, with a 16-word window and the 250 most frequent
  words skipped. On a sample of the new sources the channels separate subjects: food and
  recipes, government and politics, religion, travel and hiking, fashion and décor,
  scheduling.
- **Evaluation**: 24 new prompts that give a subject first ("I just got back from Italy. The
  food there was"), added to the 50 used so far, and every earlier arm evaluated again
  for comparison.

```powershell
& ..\.venv-rocm\Scripts\python.exe r1\runs\claude_flamew_topic_20261002\topic.py --hours 8
```

The default arm is `T2_topic`. The first run downloads about 300 MB and builds the corpus
(about 15 minutes, once).

## Result of the second run (2026-10-01 23:40 – 2026-10-02 06:30)

The first attempt stopped while building the corpus: the UltraChat download was truncated
(fixed with checks on size and SHA-256). The second attempt built v03 and trained `T2_topic`
for 500 rounds (5.9 h) from `B_lr010`. Codes from the topical windows separate subjects
(food and recipes, booking and weekdays, cities, nature, feelings, the economy;
[results/run2/](results/run2/)).

| Model | Bits/word val500 | Next-word top-1 chat / everyday |
|---|---|---|
| start | **5.309** | **27.3% / 24.0%** |
| T_topic | 5.493 | 25.6% / 22.2% |
| T2_topic | 5.538 | 26.8% / 22.3% |

Sentences (`w-sent-bos.5f`), all 74 prompts, three models in random order, scored before
unblinding. Each cell shows prompts at least partly useful / fully useful:

| Model | Holdout 20 | Dev 30 | Topic 24 (subject first) |
|---|---|---|---|
| start | 4 / 0 | 18 / 10 | 4 / 1 |
| T_topic | 3 / 1 | 18 / 10 | 3 / 1 |
| T2_topic | 3 / 0 | 15 / 12 | 3 / 1 |

**Reading.** No model uses the first sentence: "I just got back from Italy. The food there
was" → "a lot of fun." / "to be there." / "also a difference." The three are within one
judge's noise everywhere. The register carries the topic (the codes are good), but in this
wiring it barely changes the predicted word:
- it only picks which rule fires in phases 8–15;
- those rules write work channels that 96 of the 240 probes read;
- the readout's word scores are linear in those, so the topic reaches a word only through
  a few hashed buckets.

**Verdict: not promoted. The topic register as wired does not give topic coherence.**
The direct route is for the readout to read the register: each word gets learned weights
on the 8 topic channels, so the same register state favours "pasta", "delicious" and
"restaurant" after "Italy". This is a topic-conditioned word bias, exact and in integers. The
fitter needs a third kind of probe for it, because register values come from simulation
and are neither tape codes nor rule immediates.
