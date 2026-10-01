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
