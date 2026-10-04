# Changelog

Model files are identified by their SHA-256. Scores come from the exact integer engine; see
[RESEARCH.md](RESEARCH.md) for how each number is measured.

## Ember v0.3a: word assistance (2026-10-04)

- The unchanged v0.2A byte cellular automaton now has two learned integer
  readouts over its existing probes for next-word suggestions and completion
  after two typed letters. The packed heads are 1.23 MB; details and hashes
  are in [`ember/runs/ember_v03a_20261004/`](ember/runs/ember_v03a_20261004/).
- Against official Ember v0.2A on the same 200 chat and 200 everyday clean
  prompts, next-word top-1 improves 6.75% to 14.25%; two-letter completion
  improves 42.75% to 48.0%. The paired equal task/domain gain is +6.375
  percentage points, 95% CI [+4.0,+8.75]. Byte loss of the cellular base is
  unchanged.
- Ember's product role is word suggestions and typed-word completion. Flame-W
  owns sentence continuation and generation. The site now exposes both Ember
  word actions through the v0.3a integer readouts.

## v0.3.1: Flame-W memory refit and answer mode (2026-10-04)

**Models**
- **Flame-W 0.3.1 is the new official word model.** It is the same B-740 automaton with a
  refitted memory.
  - Memory: `flame/runs/claude_flamew_031_20261004/memory.npz`, 674 KB. It is v0.3's readout,
    refitted on 4.5 million word positions (7 times more) and selected on held-out text.
  - Bits per word on no-TinyStories validation: **5.208**, down from 5.268. Dev: 5.504, down from
    5.577.
  - Context use use_w(8), chat / everyday: 0.0183 / 0.0483 (v0.3: 0.0209 / 0.0407).
- **Answer mode** for multiple-choice questions: +3.25 nats for content words already in the
  context. The gain was picked on synthetic practice items. The website does not use it.
- **Tiny Theory-of-Mind:** 31.25% per word token and 31.45% per character. On the held-out half
  it is 31.80% / 32.50%.
  - That is not significantly different from v0.3 with answer mode (32.20% / 31.40%, p = 0.17).
  - It is above 6–7 of the 36 leaderboard models.
- **Tried without gain:** counting names as seen content in answer mode.

**Playground**
- The site runs B-740 with the 0.3.1 memory (no answer mode) and the same decoders.

## v0.3: Flame-W with memory (2026-10-03)

**Models**
- **Flame-W B-740 + memory is the new official word model.**
  - Automaton: `flame/runs/claude_flamew_b740_20261001/train/best.mica`, SHA `1f4d5503…`, 34 MB.
    It is B continued to round 740.
  - Memory: `flame/runs/claude_flamew_memory_20261003/memory.npz`, 674 KB.
- **New mechanism: a memory readout.**
  - It is an integer readout over the last 64 words: recall of the same words plus a
    16-channel association of related words.
  - It is fitted with the automaton frozen, on held-out selection only.
- **Context use:** use_w(8) is 0.0209 / 0.0407 on chat / everyday, up from 0.0006 / 0.0004. The
  0.010 gate is passed for the first time.
- **Bits per word on no-TinyStories validation:** 5.268, down from 5.309.
- **Tiny Theory-of-Mind (2,000 rows):** **30.95%**, up from 28.15%, paired +2.80 pp [+1.00, +4.55].
  That is above 6 of the 36 leaderboard models, up to Syn-2.6M at 30.30%.
- **Sentences:** no significant difference from B in a blind 50-prompt check.

**Playground**
- The site runs B-740 + memory with the same decoders.

## v0.2: Flame-W B (2026-10-01)

**Models**
- **Flame-W B is the new official word model**:
  `flame/runs/claude_flamew_b_20261001/train/best.mica`, SHA `4d8cab66…`, 34 MB.
  It continues full40 for 240 more rounds at `--round-lr 0.01`, without TinyStories.
  - Next-word top-1 on chat / everyday text: **27.5% / 23.8%** (full40: 23.1% / 20.0%), on the
    same 1,000 positions.
  - Bits per word on no-TinyStories validation text: 5.35 (full40: 6.22).
  - In a blind test on 20 holdout prompts, **30%** of sentence continuations were at least
    partly useful (full40: 15%).
- New sentence decoder `w-sent-bos.5f`. It penalizes stock phrases that would follow any
  prompt (MMI against the start of a record) and drops endings left hanging.

**Playground**
- The site runs Flame-W B with the new decoder.
- A "thinking" panel shows while MICA works: a small automaton seeded from the prompt, the
  engine's real steps, and a timer. After the answer, the page scrolls to the rating buttons.
- Fixed the Creativity slider showing on the Flame-W tabs.
- Added a link preview image for sharing.

**Research**
- [RESEARCH.md](RESEARCH.md): how MICA works, how to measure it, the numbers to beat, open
  problems, and what has already been tried.
- Overnight and day GPU training launchers with VRAM and RAM guards and built-in evaluation:
  [`r1/runs/claude_flamew_night_20261001/`](r1/runs/claude_flamew_night_20261001/).

## v0.1: first public release (2026-09-30)

- Engine, training and evaluation code (spec revision R1), with 139 tests.
- **Flame-W full40** (word model, 16,384-symbol vocabulary): SHA `ef969c0e…`.
- **Ember v0.2A** (byte model): SHA `3b2a94e2…`, 4.5 MB. It scores 1.86–1.88 bits per byte on
  clean chat and everyday text.
- The playground at https://mica-ai-ten.vercel.app: try both models, rate outputs, and
  suggest better continuations. Signed logs are saved for training.
- License: free for personal and research use; commercial use is forbidden.
