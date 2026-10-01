# Changelog

Model files are identified by their SHA-256. Scores come from the exact integer engine; see
[RESEARCH.md](RESEARCH.md) for how each number is measured.

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
