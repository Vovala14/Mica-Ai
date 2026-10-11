# Changelog

Model files are identified by their SHA-256. Scores come from the exact integer engine; see
[RESEARCH.md](RESEARCH.md) for how each number is measured.

## Ember v0.3.5: weight-averaged automaton and new word readouts (2026-10-11)

- Owner-authorized release on GitHub, Hugging Face and the website (Ember next word and finish word now run v0.3.5).
  Model `c2bef375…`, memory `3827c2a3…`, word heads `1f73aa85…` (8,192 words).
- Automaton: the 097b native refit plus 60 live training rounds with the learned integer rule values and readout
  averaged (the Flame-W 0.3.5 method). Byte readout and 14 KB memory refit jointly on 119,823 training records;
  word readouts retrained on the same records.
- Clean byte+EOS bits/target versus v0.3.1: chat 1.827462 → 1.807137, everyday 1.836078 → 1.812973;
  equal-domain paired change -0.021715 [-0.023154, -0.020275]. Development confirmation -0.020635 [-0.022641, -0.018629].
- Clean word benchmark: two-letter completion 48.00% → 54.50% (+6.50 points [+3.00, +10.00]);
  next-word top-1 14.25% → 15.25% (+1.00 points [-1.75, +3.75], within noise; development
  +3.08 points [+1.67, +4.58]).
  Selection used development data only; no sealed data. [Package and evidence](ember/runs/ember_v035_20261011/README.md).
- The website no longer lists Flame-W 0.3.4 separately; Flame-W 0.3.5 includes its belief memory.

## Flame-W 0.3.5: weight-averaged automaton (2026-10-11)

- Owner-authorized release on GitHub, Hugging Face and the website. One file changes: the automaton. B_swa SHA `eca9faee…` is B-740 with its learned integer rule immediates and readout averaged over 40 live fit rounds (40,000 training records per round, round-lr 0.01). Geometry, file size, memories, answer mode, belief memory, tokenizer and decoders are byte-identical to 0.3.4.
- Exact integer bits per word symbol plus EOS with the released memories, paired against 0.3.4: short500 5.189430 → 5.162197 (−0.0272 [−0.0387, −0.0147]); long500 8.086188 → 8.073772 (−0.0124 [−0.0161, −0.0086]). Automaton alone −0.048 short, −0.035 long.
- Public Tiny ToM word 638 → 645/2,000 (32.25%; +54/−47, p = 0.55), character 635 → 637. HellaSwag 27.30 → 27.70 (+188/−147, p = 0.03); ARC-Easy 25.80 → 26.18; ARC-Challenge 19.80 → 20.73; PIQA 51.41 → 51.31; ArithMark-3 26.9 → 27.0 (frozen 0.3.4 five-task protocol). Nothing was tuned on any benchmark. [Release, hashes and evidence](flame/runs/claude_flamew_035_20261011/README.md).

## Flame-W 0.3.4: experimental cellular belief memory (2026-10-10)

- Owner-authorized prototype release on GitHub and the website. Adds a separate radius-one 32×256-cell integer memory with learned COPY/SKIP rules and external English binding. Native B-740 SHA `1f4d5503…`, all language-memory banks and sentence decoder remain unchanged.
- Fresh controlled supported development versus the previous private prototype: 899→965/1,000, 66 gains/zero losses; paired scenario interval +5.1 to +8.2 percentage points. Memory-only complete pairs 465/500; unsupported definite answers 0/200. Related finite authored grammar limits transfer claims.
- Public Tiny ToM word 638/2,000 (31.90%), character 635/2,000 (31.75%). Unchanged versus the prior private prototype. Against published 0.3.3 the word difference is +0.10 percentage points [−0.10,+0.30], not statistically established. No new sentence-quality or five-task leaderboard claim; no sealed data.
- Adds a live story/question/four-choice demo with explicit memory-versus-language-model fallback reporting. [Release, hashes and limits](flame/runs/codex_flamew_034_20261010/README.md).

## Flame-W 0.3.3: conservative short-memory blend (2026-10-09)

- Released on [Hugging Face](https://huggingface.co/vynly/mica-flame-w-0.3.3) and the live playground. The B-740 learned integer-rule cellular automaton SHA `1f4d5503…`, tokenizer, long-memory bank, sentence decoder and answer mode are unchanged. An additional fitted integer short bank is blended 1:4 with the original for the first 64 fed word symbols. Model and three memory files total 36,132,566 bytes, 673,650 bytes above 0.3.2.
- Fresh disjoint development selection passed the frozen equal-domain gate. Exact integer clean short500 loss improved 5.208428 → 5.189430 bits/word symbol plus EOS (paired −0.018998 [−0.021376,−0.016611]); clean long500 worsened 8.082371 → 8.086188 (+0.003818 [+0.002786,+0.004814]). The long-record cost is real and below the predeclared cap.
- Public Tiny ToM, unchanged answer mode and 2,000 items: word-normalized 632 → 636 correct (31.60% → 31.80%), paired +0.20 percentage points [−0.25,+0.65]; character-normalized 629 → 633 (31.45% → 31.65%). The ToM gain is uncertain. Blind sentence40 was 1 win, 1 loss, 38 ties with one new unusable output; no sentence-quality gain is established. The five-task leaderboard was not rerun. Full [evaluation and limitations](flame/runs/codex_flamew_033_20261009/EVALUATION.md).

## Flame-W 0.3.2: gated long memory and decoder (2026-10-08)

- Kept the B-740 automaton and original short-memory bank through 64 symbols, then used a trained long integer memory bank. Clean long500 loss improved 8.305251 → 8.082371 bits/word symbol plus EOS, paired −0.222880 [−0.241879,−0.203769]; clean short500 was unchanged. A gated decoder adjustment passed its small single-rater blind check. See the [0.3.2 report](flame/runs/codex_flamew_032_20261008/EVALUATION.md) where available; historical details remain in the research runs.

## Site: how to find it, and a byte replay (2026-10-07)

- The playground leads with what MICA is not (no attention, no floating point, not a neural network, not a production assistant), then Tiny Theory-of-Mind at 31.25% beside the weak topics (bluffing and communication failure 16%, diverse beliefs and white lie 18%).
- GitHub, the [vynly collection](https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton), and the Flame-W and Ember model cards are linked from the top of the page.
- Ten locked rating prompts (five Flame-W sentences, five Ember next-word) use the existing signed log. Free-text prompts still work. A locked prompt stores an optional `prompt_id` (`s1`–`s5` match the replay file; Ember is `e1`–`e5`).
- `flame/replay/` replays five Flame-W 0.3.1 sentences. `replay_flamew_031.py --check` matches `out.jsonl` byte for byte.

## Ember v0.3.1 and playground update (2026-10-08)

- The public Vercel playground now runs the Ember v0.3.1 byte checkpoint for its next-word and two-letter completion modes. The existing learned integer-rule cellular architecture is retained.
- Exact integer clean byte+EOS equal-domain loss is 1.831770 bits/target versus 1.868322 for v0.2A (paired change -0.036552; 95% CI -0.038721 to -0.034442). The word heads are unchanged from v0.3a: 14.25% next-word top-1 and 48.00% two-letter completion top-1.
- The site now includes the copied landing, fixed ten-prompt rating set, and deterministic Flame-W replay from draft PR #22; Ember v0.3.1 links to its public GitHub package.

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
