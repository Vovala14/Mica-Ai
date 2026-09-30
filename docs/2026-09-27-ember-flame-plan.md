# MICA Ember v0.1 and MICA Flame v0.1: plan

Owner's decision, 27 September 2026, about 21:00 UTC. Written by Claude for
Codex and the owner. Replaces the rule-selection experiments as the main line
of work.

## Goals

1. **MICA Ember v0.1.** Retrain the 4,467,396-byte MICA (the current
   c256, lag-64 geometry, unchanged) on a much larger corpus of text that
   looks like what people type: conversations, messages, everyday speech.
   Then measure where it stands.
2. **MICA Flame v0.1.** In parallel, build a separate, larger MICA. Size
   does not matter for Flame. Its goal is very good autocomplete of
   everyday speech, and generating new sentences that make sense. It must
   remain the minimal-inference cellular automaton: learned integer rules,
   byte tape, integer readout, and no neural network at inference.
3. **Also:** only show suggestions when the model is confident; stop stock
   phrases winning everywhere (both decoder-side); keep the gentle low-rate
   refits.
4. **Paused:** longer rule offsets, and single-rule or small-block selector
   searches. Seven were rejected today.

## Who does what

| Step | Owner | Waits for |
|---|---|---|
| Shortlist corpus sources with verified licences; owner approves | Claude | owner's answer |
| Corpus build script plus a reference build in the cloud (SHA-256 of every output) | Claude | approval |
| Run the build script on the PC and check the hashes (or receive the files) | Codex | script |
| **Flame scaling pilots on the GPU**, on the current everyday corpus | Codex | nothing: can start now |
| File-format fix for more than 6 scoring terms (`spec.py`, `serialize.py`, tests), as a patch | Claude | nothing |
| Ember v0.1 retrain on the new corpus (GPU) | Codex | corpus |
| Baselines on the new corpus: KenLM 5/6/7-gram (cloud); tiny transformer (GPU, launched by Codex) | Claude / Codex | corpus |
| Decoder: confidence gating and anti-stock-phrase ranking | Claude | nothing |
| Flame long run on the new corpus (GPU) | Codex | pilots and corpus |
| Independent check of every candidate: exact bits, word metrics, blind 100-prompt judge | Claude | models |

## Corpus (shared by Ember and Flame)

Licences below were checked on the dataset pages; the owner approves which
ones to use.

| Source | What it is | Licence |
|---|---|---|
| SODA (AllenAI) | 1.5 M everyday social dialogues, machine-generated | CC BY 4.0 |
| Taskmaster-1 and -2 (Google) | about 30 k human task dialogues (ordering, bookings, films), spoken and written | CC BY 4.0 |
| Schema-Guided Dialogue (Google) | 20 k+ human-assistant dialogues | CC BY-SA 4.0 |
| Topical-Chat (Amazon) | about 10 k human-human chats on topics | CDLA-Sharing 1.0 |
| TinyStories | millions of simple, coherent short stories, machine-generated | CDLA-Sharing 1.0 |
| DailyDialog | 13 k human everyday dialogues | CC BY-NC-SA 4.0 (non-commercial) |
| EmpatheticDialogues | about 25 k human conversations | CC BY-NC 4.0 (non-commercial) |
| current everyday corpus | Tatoeba, COCO captions (to be cut back), filtered prose | as before |

Records: one utterance per record, plus some two-turn pairs joined by a
newline, at most 256 bytes; plain text without speaker tags; exact
duplicates removed. Splits are made **by dialogue**, so no dialogue appears
in two splits.

Evaluation sets:
- the existing clean everyday val1000, unchanged, for comparison with every
  earlier number;
- a new clean chat val1000 from held-out dialogues, never in training.

## Ember v0.1 recipe (geometry unchanged, 4,467,396 bytes)

Use the recipe that produced d6cedad8:
- the lag-64 tape layout, probe window 64;
- `--rule-max-back 1,2,2,3,2,3,3,4,1,2,3,4,2,3,4,4`;
- fit mode, 40,000 fit records, 1,500 fit steps;
- 40 rounds at `round_lr` 0.3;
- then low-rate (0.01) refits on fresh samples until the development set
  stops improving.

Accept on a fresh dev1000 from the new corpus's validation split. Report
both clean sets. A round now takes about 50 s, so the whole run is about an
hour.

## Flame v0.1 scaling pilots

Run each pilot for one round, with the same seed, records and budget as the
control, on the current everyday corpus (the ranking of the knobs should
carry over to the new corpus). The control is the one-round lag-64 run
(`24149019…`), 1.797631 on the clean everyday val1000. Run `--dry-run`
first for memory. The soft model of F1 or F2 is about 4 times Ember's 814 MB.

| Pilot | Change | Rule book | Why |
|---|---|---|---|
| F1 | `MICA_CANDIDATES=1024` | 16.8 MB | finer partition of contexts on every page |
| F2 | `MICA_ROUTING_CHANNELS=0,1,2,3,4,5,6` (7 routing bits, 2,048 pages) | 16.8 MB | the page encodes the current byte more finely, freeing scoring terms for older bytes |
| F3 | wider readout: `--work-lags 1:16,2:16,3:8,4:8,8:8,16:8`, `MICA_PROBE=432` | +0.2 MB | rule features read at more lags, a way to use longer context |
| F4 | `MICA_PHASES=32`, `MICA_TICKS=32`, `MICA_CHANNELS=208` | 8.4 MB | twice as many rule features per byte |
| F5 | 12 scoring terms (after Claude's format fix) | 6.3 MB | a rule can read far bytes without giving up near ones; both long-offset pilots failed on exactly that trade |

Gate: keep a knob that gains at least 0.02 bits on the clean everyday
val1000 and on a fresh dev1000. Then combine the winners and train Flame
long on the new corpus. If the combination does not fit in VRAM (F1 with F2
is about 16 times the soft selectors), fit mode needs a memory-lean path
that keeps selectors as integer indices instead of logits. Claude can draft
it.

## Reporting (unchanged)

- Exact integer engine; bytes plus EOS, from BOS.
- Every score names the model path and SHA-256.
- Paired record bootstrap for every difference.
- Decisions are made on development sets; clean sets are for reporting only.
- Raw samples with every report.
- The sealed completion test set stays untouched until a release.

## Codex's question: a residual objective for later selector work

Paused for now; kept here for when it resumes. Score a proposed rule change
by the drop in the current model's loss, not by entropy of the winner alone:
1. Keep the current logits `z` at every training position as an offset.
2. At every position the change hands to candidate `c`, replace the old
   winner's contribution `W v_old` with `W v_c`. Refit only `v_c` (a few Adam
   or Newton steps) on all positions that `c` now wins.
3. The score is the sum, over those positions, of
   `log softmax(z − W v_old + W v_c)[y] − log softmax(z)[y]`.
4. Accept a page only if this gain holds on a disjoint training screen.
This measures what the readout does not already know.
