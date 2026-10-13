# MICA code handoff integration

Integrated on `codex/mica-handoff-20261013` from current main
`dd729762f233789559e4314a26538fc3210714c1`. No publication/deployment.

## Adaptations

- Applied corrections-only ingestion, bounded deterministic correction mixing,
  their tests and preparation documentation from improvements.patch.
- Ported runtime work to Ember037 instead of replacing its word memory/rules
  with the old031 class. Imported the previously verified exact prefix-scoring
  and bounded-session implementation; added case-insensitive unique suggestions.
- Current037 already accepts arbitrary-length prefixes and returns the correct
  missing-letter suffix. Kept this backend behavior, current model versions,
  Flame-W036 features and frontend layout. Updated the completion explanation.
- Fixed full-word correction ingestion for current037 one-letter prefixes too.
- Integrated previously confirmed9:7 existing-head ranking and gain12/16
  probability calibration as a separately pinned completion_policy.json.
  Next-word and one-letter rankings remain unchanged. Returned probability
  metadata identifies the actual policy, including unscaled one-letter behavior.
- Archived candidate-experiment.patch unchanged, with LOCAL_HANDOFF.md explaining
  failed gates and reproduction limits. The runnable candidate ZIP mentioned by
  the outer handoff README was absent; raw corpora/features/weights also absent.
  Those experiments were not rerun, retrained or promoted.

## Verification

- Full `python -m pytest r1/tests -q`:178 passed plus11 subtests.
- Native Ember037 packaged word/model conformance passed.
-136 real-prompt ranking comparisons and136 probability comparisons match the
  previously verified local decoder;32 native next-word full-int64-score checks,
 8 bounded native typing sessions, and a real local backend completion passed.
- Checkpoint, word-head and byte-memory SHA256 unchanged; policy hash matches
  the accepted local calibration artifact. No GPU or sealed dataset used.
- `git diff --check` clean. See ember-handoff-runtime-verification.json.

Source artifacts and full score protocol are documented in
ember-completion-calibration.md. These are integration/parity checks, not a new
quality experiment. Conditional word loss must not be called byte loss, and
probability calibration must not be called an additional accuracy or ToM gain.

The corpus mixer enforces encoded exact exclusions and a maximum5% correction
share by both records and targets includingEOS. It does not detect paraphrases
or partial-record overlap; preparation establishes no quality improvement.
