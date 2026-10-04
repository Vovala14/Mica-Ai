# Ember v0.3a: word assistance

Ember v0.3a uses the unchanged v0.2A byte-level Minimal Inference Cellular
Automaton. The model reads UTF-8 bytes, updates its ring with learned integer
rules, and exposes the same 240 cellular probes. Two additional learned integer
readouts score a training-only vocabulary of 4,339 words: one at a word boundary
and one after two letters have been typed. At the second position, only words
matching the typed letters can be suggested. This is an output extension of
the existing automaton; it does not add a transformer, GRU, external language
model or retrieval at inference.

In the MICA model hierarchy, **Ember supplies word suggestions and typed-word
completion**. **Flame-W supplies sentence continuation and generation**. Ember
v0.3a is a research word-assistance release and is not a sentence generator.

## Files

- `word_heads.npz`: 1,233,028-byte packed integer word readouts, SHA-256
  `cc8e0135ac38d8afcb11bce34d226e96372b8bd2fe377cae9c6a51dadadaa5d5`.
- `word_model.py`: exact integer runtime using the existing v0.2A `best.mica`,
  SHA-256 `3b2a94e203fc6b0cd20365e4d5c46085035722f578161a2213b8cf148a126e72`.
- `manifest.json`: version, role, model hashes and provenance.
- `test_word_model.py`: hash, scalar probe and suggestion conformance check.
- `official_v02a_dev_report.json` and `official_v02a_clean_report.json`:
  paired machine-readable summaries.

## Matched results against official Ember v0.2A

The same training-only 5,000-word reference lexicon and beam-4 one-word decoder
were used with the v0.2A byte model. v0.3a used its fixed integer word readouts.
Both were tested on the same eligible prompts. Top-1 means the first suggestion
matches the next word or the word after two typed letters. Each set below has
200 chat and 200 everyday prompts. The difference gives chat and everyday equal
weight. Intervals are paired 95% record bootstrap intervals.

| Set | Task | v0.2A | v0.3a | Gain and 95% interval |
|---|---|---:|---:|---:|
| Disjoint development | next word | 5.25% | 13.50% | +8.25 pp [+5.50, +11.25] |
| Disjoint development | two-letter completion | 46.00% | 52.75% | +6.75 pp [+3.00, +10.50] |
| Independent clean | next word | 6.75% | 14.25% | +7.50 pp [+4.75, +10.50] |
| Independent clean | two-letter completion | 42.75% | 48.00% | +5.25 pp [+1.50, +9.00] |

The equal task/domain mean gained +7.50 pp [+5.125, +9.875] on development
and +6.375 pp [+4.0, +8.75] on clean prompts. Development selection was fixed
before the clean check. The chat/everyday development record hashes are
`061aed46cc155ebd0aca85dfa2a34c8406467ed727341981054aa658636f8027`
and `5d228442623901b70b8a16319805d878b0d392a47c7a1972b442e0487b31d8f8`.
The clean hashes are
`a8eb6d3553656a2cf78e8a2f61c66a4b02a6dd1a124f5f2b8b938a1d3c77423e`
and `c6107c8e80f28156c0314db180576e7311ff3f7a9e2843d711709c9a7d2f4d26`.
The sealed test was not accessed.

The cellular byte predictor itself is v0.2A: its exact clean loss remains
1.860907 bits per byte plus EOS on chat and 1.875736 on everyday. The 097b
experimental byte refit has lower byte loss but did not improve word top-1
under the earlier fixed decoder. This v0.3a change targets word assistance.

## Try it

From the repository root, with NumPy installed:

```bash
python ember/runs/ember_v03a_20261004/test_word_model.py
python ember/runs/ember_v03a_20261004/word_model.py
```

Or import `EmberWord` from `word_model.py` and call
`suggest("Thank you for ")` or
`suggest("Thank you for be", prefix2=True)`.

The vocabulary contains common training words only. These top-1 tests do not
measure relevance beyond the exact target, calibrated abstention, sentence
coherence or open-vocabulary coverage. The website now offers separate next-word
and two-letter completion routes backed by this integer readout.
