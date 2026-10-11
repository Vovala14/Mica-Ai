# MICA Ember v0.3.7

Ember is the byte-level MICA model: a learned integer-rule cellular automaton that reads text one UTF-8 byte at a
time. It suggests the next word and finishes the word you are typing. It is not a transformer and not a neural network,
and it is not a sentence generator.

## What changed since v0.3.5

1. **Word rules over the previous words.** Before each word readout, 16 extra MICA rule ticks run. Each previous word
   gets a fixed 16-value integer code (a hash of the word, the way the automaton keeps a 16-value code for every byte).
   A tick picks a page by comparing pairs of values, scores its 256 candidates with six ±1 terms plus an integer bias,
   and the winning rule writes six learned integers that the readout also reads. The terms read the codes of the
   previous three words and the automaton's probes at the current position, so a rule can combine "weather" + "today"
   + "is". It is the same rule mechanism the automaton uses; everything is integer arithmetic.
2. **A word memory.** The word readouts read the automaton's 240 probes at the current position and at the starts of
   the previous 2 words.
3. **A bigger vocabulary.** 32,768 words (v0.3.5: 8,192), all learned from training text.
4. **Completion after any number of letters**, with prefix indexes and exact float32 products of the integer weights.
5. **More training.** The word readouts and the rules' values were fitted together on the exact integer probe states of
   697,070 training texts (v0.3.5: 119,823).

The automaton, its byte readout and its 14 KB byte memory are unchanged from v0.3.5. Ember v0.3.6 is the step in
between (items 2–4, 299,227 training texts, no word rules); it was not released.

## Evaluation

Every choice was made on development prompts with a pass mark fixed before the run; the clean sets were used only for
reporting. The sealed test set was not accessed.

Word assistance, first 200 eligible prompts per clean domain (the same benchmark as v0.3a, v0.3.1, v0.3.5 and v0.3.6):

| Clean word benchmark (400 prompts) | v0.3.5 | v0.3.6 | v0.3.7 | v0.3.7 vs v0.3.6 (95% CI) | v0.3.7 vs v0.3.5 (95% CI) |
|---|---:|---:|---:|---:|---:|
| Next word, top-1 | 15.25% | 20.75% | 21.50% | +0.75 [−2.75, +4.00] | +6.25 [+2.50, +10.00] |
| Next word, in top 5 | 34.75% | 37.75% | 43.25% | +5.50 [+2.25, +8.75] | +8.50 [+4.50, +12.75] |
| Finish after 1 letter, top-1 | 43.50% | 45.00% | 50.75% | +5.75 [+2.50, +9.01] | +7.25 [+3.25, +11.25] |
| Finish after 2 letters, top-1 | 54.50% | 59.00% | 63.75% | +4.75 [+2.00, +7.75] | +9.25 [+5.25, +13.00] |
| Finish after 3 letters, top-1 | 73.00% | 73.75% | 77.25% | +3.50 [+0.50, +6.75] | +4.25 [+0.25, +8.00] |

Development prompts held back from every choice (739): next word 22.68% → 25.34% and two-letter completion
61.59% → 66.63% against v0.3.6 (mean of the two: +3.85 [+2.01, +5.73] points).

Older development prompts that no choice used (1,920; chat dev1000 and everyday dev_fresh1000, none of whose records
appears in the training texts), the same protocol, checked after the choice:

| Older development prompts (1,920) | v0.3.5 | v0.3.6 | v0.3.7 |
|---|---:|---:|---:|
| Next word, top-1 | 13.92% | 19.31% | 22.65% |
| Next word, in top 5 | 29.14% | 36.62% | 42.04% |
| Finish after 1 letter, top-1 | 38.77% | 44.38% | 49.39% |
| Finish after 2 letters, top-1 | 53.06% | 60.12% | 66.03% |
| Finish after 3 letters, top-1 | 72.19% | 76.66% | 81.37% |

Mean of next word and two-letter completion there: v0.3.7 vs v0.3.5 +10.86 [+9.48, +12.22] points, vs v0.3.6
+4.63 [+3.52, +5.73].

**What was tried and not kept.** Rules reading the automaton's probe states at earlier word starts (instead of word
codes) memorised the 299,227 training texts and did not pass on held-back prompts; with 697,070 texts they passed, but
the word-code rules scored higher on the selection prompts and, checked afterwards, on the 1,920 older development
prompts (+2.43 [+1.54, +3.34] points). One model with both rule kinds (32 ticks) was not clearly better than v0.3.7:
+0.42 points on the selection prompts, short of the +1.0 fixed beforehand, and +0.17 [−0.56, +0.90] on the older
development prompts. More training text alone, without the rules, was not clearly better than v0.3.6 (+0.65 [−0.28,
+1.57] points). A bigger byte memory (256 bytes, rank 32) matched but did not beat v0.3.5's byte model.

## Files

| File | What it is |
|---|---|
| `model.mica` | The automaton with its byte readout (unchanged from v0.3.5), SHA-256 `c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506` |
| `memory.npz`, `memory.py` | Integer byte memory for byte scoring (unchanged from v0.3.5), SHA-256 `3827c2a38981347bff17f9666543c333178c5a89fdd089d216e5c6f8ab8847c8` |
| `word_heads.npz` | Integer word readouts (32,768 words, 2 previous word starts) and the 16 word-rule ticks per readout, SHA-256 `40f3bfa6f9692f4424d05cc3666afdca7ecbd648244df059ce2d181f66561e67` |
| `word_model.py`, `mica_r1/` | CPU-only exact integer runtime (numpy only); checks the hashes when it loads |
| `test_word_model.py`, `test_memory.py` | Conformance tests |
| `results/` | Machine-readable evaluation summaries, the candidate choice rule and every pass mark, and the later checks (both rule kinds; older development prompts) |
| `CHECKSUMS.json` | SHA-256 of every file |

## Try it

```bash
python -m pip install -r requirements.txt
python test_word_model.py
python test_memory.py
python word_model.py "Thank you for "
python word_model.py --complete "Are you coming to the pa"
```

## Limitations

The vocabulary contains training words of four or more letters. Top-1 accuracy on these prompts does not measure
relevance beyond the exact target, sentence coherence or open-vocabulary coverage. The automaton itself still uses only
the last few bytes; the word rules give the suggestions their view of the previous words. Ember is a research model.
