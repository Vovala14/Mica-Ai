# MICA Ember v0.3.5

Ember is the byte-level MICA model: a learned integer-rule cellular automaton that reads text one UTF-8 byte
at a time. It suggests the next word and finishes a word after its first two letters. It is not a transformer
and not a neural network, and it is not a sentence generator.

## What changed from v0.3.1

1. **A weight-averaged automaton.** Starting from the 097b native refit of Ember's learned integer rules,
   60 more training rounds were run (40,000 records each, round rate 0.01) and the learned integer rule values
   and readout were averaged over those rounds. This is the stochastic weight averaging that produced Flame-W 0.3.5.
   The geometry, wiring and file size are unchanged.
2. **A refit byte readout and memory.** On the averaged automaton, the integer byte readout and the 14 KB integer
   memory sidecar were refit together on the exact integer probe states of 119,823 training records, with the
   recipe that produced v0.3.1 (selected on development data: the running average after 4 of 8 epochs).
3. **New word readouts.** The next-word and two-letter-completion readouts were retrained over the new automaton's
   240 cellular probes on the same 119,823 records (the previous heads used about 8,000), with a
   training-only vocabulary of 8,192 words (selected on development data: weights averaged over epochs 2 to 4).

## Evaluation

All selection used disjoint development data (dev2 first 500 records per domain for bytes; dev2 word prompts for
the word readouts). The clean sets were scored once, after selection. The sealed test set was not accessed.

Exact integer engine, bits per target byte including EOS (lower is better):

| Clean set | Ember v0.3.1 | Ember v0.3.5 | Paired change (95% CI) |
|---|---:|---:|---:|
| Chat val1000 | 1.827462 | 1.807137 | -0.020325 [-0.022023, -0.018659] |
| Everyday val1000 | 1.836078 | 1.812973 | -0.023105 [-0.025522, -0.020739] |
| Equal-domain mean | 1.831770 | 1.810055 | -0.021715 [-0.023154, -0.020275] |

On the independent last-500-per-domain development confirmation the equal-domain change was
-0.020635 bits/target [-0.022641, -0.018629].

Word assistance, first 200 eligible prompts per clean domain (the same benchmark as v0.3a and v0.3.1):

| Clean word benchmark (400 prompts) | Ember v0.3.1 | Ember v0.3.5 | Paired gain (95% CI) |
|---|---:|---:|---:|
| Next word, top-1 | 14.25% | 15.25% | +1.00 [-1.75, +3.75] points |
| Finish a word after two letters, top-1 | 48.00% | 54.50% | +6.50 [+3.00, +10.00] points |

The completion gain is significant. The next-word change on these 400 clean prompts is within noise; on the larger
development set (1,200 prompts, used for selection) it was +3.08 [+1.67, +4.58] points.

## Files

| File | What it is |
|---|---|
| `model.mica` | The automaton with its byte readout: int8 rules, probes and readout. 4,467,396 bytes, SHA-256 `c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506` |
| `memory.npz` | Integer byte-memory sidecar for byte scoring, SHA-256 `3827c2a38981347bff17f9666543c333178c5a89fdd089d216e5c6f8ab8847c8` |
| `word_heads.npz` | Two integer word readouts over the 240 cellular probes (8,192 words), SHA-256 `1f73aa859fa9b141bdeb5a87c867cdb639a2d885166e3fd124d72989a05ee9ba` |
| `word_model.py`, `mica_r1/` | CPU-only exact integer runtime (numpy only); checks the hashes when it loads |
| `memory.py` | Validates and applies the integer memory tables |
| `test_word_model.py`, `test_memory.py` | Conformance tests: hashes, one exact probe state, known suggestions, memory behaviour |
| `results/` | Machine-readable evaluation summaries |

## Try it

```bash
python -m pip install -r requirements.txt
python test_word_model.py
python test_memory.py
python word_model.py "Thank you for "
python word_model.py --complete "Are you coming to the pa"
```

## Limitations

The vocabulary contains common training words only. Top-1 accuracy on these prompts does not measure relevance
beyond the exact target, sentence coherence or open-vocabulary coverage. Ember is a research model; its raw
byte-level generation is not coherent and is not offered as a feature.
