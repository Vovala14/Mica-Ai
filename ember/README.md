# MICA Ember (byte engine, word assistance)

Ember is the byte/letter-level MICA model: an integer cellular automaton that
reads and writes one UTF-8 byte at a time (258 symbols, original c256 lag64
geometry).

## Latest release: v0.3.5

[Ember v0.3.5](runs/ember_v035_20261011/) averages the automaton's learned integer rule values and readout
over 60 live training rounds (the Flame-W 0.3.5 method), refits the byte readout and the 14 KB integer
memory on 119,823 training records, and retrains both word readouts (8,192 words) on the same
records. Clean word benchmark: two-letter completion 48.00% → 54.50% (paired 95% CI
+3.0 to +10.0 points); next-word top-1 14.25% → 15.25% (within noise). Clean equal-domain byte+EOS loss
1.831770 → 1.810055 bits/target (paired change -0.021715; 95% CI
-0.023154 to -0.020275). The website runs v0.3.5.

## Word-assistance version: v0.3a

[Ember v0.3a](runs/ember_v03a_20261004/) uses the unchanged v0.2A cellular
checkpoint with two additional learned integer readouts over the same cellular
probes. It suggests a next word or completes a word after two typed letters.
The packed readouts add 1.23 MB. On independent clean prompts, next-word top-1
rose from 6.75% to 14.25%, and two-letter completion from 42.75% to 48.00%,
compared with v0.2A and the same training-only word vocabulary/beam protocol.
The paired combined improvement is +6.375 percentage points [95% CI +4.0,
+8.75]. See the [v0.3a report](runs/ember_v03a_20261004/README.md) for
model hashes, prompt counts, domain scores and limitations.

Ember owns word suggestions and typed-word completion. Flame-W owns sentence
continuation and generation. The first website word interface used the v0.3a
heads; the current site runs the v0.3.1 checkpoint described below.

## Latest byte/readout release: v0.3.1

[Ember v0.3.1](runs/ember_v031_20261008/) keeps the learned integer-rule cellular
architecture and the existing word heads, while updating the byte checkpoint
and adding a compact integer memory sidecar for byte scoring. On the exact
integer clean byte+EOS evaluation, its equal-domain mean is 1.831770 bits/target,
an improvement of 0.036552 bits/target over v0.2A [paired change -0.036552,
95% CI -0.038721 to -0.034442]. The word-assistance metrics are unchanged from v0.3a: 14.25%
next-word top-1 and 48.00% two-letter completion top-1. The website uses the
v0.3.1 checkpoint and word readouts for its interactive word modes; the memory
sidecar is for the byte-scoring runtime.

## Cellular base: v0.2 balanced mix A

`runs/codex_ember_balanced_v02a_20260928/train/best.mica`
SHA-256 `3b2a94e203fc6b0cd20365e4d5c46085035722f578161a2213b8cf148a126e72`, 4,467,396 bytes.

Trained on 40 native rounds of a byte-balanced mix: conversation 45%, everyday 40%, TinyStories 15%.
It was selected over mix B on the development gate. Scores come from the exact
integer engine in bits/target, counting bytes plus EOS from BOS (see `REPORT.md`):

| Set | Bits/target |
|---|---:|
| Chat dev1000 | 1.8593 |
| Everyday dev_fresh1000 | 1.8680 |
| Chat clean val1000 | 1.8609 |
| Everyday clean val1000 | 1.8757 |

Greedy output (R1 §8 reference decoder, reproduced from this repo):

```
python r1/generate_bytes.py ember/runs/codex_ember_balanced_v02a_20260928/train "I don't know"
I don't know what to do that the street.
```

This is a research model. It reuses stock phrases and makes grammar errors.
