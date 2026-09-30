# MICA Ember (byte/letter model)

Ember is the byte/letter-level MICA model: an integer cellular automaton that
reads and writes one UTF-8 byte at a time (258 symbols, original c256 lag64
geometry).

## Official model: v0.2 balanced mix A

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
