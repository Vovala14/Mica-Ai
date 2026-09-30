# Flame-W full40 report

## Model and training

The owner-authorized second Flame-W training run completed all 40 native
integer-rule MICA rounds on the verified mix-A word corpus.

- Model: `train/best.mica`
- SHA-256: `ef969c0e96173fc11ecc3b5cf0d1d28d04e21fe846a0752297d5549e90f7e6b7`
- Size: 34,111,616 bytes
- Geometry: Flame F1, 16,384 word symbols, learned integer rules and integer
  readout; no neural network at inference
- Trainer-only best validation: 5.7861 bits/byte (not the final metric)

The preceding P1 run was SHA `cdb53096720b0c1b4c3149a000a9294dafb8d540b1886a4c96e7b822a06b7469`.

## Exact development metrics

All metrics use the exact exported integer word engine. Next-word and
completion use 2,000 fixed letter-model positions per set; bits/token counts
word tokens plus EOS from BOS.

| Set | Next-word top-1 | Next-word top-3 | Completion top-1 | Bits/token |
|---|---:|---:|---:|---:|
| Chat dev1000 (22,832 targets) | 20.70% | 31.50% | 43.64% (1,132 positions) | 6.9287 |
| Everyday dev_fresh1000 (10,622 targets) | 18.45% | 29.50% | 40.84% (1,185 positions) | 7.0222 |

Equal-domain means: next-word top-1 **19.58%**, completion top-1
**42.24%**, and bits/token **6.9754**.

Compared with the one-round P1 run (17.10% chat / 16.40% everyday next-word
top-1), full40 improves by 3.60 and 2.05 percentage points respectively.

On the identical 2,000 sampled positions, paired P1→full40 top-1
differences were also positive: chat **+3.60 percentage points** (95% paired
bootstrap CI **[+2.15, +5.05] pp**, 1,776 ties) and everyday **+2.05 pp**
(95% CI **[+0.60, +3.50] pp**, 1,785 ties). These intervals compare the two
development runs only; they are not a clean-set or sealed-test claim.

## Independent clean reporting

The clean sets were not used for selection. Exact word bits/token were:

- Chat clean val1000: **6.8414 bits/token** (22,660 targets)
- Everyday clean val1000: **6.9972 bits/token** (10,657 targets)
- Equal-domain clean mean: **6.9193 bits/token**

No sealed test was opened. These word-token bits are not numerically
interchangeable with the byte-level Flame bits/target.

## Unedited raw samples

Using the native word decoder `w-sent-mmi.3`:

- Prompt: `I was thinking about`  → `what you said.`
- Prompt: `The best part of the weekend was`  → `an accident and I just wanted to let you know that I need to make some changes.`

These two samples are evidence of decoding behavior only, not a blind quality
score. A 100-prompt blind judge has not yet been run for full40, so this run
cannot be declared a sentence-quality replacement for letter Flame yet.
