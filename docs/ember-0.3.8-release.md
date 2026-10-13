# MICA Ember v0.3.8 — better word completion, same cellular architecture

Ember provides **next-word suggestions and typed-word completion**. Flame-W
handles sentence continuation. This is a research model, not a general assistant.

## Architecture

The Minimal Inference Cellular Automaton reads UTF-8 bytes into a ring of 768
cells with 112 integer channels. Sixteen phases select learned integer rules;
240 probes feed integer readouts. The unchanged word component retains 32,768
words, two previous-word-start anchors, and 16 native word-rule ticks whose six
signed terms read current probes and fixed codes of the previous three words.
Winner rules write six learned integer values. No neural layers, transformer,
GRU, external language model or corpus retrieval were introduced.

0.3.8 refits existing prefix2 context coefficients and biases, combines the two
existing word heads 9:7, and uses a training-calibrated 12/16 probability gain.
Cellular transitions, geometry, word-rule arrays, vocabulary, next-word head,
byte checkpoint and integer byte-memory are unchanged. Optional probabilities
use floating-point softmax; the scores and cellular updates remain integer.
Restricted prefix scoring and optional bounded typing caches avoid unnecessary
work. No new general speed benchmark is claimed.

## Matched comparison with actual published 0.3.7

After two typed letters; top1 includes out-of-vocabulary targets. Conditional
bits/word normalize over **only legal prefix words**, excluding OOV. These are
not bits/byte or unconditional language-model perplexities. Lower bits is better.

| Frozen word-stage confirmation | Prompts | Top1 0.3.7 → 0.3.8 | Conditional bits/word 0.3.7 → 0.3.8 |
|---|---:|---:|---:|
| TinyStories, varied word positions |2000|70.65% → 71.75%|2.530207 → 1.826396|
| Conversation, first eligible word |2000|80.40% → 81.05%|1.753580 → 1.268459|
| Everyday, first eligible word |1999|60.58% → 63.03%|4.240615 → 2.977478|

Equal-domain accuracy gain **+1.4004pp**, paired95%CI[+0.8503,+1.9345].
Conditional loss difference **−0.817356bits**, CI[−0.880697,−0.757175].
These are the existing frozen candidate-confirmation sets, transparently reused
for direct public-version comparison, not a newly blind release test. One of 2,000
raw everyday records has no eligible ASCII word; both models use the same1999.
Loss denominators are 1,997/1,972/1,982; one probe per normalized-unique record.

| Clean reporting only | Prompts | Top1 0.3.7 → 0.3.8 | Conditional bits/word 0.3.7 → 0.3.8 |
|---|---:|---:|---:|
| Conversation |968|68.90% → 70.56% (667 → 683)|3.228380 → 2.250338|
| Everyday |937|60.73% → 63.61% (569 → 596)|4.302341 → 3.026711|

Equal-domain clean accuracy gain **+2.2672pp**, CI[+1.2601,+3.3056].
Clean conditional loss difference **−1.126836bits**, CI[−1.262410,−0.993176].
Loss denominators 953/926. Earlier0.3.7 release tables used a smaller 400-prompt
sample and are not directly interchangeable with this full eligible-set table.

## Decoder versus newly fitted coefficients

Most gains over public0.3.7 come from the improved decoder and calibration.
Compared with ORIGINAL 0.3.7 weights using the same improved decoder, the new
coefficients lower confirmation conditional loss by 0.015394bits
CI[−0.022877,−0.008062], but the additional accuracy change is inconclusive
(+0.1668 pp, CI[−0.2333,+0.5502]). On clean sets, coefficient-only loss changes
by −0.022628bits CI[−0.037214,−0.008116]; chat accuracy is 0.7231 pp lower and
everyday 1.1740 pp higher than that decoder-only baseline. All domains improve
against the ACTUAL public0.3.7 version. This is not a claim that every step or
every prompt improves.

## What did not improve

Next-word/one-letter behavior and byte loss are unchanged by file/array identity.
Full2000-item public Tiny Theory of Mind: **524/2000=26.20%** with byte or
character length normalization (Wilson95%CI24.32–28.17%). Secondary word-token
normalization:545/2000=27.25%. The completion heads are unused by this no-hint
byte-scoring task, so0.3.7 and0.3.8 have identical scores. One exact full pass
plus byte-path identity audit; no claim of separate independent reruns or ToM
improvement. Do not compare normalization conventions without disclosure.
No sentence-coherence or free-generation improvement is established.

## Data and verification

Learning uses original training text only. Word-stage holdouts exclude original
word training, additional fitting/screens and earlier word holdouts, with
normalized-overlap checks; their source population was historically used for
byte-trainer validation. Candidate fixed before direct comparison; no clean
retuning. Paired record bootstrap 5,000 replicates; equal weight per domain.
Sealed completion test never accessed. Detailed hashes, outcomes, protocol and
ToM audit are in evidence/. Native runtime ranks/probabilities and invariant
components are checked separately in evidence/package_verification.json.

## Run locally

Only NumPy is required; use the provided native runtime, not AutoModel.

```bash
pip install numpy
python test_word_model.py
python test_memory.py
python word_model.py --complete "Please send me the add"
python word_model.py "Thank you for "
```

```python
from word_model import EmberWord
model = EmberWord()
print(model.complete("The little rabbit went into the fo"))
print(model.complete_with_probabilities("Please send me the add"))
print(model.next_words("Thank you for "))
```

Completion probabilities are conditional over ALL matching vocabulary words;
top 5 probabilities need not sum to 1. One-letter uses unchanged next-word scores.
Longer prefixes filter the same causal two-letter score. Vocabulary/English
coverage is limited, and suggestions can be wrong. MICA AI Non-Commercial
License: personal and research use; commercial use forbidden.

Byte checkpoint SHA256:
`c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506`

Word-head SHA256:
`44c60a2e64a5d671a03dfc162fca0c9c435b75c2059ebca95eec47ab4a2a92ce`

Byte-memory SHA256:
`3827c2a38981347bff17f9666543c333178c5a89fdd089d216e5c6f8ab8847c8`

[Playground](https://mica-ai-ten.vercel.app/) ·
[GitHub release](https://github.com/Vovala14/Mica-Ai/releases/tag/ember-v0.3.8) ·
[Hugging Face model](https://huggingface.co/vynly/mica-ember-0.3.8) ·
[Collection](https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton-6ac3f2f24567a92325cb45f2)
