# MICA Flame-W 0.3.4

**Experimental cellular belief-memory release — 10 October 2026.**

0.3.4 packages the original B-740 learned integer-rule Minimal Inference
Cellular Automaton, the unchanged 0.3.3 language-memory banks and decoder, and
an additional **32 × 256-site radius-one integer cellular belief-memory plane**.
Learned bounded integer rules decide COPY/SKIP for agent–object–place evidence.
An external deterministic English adapter binds story mentions and choices.
The additional plane is **not compiled into B-740's native rule VM**.
There is no transformer, GRU, external language model or retrieval generator.

The new feature is four-choice question answering. **Sentence generation,
language-model probabilities and byte/word loss are unchanged from 0.3.3.**
This is not a general reasoning assistant or a sentence-quality improvement.

## Results and limits

| Measurement | Comparator | 0.3.4 |
|---|---:|---:|
| Fresh controlled supported development, previous private prototype | 899/1,000 (89.9%) | 965/1,000 (96.5%) |
| Complete supported scenario pairs, memory-only | — | 465/500 |
| Unsupported prompts receiving definite memory answers | 0/200 | 0/200 |
| Public Tiny ToM, published 0.3.3, per word | 636/2,000 (31.80%) | 638/2,000 (31.90%) |
| Public Tiny ToM, published 0.3.3, per character | 633/2,000 (31.65%) | 635/2,000 (31.75%) |

Controlled development adds 66 correct answers with no previously correct
answers lost; paired scenario-bootstrap 95% interval **+5.1 to +8.2 percentage
points**. Training used 24,000 independently generated stories/72,000 location
links, following 12,000 stories/34,000 event labels for the frozen cellular
update rule. Fresh names, objects, locations and surface forms, zero normalized
train/development story overlap. The authored finite semantic grammar is related
across splits; these scores do not measure unrestricted English.

Public ToM adds only two correct answers versus published 0.3.3, paired 95%
interval **−0.10 to +0.30 percentage points**. It is **unchanged from the previous
private prototype**: zero new answers. Only 12 of 2,000 items use a definite
memory answer (five correct); others use the exact packaged 0.3.3 word-model
fallback. No statistically established public ToM gain; the >33% goal is unmet.
The benchmark has been measured repeatedly and is descriptive, not a fresh
selection gate. No benchmark labels/templates were used for training; no sealed
test was accessed. The five-task leaderboard was not rerun.

## Try it

- [Live website](https://mica-ai-ten.vercel.app/#belief): story, question, four choices.
- [Download the standalone package](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.4).

Standalone package:

```bash
pip install numpy
python generate.py "I was thinking about"  # same sentence decoder as 0.3.3
python answer.py < request.json
python run_tom.py --data tiny_theory_of_mind_2000.jsonl --out results.json
```

`request.json` accepts `context`, optional `question` and exactly four `choices`.
If `question` is omitted, the last sentence boundary separates story and query,
as in public Tiny ToM's full-context completion protocol. Ambiguous binding or
unsupported queries fall back to the unchanged exact integer word model with
the released answer-mode memory. Endings use mean word-token log likelihood,
without EOS; the ToM script also reports character normalization.

From a repository clone, `python flame/runs/codex_flamew_034_20261010/answer.py < request.json`
uses the same website backend. The original standalone 0.3.3 checkpoint and
sentence-generation dependencies are included in the downloadable archive.

## Identity and architecture

- Native B-740: `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`
- Cellular update/query rule: `9a0ad7b1cc0c607176092d333f29f1ce4b993f460fde1a57fab338aca1400a9c`
- Trained grounding rule: `0a2edf02857ca8a021aadb4a0e1f9d7b4cc918957e2ff32cfaabdb6b4e981c0c`
- Existing checkpoint + three language-memory banks: **36,132,566 bytes**, unchanged.
- Added learned JSON parameters and runtime files are itemized in `MANIFEST.json`.

The ring update rule is local, homogeneous and integer. Input text binding and
tagged-cell readout are external interfaces; scoring/log probability conversion
and the existing sentence search may use floating-point normalization. Do not
describe the complete Python wrapper as having no floating point anywhere.

See `RESEARCH_REPORT.md`, `component_summary.json`, `comparison_summary.json`,
`tom_results.json`, `package_verification.json` and `package.json` for the exact
protocol, per-item predictions, hashes and limits. The research report predates
this owner-authorized release and records the original no-promotion decision;
this README describes the later explicit decision to publish the prototype.

Personal and research use only. Commercial use is forbidden by the repository
[license](../../../LICENSE).
