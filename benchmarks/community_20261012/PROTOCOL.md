# Frozen leaderboard submissions, 2026-10-12

No training, tuning, model promotion, website deployment or sealed MICA completion-test use. Public benchmark test splits are evaluated as required by their boards. These are self-reported results pending maintainer verification.

## Targets and actual recent activity

* [AxiomicLabs/Open_SLM_Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard): model submissions merged October 12, 11, 10, 9 and 8. Snapshot aa3ed1ebbdd1912e616413804f5a07a72324147a. The pinned discussion #162 requests actual PRs containing measured results.
* [Glint-Research/Tiny-ML-Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard): new model PRs merged October 10, 9, 8 and 6. Snapshot 8562ace3ea37e1f190837154a4d4180be5e0c095. Uses its own benchmark protocol, which differs from Open SLM.

Old curated-list PRs are not counted as leaderboard submissions. Tiny Theory of Mind's dataset table is a useful benchmark, but its independent table has much less ongoing submission activity; it is not a target for this request. Closed competitions and Transformer-only loaders do not qualify merely because the model is small.

## Flame-W 0.3.6: Open SLM

Native integer automaton model.mica SHA256 eca9faee6f44e3b2e4c10399ec99d202fa941a10ef78e64fb2094b0de70a4616. All 67 files in its release manifest verified locally. Pinned release and source dataset revisions are in release_pins.json and slm_data_manifest.json.

Configuration: word automaton + released blended short/long integer memories + frozen answer-mode calibration + previous-word association bank + phrase-copy table. **No English belief adapter, no answer override, no retrieval, no external neural model.** Four standard tasks use lm-eval 0.4.13 canonical task YAML, prompt preprocessing and metrics. Only dataset loading changes to pinned local JSON copies of the exact official splits. ArithMark-3 uses its official per-model-token normalization via a custom harness task. Zero examples, no chat template. All rows are scored; no subset scores are submitted.

The frozen original per-choice likelihood cache is keyed by complete context/continuation hashes, bound to the entire package MANIFEST, and derived from the already independently reproduced release evaluation. The new LM subclass consumes those exact cached likelihoods, computes greedy flags live and re-scores every 1,000th request live; uncached requests are always evaluated live. The cache is optional: omit --cache to recompute every likelihood. The smoke check recomputes every likelihood on the first four items of every task. Results must reproduce the published counts before submission. Score cache use is disclosed, not presented as a new full uncached evaluation.

Whole-word tokens casefold text; unknown words map to 256 stable hash buckets. Choice scores are native symbol likelihoods. This is explicitly disclosed rather than treated as lossless byte compression. The Open SLM row's size is a **conservative count of all stored numeric scalar entries**, including fixed wiring and metadata (31,523,432), not a claim of 31.5M neural trainable weights. The automaton was trained from scratch; SWA averages rule immediates/readouts within one training trajectory, not independently pretrained models. The license is noncommercial research/personal use, not Apache/MIT.

## Ember 0.3.7: Tiny-ML

Byte automaton SHA256 c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506, released memory checked by SHA. Frozen byte foundation only; the optional 32,768-word suggestion/completion heads are not invoked. A separate entry label makes this distinction visible.

Reference: Glint-Research/Glint-1.3 benchmark.py at c3f99e246aa2c64382f9668dc864533617482d0a. Exactly its 67 BLiMP configurations / 67,000 pairs, strict good-score > bad-score, sentence inputs truncated to first 256 native tokens, unnormalized likelihood, no BOS and no scored first token. ARC-Easy test 2,376 records, score question + space + choice minus the constant question likelihood, first 256 tokens, no length normalization. WikiText-2 raw test joined by a single space and stripped, 257-token chunks at stride 256, no cross-chunk state. Native tokens are UTF-8 bytes, so byte perplexity is exp(total NLL / scored bytes), with no guessed bytes-per-token constant. EOS remains in the predictive alphabet but is not a target; BOS is excluded. The native engine's zero session is used to honor the benchmark's no-BOS protocol. This initialization differs from the normal MICA BOS-start API and is explicitly part of the submitted configuration.

Independent verification checks the optimized runner's readout against generic exported engine.probe_scores and checks prefix sharing against independent sequence scoring (both max error zero). All VM/state/memory/readout operations are the released integer engine; final log-softmax is float64. Sentence scoring shares exact prefixes only as a computation optimization. No training or inference weights are changed.

Size accounting reports both the active byte engine/memory (4,276,653 numeric scalars, includes fixed wiring) and the full distributed package including inactive word heads (62,800,623 numeric scalars). The row conservatively uses the **full-package count**, so the unused heads do not create an artificial size bonus. These counts are not neural parameter equivalence. Model/runtime files keep their published license.

## Release/result claims

PRs are proposals, not admission or independent verification. No public rank is claimed before merge. Scores on these tasks do not establish coherent generation or human-level reasoning. ToM's hybrid 33.20% result is not substituted into plain-language-model leaderboards.
