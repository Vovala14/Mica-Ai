# MICA community leaderboard submissions — 2026-10-12

Frozen released checkpoints, CPU only. No retraining or inference weight changes.
Scores are self-reported until the leaderboard maintainers verify and merge them.
See [PROTOCOL.md](PROTOCOL.md) for components, size accounting, initialization,
data splits, scoring and limitations. See `release_pins.json` for exact HF revisions.

## Open SLM: Flame-W 0.3.6

| Task | Split / count | Accuracy |
|---|---|---:|
| HellaSwag | validation / 10,042 | 27.6439% acc_norm |
| ARC-Easy | test / 2,376 | 27.2727% acc_norm |
| ARC-Challenge | test / 1,172 | 20.2218% acc_norm |
| PIQA | validation / 1,838 | 52.1763% acc_norm |
| ArithMark-3 | full / 1,000 | 27.10% per-word-token normalization |

The `lm_eval.api.model.LM` adapter runs the released integer word model with its
frozen answer-mode calibration, without the English belief adapter. Full splits,
zero-shot, no chat template. Native symbols casefold words and hash OOV words;
these are not lossless byte likelihoods. The release uses training-time SWA within
one training trajectory. Its conservative size proxy includes all stored numeric
entries, including fixed wiring; it is not a neural parameter-equivalence claim.

`flame_harness_summary.json` contains actual harness outputs and standard errors.
`flame_harness_evidence.json.gz` contains per-item scores, labels, outcomes and
document/request hashes without redistributing benchmark question texts.

### Reproduce without score cache (recommended for independent verification)

Obtain the pinned published [Flame-W 0.3.6 package](https://huggingface.co/vynly/mica-flame-w-0.3.6)
or [GitHub release](https://github.com/Vovala14/Mica-Ai/releases/tag/flame-w-0.3.6).
Internal project agents must use the project's audited, cache-first downloader.

```sh
pip install lm_eval==0.4.13 numpy pyarrow torch
python fetch_slm_data.py
python flame_lm_eval.py --package /path/to/flame-package --data data --workers 8
```

PyTorch is used by the evaluation infrastructure, not by MICA inference. The
native model remains the published NumPy/integer cellular automaton.

The completed harness reproduction used exact release likelihoods cached by
complete context/continuation hash. All package files were hash-verified, greedy
flags computed live, and 67 likelihoods re-evaluated live systematically; the
small smoke run recomputed every request. This is disclosed cache reuse, not a
claim of a second complete uncached engine evaluation. For the faster check:

```sh
python flame_lm_eval.py --package /path/to/flame-package --data data --workers 8 \
  --cache flame_likelihood_cache.json.gz --verify-every 1000
```

The initial custom ArithMark metric compared integer predictions against string
labels from JSON. It was corrected to cast the label, and all 1,000 ArithMark
items were rerun through the harness; the four standard task results were
unchanged. The final counts exactly reproduce the released evaluation.

## Tiny-ML: Ember 0.3.7 byte foundation

Fresh complete native CPU evaluation (1,170 seconds, 12 workers):

| Task | Full count | Result |
|---|---|---:|
| BLiMP | 67,000 pairs | 52.5597% (35,215 correct) |
| ARC-Easy | 2,376 questions | 25.9259% (616 correct) |
| WikiText-2 raw test | 1,292,007 scored bytes | byte perplexity 7.87131116; 2.97660397 bits/byte |

199 ARC items encounter the reference's 256-native-token input clipping.
`ember_glint_summary.json` contains dataset/model hashes and the exact protocol;
`ember_glint_evidence.json.gz` contains all per-item scores and outcomes.
All 29 release checksum files verified. These are new public-benchmark measurements,
not evidence of improvement over another model; no such comparison was run.

This configuration uses the released byte automaton + byte memory, with no word
suggestion/completion heads. It is labeled accordingly. For conservative size
accounting the proposed row counts the entire distributed package including
unused word heads (62,800,623 numeric scalars); the active byte component has
4,276,653 stored numeric scalars, including fixed wiring and metadata.

```sh
pip install numpy pyarrow
python fetch_slm_data.py
python prepare_glint_data.py
python ember_glint.py --package /path/to/ember-package --workers 12
```

It follows the pinned Glint-1.3 protocol: 67,000 BLiMP pairs; 2,376 ARC-Easy test
items; WikiText-2 raw test; 256-token windows; no BOS; no length normalization.
Each token is one UTF-8 byte. Unlike the normal BOS-start Ember API, the benchmark
starts a zero session and leaves the first byte unscored. Byte perplexity uses
actual scored bytes, with no estimated bytes-per-token factor. Results do not
measure Ember's dedicated word heads or sentence quality.

The optimized integer readout matched the exported engine's generic probe_scores
exactly; prefix sharing matched independent sequence scoring exactly. See
`ember_verification.json`, `ember_size_accounting.json` and PROTOCOL.md.

## Submission status

Open SLM submission: [PR #209](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/209). Tiny-ML PR link will be recorded after submission. Both require maintainer review. Reproduction code is in [GitHub PR #37](https://github.com/Vovala14/Mica-Ai/pull/37). The board
maintenance snapshots show actual model merges on October 12 (Open SLM) and
October 10 (Tiny-ML), with repeated additions on earlier days.

MICA weights/runtime retain the [MICA AI Non-Commercial License](https://huggingface.co/vynly/mica-flame-w-0.3.6/blob/main/LICENSE).
Benchmark datasets and the externally downloaded reference script retain their
original licenses. No third-party model weights are included here.
