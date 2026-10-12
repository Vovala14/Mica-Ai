# Maintained leaderboard submissions — complete

Date: 2026-10-12. Owner requested suitable leaderboards with continual, recent updates and actual PRs.

## Submitted

| Board | Latest observed maintainer model additions | Model / configuration | PR | Verified state |
|---|---|---|---|---|
| Open SLM | October 12, 11, 10, 9, 8 | Flame-W 0.3.6 word LM + frozen answer calibration, no belief adapter | [#209](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/209) | Open, not merged |
| Tiny-ML | October 10, 9, 8, 6 | Ember 0.3.7 byte foundation + released byte memory, no word heads | [#97](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard/discussions/97) | Open, not merged |

Both boards direct submissions through Hugging Face Space PRs; these are real PRs, not ordinary discussion posts. Their underlying leaderboard code is hosted on HF, so the inclusion PRs are there. The reproducible scripts/evidence are also available in [GitHub PR #37](https://github.com/Vovala14/Mica-Ai/pull/37), left open. No PR was merged and no deployment occurred.

Actual activity is recorded in maintenance_activity.json, from main-branch commit histories, not search crawl dates or the timestamps of our own submissions. TinyToM's standalone table, old curated lists, and incompatible or closed competitions are not counted toward this request.

## Exact results

Flame-W automaton SHA256 `eca9faee6f44e3b2e4c10399ec99d202fa941a10ef78e64fb2094b0de70a4616`.
All 67 release manifest files verified. lm-eval 0.4.13 canonical task preprocessing/metrics, zero-shot, entire official splits; official token-normalized ArithMark-3 adapter.

| Task | Correct / full count | Score |
|---|---:|---:|
| HellaSwag acc_norm | 2776 / 10042 | 27.643896% |
| ARC-Easy acc_norm | 648 / 2376 | 27.272727% |
| ARC-Challenge acc_norm | 237 / 1172 | 20.221843% |
| PIQA acc_norm | 959 / 1838 | 52.176279% |
| ArithMark-3 per-word-token normalization | 271 / 1000 | 27.10% |

Full-split harness reproduction reused exact release likelihoods keyed by complete request hashes, verified against all package file hashes; greedy flags computed live and 67 likelihoods re-scored live, plus the all-live smoke run. All counts reproduce the independently checked release evaluation. It is not falsely described as a fresh full uncached model run. An uncached native LM adapter and pinned source data downloader are supplied for maintainer verification. Initial ArithMark string-label type issue corrected and the complete 1,000-item task rerun.

Ember automaton SHA256 `c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506`; memory SHA256 `3827c2a38981347bff17f9666543c333178c5a89fdd089d216e5c6f8ab8847c8`. All 29 checksum files verified. Fresh uncached CPU run completed in 1,170 seconds, 12 workers, reference Glint-1.3 protocol with native UTF-8 byte tokens.

| Task | Full count | Result |
|---|---:|---:|
| BLiMP | 67000 pairs | 35215 correct / 52.559701% |
| ARC-Easy | 2376 questions | 616 correct / 25.925926% |
| WikiText-2 raw test | 1292007 scored bytes | byte perplexity 7.87131116 / 2.97660397 bits per byte |

Reference no-BOS initialization and first-256-token clipping are explicit; 199 ARC questions are clipped. Native generic-readout and independent-prefix checks both match exactly. The word suggestion/completion heads are not evaluated. No comparison or model improvement is inferred from these new benchmark results.

## Qualifications and verification

Both submissions disclose that MICA is non-neural and ask maintainers to confirm sizing/checkpoint eligibility before merge. `params` uses conservative numeric stored-value counts, not a claim of neural trainable-parameter equivalence. Flame counts 31,523,432 values including wiring; Ember conservatively counts the entire package, including its unused word heads, at 62,800,623 (active byte component: 4,276,653). Training-time averaging is within a single trajectory, not a merge of independent pretrained models. The published noncommercial research/personal license is stated.

Uploaded files were fetched back from each PR branch and matched exactly; statuses verified as actual open PRs. Inline JavaScript passed node --check and isolated model/org data checks. Open SLM diff: 2 additions, 0 removals. Tiny-ML adds the model/maker/color; one original trailing comma is required. Per-item evidence and model/dataset hashes are public in the GitHub reproduction PR.

No native training/shared code or weights edited, no GPU use, no retraining, no sealed MICA completion test. CPU evaluations finished; no evaluator remains active. No HF model/config/weight requests were made: existing local release artifacts were used, plus metadata APIs and public benchmark data/docs. Public download counters were not artificially increased.
