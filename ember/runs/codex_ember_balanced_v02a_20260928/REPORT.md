# Ember v0.2 balanced mix A: exact integer evaluation

## Model and provenance

- Original c256 lag64 learned integer-rule Minimal Inference Cellular Automaton, unchanged inference geometry.
- Exported model: `train/best.mica`, SHA-256 `3b2a94e203fc6b0cd20365e4d5c46085035722f578161a2213b8cf148a126e72`, 4,467,396 bytes.
- Corrected chat corpus v2 manifest SHA-256 `15d19d54519ce322c310fd78b6ab640979c667992c77ce7a844e35438bf0c652`; mix A manifest SHA-256 `74158d323157828c7f6d2fff511e65a9a9447b6c86b52cc6958be59b625fcd8a`. Target byte shares: conversation 45%, everyday 40%, TinyStories 15%. All input hashes and normalized held-out disjointness passed preflight; sealed test was not read.
- 40-round native fit finished. Trainer validation 1.7045 was preliminary. All results below use the exported exact integer C engine, pooled bytes plus EOS from BOS. Each set has 1,000 records. Differences use 5,000 paired record bootstrap resamples.

## Development scores for A/B selection

| Disjoint development set | Targets | A bits/target |
|---|---:|---:|
| Chat dev1000 | 96,543 | 1.859332978 |
| Everyday dev_fresh1000 | 44,938 | 1.867953661 |

Equal-weight mean: **1.863643319**. Mix B is still training; this is not an A/B selection.

## Clean reporting sets

| Clean set | Targets | A | Comparator | A minus comparator, 95% paired CI |
|---|---:|---:|---:|---:|
| Chat val1000 | 94,901 | 1.860907270 | F1 Flame SHA `15767181…`: 1.786366261 | +0.074541009 [+0.066890385,+0.082318110] |
| Chat val1000 | 94,901 | 1.860907270 | Ember v0.1 refit SHA `f863bf61…`: 1.847477932 | +0.013429337 [+0.006574956,+0.020520149] |
| Everyday val1000 | 44,795 | 1.875735890 | F1 Flame SHA `15767181…`: 1.965974742 | -0.090238852 [-0.102695649,-0.077679285] |
| Everyday val1000 | 44,795 | 1.875735890 | Mature everyday MICA SHA `d6cedad8…`: 1.742303318 | +0.133432573 [+0.122612619,+0.144554255] |

Balancing improves everyday loss against chat-trained Flame and Ember, but A does not beat Flame on chat or the mature everyday checkpoint on everyday. Clean sets are reporting evidence, not the A/B selection criterion. Machine-readable per-record evidence and comparison SHA identities are in `eval_*.json`, `eval_results.json`, and `comparison_prior.json`.

## Unedited raw English generations

These are full reference greedy outputs from the exact integer model:

```text
I don't know what to do that the street.
The doctor and said, "I will be a good to help you think that the street.
```

The ten outputs in `generation.json` reuse stock phrases and contain grammar errors. A is not a sentence-quality success. Compare mix B on disjoint development records before choosing a balanced variant; then use both clean sets and raw generation for the final report. Do not deploy.

Claude's independent blind 100-prompt development benchmark with the same whole-word plus anti-stock-phrase decoder scored A relevance `0.64`, grammar `1.37`, usefulness `0.24` (0–2 scale), and `19%` at least partly useful. The mature everyday MICA scored `0.82`, `1.56`, `0.25`, and `20%`. A's relevance deficit was `-0.18` with paired 95% CI `[-0.35,-0.02]`; its usefulness difference was inconclusive. See Claude's 2026-09-28 06:34 UTC board update in `COORDINATION.md`. This confirms A has not established a quality gain.
