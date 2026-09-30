# English prose corpus and context pilot (2026-09-26)

## Why this experiment was run

The old `r1/data/bulk` ingest took 150 MB from each of eight downloaded
Parquet shards. Two shards were OpenMathInstruct, so about 300 MB of the
1.2 GB raw input was math Q/A. The best saved model,
`r1/runs/sweep/everyday_c256/best.mica`, was trained on a different corpus:
Tatoeba sentences, COCO captions, and 200,000 filtered prose records. Its
`run_info.json` names `r1/data/everyday/train.jsonl`. Thus math content is a
plausible contributor to the earlier bulk model's “sum of sums” outputs, but
does not by itself explain the best everyday model's poor sentence coherence.

`r1/run_train.py` now defaults to the everyday corpus. Bulk must be selected
explicitly with `--corpus bulk`; resuming a checkpoint with a different or
unknown corpus is refused. The old `r1/runs/soft` checkpoint remains intact.

## Corpus and evaluation

`r1/data/build_prose.py` streamed the already downloaded Simple English
Wikipedia and FineWeb Parquet shards. The run excluded OpenMath by source,
filtered math-like/boilerplate text, assigned complete normalized documents
to 90/5/5 splits by SHA-256, and removed exact normalized document and record
duplicates across splits. It capped each record at 256 UTF-8 bytes and each
document at eight retained records. Both source shards were fully scanned:

| Split | Documents | Records | UTF-8 text bytes |
|---|---:|---:|---:|
| Train | 232,735 | 1,262,067 | 232,958,372 |
| Validation | 13,032 | 70,463 | 12,994,125 |
| Test | 13,018 | 70,242 | 12,962,436 |

The complete build totals 258,914,933 text bytes (247 MiB) across 1,402,772
records. Its source files, caps, rejection counts and split hashes are in
`r1/data/prose_v1_320/manifest.json`. Exact deduplication does not catch
paraphrases or near duplicates. FineWeb remains noisy web text despite the
filters; spot checks include news and job-ad prose as well as expository text.

`r1/checks/make_prose_eval.py` sampled 256 validation records from across
the *whole* validation file for checkpoint selection, plus disjoint audit
samples of 1,000 validation and 1,000 test records. The split files and sample
hashes are recorded in `r1/data/prose_eval_v1/manifest.json`. Evaluating the
audit validation file requires `--val-skip 0`; it has already excluded the
checkpoint records.

Separately, `r1/checks/make_clean_eval.py` removed exact normalized overlap
from the older everyday evaluation splits. The best everyday model scored
1.790994 bits/target on the new 1,000-record validation sample and 1.798603
on the new 1,000-record test sample. These are valid standalone measurements;
they cannot be subtracted from the old 1.777176/1.788539 scores to estimate
the effect of leakage, because the samples contain different records. The
exact-dedup check also cannot exclude near-duplicate captions.

## First bounded training comparison

The best everyday model scored 2.353609 pooled bits/target on the 128-record
prose audit subset. A fresh three-round prose fit with 10,000 records and 300
fit steps per round (`r1/runs/ablation/prose_default_r3`) reached 2.2247 on
its checkpoint-selection records. On the separate prose audit subset it
scored 2.217029; on the clean everyday 128-record subset it scored 2.208237.
This short fit used only 30,000 sampled records from the larger training set,
so it is a probe of the data direction, not a converged large-corpus model.
The paired prose-audit improvement over the old everyday model is 0.13658
bits/target on 24,134 targets (record-bootstrap 95% interval [0.11724,
0.15723]). The models had different training corpora and budgets, so this is
an empirical domain comparison, not an isolated causal estimate.

Using the same fixed English prompts and everyday training vocabulary as the
prior sentence check, the prose model still produced incoherent sentences:
`The meeting has been moved to be the state of them.` and
`Please remember to bring your home of a few week.` The earlier everyday
model also failed this check, for example `The meeting has been moved to be a
lot of people.` The prose-only pilot is therefore not a release candidate.

The new-corpus test audit remains unused for model selection. The numeric
release goal is under 2 pooled bits/target on an independent evaluation set,
and the product goal also requires sensible short continuations and full
sentences on the fixed English prompts.

## Long-lag work-feature pilot

`--work-lags 1:16,2:8,8:4,16:4` optionally allocates the same 192 work
probes as 96 at lag 1, 48 at lag 2, 24 at lag 8 and 24 at lag 16. Each
selected group contains all six VSET channels. The default layout and model
format are unchanged. `r1/runs/ablation/prose_worklags_r3` used exactly the
same corpus, seed, geometry, three rounds, 10,000 records and 300 fit steps
per round as the default prose pilot.

| Check | Default prose | Long-lag work probes |
|---|---:|---:|
| Checkpoint validation, best mean bits/target | 2.2247 | 2.2272 |
| Separate 128-record prose audit, pooled bits/target | 2.217029 | 2.215418 |
| Fixed topic pairs with different raw top suggestions | 4/20 | 5/20 |
| Score pairs identical after eight shared bytes | 6/6 | 1/6 |
| Score pairs identical after 32 shared bytes | 6/6 | 6/6 |

On the 24,134 shared audit targets, the paired default-minus-long-lag
difference is +0.00161 bits/target, with a record-bootstrap 95% interval
[-0.00481, +0.00852]. The extended probes carry earlier information into
the exported integer scores, but the measured loss is effectively tied. The
full-sentence results are still incoherent, e.g. `Please remember to bring
your company in the first time.` The candidate is not promoted.

The user approved a separately labeled retrieval-assisted decoder: it may
draw candidate English continuations from the **training** prose corpus and
use MICA to rank them. This can improve demo output quality while keeping
MICA's byte-prediction score and corpus-assisted generation clearly distinct.
It has not yet passed the sentence-quality gate.
