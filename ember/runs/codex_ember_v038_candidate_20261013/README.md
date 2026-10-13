# Ember 0.3.8 experimental candidates

**Neither candidate passed the predeclared confirmation gates. Keep released
Ember 0.3.7 as the best validated model. This directory is research evidence,
not a new release.**

Baseline repository revision: `dd729762f233789559e4314a26538fc3210714c1`.
Baseline word heads: `40f3bfa6f9692f4424d05cc3666afdca7ecbd648244df059ce2d181f66561e67`.
The earlier UI/readout improvements remain untouched in the separate
`/workspace/Mica-Ai` checkout.

Two architecture-preserving attempts were run:

1. Refit the two existing integer word biases with three regularization
   strengths. Selected on 400 prompts; confirmed on 400 others. Primary gain
   +0.625 percentage points, cluster-bootstrap 95% interval [-0.971,+2.192].
   Rejected: gain below one point and interval includes zero.
2. Reweight the existing current-probe, previous-word-probe and word-rule
   coefficient groups and the existing bias; quantize to the same int8/int32
   formats. Five fitted scalars per head, three regularization strengths.
   Selected on a new 400-prompt block and confirmed on another new 400-prompt
   block. Neither block contains experiment-1 prompts.

## Second candidate, independent confirmation

Each domain has 200 prompts. Scores are equal-domain means.

| Metric | Released 0.3.7 | Candidate | Change |
|---|---:|---:|---:|
| Next word top-1 | 9.75% | 10.75% | +1.00 pp |
| Next word top-5 | 21.00% | 21.25% | +0.25 pp |
| Two-letter completion top-1 | 40.25% | 41.25% | +1.00 pp |
| Two-letter completion top-5 | 65.25% | 63.75% | -1.50 pp |
| Primary: mean of next-word and two-letter top-1 | 25.00% | 26.00% | +1.00 pp |

Primary cluster-bootstrap 95% interval: **[-0.014,+2.140] percentage points**.
The >=1-point gain and no-regression gates passed; the positive-confidence
gate failed. Selection primary also decreased: baseline 24.875%, selected
candidate 24.500%. The confirmation gain does not establish a reliable win.

Candidate word-head SHA-256:
`e7d2ebf8ea770c8c76ff9509a6c7deeab53a544dd54e6985e93538c47ab241a7`.
Only existing word-readout coefficients and biases changed. The cellular model,
byte memory, vocabulary, 16 word-rule ticks and their learned tables are
unchanged. No new inference state or rule mechanism was added.

## Data and limits

800 training prompts from Taskmaster-3 and Brown, with conversation/document
hash splits (60/20/20); 400 selection and 400 confirmation prompts per attempt.
Training prompts are reused between attempts; scored selection/confirmation
prompts are not. The second blocks use new prompts from the same source-group
splits, not new corpora. Exact normalized prompt-target duplicates are removed.

These are **additional-domain scores, not the published clean benchmark**.
The original training/evaluation records are absent, so pretraining overlap
cannot be ruled out and the released benchmark cannot be reproduced. Brown
contains older mixed-genre prose and Taskmaster-3 movie-ticket dialogues;
performance here is considerably lower than on the original benchmark.
No sealed test was accessed. No general-quality, compression, reasoning or
sentence-quality improvement is claimed. Real site correction logs are also
unavailable and were not fabricated or used in this run.

Taskmaster-3 source revision and file URLs/hashes are in each data manifest.
Brown is distributed by NLTK with permission of its copyright holder; its
README states redistribution is permitted. The recorded archive SHA-256 pins
the downloaded content. Raw corpora and features stay under
`/workspace/scratch/mica038`, outside this repository.

## Reproduce

Dependencies: numpy and scipy. Place the two source files and `sources.json`
listed in the data manifest in a local data directory. Use the pinned baseline
package under `ember/runs/ember_v037_20261011`.

```bash
python experiment.py prepare --data /path/to/data --work /path/to/run1
python experiment.py fit --work /path/to/run1
python experiment.py confirm --work /path/to/run1
python verify.py --work /path/to/run1

python experiment.py prepare --data /path/to/data --work /path/to/run2 \
  --offset 200 --reuse /path/to/run1
python calibrate.py fit --work /path/to/run2
python calibrate.py confirm --work /path/to/run2
python calibrate.py verify --work /path/to/run2
```

The scripts refuse to overwrite an existing confirmation result. Use a new
work directory for exact reproduction, never to tune using confirmation data.
`results/` retains source/feature hashes, selection results, confirmations,
and per-prompt integer top-5 rankings for both baseline and candidates.

The runnable second-candidate package is at
`/workspace/scratch/mica038/ember-v038-candidate`; its manifest explicitly
labels the failed gate. It uses the released numpy-only runtime and verifies
artifact hashes at loading. Example:

```bash
python /workspace/scratch/mica038/ember-v038-candidate/word_model.py \
  --complete "Thank you for bei"
```

Verification: released word and byte-memory conformance checks pass; fitted
artifact checks verify unchanged non-readout arrays, exact int64 scalar scores,
and runtime top-5 agreement on sampled confirmation prompts. This candidate
was neither pushed, deployed nor promoted to an official version.
