# Further prototype training: controlled gain, no new public ToM gain

Completed CPU-only on 2026-10-09 UTC. No active training/evaluation remains.

## What was trained

24,000 independent synthetic stories (12,000 paired scenarios), with 72,000
location-to-ending link labels. A bounded integer branch rule learns option
grounding, including articles and whole-sentence endings. The existing learned
cellular update rule and query binding remain frozen. No public benchmark
examples, labels or templates were used for training; no sealed data accessed.

The first attempt was rejected on its predeclared component gate (839/1000
supported answers). Duplicate location identities caused by overlapping
article prefixes were fixed before generating and freezing this second run.
The failed attempt is preserved in codex_completion_grounding_20261010.

## Fresh controlled development

1,000 supported items in 500 scenario pairs and 200 unsupported prompts.
Names, objects, locations and surface variants differ from training. Normalized
story overlap is zero; the finite semantic generator is related to training,
so this is not a general-English accuracy claim.

| Measurement | Previous prototype | New trained grounding |
|---|---:|---:|
| Memory-only supported answers | 835/1000 | 965/1000 |
| Exact same-source hybrid answers | 899/1000 | 965/1000 |
| Complete supported pairs, memory-only | — | 465/500 |
| Unsupported definite answers | 0/200 | 0/200 |

Hybrid gain: +66 correct, zero previously correct answers lost, +6.6 percentage
points, paired scenario-bootstrap 95% CI [+5.1,+8.2] points. All predeclared
development gates passed. 1,200 reversed-option checks, 600 prior development
retention checks (500 definite decisions), article-identity and negative-link
checks passed. Exact packaged integer source fallback is identical for both.

## Frozen public Tiny Theory of Mind measurement

All 2,000 original full contexts and original endings were passed to the frozen
candidate before labels were scored. The packaged source cache was checked for
all record IDs, labels, ending word/character counts and source predictions;
eight predetermined exact-engine audits matched with 0.0 nats error.

| Scoring normalization | Published 0.3.3 | Previous private prototype | New prototype |
|---|---:|---:|---:|
| Mean ending log likelihood per word | 636/2000 (31.80%) | 638/2000 (31.90%) | 638/2000 (31.90%) |
| Mean ending log likelihood per character | 633/2000 (31.65%) | 635/2000 (31.75%) | 635/2000 (31.75%) |

**No public answers changed versus the previous prototype.** The new grounding
created zero additional definite memory answers. Existing memory routing is
12/2000, of which 5 are correct. 1,891 items failed the distinct-grounded-place
requirement; 55 reached no memory answer, 30 abstained at the boundary, and
12 had ambiguous endings. These counts diagnose coverage; they do not justify
learning benchmark-specific rules.

Against the release, the unchanged word result has 3 gains/1 loss and paired
95% CI [-0.10,+0.30] percentage points; character has 4 gains/2 losses,
CI [-0.15,+0.35]. This is a repeated, descriptive public measurement, not a
fresh selection gate or a statistically established improvement. The >33%
objective remains unmet.

## Architecture and decision

Private 32-ring x256-site radius-one integer-rule cellular memory remains
unchanged. The newly trained English/option grounder is an external adapter.
It is not compiled into the native B-740 rule VM. Original Flame-W B-740 and
released memory files remain byte-identical; no shared engine edits or GPU use.
No sentence-quality, native-model improvement, release, or deployment claim.

Keep v6 as the current public-scored prototype. Preserve this independently
passing controlled-grounding candidate as research, without promoting it as
a ToM upgrade. Repeating more of this same grammar is not supported by these
results; transfer needs broader independent English event/role/query coverage.

## Provenance

Native model SHA-256:
`1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`

Frozen cellular rule SHA-256:
`9a0ad7b1cc0c607176092d333f29f1ce4b993f460fde1a57fab338aca1400a9c`

New integer grounding rule SHA-256:
`0a2edf02857ca8a021aadb4a0e1f9d7b4cc918957e2ff32cfaabdb6b4e981c0c`

Grounder code SHA-256:
`633f40ecae3afa2f24f4af2b1434d26737f5ccc91f4c14f406d757733aeb092d`

Train SHA-256:
`1e7f7e3818246fc6e2acdfc360c4c008e1e4f526cc275b8b7d42380cd9a1f0ed`

Fresh development SHA-256:
`d038373f4c082b1f5e9298c5c1ee85714b94be4335707b1d5d7aaf59e64816f8`

Public dataset SHA-256:
`58827a0361597ae4d5ffa79ddbab0936d1509cafcfc02cbb731e4d7a7f3584b8`

Complete source hashes, inference artifacts, checks and result hashes are in
MANIFEST.json. Raw controlled QA examples (not free-form generations):

### Example 1

Before lunch, Ludovic placed the buckle by the silk pouch. Oriana later told Ludovic the buckle was in the purple trunk. Then Oriana carried the buckle over to a cotton bag.

Ludovic thinks the buckle is currently at

Selected ending: Ludovic would look for the buckle at the purple trunk.

### Example 2

At the outset, Nestor had stored the kettle inside the hallway locker. Later, Oriana informed Nestor that the buckle could be found at silk pouch. Afterward, Oriana moved the kettle inside silk pouch.

Nestor thinks the kettle is currently at

Selected ending: Nestor would look for the kettle at the hallway locker.
