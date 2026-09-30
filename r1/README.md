# MICA R1

Reference implementation of *MICA exact mechanism and learning specification*,
revision R1 (20 September 2026). Sections referenced below are that document's.

The current MICA-only sentence-generation and integer-rule memory experiments
are recorded in
[`docs/2026-09-27-mica-only-generation.md`](../docs/2026-09-27-mica-only-generation.md).
Earlier suggestion results remain in
[`docs/2026-09-26-context-and-suggestions.md`](../docs/2026-09-26-context-and-suggestions.md).

R1 supersedes the v0.1 proposal for this candidate. The v0.1 differentiable
work lives in `../train/` and is kept only as the relaxation R1 §14 discusses;
it is not R1 and its numbers are not R1 numbers.

The English generation candidate remains MICA's learned integer-rule cellular
automaton. `research_baselines/` contains archived word-count and GRU experiments
for comparison only; neither is a MICA training component or release decoder.
Sentence quality and bits/target are evaluated separately before any demo
promotion.

## Layout

```
mica_r1/spec.py        dimensions, opcodes, file layout; nothing learnable
mica_r1/engine.py      the exact integer state transition (§3-§8)
mica_r1/serialize.py   86,820-byte model file, writer and validating loader (§10, §11)
mica_r1/decode.py      greedy reference decoder and UTF-8 validator (§8, §9)
mica_r1/suggest.py     optional whole-word suggestion decoder
mica_r1/batch.py       many sessions under many models in one pass; §12 objective
mica_r1/torch_batch.py GPU backend; same results, dense masked ticks
mica_r1/search.py      direct discrete mutation search, xorshift32 RNG (§13)
mica_r1/diagnose.py    routing usage, active set, ties, saturation (§14)
data/make_records.py   90/5/5 document split, <=256-byte UTF-8 records (§13)
data/ingest.py         YOUR OWN text or chat data -> training records
scale_study.py         sweep field size or rule-book size at a matched budget
occupancy.py           how full is the field, byte by byte (see findings 11-12)
tests/                 conformance fixtures, including the §7 worked example
checks/eval_suggest.py resumable fixed-prompt suggestion benchmark
run_diagnostic.py      the §14 first diagnostic
run_search.py          multi-restart §13 search, CPU or GPU
bench.py               time one full round on this host, project the schedule
```

## Running it

### Training on your own data

```bash
python3 r1/data/ingest.py --input ~/my_texts --out r1/data/mydata
python3 r1/run_search.py --records r1/data/mydata/train.jsonl \
        --val-records r1/data/mydata/val.jsonl \
        --bias-init spread --probe-init zero --routing pairdiff
```

`ingest.py` takes folders or files of UTF-8 text, deduplicates, splits whole
documents 90/5/5, and cuts records at UTF-8 boundaries. `--format chat` reads
`.jsonl` of `{"user": ..., "assistant": ...}` and emits the `User:` /
`Assistant:` markers §14 describes.

**Keep records at 256 bytes.** Finding 11 in `../docs/r1-findings.md` measures
what happens if you economise: a model trained on 24-byte records gets better
at 24 bytes and *worse* at 256, by 1.28 bits.

### Everything else

```bash
python3 -m pytest r1/tests -q          # 11 conformance fixtures
python3 r1/data/make_records.py        # 90/5/5 records from data/corpus
python3 r1/run_diagnostic.py --rounds 120 --record-bytes 16

python3 r1/bench.py                    # what will a round cost here?

# conformant run: every default is R1 exactly as written
python3 r1/run_search.py --backend torch --device cuda

# with the four changes argued for in ../docs/r1-findings.md
python3 r1/run_search.py --backend torch --device cuda \
        --routing pairdiff --bias-init spread --probe-init zero \
        --activity-cap 600
```

`search_result.json` records a `conformant` flag, true only when no flag
departs from the specification, so a modified run can never be mistaken for
an R1 run.

## Status

Conformant to §3-§11 and verified against the §7 worked example. The §14 first
diagnostic runs and lowers target loss. The §13 full search schedule is
measured at 290 days for its eight restarts and has not been run.

`../docs/r1-findings.md` has the measurements and the four specification
changes they argue for.

## Deviations, all deliberate and all listed

1. **A batched evaluator** (`batch.py`) that steps many sessions at once. A
   scalar interpreter cannot run §13 at all. It agrees bit-for-bit with the
   scalar engine, which is what makes it a speed change rather than a
   semantic one.
2. **Pluggable routing** (`engine.ROUTING_MODE`, default `"r1"`). The
   alternatives exist so the §5 routing degeneracy could be measured rather
   than asserted. The default is the specification.
3. **Corpus provenance.** §13 asks for 20 MiB of authorised UTF-8 text. The
   corpus here is 19.83 MB of permissively licensed technical documentation
   and Python source built from PyPI source distributions, with a manifest.
   It is not dialogue data and it is a single natural language.
