# User corrections for readout fitting

Use an owner export from the existing signed site-log endpoint. The export
contains user text; keep the export and generated corpora outside Git.

```bash
python r1/data/ingest_site_logs.py /path/to/site-logs.jsonl \
  --corrections-only --out /path/to/site-corrections

python r1/data/mix_site_corrections.py \
  --base /path/to/ember/base-train.jsonl \
  --site /path/to/site-corrections --model ember \
  --heldout /path/to/ember/dev.jsonl \
  --heldout /path/to/ember/validation.jsonl \
  --out /path/to/ember/mixed-train.jsonl
```

For Flame-W, use `--model flamew` and word-token base and held-out files.
The mixer emits the same hex-record format as the original corpus. Supply
all development and validation corpora used by the fitting experiment via
repeatable `--heldout` arguments. The site validation split is excluded
automatically. Never supply the sealed test set to this development pipeline.

Only the latest user-written correction for a generation is admitted;
thumbs-up model outputs and uncorrected prompts are excluded. Fixed rating
prompts remain excluded. Ingestion normalizes text for deduplication and
splits by normalized prompt hash. Word-completion corrections accept either
missing letters or a full word, including one-letter and longer prefixes.

The mixer removes exact encoded duplicates of the base and supplied held-out
records. It does not detect paraphrases or partial-record overlap. It verifies
site file hashes, samples without replacement in a seeded deterministic order,
and caps corrections at 5% of both records and targets, counting EOS. Smaller
caps can be set with `--max-fraction`. Small base corpora may admit no corrections.
Base records retain their order, with selected corrections spread between them.
The companion manifest records counts, source hashes, and the output hash.

Use the mixed corpus as input to frozen-automaton feature extraction and
word-head or memory-readout fitting. Keep the cellular checkpoint and vocabulary
fixed. The historical fitting scripts have machine-specific paths and feature
inputs; this command prepares records, rather than launching those scripts.
Compare a base-only control and the mixed-data readout on identical fresh
development records, export integer coefficients, and verify with the exact
integer runtime. Evaluate general-text regression and blind sentence usefulness
before replacing any released readout. Preparing this corpus does not establish
a model-quality improvement or change the deployed model.
