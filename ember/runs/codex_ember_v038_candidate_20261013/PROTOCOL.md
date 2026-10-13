# Ember 0.3.8 candidate: frozen-rule bias refit

Written before feature extraction, fitting, or candidate results.

Baseline: the committed Ember v0.3.7 package, word-head hash
`40f3bfa6f9692f4424d05cc3666afdca7ecbd648244df059ce2d181f66561e67`.
Keep byte model, memory, vocabulary, head weights and all word rules unchanged.
Only the two existing integer bias vectors may change.

Use Taskmaster-3 (conversation) and Brown (mixed prose), which are not named in
the documented original source list. Original training record hashes are not
available: absence of baseline pretraining overlap cannot be proved. No claim
of reproducing the released clean benchmark follows from this experiment.

Split by conversation ID / Brown file ID using SHA-256 into training (60%),
selection (20%), confirmation (20%). For each domain choose 400 training,
200 selection and 200 confirmation distinct normalized prompt-target records,
deterministically by hash. Remove duplicate normalized texts across splits.
Use one eligible word (ASCII letters, >=4 letters, in the released vocabulary)
per sentence/utterance. Context is at most 160 UTF-8 bytes and cut at a word
boundary; select the eligible position by hash rather than prediction quality.
These counts can fall short if a domain cannot supply them; report actual sizes.

Fit per-word bias deltas on training records only using cross-entropy plus
L2 regularization, in score/1024 units. Screen exactly three strengths:
0.001, 0.01, 0.1. Completion fitting restricts softmax to the matching
two-letter candidate vocabulary, as the interactive runtime does.
Quantize deltas to integer units before measuring. Selection criterion:
equal-domain mean of next-word top-1 and two-letter completion top-1.
Ties choose the stronger regularizer. No decoder or prefix tuning.

Confirm only the selected candidate once. Success requires a >=1 percentage
point gain on the primary mean, paired-bootstrap 95% lower bound >0, no
domain primary regression over 2 points, and no individual next-word or
two-letter top-1 regression over 2 points. Bootstrap by source conversation /
document, not individual prompt, within each domain (10,000 draws, seed 38).
Also report next-word top-5 and one/three-letter completion top-1 and top-5.
Those are diagnostic, not selection criteria. Preserve all per-prompt integer
rankings, split identities and hashes. If the confirmation fails, mark the
candidate rejected; do not adjust it on confirmation records.

Passing is evidence on these additional domains only. The original general
text / word benchmark and byte conformance must be checked before release.
No sealed test access, deployment, or published version change in this run.
