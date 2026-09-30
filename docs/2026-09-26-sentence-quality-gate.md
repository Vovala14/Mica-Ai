# English demo sentence-quality gate

The 20 prompts in `r1/checks/sentence_holdout_20260926.json` are a qualitative
holdout. Do not tune decoder rules, candidate filters or weights to individual
holdout prompts. The earlier `everyday_dev_greedy.json` prompts are for
development only.

For each holdout prompt, inspect the top continuation in two modes: a short
next-word/phrase suggestion and a complete sentence. Mark an output acceptable
only when it joins the prefix grammatically, keeps the subject and situation
plausible, contains no obvious unrelated topic jump or repeated filler, and
(for sentence mode) ends as a complete sentence. An abstention is safe but
does not count as a successful completion. Record the exact output and a
one-sentence reason for each failure.

The quality gate for the example site is at least 18 acceptable outputs out
of 20 in **each** mode, with no severe failure on the remaining prompts. The
retrieval index must be built only from the training split; source-assisted
generation must be labeled in the UI. Separately, the exported integer MICA
checkpoint used for byte prediction must score below 2 pooled bits/target on
an independent, exact-deduplicated English evaluation set. Report the two
measurements separately; a retrieval result does not change bits/target.
