# Ember 0.3.8 release qualification

One fixed candidate, word_heads SHA256
44c60a2e64a5d671a03dfc162fca0c9c435b75c2059ebca95eec47ab4a2a92ce,
with previously TRAIN-fitted 9:7 prefix2:next completion scores and gain12/16.
Compare with the actual public 0.3.7 prefix2-only decoder, gain16/16, using
the original 40f3bfa6 word heads. Also show original heads with the improved
decoder to separate decoder gains from coefficient refitting.

No fitting or model selection in this qualification. Reuse the already-frozen
story-bias confirmation sets (2000 per domain, 1999 eligible everyday prompts)
and their exact native probe caches. These were evaluated in the preceding
candidate experiment; do not call them a newly blind release test. They exclude
word training, refit/screen and prior word heldouts; historical byte-validation
lineage is disclosed. Story uses varied word positions; chat/everyday use the
first eligible word. All comparisons are matched within each set.

Gate, frozen before direct-public037 scoring: equal-domain conditional word
loss gain at least 0.01 bits with paired95% upper below zero; no domain loss
regression over 0.01 bits or top1 regression over 0.2 percentage points.
Top1 includes OOV; conditional loss excludes OOV and normalizes over the legal
two-letter-prefix vocabulary. Resample paired records within domains, 5000
bootstrap replicates, seed20261013251. Significant accuracy gains require a
positive paired lower confidence bound; otherwise report as descriptive.

Only after passing, score fixed clean chat/everyday sets for reporting, never
selection. A clean domain accuracy regression over 0.2pp or loss regression over
0.01bits vetoes replacement. These clean sets have earlier public037 research
reports, but have not selected the latest fixed candidate. No sealed test.

Verify actual packaged runtime ranks/probabilities against native integer
features and scalar sums; preserve one-letter/next-word/byte model/memory,
geometry, rule predicates/immediates/vocabulary. Reuse completed full2000
ToM result by byte-path identity, clearly mark no ToM gain. Publish only if
qualification, clean reporting and packaging conformance pass. Release is a
word-completion upgrade, not a sentence/reasoning or byte-loss improvement.
