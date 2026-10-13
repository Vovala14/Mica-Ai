# Local archive integration

These scripts and result files were imported from the owner-supplied handoff.
Both candidates failed their confidence gates. They are archived evidence only;
no candidate weights are enabled or promoted.

The ZIP contains four files: README.md, BASES.json and two patches. The runnable
candidate ZIP mentioned by its outer README was **not included**. Raw corpora,
cached features and fitted candidate weights are also absent. Their training or
benchmarks have not been rerun locally, and these reported results are not new
local validation.

For exact research reproduction, use the frozen baseline revision dd729762f233789559e4314a26538fc3210714c1
and original Ember037 decoder, as recorded in this archive. Current checkout has
a separately validated consensus/probability decoder, so it is not the identical
runtime used by the original archived experiment. Source automaton/heads stay
unchanged. Do not tune or promote a failed candidate using its confirmation data.
