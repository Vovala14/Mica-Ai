# MICA-only English generation check (27 September 2026)

## Model boundary

The candidate is **Minimal Inference Cellular Automaton**: its inference
state, rule selection, integer rule operations and readout come from the
exported `.mica` model. The user's release constraint excludes a GRU,
word-count model, retrieval system or other language model from the MICA
generation path. The exploratory word-count and GRU code was moved to
`r1/research_baselines/` after that clarification. The GRU run was stopped;
none of those experiments is a MICA candidate.

No model has been packaged or put on the example site.

## Existing numeric result and direct generation

The strongest exported integer MICA checkpoint remains
`r1/runs/sweep/everyday_c256/best.mica`: 1.790994 pooled bits/target on
the clean 1,000-record validation set and 1.798603 on the separate clean
1,000-record test set. That meets the numeric target of less than 2 on these
sets. It does not meet the requirement to create sensible new sentences.

`r1/checks/eval_mica_raw_generation.py` feeds each fixed development prompt
to MICA's exact integer engine and then calls its reference greedy decoder.
It uses no vocabulary, retrieved text or auxiliary model. At a 100-byte
limit, the best checkpoint generated, for example:

- `The dog ` → `The dog with a bus stop sign and a book at the train station
  of the country in the state the back to the cou…`
- `I decided to ` → `I decided to the country in the state the back to the
  country in the state…`

These are continuations, not complete plausible sentences. The exact outputs
are in `r1/runs/checks/generation_dev10_everyday_raw100.json`.

## Matched integer-rule memory layout pilot

`r1/checks/run_mica_long_lag_pair.py` trained two MICA models with the same
geometry, seed, everyday training split, 40,000 training records, one full
fit round and 1,500 fit steps. Both use 48 tape probes and 192 probes of
rule-written channels. The single change is the tape probe lag allocation:
the control reads at lags 1–7, while the other also reads at 8, 16, 24 and
32 bytes. Both export 4,467,396-byte integer-rule `.mica` files.

| Measure | Standard lags | Lags through 32 |
|---|---:|---:|
| Trainer-selected validation, bits/target | 1.7910 | 1.7944 |
| Separate clean validation, pooled bits/target (1,000 records) | 1.826877 | 1.817006 |
| Prefix pairs with exactly equal next-byte scores after 8 shared bytes | 6/6 | 1/6 |
| Same after 16 shared bytes | 6/6 | 1/6 |
| Same after 32 shared bytes | 6/6 | 6/6 |
| Repeated four-word phrases in 100-byte greedy continuations, mean fraction over 10 prompts | 0.601 | 0.244 |

The longer layout carries more of the earlier prompt into the next-byte
scores, reduces mechanical repetition, and improves independent clean
validation by 0.009871 bits/target at the matched one-round budget. A paired
record bootstrap (5,000 resamples) gives a 95% interval of [-0.016338,
-0.003167] for long minus standard. Both one-round models remain worse than
the existing best checkpoint's 1.790994 on this same clean set. The longer
layout has not made the raw sentences reliably grammatical or sensible.
Examples from the 32-byte model:

- `Our flight ` → `Our flight be able to do that.`
- `After the storm, ` → `After the storm, and the station of the station of
  the computer and a computer and a computer…`

The fixed-prompt outputs are in
`r1/runs/checks/generation_dev10_mica_tape_default_full_round1_raw100.json`
and `r1/runs/checks/generation_dev10_mica_tape_long32_full_round1_raw100.json`.
The score-horizon checks are in
`r1/runs/checks/mica_tape_default_full_round1_horizon.json` and
`r1/runs/checks/mica_tape_long32_full_round1_horizon.json`.
The independent paired validation numbers are in
`r1/runs/checks/mica_tape_full_round1_clean_val1000.json`. The
trainer-selected validation numbers use different records and should not
replace the clean comparison.

## Wider MICA probe-window preflight

A further matched, three-round pilot kept the same integer-rule engine and
expanded the allowed probe window to 64 bytes. Both arms had the same
geometry and budget (500 records and 30 fit steps per round); one allocated
its farthest four tape probes at lag 24, the other at lag 64. On 128 clean
validation records the lag-32 arm scored 2.552918 and the lag-64 arm scored
2.537030 bits/target. The paired difference, -0.015888, has a 95% bootstrap
interval of [-0.049593, +0.017233], so this short pilot does not establish a
byte-prediction win. After 32 shared bytes, all 6/6 prefix pairs had exactly
equal scores with lag 32, versus 0/6 with lag 64; after 64 shared bytes both
arms were 6/6 equal. Raw sentence quality remained poor in both arms.

The two short `.mica` runs are `r1/runs/ablation/mica_probe64_lag32_short`
and `r1/runs/ablation/mica_probe64_lag64_short`, with paired validation in
`r1/runs/checks/mica_probe64_short_clean_val128.json`.

The matched full-round comparison has completed. The lag-32 control's
exported rules and readout are byte-identical to the previous full-round
lag-32 model; its `.mica` file differs only at byte 88 of the geometry
header, which records the allowed probe window. The lag-64 arm's
trainer-selected validation is 1.7751 versus 1.7944 bits/target for that
exact control. On the same independent 1,000-record clean validation set,
lag 64 scores 1.797631 versus 1.817006 bits/target for lag 32. The paired
difference is -0.019376 with a 95% record-bootstrap interval of [-0.025583,
-0.013080]. The existing best MICA checkpoint scores 1.790994 there; lag
64 minus best is +0.006637 with an interval of [-0.001923, +0.015423], so
this run does not establish a new numeric best.

On the same 10 direct-generation development prompts, lag 64 stops at EOS
for 9/10 continuations (versus 3/10 for lag 32), and generates a mean of
40.1 bytes (versus 76.9). Its repeated four-word fraction is 0.020 (versus
0.244), but the shorter outputs and new fragment failures make that number
a poor proxy for quality. Examples include `The dog is the state of food to
do that.` and `Our flight and a woman sitting on a street.`. Neither arm is
suitable for the example site. The full-round raw outputs are in
`r1/runs/checks/generation_dev10_mica_probe64_lag64_full_round1_raw100.json`.
The integer evaluation is in
`r1/runs/checks/mica_probe64_full_round1_clean_val1000.json` and the longer
context check is in
`r1/runs/checks/mica_probe64_lag64_full_round1_horizon.json`.

## Cross-word rule-state pilot

The native MICA `--rule-state stream` setting was paired with the lag-64
layout for a short three-round pilot, again with 500 records and 30 fit steps
per round. It changes rule-state initialization and the work-probe layout, so
it is a diagnostic of the combined mechanism rather than a single-factor
comparison. It scored 2.538959 pooled bits/target on the same clean 128
records where the no-state lag-64 pilot scored 2.537030. Crucially, all six
prefix pairs had exactly equal next-byte scores after a common 64-byte
suffix, just as with no stream state. Its raw outputs remain incoherent.
There is no reason to run this stream setting at a full budget without a
redesign of how the integer rules preserve state across words.

The run and checks are
`r1/runs/ablation/mica_probe64_stream_lag64_short`,
`r1/runs/checks/mica_probe64_stream_short_clean_val128.json`, and
`r1/runs/checks/mica_probe64_stream_lag64_short_horizon.json`.

## Decision

Do not promote any new model. The output quality target is unmet. Any
next experiment must keep generation inside MICA's learned integer-rule
machine and report held-out bits/target alongside directly generated
sentences; lower repetition alone is insufficient.

## Overnight continuation and next experiment

The unchanged 32-byte-probe MICA run continued from its best checkpoint to
round 145. The best trainer-selected validation was 1.7388 at round 52 and
did not improve thereafter. Its exported integer model scored 1.787362
pooled bits/target on the clean 1,000-record validation set, versus 1.790994
for the prior checkpoint. The paired difference is -0.003632 with a 95%
record-bootstrap interval [-0.009097, +0.001849]. The change is not
established. Direct completions still repeat broken phrases, such as
`The dog is a computer sitting on the station of the country...`.
See `r1/runs/overnight/mica_original_20260927/REPORT_HE.md`.

The exact integer context-horizon check on the overnight model found identical
next-symbol score vectors in all six prefix pairs after eight common suffix
bytes. See `r1/runs/overnight/mica_original_20260927/context_horizon.json`.
The current fit procedure optimizes readout coefficients, biases and VSET
immediates; the candidate score selectors are initialized and balanced, then
held fixed. A recurrent MICA state pilot used random frozen state transitions
and also lost prefix influence by 64 common bytes. More rounds of either fit
scheme have no demonstrated path to the one-bit or sentence-quality targets.

The next controlled experiment should keep the same exact integer engine,
geometry and instruction set, but train candidate scoring and state-transition
rules so information can pass from one byte's work channels to the next.
First prove that exported integer scores retain a prefix influence beyond 32
and 64 common bytes. Then compare pooled clean validation loss on the same
records with a paired interval, and inspect the fixed ten raw-generation
prompts. Run a long fit only after the short pilot passes these checks. Clean
complete-sentence training data can be tested separately to improve output
style; that data change must not be mixed into the rule-learning comparison.

A bounded first attempt was run in
`r1/runs/ablation/mica_state_learning_pilot_20260927/REPORT.md`: the existing
hard straight-through trainer continued a small stateful MICA fit for 60
gradient steps. Its selected best `.mica` checkpoint was identical to the
control. The final checkpoint changed no hard `sc_nb`, `sc_ch`, or `sc_co`
selectors and worsened clean val128 from 2.558919 to 2.648082 bits/target.
This setting does not solve the selector-learning problem; do not scale it.
