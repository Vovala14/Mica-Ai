# MICA R1: context and suggestion check (26 September 2026)

This note records the work resumed after the interrupted development session.
All reported MICA results use exported integer `.mica` files. The 100-prompt
completion set is a development set; no result here is a release claim.

## Starting point

`r1/runs/sweep/everyday_c256/best.mica` is trained on the everyday corpus.
The original greedy 100-prompt check produced 41 distinct continuations, but
only 5 of 20 matched topic pairs received different continuations. Many
continuations were unrelated to the prompt. A direct score diagnostic found
that all six tested pairs became **exactly identical** after eight shared
bytes, despite a 32-byte probe window. The fitted readout used tape lags 1–7;
training coefficients alone cannot read a more distant byte through that
layout.

On the same 1,000 everyday validation records beyond the first 256 (44,965
byte/EOS targets), the integer model scored **1.777176 bits/target** and the
5-gram Kneser–Ney baseline scored **1.828412 bits/target**. These numbers
compare the saved systems on the same text. The 5-gram fit used 200,000
training records; this is not a controlled capacity or training-budget
comparison. The first 256 validation records were used to select MICA
checkpoints, so their scores should not be treated as an independent test.

## Suggestion decoder

`r1/mica_r1/suggest.py` is an optional whole-word beam decoder over a
vocabulary built only from the everyday **training** split. It runs against
the exported integer engine and does not alter MICA's reference decoder.
It supports partial-word completion and an explicit `--complete-final-word`
mode for next-word requests. The score excludes BOS, which cannot be emitted.
Its reported confidence is relative to found beam completions and is **not
calibrated** for correctness.

`r1/checks/eval_suggest.py` checkpoints a 100-prompt benchmark after every
prompt so a crash can be resumed. With the final word treated as complete,
3 maximum generated words, a 5,000-word vocabulary, beam width 2 and the
default 0.55 search-confidence threshold, the everyday model offered
26 of 100 top suggestions and abstained on 74. It changed the top suggestion
for 4 of 20 topic pairs. The outputs still include irrelevant suggestions
such as `"I need to buy some milk and" → " white."`; the decoder has not
solved semantic relevance.

The decoder also supports an experimental full-sentence constraint: a minimum
number of generated words and terminal punctuation. On 20 fixed everyday
prompts with at least five generated words, up to eight, both the best model
and the corrected stream pilot produced complete but mostly nonsensical
sentences. For example, the best model gave `"The meeting has been moved to
be a lot of people."`, and the stream pilot gave `"Please remember to bring
your back for from a lot."` The sentence constraint is a test of the model's
language structure, not a cure for it.

## Short matched tape-lag experiment

An optional `--tape-lags` argument now specifies the allocation of the 48
tape probes. The default is unchanged. The candidate layout was
`1:16,2:8,3:4,4:4,8:4,16:4,24:4,32:4`; both models kept the other 192 probes
on work channels, the same geometry, data, seed, model format, and fit
settings. Each completed 3 short rounds using 500 records and 30 fit steps
per round. The output directories are `r1/runs/ablation/tape_default_short`
and `r1/runs/ablation/tape_long_short`, with exact settings in each
`run_info.json`.

| Measure | Default tape lags | Lags through 32 |
|---|---:|---:|
| Score pairs identical after 8 shared bytes | 6/6 | 1/6 |
| Score pairs identical after 16 shared bytes | 6/6 | 1/6 |
| Score pairs identical after 32 shared bytes | 6/6 | 6/6 |
| Pooled bits/target on 128 separate validation records (5,821 targets) | 2.579180 | 2.594727 |
| Topic pairs with different top suggestions, threshold disabled | 4/20 | 8/20 |

The longer layout demonstrably carries earlier bytes into the output, and
produced more different answers across topic pairs. It did **not** improve
held-out byte prediction in this short run. The long-minus-default difference
was +0.01555 bits/target, with a paired record-bootstrap 95% interval of
[-0.02001, +0.05043]. The interval includes zero. The extra pair differences
were often still inappropriate to the topic; they are a context-sensitivity
measure, not a usefulness score. A longer controlled run is required to
judge this layout as a model-quality change.

## Short recurrent stream experiment

`--rule-state stream` lets the phase-0 rule state update across spaces and
punctuation; the earlier `word` setting reset there. Simulated balancing now
samples one tick in 31, which bounded its memory use after the first attempt
ran out of GPU memory. The readout fit keeps phase-0 state immediates fixed:
its provenance is computed before fitting, so changing those immediates
would make the training features disagree with the exported integer model.
The first completed `stream_state_short` run used the old fit and is retained
only as a debugging artifact. The corrected run is
`r1/runs/ablation/stream_state_fixed_short` (3 rounds, 500 training records and
30 fit steps per round). It used the same model geometry and validation set
as the tape-lag pilot.

| Measure | Default tape pilot | Corrected stream pilot |
|---|---:|---:|
| Score pairs identical after 8 shared bytes | 6/6 | 3/6 |
| Score pairs identical after 32 shared bytes | 6/6 | 4/6 |
| Pooled bits/target on 128 separate validation records | 2.579180 | 2.585016 |
| 100 prompts passing the decoder's uncalibrated display threshold | not measured here | 51 |
| Topic pairs with different raw top suggestions | 4/20 | 12/20 |

The stream state changes more outputs across contexts, but has not improved
held-out prediction in this short test. Its difference from the default tape
pilot is +0.00584 bits/target, with a paired record-bootstrap 95% interval
of [-0.05273, +0.06442]. Its suggested words are still often
inappropriate: `"Please remember to bring your" → " back."` and
`"The chef tasted the soup and said it needed more" → " first."` both passed
the display threshold. Raw 120-byte samples remain fragmented or invent words
(for example `"The weather " → "reactive to make of Mesting. ..."`).
Passing the display threshold is **not** a semantic quality score.

The current best model remains `everyday_c256`: 1.777176 bits/target on the
separate 1,000-record validation check (and 1.802545 on the pilot's 128
records). A further untouched 1,000-record test check scored **1.788539
bits/target** over 44,299 targets. It meets the numerical goal of under 2
bits/target on both measured splits, but it does **not** yet meet the
requirement for sensible sentences.

## One full-budget stream round

With the ROCm Python environment available, one corrected stream-state round
used 40,000 training records and 1,500 fit steps, matching the *first round's*
fit budget of `everyday_c256`. The run is
`r1/runs/ablation/stream_state_full_round1`; it ended normally and left no
background training process. The best everyday model had 19 completed rounds,
so the two final models do **not** have matched total training budgets.

| Split (1,000 records) | Best everyday model | Stream, one full round |
|---|---:|---:|
| Validation beyond checkpoint-selection records | 1.777176 | 1.823099 |
| Separate test (44,299 targets) | 1.788539 | 1.840921 |

The stream-minus-best pooled difference is +0.04592 bits/target on validation
(paired record-bootstrap 95% interval [+0.03758, +0.05424]) and +0.05238 on
test ([+0.04339, +0.06113]). Both models satisfy the numeric target of under
2 bits/target on these measured splits. After eight shared bytes, all six
prefix pairs still had different score vectors in the full-round stream
model; after 32 bytes all six had identical vectors. On the fixed next-word
prompts, 49/100 passed the uncalibrated display threshold and 5/20 topic
pairs had different raw top suggestions. The greater display count did not
yield sensible sentences: `"I need to buy some milk and" → " white photo of"`
and constrained sentence output `"I need to buy some milk and white photo of
a car."` are representative failures. The best everyday model remains the
best candidate for numeric prediction, and neither model is ready for the
requested demo quality.

## Word-start-weighted fit

`--word-start-weight 4` is an optional fitted-readout objective that weights
the first letter after a space or BOS four times as much as other byte
targets. In the 40,000-record sample, 437,503 of 2,455,663 targets (17.8%)
received that weight. `stream_state_wordstart4_round1` used the same geometry,
data sample size, fit steps, and one-round budget as the unweighted full-round
stream run. On the same 128 separate validation records it scored **1.882988**
versus **1.840059** bits/target unweighted; the paired difference was
+0.04293 bits/target with a 95% bootstrap interval of [+0.02658, +0.06194].
Only 32/100 next-word prompts passed the decoder's display threshold, versus
49/100 unweighted. Full-sentence examples were still incoherent, including
`"The meeting has been moved to be able to do that."` This pilot does not
support promoting the weighted objective; it remains an optional research
setting. A 1,000-record test run was not warranted after the separate
validation and sentence checks both worsened.

## Reproduce and continue

- `python -m pytest r1/tests -q` covers the exact engine and the new decoder
  and tape-lag and recurrent-state preflights; 30 tests and 11 subtests passed.
- `python r1/checks/eval_suggest.py --run r1/runs/sweep/everyday_c256 --out
  r1/runs/checks/everyday_suggest_nextword_current.json` resumes the development
  prompt check. The output records hashes of the model, prompts, corpus, and
  decoder.
- `r1/checks/context_horizon.py` probes exact score equality with aligned
  prefixes. The new results are in `r1/runs/checks/tape_*_horizon.json`.
- The matched validation outputs are in
  `r1/runs/checks/tape_short_val128_int.json`,
  `r1/runs/checks/stream_fixed_val128_int.json`,
  `r1/runs/checks/stream_fixed_horizon.json`,
  `r1/runs/checks/stream_fixed_suggest100_current.json`,
  `r1/runs/checks/stream_fixed_sentence_samples.json`,
  `r1/runs/checks/everyday_sentence20.json`,
  `r1/runs/checks/stream_fixed_sentence20.json`,
  `r1/runs/checks/stream_full_round1_val1000.json`,
  `r1/runs/checks/stream_full_round1_test1000.json`,
  `r1/runs/checks/stream_full_round1_horizon.json`,
  `r1/runs/checks/stream_full_round1_suggest100.json`,
  `r1/runs/checks/stream_full_round1_sentence20.json`,
  `r1/runs/checks/stream_wordstart4_val128.json`,
  `r1/runs/checks/stream_wordstart4_suggest100.json`,
  `r1/runs/checks/stream_wordstart4_sentence20.json`,
  `r1/runs/checks/everyday_test1000_int.json`,
  `r1/runs/checks/everyday_val1000_int.json`, and
  `r1/runs/checks/everyday_ngram_val1000.json`.

The short experiments were explicitly authorised after a `STOP_MICA` marker
was found. That marker was removed for the manual runs. `SUPERVISOR_EXIT`
remains in place, so the unattended supervisor is still disabled. No
background training was started.

Next, redesign how the recurrent state preserves and uses word information,
then test it with a human semantic review of fixed prompts and full sentences
alongside bits/target. Package the best model for the English demo site once
both numeric and sentence-quality criteria are met. The demo needs short
continuation and full sentence generation modes.
