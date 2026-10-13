# Exact Ember completion optimization

The Ember037 word runtime scores only vocabulary rows compatible with the typed
prefix and processes the causal byte history only through the feature position
it actually uses. Integer rules, weights, vocabulary and tie behavior are unchanged.
The optimization itself preserves source scores with CompletionEngine(alpha=0).
This checkout also enables a separately validated9:7 completion decoder policy;
see ember-completion-calibration.md for its accuracy and conditional-loss evidence.

`complete()` stays stateless. Local applications can opt into a typing session:

```python
from word_model import EmberWord
model = EmberWord()
typing = model.new_completion_session()
typing.complete("The weather is su")   # computes cellular features
typing.complete("The weather is sun")  # reuses the same exact features
typing.clear()                         # discard context at logout/end of typing
```

A session is bound to one frozen model. Replacing/refitting a model requires a
new session. Context edits, case changes, task/first-two-letter changes and TTL
expiration invalidate feature reuse. Default TTL60seconds, max8192UTF8bytes;
larger contexts are computed without caching. TTL limits reuse: references are
cleared on the next call; clear() removes them immediately. No disk persistence,
global user cache, cross-user sharing or automatic website session is introduced.

Local validation:1028 source/integrated-runtime parity checks, typing/backspace/context/case edits,
Unicode/whitespace, empty/out-of-vocabulary cases, expiration/limits, independent
sessions and concurrent use. On four prompts (one BLAS thread, nine repeats),
native cold51.9–65.2ms, optimized cold43.3–59.9ms, warm0.048–0.155ms.
Warm numbers exclude initial computation and network time. This is a local
four-prompt benchmark, not a general hardware/API speed or quality claim.

Evidence: `r1/runs/codex_ember_rulelearn_20261012/speed_results.json` in the
research workspace. Published Ember037 artifacts and live website are unchanged
until an explicit release. A separate accuracy pilot is not evidence for this
runtime's accuracy. The historical timing figures above measure the alpha0
optimization; they must not be presented as timings for the new two-head policy.
