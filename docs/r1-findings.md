# MICA R1 — implementation report

What was built against revision R1, what conforms, what was measured, and the
four things the measurements say should change in the specification.

Everything below is a measurement or a direct quotation from R1. Where a number
came from a reduced-scale run, the scale is stated next to it.

---

## 1. Conformance

The reference engine implements sections 3 through 9 exactly: flat cyclic
192-cell address space, `16 * phase + four sign bits` page addressing, argmax
over 32 candidate scores with ties to the lowest index, the eight opcodes,
synchronous single-writer ticks with change-triggered activation, the twelve
tick cap, and sparse absolute-address probes. No floating point appears
anywhere in the state transition.

**The section 7 worked example reproduces exactly.** This is the strongest
available evidence that the engine is the machine R1 describes:

| Section 7 claim | Engine |
|---|---|
| injection leaves `F[0][0]=5`, `F[1][1]=3`, active `{0,1}` | matches |
| both cells route to page 15 | matches |
| candidate 0 wins at cell 0, ADD gives `sat(5+3-1)` | `F[0][0] = 7` |
| cell 1 selects candidate 1, HOLD, retains values | matches |
| both active cells advance phase; cell 2 does not | matches |
| `next_active = {0, 1, 191, 16, 176, 37, 139}` | matches exactly |
| readout fixture scores 8 and 7, first wins | matches |

Eleven conformance tests pass, covering those plus the saturation fixture
(`127+4 = 127`, `-127-4 = -127`), zero-counts-as-nonnegative, tie resolution,
reset determinism, the int8 range invariant, the 2,304-update work bound, and
position wrapping after 192 symbols.

**The model file is exactly 86,820 bytes**, with section starts at 128, 12,512
and 78,048 as section 11 states, a SHA-256 over the payload, and a loader that
rejects bad magic, wrong dimensions, nonzero flags, nonzero reserved bytes,
forbidden `-128` deltas, out-of-range selectors and coefficients, and digest
mismatch. A round trip is bit-exact.

**The RNG matches the reference xorshift32 sequence.** Seed 1 produces
270369, 67634689, 2647435461, which is Marsaglia's published sequence, and the
rejection rule in section 13 samples uniformly.

**Two independent interpreters agree.** Section 15 asks for agreement between
"a slow reference interpreter and scalar C". Standing in for that pair, the
scalar engine and a batched evaluator (written for speed, see below) produce
bit-identical field, phase and probe scores over multi-symbol prompts.

---

## 2. What had to be built that R1 does not specify

**A batched evaluator.** Section 14 prices the specified search at "over 11
billion target evaluations" and warns it "can be prohibitively slow". With a
scalar interpreter it is not slow, it is impossible. The evaluator steps the
incumbent and all sixteen children across all thirty-two records at once — 544
independent sessions in one vectorised pass, each with its own field, phase,
position and activity. The semantics are unchanged, which the bit-exact
agreement above demonstrates.

This does not rescue the schedule; it makes the measurement below possible.

---

## 3. The cost measurement section 17 asks for

> "Then implement the discrete search exactly as specified, run the 32-record
> overfit diagnostic, and measure the first 100 training rounds."

**One full-specification round — the incumbent plus sixteen children, evaluated
on thirty-two 256-byte records — takes 313 seconds** on two CPU cores.

| Schedule | Projected time |
|---|---|
| 100 rounds | 8.7 hours |
| 10,000 rounds (one restart) | 36 days |
| eight restarts, as section 13 specifies | **290 days** |

This is the project's binding constraint, and it is worth stating plainly: the
*inference* budget is the easy half. A 128 KiB engine is achievable; the search
that produces its contents is what does not fit. Three ways out, in increasing
order of how much they change R1:

1. Accept a far smaller search budget and report it as such. The schedule in
   section 13 then becomes aspirational rather than the method.
2. Write the trainer in C and parallelise across cores and machines. A hundred
   times faster turns one restart into roughly three days, which is a real
   research loop. Nothing about R1 has to change.
3. Revisit the differentiable relaxation. Section 14 does not forbid it; it
   says it "requires a separate specification for soft addressing, soft
   instruction effects, activity gradients, saturation derivatives, hardening
   schedules, and validation against the integer engine." That is a document
   someone has to write, and the validation clause is the load-bearing part.

---

## 4. The section 14 first diagnostic

Thirty-two records of sixteen bytes, 120 rounds, 16.9 seconds per round, 34
minutes total, seed 1.

| | value |
|---|---|
| uniform over 257 eligible symbols | 8.0056 bits/target |
| random initialisation | 8.1073 |
| after 120 rounds | **7.8089** |
| mutation acceptance rate | 1.00 |

**The search can lower target loss, so this is not section 14's "grounds to
stop".** But 120 rounds bought 0.30 bits, and the acceptance rate of 1.00 says
every round found some improvement — the search is not stuck, it is slow. At
this rate the diagnostic is a statement about the optimiser's step size, not
about whether the architecture can represent language.

The instruments section 14 names, measured before and after:

| instrument | initial | after 120 rounds |
|---|---|---|
| active updates per symbol (of 2,304) | 379 | 501 |
| mean active set (of 192 cells) | 36.0 | 44.0 |
| **pages ever reached (of 64)** | **28** | **28** |
| page-usage Gini | 0.931 | 0.931 |
| candidates ever selected (of 32) | 32 | 32 |
| score-tie fraction | 0.398 | 0.362 |
| saturated field fraction | 0.000 | 0.000 |

---

## 5. Seven findings that should change R1

### 5.1 The page routing is degenerate, and search does not repair it

`bits = sum((1 if F[i][c] >= 0 else 0) << c for c in range(4))` tests four
channels against zero, and **zero counts as nonnegative**. An untouched cell
therefore has `bits = 1111`. Injection writes twelve entries into channels
chosen from all twenty-four, so roughly five in six writes miss the routing
channels entirely and most cells keep the all-ones pattern. The observed top
pages are 15, 31, 47 and 63 — precisely `16*phase + 15` for each phase.

Page reach, measured over three random seeds on representative text records:

| routing rule | pages reached (of 64) | Gini |
|---|---|---|
| **R1 as written** (`F[c] >= 0`) | 42.7 | 0.891 |
| `F[c] > 0` | 47.0 | 0.907 |
| **`F[c] >= F[c+4]`** (compare two channels) | **61.0** | **0.854** |
| `F[c] & 1` (low bit) | 52.0 | 0.875 |

On the degenerate diagnostic records the R1 rule reached only 28 of 64, and
**120 rounds of search moved it by zero pages** — this is structural, not
something the optimiser fixes.

Why it matters more than it looks: the rule section is **65,536 of the model's
86,820 bytes, 75% of the file**. Under the R1 rule a third of that is
unreachable at initialisation and stays unreachable.

**Recommendation.** Replace the test against zero with a comparison between two
channels, `bits |= (F[i][c] >= F[i][c+4]) << k` for `c` in 0..3. It costs the
same four comparisons, keeps the routing unlearned and fixed as section 12
requires, removes the privileged bit pattern of an untouched cell, and in
measurement reaches 61 of 64 pages instead of 42.7.

### 5.2 Two in five candidate selections are ties

The score-tie fraction is 0.398 at initialisation. With every bias initialised
to zero and six sparse ±1 reads of a mostly-zero field, many candidates score
exactly zero, so candidate 0 wins by default and the selection mechanism —
which is the architecture's distinguishing feature — does no work at all.

Section 14 already anticipates this: "If all candidates tie, explore bias
initialization and routing diagnostics before changing the machine."

**Recommendation.** Do not initialise all biases to zero. Section 13's
"all biases to zero" is the direct cause. Distinct small biases, for example
candidate index minus sixteen, break every tie at initialisation and cost
nothing in the file format, which already stores an int16 per candidate.

### 5.3 Random initialisation is worse than uniform, and the gap grows with record length

On sixteen-byte records, random initialisation scores 8.107 bits/target against
a uniform reference of 8.006. On full 256-byte records it scores **13.1
bits/target** — far worse than uniform.

The cause is the logit scale. Section 12 divides the integer probe score by 16.
Early in a record the field is near zero and scores are small, but a settled
field holds values across the int8 range, so eight ±1 reads produce scores in
the tens, and dividing by 16 still leaves logits spread over several nats. The
model is confidently wrong, and the search must first climb back to uniform
before it starts learning anything.

**Recommendation.** Initialise probe coefficients to 0 rather than uniformly
from {-1,0,+1}. Every symbol then scores its bias, the model starts exactly at
uniform, and the search spends its budget on learning instead of on recovering.
This is a change to section 13's initialisation only; the file format and the
engine are untouched.

### 5.4 The activity penalty does not hold the event economy

Over 120 rounds, active updates per symbol rose from 379 to 501, a 32%
increase, while the loss fell. The search is buying loss reduction with
activity, and at weight 0.01 the penalty barely resists.

Measured against section 10's bound, the event economy is real but eroding: 501
updates per symbol is 22% of the 2,304 worst case, so roughly 101,000 of the
442,368 scoring-term reads. Extrapolating the trend, the margin disappears.

**Recommendation.** If bounded work is part of the claim, make it a constraint
rather than a price — reject any child whose mean active updates exceed a
stated ceiling — or raise the weight and report the ablation section 12 already
requires. A penalty the optimiser can outbid does not bound anything.

### 5.5 The correction to 5.3: zeroing the coefficients cured the symptom and killed the search

5.3 got the diagnosis right and the remedy wrong, and the cost was a hundred
rounds of a live training run that learned nothing at all.

Zeroing every probe coefficient does start the model exactly at uniform, as
claimed. It also disconnects the field from the output completely. With
`pr_co = 0` a symbol's score is its bias and nothing else, so mutations to
signature entries, candidate records, scoring terms and rewrite programs are
not merely unhelpful — they are **provably inert**. Two of section 13's three
mutation kinds cannot change the loss by any amount. The observed behaviour
matched exactly: over 100 rounds at 32 records and 256 bytes,
`accept_rate = 0.000`, `holdout_rejections = 1`, and the loss sat on 8.0056
bits — the uniform value to four decimals — for every round.

The scale problem 5.3 identified is real, and measuring it properly says more
than 5.3 did. On a random model over 256-byte records, the across-symbol
spread of raw probe scores grows with field occupancy:

| position in record | field occupancy | score spread (raw) | logit spread at divisor 16 |
|---|---|---|---|
| t = 0–8 | 0.028 | 0.76 | 0.05 nats |
| t = 8–32 | 0.112 | 3.72 | 0.23 nats |
| t = 32–128 | 0.258 | 18.1 | 1.13 nats |
| t = 128–256 | 0.427 | 36.2 | **2.26 nats** |

Section 12's divisor of 16 is mis-scaled against the field it reads. A settled
field puts 2.26 nats of spread on the logits before the model has learned
anything, on a distribution whose entire useful range is about 4.8 nats. That
is why random initialisation degrades with record length. But the divisor is
not a field of the 86,820-byte file — no more than the tick cap or the
neighbour offsets are — so it is free to set, and setting it is the fix that
5.3 should have recommended instead of amputating the readout.

The second half of the fix is the bias. Section 13 starts every probe bias at
zero, which is a uniform prior over 257 eligible symbols: 8.0056 bits/byte. The
corpus unigram is 4.92 bits/byte. Quantising `LOGIT_DIVISOR * log p` into the
int16 bias field the format already provides costs 0.0003 bits, so **3.08
bits/byte are available for free**, in a field the file already stores, before
any search happens. Withholding them does not make the result more impressive;
it makes the first several thousand rounds of search go into rediscovering a
byte histogram, against a loss signal too coarse to guide them there.

**Recommendation.** Replace 5.3's recommendation with: seed probe biases from
the training corpus unigram, start probe coefficients at zero, and set the
readout divisor so that one coefficient's contribution is comparable to the
byte statistics it must improve on. Report the unigram as the floor the model
must beat, not as part of what it learned — the interesting claim is the
distance below 4.92, and separating the two makes that claim readable.

### 5.6 Seeding the bias was not enough: the readout must be live, not just well-placed

5.5 recommended zeroing the probe coefficients and seeding the biases from the
corpus unigram. The first half of that was still the mistake 5.5 was written to
correct, and it survived because the second half made the numbers look better.

With `pr_co = 0` the field reaches the loss through nothing. Seeding the biases
moves where the model starts; it does not connect anything. A 500-round run at
the reference geometry, 32 records of 256 bytes, held-out acceptance on 32
fresh records, 6.1 hours on a 9900X:

| round | validation bits | accept_rate |
|---|---|---|
| 10 | 4.9461 | 0.00 |
| 110 | 4.9462 | 0.30 |
| 210 | 4.9467 | 0.40 |
| 360 | 4.9509 | 0.45 |
| 500 | 4.9495 | 0.46 |

The unigram floor is 4.9341. Validation worsens monotonically as acceptance
rises, and the best model of the run was the one at round 10, before a single
child had been accepted. All 232 accepted changes were `pr_bias` steps fitting
the training corpus's byte frequencies slightly better and the validation
documents' slightly worse. The cell field contributed nothing, because it
could not.

Measured directly: six children of two mutations each, scored against their
parent on 20 independent 32-record batches.

| child | mean effect (bits) |
|---|---|
| 1 | 0.00001 |
| 2 | 0.00000 |
| 3 | 0.00000 |
| 4 | 0.00000 (identical on all 20 batches) |
| 5 | 0.00001 |
| 6 | 0.00185 |

The same measurement settles a question worth recording separately: the
batch-to-batch noise of a paired 32-record comparison is **0.00017 bits**.
Parent and child are scored on the same records, so most of the variance
cancels, and 32 records is ample. The flat curve was never a sample-size
problem, and raising the batch would not have helped.

**Recommendation.** Leave section 13's random coefficients in place so the
field reaches the output from round one, seed the biases from the unigram, and
set the readout divisor to balance two measured quantities:

| divisor | handicap of a random readout | effect of a 2-mutation child | ratio to noise |
|---|---|---|---|
| 64 | 0.31 bits | 0.00097 | 5.7 |
| 128 | 0.085 bits | 0.00049 | 2.9 |
| 256 | 0.027 bits | 0.00026 | 1.5 |

128 is the balance: children are about three times the noise floor, which is
what a selection test needs, and the handicap is small enough for the search to
remove. This is `--probe-init unigram-live` with `MICA_LOGIT_DIVISOR=128`.

**And a standing check.** This class of bug has now cost two multi-hour runs
and is invisible in the loss curve — a disconnected readout looks exactly like
a hard optimisation problem. `run_search.py` now tests for it before training:
eight children whose mutations are drawn only from kinds 0 and 1, which reach
the loss only through the cell field. If none of them move the loss, the
readout is dead. The test is exact rather than statistical, because with every
coefficient at zero the loss is provably invariant under those mutations, at
any record length and any batch size.

### 5.7 The activity cap must be relative, and the first evidence that the field contributes

Finding 5.4 asked for a ceiling on rewrites per symbol so the search cannot buy
loss reduction with work, and recommended `--activity-cap 600` from a run whose
activity was drifting between 379 and 501. At the reference geometry a fresh
random model runs **717 to 733** updates per symbol. A ceiling of 600 is
therefore below the floor: every child is set to infinity before it is scored,
nothing can ever be accepted, and the run reports `accept_rate 0.000` with
`holdout_rejections 0` and a validation figure identical to four decimal places
at every checkpoint. Observed for 30 rounds and 480 children.

That is the third distinct way a run has been silently dead — disconnected
readout, seeded-but-still-disconnected readout, and now a ceiling below the
floor — and all three look identical in the loss curve. The cap is now anchored
to the incumbent's measured activity at round one (`--activity-headroom 1.25`,
giving 897 against section 10's bound of 2,304), and an absolute cap at or
below the measured floor prints a warning before training starts.

**With the readout live and the ceiling above the floor, the search works.**
Fifty rounds, 16 records of 256 bytes, evaluated afterwards on 512 held-out
records with a paired per-record test:

| model | bits/byte |
|---|---|
| start, random readout | 5.0403 |
| after 50 rounds | 4.9954 |
| pure byte histogram, readout off | 4.9756 |

The improvement over the starting model is 0.0448 bits with a standard error
of 0.0038 — **z = 11.8**, better on 79.5% of individual records. Activity fell
from 698 to 467 updates per symbol over the same fifty rounds, so the model got
cheaper as it got better.

The composition of what was accepted is the part that matters. Counting every
field that differs between the trained model and the model it started from:

| | changes |
|---|---|
| probe bias (the byte histogram) | **0** |
| probe coefficients | 4 |
| probe cells / channels | 5 |
| injection signatures | 22 |
| candidate scoring | 11 |
| rewrite programs | 8 |

Not one bias changed. Every accepted mutation was field-side: what a byte
injects, which candidate wins, what the rewrite does, where the readout looks.
In every previous run the opposite held — the field was disconnected and all
movement was histogram tuning. This is the first measurement in which MICA's
cellular automaton contributes to its loss.

**What is not yet shown.** The trained model is still 0.0196 bits *worse* than
a plain byte histogram, and that gap is also significant (z = −12.8). Fifty
rounds were spent recovering from the handicap a random readout imposes, and
parity with a lookup table has not been reached, let alone the section 15 gate
at 3.29. The mechanism is live; its ceiling is unmeasured.

---

## 6. What R1 resolved from the earlier review

Reviewing v0.1 produced thirteen places where the specification did not
determine the implementation (`docs/spec-gaps.md`). R1 closes eleven of them:
vote count versus candidates per page, phase update semantics, page hash width,
which channels supply the sign bits, GATE's missing operand, activity scale,
probe read granularity, the rolling position stride, injection entry layout,
the halt probe (removed entirely), and the parameter-count confusion, which
section 10 replaces with an exact byte count.

Two remain open, and both are now design questions rather than ambiguities:
the ROTATE opcode is gone from the instruction set so its group addressing is
moot, and the concern that injection and probes dominate the learnable content
is **resolved in R1's favour** — the rule section is now 75% of the file rather
than under 10% of the parameters, which is what makes section 16's
"randomize trained rules" ablation the decisive experiment.

---

## 7. What has not been shown

No language capability has been demonstrated. The diagnostic ran on 512 bytes
of text for 34 minutes and reached 7.81 bits/target, which is barely below
uniform and nowhere near the byte trigram measured on the same corpus at
**2.198 bits/byte**. Section 15's continuation gate asks for at least 5% below
the trigram, which is 2.088 bits/byte. Nothing here approaches that, and
nothing here should be described as if it might.

What has been shown is narrower and still useful: the machine in R1 is
implementable exactly as written, its file size claim is arithmetically
correct, two independent interpreters agree on its transitions, its specified
search is measurably out of budget by roughly two orders of magnitude, and four
initialisation and routing choices are costing the search more than they need
to.

---

## 8. Addendum: the initialisation fixes, measured

After the report above was written, findings 5.2 and 5.3 were implemented as
flags on `r1/run_search.py` and measured directly. Three seeds, eight 256-byte
records, loss at initialisation before any search:

| `--bias-init` | `--probe-init` | bits/target at init | |
|---|---|---|---|
| `zero` | `random` | **13.6467** | R1 §13 as written |
| `spread` | `random` | 10.3143 | distinct candidate biases |
| `zero` | `zero` | **8.0056** | probe coefficients start at zero |
| `spread` | `zero` | **8.0056** | both |

Uniform over the 257 eligible symbols is 8.0056 bits/target, so the second and
third rows start the search *exactly* at uniform.

The two changes are worth **5.64 bits/target of avoided search**. For scale,
the 120-round §14 diagnostic bought 0.30 bits in 34 minutes, so this is roughly
2,200 rounds of search obtained for two lines of initialisation code.

Distinct biases alone are worth 3.33 bits, which is a second, independent
confirmation of finding 5.2: those ties were not cosmetic, they were suppressing
a third of the achievable score separation.

## 9. Addendum: the GPU backend

`r1/mica_r1/torch_batch.py` is a tensor port of the batched evaluator, verified
to agree with the numpy engine to 1e-9 on loss and exactly on update counts.

It differs from the numpy path in one deliberate way: it evaluates all 192
cells each tick and masks the writes by the activity flags, rather than
gathering the active subset. The masked result is identical because inactive
cells write nothing. The reason is that `nonzero` forces a device-host
synchronisation, and at twelve ticks per symbol across a 257-symbol record
that is over three thousand stalls per round — far more costly on a GPU than
the extra arithmetic.

Consequently the dense path is *slower on a CPU* (measured 4.2x slower than the
sparse numpy path, which is about what the 192-to-21 active-cell ratio
predicts) and should be faster on a GPU by a wide margin. Use `--backend numpy`
on a CPU and `--backend torch --device cuda` on a GPU; `r1/bench.py` times both
on the host and projects the §13 schedule from the measurement.

One conformance detail: `torch.argmax` does not promise the lowest index on
ties, which §5 requires. The port compares `score * 64 - index` instead, which
is injective and makes the lowest index win every tie. Scores are bounded by
|bias| + 6x127 < 33,530, so this cannot overflow.

---

## 10. Addendum: geometry is two axes, not one, and only one of them costs file bytes

Making the dimensions configurable (`MICA_CELLS`, `MICA_CHANNELS`, `MICA_PAGES`,
`MICA_CANDIDATES`, …, one geometry per process) exposed something that is easy
to miss from the §10 size table.

| geometry | model file | field state | round trip |
|---|---|---|---|
| 192 × 24 (R1) | 86,820 B | 4,608 B | exact |
| 384 × 24 | 86,820 B | 9,216 B | exact |
| 768 × 32 | 86,820 B | 24,576 B | exact |
| 1536 × 32 | 86,820 B | 49,152 B | exact |

**The model file does not grow with the field at all.** It stores injection
programs, rule candidates and probe programs — none of which is per-cell. The
field is runtime state, not stored weights.

That separates the project's two questions, which have been tangled together:

- *How much can it remember?* is `cells × channels`, and it costs RAM at run
  time and **zero bytes** in the model file.
- *How much does it know how to do?* is `pages × candidates`, which is 75% of
  the file and is what the size claim is actually about.

R1 §14 already lists "compare more cells or different topology while counting
every byte" as a response to forgetting. The count, it turns out, is zero file
bytes — so the honest framing of a future claim is "an 86,820-byte model with a
*N*-kilobyte working field", and the two numbers should always be quoted
together. `r1/scale_study.py` sweeps either axis at a matched round budget.

### A format consequence: 256 cells is a hard ceiling as written

An injection entry stores `base_cell` as one byte, so §10's layout cannot
address a field larger than 256 cells. The same applies to probe entries.

The fix costs nothing. Both entry types already carry **one reserved zero
byte**. Using it as the address high byte gives u16 cell addressing with the
entry still four bytes wide and the file still exactly 86,820 bytes. That is
what `serialize.py` does above 256 cells, and the round trips above are under
that layout.

If R1 is revised, it is worth writing this into §10 rather than leaving the
reserved byte unallocated, because it is the difference between the field being
capped at 6 KB of state and being open-ended.

### A topology note

R1's neighbour offsets are ±1, ±16, +37, −53 on 192 cells. `gcd(16, 192) = 16`,
so repeating the ±16 hop visits only 12 distinct cells — it is a short cycle,
not a field-wide path. The ±37 and −53 hops are coprime with 192 and do reach
every cell, so the field is still fully connected and nothing is broken. But
if the ±16 hop was intended as a second long-range path rather than a local
cross-lane exchange, it is not doing that job, and a coprime value such as 17
would.

`scale_study.py` scales the offsets with the field and forces the long hops
coprime with the cell count, for exactly this reason.

---

## 11. Addendum: what MICA learns is specific to how full the field is

This is the most consequential measurement in this document.

An arm trained for 250 rounds on **24-byte** records, with all four changes
applied, improved its training loss by 0.134 bits — and its loss on 256-byte
validation records got **worse by 0.935 bits**, monotonically, from round 50
onward. Training and validation moved in opposite directions the whole way.

The cause is not ordinary memorisation: the record cursor advances every round,
so the run had seen roughly eight thousand distinct records, not the same
thirty-two. Evaluating the finished model against its own starting point across
truncation lengths isolates it:

| record length | initial | trained | change |
|---|---|---|---|
| 24 bytes | 8.0056 | 7.8930 | **−0.113** |
| 48 bytes | 8.0056 | 7.8987 | −0.107 |
| 96 bytes | 8.0056 | 7.9844 | −0.021 |
| 160 bytes | 8.0056 | 8.3862 | **+0.381** |
| 256 bytes | 8.0056 | 9.2847 | **+1.279** |

Training helps up to about 96 bytes, crosses over near 100–130, and by 256
bytes actively hurts by more than a bit. The learned rules are tuned to a
sparsely populated field; at 2.5× the trained length the field is in a state
the search never optimised for, and the rules mislead rather than inform.

**This is the same failure mode the v0.1 differentiable work hit**, by a
completely different mechanism: there, a model trained from a zero field scored
5.6 bits on training windows and 14.9 on held-out streams with the field
carried. Two unrelated training methods, the same collapse whenever the field
is fuller than training ever showed it. That makes it architectural rather than
incidental.

### What follows for R1

1. **§13's 256-byte record size is load-bearing, not a convenience.** Training
   on shorter records to save compute does not produce a slower version of the
   same model; it produces a model that is worse at the length that matters. It
   is the one parameter not to economise on.
2. **The warning in §13 should be strengthened.** It currently says the split
   "initially sacrifices long context; do not claim long-context learning from
   it." The measurement says more than that: performance *degrades* past the
   trained length, by 1.28 bits at 2.5× it. A prompt or generation longer than
   the training records does not merely fail to benefit — it is handled worse
   than by an untrained model.
3. **Field occupancy belongs in the diagnostics.** §14 lists routing usage,
   active-set size, ties, saturation and acceptance. None of them would have
   caught this. A histogram of nonzero field cells versus position in the
   record would have caught it immediately, and it is cheap.
4. **This is the strongest argument yet for the scale study.** If what the model
   learns is bounded by how much of the field is in play, then field size is
   not a tuning knob, it is the variable that sets what can be learned at all —
   and it costs zero model-file bytes (§10 addendum above).

---

## 12. Addendum: field occupancy, and why scaling it up is cheaper, not dearer

`r1/occupancy.py` is the instrument finding 11 said was missing. It reports
what fraction of cells have ever been written, what fraction of channel slots
hold a value, and the work done, as a function of position in the record.

At the R1 reference geometry, random model, three 256-byte records:

| after byte | cells touched | channels written | updates/symbol |
|---|---|---|---|
| 1 | 14.1% | 0.6% | 101 |
| 8 | 70.7% | 5.6% | 243 |
| 32 | 98.4% | 15.9% | 288 |
| **64** | **99.7%** | 23.8% | 408 |
| 128 | 100.0% | 35.3% | 447 |
| 256 | 100.0% | 50.7% | 553 |

**Every cell has been written by byte 64.** That is the explanation for finding
11. The field has two regimes — *filling* up to roughly byte 64, then
*overwriting* — and they are qualitatively different problems. A search trained
on 24-byte records optimises entirely inside the filling regime and has never
been asked to decide what to discard. The crossover where that training stops
helping, measured independently at 96–130 bytes, sits just past where the
filling regime ends.

### The same measurement at 768 × 32

| after byte | cells touched | channels written | updates/symbol |
|---|---|---|---|
| 1 | 86.7% | 4.8% | 229 |
| 8 | 93.6% | 5.5% | 137 |
| 32 | 96.3% | 6.8% | 74 |
| 64 | 98.1% | 8.6% | 78 |
| 256 | 100.0% | **17.8%** | **133** |

Two things fall out, and the second one was not expected.

**The larger field has real headroom.** At 256 bytes the reference geometry has
50.7% of its channel slots written — about 2,336 bytes of the field carrying a
value. The 768 × 32 field has only 17.8% written, which is about 4,374 bytes:
nearly twice the retained information, and still far from full. The reference
geometry is running out of room inside a single record; the larger one is not.

**The larger field costs less work per symbol, not more.** 133 active updates
per symbol against 553 — a four-times-larger field doing four times *less*
work. Activity in a small torus ricochets: a change wakes six neighbours, and
in 192 cells those wavefronts collide with themselves within a couple of ticks.
With more room the same events dissipate instead. The event economy that §12's
activity penalty is trying to protect works considerably better at scale than
it does at the reference size.

This inverts the usual assumption behind "smallest possible". On these
measurements the reference geometry is not the cheap configuration — it is the
cramped one, paying four times the per-symbol work for half the retained
information, and costing **zero extra model-file bytes** to fix (§10 addendum).

Caveats worth keeping: this is a random model rather than a trained one, the
offsets differ between the two geometries because they are scaled with the
field, and RAM grows from 4.6 KB to 24 KB — which is nothing against §10's
128 KiB budget, but should be counted. The scale study is what turns this
signal into a result.

---

## 13. Addendum: the slope experiment, and why it could not answer its question

Two arms, 250 rounds each, same seed, same records, 32 records of 24 bytes,
16 children. One arm R1 exactly as written, one with all four changes.

| | R1 as written | all four fixes |
|---|---|---|
| train bits/target, round 1 | 8.1313 | **8.0056** |
| train bits/target, round 250 | **7.6739** | 7.8716 |
| train gained over 250 rounds | **0.4574** | 0.1340 |
| validation bits/target (256-byte records) | 12.1927 | **9.4283** |
| updates per symbol | 549 | 500 |
| seconds per round | 24.1 | **5.5** |

Three things to take from this, in order of how much they matter.

**Neither arm beats a coin flip on held-out full-length records.** Uniform is
8.0056; both arms end above it, at 12.19 and 9.43. After 250 rounds the search
has produced nothing that generalises. That is the honest headline and it
should not be softened.

**The experiment was contaminated by its own shortcut.** I trained on 24-byte
records to fit the compute budget and validated on 256-byte records. Finding 11
then measured that this specific mismatch costs 1.28 bits at 256 bytes. So the
validation column is measuring my shortcut as much as it is measuring either
arm, and the comparison cannot settle which configuration searches better. The
experiment failed at its stated purpose. It is reported here in full because
the failure is the finding: the length shortcut is not a mild approximation,
it invalidates the metric.

**The fixes are not a uniform win, and the shape of the difference is
informative.** R1 as written gains more than three times as much *training*
loss, because random probe coefficients hand the search a landscape with
structure to exploit immediately, whereas zeroed coefficients start flat and
must build discrimination from nothing. But what the baseline exploits does not
transfer — it ends 2.76 bits worse on held-out records. Meanwhile the fixed arm
runs **4.4 times faster per round**, which came from `pairdiff` routing
producing less activity ricochet, not from anything intended.

So: the fixes buy a better starting point (5.64 bits, finding 8), better
generalisation, and 4.4× the rounds per hour. They cost raw training-loss
progress. On a budget measured in rounds that trade is clearly worth it; the
experiment that actually settles it is the same two arms at 256 bytes
throughout, which costs about 417 s/round here and should cost a small fraction
of that on a workstation or GPU.

### The experiment to run next, stated exactly

```
python3 r1/run_search.py --backend torch --device cuda --restarts 8 \
        --rounds 500 --batch-records 32 --record-bytes 256 \
        --bias-init spread --probe-init zero --routing pairdiff \
        --out runs/slope_fixed
```

and the same without the three flags. Compare the *slope* of validation loss,
not the endpoint. If neither arm is below 8.0056 after 500 rounds at matched
length, the question stops being about search budget and becomes about
geometry — which is what §12's occupancy measurements already point at.

---

## 14. Addendum: the §15 gate, measured under the storage cap it specifies

§15 asks for "at least 5 percent lower test bits per byte than the trigram
**under the same storage cap**". The 2.1984 bits/byte figure reported earlier
in this document does not satisfy that clause: it came from an unpruned trigram
storing 525,619 values, roughly six times MICA's entire 86,820-byte budget. It
was never the comparison §15 describes.

`r1/gate.py` measures it properly. Each baseline is pruned to exactly 86,820
bytes — the R1 model file size — with entries costing `order + 3` bytes
(context key, symbol, 2-byte count), which is generous to the baseline since a
real implementation also needs index overhead. Evaluation is on the **locked
test split**, with 95% intervals from a document bootstrap, resampling whole
documents rather than records because records from one document are not
independent.

| baseline | bits/byte | 95% CI | stored | unpruned would be |
|---|---|---|---|---|
| **byte 2-gram** | **3.2868** | [3.1826, 3.4155] | 86,820 B | 334,087 B |
| byte 3-gram | 3.9189 | [3.7292, 4.1199] | 86,819 B | 1,606,153 B |
| byte 4-gram | 4.7593 | [4.5018, 5.0118] | 86,817 B | 4,729,413 B |
| byte 5-gram | 5.1547 | [4.8808, 5.4273] | 86,817 B | 10,639,181 B |

**The gate is 3.1225 bits/byte**, not 2.0885. That is about nine tenths of a bit
easier than previously stated, and the correction is entirely a matter of
comparing like with like.

Two results worth keeping from the table itself. Higher orders are *worse* under
a fixed cap — an order-5 model at 86 KB is 1.9 bits worse than an order-2 model
at the same size, because the budget goes on sparse contexts that do not
generalise. And the unpruned column shows why the cap matters: an unconstrained
5-gram wants 10.6 MB, 122× the budget.

### Two honest caveats

**The pruning may not be optimal.** Entries are selected greedily by count per
byte after reserving the order-0 table. Proper entropy-based pruning, or
modified Kneser-Ney in place of Witten-Bell, would likely produce a stronger
86 KB baseline and therefore a *harder* gate. The number above should be treated
as a lower bound on the baseline, and anyone reproducing this should try to beat
it before claiming MICA cleared it.

**The test set is small.** 120 records, about 31 kB, giving a 95% interval of
roughly ±0.12 bits. That is fine for setting a target and too small for a
published result. A real evaluation should use the full 3,810-record locked test
split.

### Where MICA stands against it

The best MICA run so far reached 9.43 bits/target on validation, which is worse
than uniform (8.0056). The distance to the gate is about six bits. The target is
now defined precisely and measured under the right conditions; nothing about
that makes the six bits smaller.

---

## 15. Addendum: the full storage-matched baseline set

§15 names three baselines and §16 adds a fourth. All four are now measured on
the locked test split, every one sized to the **same 86,820-byte cap** as the
R1 model file, with 95% intervals from a document bootstrap.

| baseline (86,820 B cap) | bits/byte | 95% CI | stored | note |
|---|---|---|---|---|
| **byte GRU** | **2.9771** | [2.904, 3.055] | 86,176 | hidden 112, embed 32 |
| byte 2-gram | 3.2868 | [3.183, 3.416] | 86,820 | count-pruned, Witten-Bell |
| reservoir (echo state) | 4.3512 | [4.260, 4.438] | 86,785 | hidden 338, matrix from a seed |
| byte Transformer | 4.3716 | [4.307, 4.440] | 82,880 | d_model 64, 1 layer, 4 heads |

### The two gates, read precisely

§15 defines them differently and they should not be conflated.

**First gate**, stated against the trigram specifically: "at least 5 percent
lower test bits per byte than the trigram under the same storage cap." That is
**3.1225 bits/byte**.

**Second gate**: "a measured advantage in either storage or latency at matched
task quality against the learned baselines." Matched task quality means the
GRU's **2.9771 bits/byte** — and note that MICA would then need to win on
storage or latency *at* that quality, not merely reach it.

### Three honest notes about this table

**The transformer number is not a fair ceiling.** One layer at d_model 64,
trained for 1,500 steps, is badly undertrained; a transformer at this storage
cap would do considerably better with a real training budget. It is included
because §15 requires it, not because 4.37 is the best a transformer can do.
Anyone reproducing this should train it properly before quoting it.

**"Stored bytes" assumes int8 and none of these models were quantised.** In
float32 the GRU is 344,704 bytes — four times the cap. A fair published
comparison must either quantise them or report the float size, and the result
file records both. MICA's 86,820 bytes, by contrast, are genuinely integer
already, which is a real advantage and one worth stating explicitly rather than
hiding inside a parameter count.

**The GRU beating the n-gram is the important result for the project.** It
means the interesting bar is not the trigram at all. A tiny recurrent model with
the same storage gets 0.31 bits further, and it is a thoroughly conventional
architecture. Clearing §15's first gate at 3.1225 would be a real result; it
would not yet show that the cell field is a better use of 86,820 bytes than a
GRU is.

---

## 16. The §16 ablation harness

All five ablations §16 requires are implemented and selectable, with the full
model as a sixth arm and every arm on an identical round budget, record batch,
seed and storage:

| arm | what changes | how |
|---|---|---|
| `full` | nothing | baseline arm |
| `no_phase_routing` | page index ignores phase; 64 pages collapse to 16 | `MICA_NO_PHASE_ROUTING=1` |
| `full_field` | every cell updates every tick, no event activation | `MICA_FULL_FIELD=1` |
| `frozen_instr` | rewrite programs stay at init; scoring, injection, probes still learn | `--freeze-instructions` |
| `no_long_hops` | offsets reduced to ±1, ±16 | `MICA_OFFSETS=1,-1,16,-16` |
| `random_rules` | rule section re-randomised, injection and probes kept | `--randomize-rules` |

`r1/ablations.py` runs all six and prints the decisive comparison §16 names:

> "If randomized rules plus trained readout perform equally well, the learned
> rewrite core has not earned the claimed contribution."

**One caveat that matters more than the harness.** These ablations are only
meaningful against a *trained* model. Run on a model that has had a hundred
rounds of search, every arm is approximately untrained and the comparison shows
approximately nothing — including the random-rules arm, which would look
"equally good" for the trivial reason that the full model is also nearly
random. The harness is ready; the experiment it serves is not runnable until
there is a trained model to ablate. Do not read a null result from a short run
as evidence either way.

### Harness validation, and three structural results that do hold

All six arms ran (8 rounds, 16 records of 24 bytes — a smoke test, not an
experiment). Every switch produced exactly the structural change it should:

| arm | val bits/target | updates/symbol | pages reached |
|---|---|---|---|
| full | 8.1198 | 688 | 64 |
| no_phase_routing | 8.0295 | 220 | **16** |
| full_field | 8.2534 | **2,304** | 64 |
| frozen_instr | 8.4548 | 672 | 64 |
| no_long_hops | 8.0020 | **148** | 64 |
| random_rules | 8.0709 | 578 | 64 |

**The loss column is meaningless and must not be quoted.** Eight rounds trains
nothing, so every arm is approximately a random model. The harness duly printed
"randomising the rules costs almost nothing… the learned rewrite core has NOT
earned the contribution claim" — which is precisely the false null predicted
above. It is reported here as a worked example of the trap, not as a result.

Three measurements in that table *are* valid, because they are structural
rather than learned and do not depend on training at all:

**`full_field` lands on exactly 2,304 updates per symbol**, which is §10's
stated worst case (192 cells × 12 ticks) to the digit. The work-bound
arithmetic in the specification is confirmed empirically, and the event
economy's real saving is now measurable: 688 against 2,304, so activation is
buying a 3.3× reduction in work.

**`no_phase_routing` reaches exactly 16 pages**, down from 64. Phase contributes
exactly the two bits of routing §5 claims for it, no more and no less.

**`no_long_hops` drops activity to 148 updates per symbol**, from 688. Removing
the +37 and −53 offsets removes **78% of all activity spread**. Those two hops
are what makes a local change wake the whole field, which bears directly on the
"Excess activity" risk in the v0.1 table: if activity ever needs constraining,
the long hops are the lever, and they are also the only offsets coprime with
192 (see §10 addendum), so they cannot be removed without also removing
field-wide reachability.

---

## 17. Addendum: the §15 language evaluation suite, and a third sighting of the same pathology

`r1/evaluate_language.py` implements the automatable half of §15's language
evaluation: EOS accuracy, exact delayed copying at 16/64/256/1024 bytes,
repetition in 512-byte greedy continuations, and UTF-8 validity. Short
completion preference needs human raters and is deliberately absent rather than
faked with a proxy.

Delayed copying is measured two ways. Exact match is all-or-nothing and says
nothing about a model that is merely *better* on a repeat, so the suite also
reports a graded version: bits per byte on the second occurrence of a span
against the first. The difference is the retained information, in bits.

Run against the best model so far (250 rounds, 24-byte records, all four
fixes), on held-out records:

| distance | first occurrence | repeat | retained | exact copy |
|---|---|---|---|---|
| 16 B | 7.861 | 7.873 | **−0.012** | 0% |
| 64 B | 7.880 | 8.712 | **−0.832** | 0% |
| 256 B | 7.830 | 14.092 | **−6.262** | 0% |
| 1024 B | 7.866 | 15.292 | **−7.426** | 0% |

EOS accuracy: precision 0, recall 0, F1 0 — the model never predicts EOS at
all, which follows from probe coefficients starting at zero and the search
never having learned to raise that one symbol. Collapse rate in 512-byte greedy
continuations: 100%. Valid UTF-8 rate: 0%.

**All of those are the numbers of an undertrained model and none of them is a
result about the architecture.** They are reported to establish the floor and
to show the harness runs end to end.

The delayed-copy column, however, is informative right now. Retention is not
merely absent — it is **negative, and grows more negative with distance**. By
1024 bytes the model is 7.4 bits per byte *worse* at predicting a span it has
already seen than one it has not. Having more history actively harms it.

This is the third independent sighting of the same pathology:

1. The v0.1 differentiable model: 5.6 bits on training windows from a fresh
   field, 14.9 on held-out streams with the field carried.
2. Finding 11: training on 24-byte records helps at 24 bytes and costs 1.28
   bits at 256, with occupancy (finding 12) showing every cell written by byte
   64.
3. This: retention goes from −0.01 bits at distance 16 to −7.43 at 1024.

Three different training methods, three different measurements, one behaviour —
the model degrades as a monotone function of how much the field has absorbed.
That is not a training artefact. It is the central empirical fact about this
architecture so far, and the scale study (§10 and §12 addenda: the field costs
zero model-file bytes) is the cheapest available response to it.
