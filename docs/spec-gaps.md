# Specification gaps found while implementing MICA 0.1

These are places where the technical design document does not determine the
implementation: two engineers reading it would build different models. Each
entry names the text, says what is ambiguous, and records the choice this
repository made so the choice can be reviewed rather than silently inherited.

The document is in good shape. Everything below is a gap in *precision*, which
matters here more than in a typical spec because the doc itself demands exact
reproducibility: "Saturation, integer width, right shifts, and queue ordering
are specified exactly."

---

## 1. Vote count does not match instructions per page

**Text.** "A rule page contains 32 candidate instructions... Within the page,
six sparse votes choose the winning instruction." Also: "Threshold comparisons
select one instruction from a fixed instruction set."

**Gap.** If each vote contributes one threshold comparison, six votes address
2^6 = 64 instructions, not 32. Either pages hold 64 instructions, or two votes
are combined, or selection is not a plain binary code over the vote bits.

**Choice here.** Five votes, 32 instructions, one bit per vote, index = the bit
pattern. `MicaConfig.validate` asserts `instr_per_page == 2 ** n_votes` so the
inconsistency cannot be reintroduced by accident.

**Decision needed.** Whether MICA XS keeps six votes (then pages hold 64) or
keeps 32 instructions (then it has five votes). This changes the packed model
size, so it should be settled before the format is frozen.

---

## 2. Nothing updates the phase tag

**Text.** The cell field table lists "Phase | 2 bits | Prevents one rule from
firing every tick." The update cycle never writes it.

**Gap.** Phase is presented as cell *state*, but no instruction in the eight-opcode
set modifies it and the settling loop does not advance it. As written, phase is
a constant, and then it does not prevent anything.

**Choice here.** Phase is derived, not stored: `phase(cell, tick) = (cell_index
+ tick) mod 4`. This gives the stated effect (a cell's rule page changes every
tick, and neighbouring cells are out of step) and costs no state at all.

**Decision needed.** If phase is meant to be written by rules, that is a
different and considerably more expressive mechanism, and it needs an opcode.
If it is derived, the field table should not list it as stored state — which
also frees two bits per cell.

---

## 3. The page hash addresses the wrong number of pages

**Text.** "The cell hashes its phase and three sign bits into a page index...
MICA XS uses 64 pages."

**Gap.** Two phase bits plus three sign bits is five bits: 32 distinct
combinations. With 64 pages, half the rule book is unreachable. With the Phase
One configuration's 16 pages, two combinations collide on every page.

**Choice here.** `page = (phase * 8 + sign_bits) mod n_pages`, which is honest
about the collisions but does not pretend to reach 64 pages.

**Decision needed.** Either more sign bits (six sign bits plus phase gives 256
combinations, enough for 64 pages with a stated fold), or fewer pages. This one
directly caps how much rule diversity the architecture can express, so it is
worth resolving early.

---

## 4. Which channels supply the sign bits is unspecified

**Gap.** "Three sign bits" — of what? The doc defines four control channels but
does not say that these are the hash inputs.

**Choice here.** The first three control channels. This makes the page choice a
learnable function of a dedicated control lane rather than of semantic content.

---

## 5. GATE has no "following instruction" to gate

**Text.** Instruction table: "GATE | Apply a following instruction only above a
threshold."

**Gap.** The update cycle selects exactly one instruction per cell per tick.
There is no following instruction for GATE to apply. As specified, GATE is a
no-op.

**Choice here.** GATE is treated as a gated ACCUM: the accumulate happens only
when the source value clears the threshold. This preserves the intent (a
conditional write) inside the one-instruction-per-tick rule.

**Decision needed.** Either instructions come in pairs — which doubles the
selection cost and changes the packing — or GATE becomes a *flag* on an
instruction rather than an opcode, which is cheaper and is what the choice here
amounts to. If GATE becomes a flag, the opcode field drops to seven values.

---

## 6. Activity has no stated scale

**Text.** "Activity | 1 uint8"; "Each rewrite spends one activity unit unless it
receives a pulse"; "PULSE | Increase activity of a target neighbor".

**Gap.** Nothing says what activity a byte injection deposits, or how large a
pulse is. Those two numbers set the per-byte tick budget and therefore the
runtime's whole cost model — the thing the doc's "Core update work | Under 1.5
million simple integer operations per byte" target is measured against.

**Choice here.** Activity normalised to [0, 1], injection deposits 1.0, each
rewrite spends 0.25, a pulse adds 0.5. A freshly injected cell can therefore
fire four times, matching the Phase One tick cap.

**Decision needed.** The integer values, stated in the model header, since the
loader must validate them.

---

## 7. "Eight field locations" is ambiguous in the probe bank

**Text.** "A program reads eight field locations, applies ternary signs, adds a
byte specific bias, and returns an int16 score."

**Gap.** A "location" could be a cell (all 24 channels) or a (cell, channel)
pair. The two differ by a factor of 24 in both read cost and expressiveness.

**Choice here.** A (cell, channel) pair, which is what makes the int16
accumulator arithmetic work out and keeps decoding as cheap as the doc claims.

---

## 8. ROTATE group addressing

**Text.** "ROTATE | Rotate a four channel group by one position."

**Gap.** Whether groups are a fixed aligned partition of the channels or an
arbitrary set of four channel indices. Aligned groups need two bits to name;
arbitrary sets need four selectors.

**Choice here.** Aligned partition, one group index per instruction.

---

## 9. The rolling position stride is unspecified

**Text.** "Position is represented by a cheap rolling counter that rotates
anchor choices."

**Gap.** No stride is given. A stride sharing a factor with N makes anchors
revisit the same cells quickly, which would make position nearly invisible.

**Choice here.** Stride 7, coprime with 96, so the rotation has full period.
For any tier, the stride must be coprime with the cell count; that constraint
belongs in the format validation rules.

---

## 10. Injection channel indices: per anchor or global?

**Text.** "A byte selects a compact signature with four anchor indices, twelve
channel indices, twelve signed magnitudes, and a phase mask."

**Gap.** Twelve channel indices across four anchors is three per anchor if they
are partitioned, or twelve shared choices if they are global. The doc does not
say, and the packing format section does not disambiguate.

**Choice here.** Partitioned: each anchor owns its own writes.

---

## 11. The halt probe's form and threshold

**Text.** "the global halt probe crosses its threshold".

**Gap.** Neither the probe's structure nor its threshold is given, and "global"
suggests it reads the whole field while every other probe in the design reads
eight locations.

**Choice here.** Same shape as a byte probe — eight sparse reads, ternary signs,
one bias — with a learned threshold. Reading the whole field on every tick
would be the single most expensive operation in the engine, which is unlikely
to be the intent.

---

## 12. The two parameter counts differ by more than the doc implies

**Text.** "Counts include ternary vote signs, selectors, thresholds, byte
signatures, biases, and probe programs... Published comparisons must report
both stored bits and executed operations."

**Observation, not a gap.** The doc is right to warn about this, and the size of
the effect is worth recording. In the Phase One configuration built here, the
trainable surrogate has about 418,000 parameters while the hardened model stores
about 17,800 discrete values — a ratio of roughly 24 to 1, because every
categorical selector collapses from a distribution to a single index.

Both numbers are real and they answer different questions. The surrogate count
is what a baseline must be matched against to make the *training* comparison
fair; the hardened count is what the deployment claims rest on. Reporting only
one of them would flatter MICA in one direction or the other, so
`train/compare.py` prints both.

---

## 13. Where the byte probes' cost actually lands

**Observation.** The doc says "Because the vocabulary is tiny, evaluating every
candidate is inexpensive and avoids a large output matrix." That is true of the
*runtime*, but in the surrogate the injection table and probe bank together hold
about 97% of the trainable parameters (170k + 238k of 418k). The rule book —
the part the architecture's novelty claim rests on — is under 10%.

This is worth knowing before the ablations in the "Minimum credible publication
package" are designed: if MICA does well, the first question a reviewer will
ask is whether the byte signatures and probes are doing the work and the cell
field is along for the ride. An ablation that freezes the rule book at random
initialisation and trains only signatures and probes would answer it directly,
and it is cheap to run.
