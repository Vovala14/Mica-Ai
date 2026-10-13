# Second candidate: existing readout coefficient calibration

Written after the bias-only candidate failed its independent confirmation.
That confirmation block is retired and will not select or validate this model.

Keep the v0.3.7 automaton, memory, vocabulary and word rules frozen. Reweight
the existing integer readout coefficient groups: current 240 probes, first
previous-word 240 probes, second previous-word 240 probes, and 96 rule outputs.
One additional scalar reweights the existing integer bias. Fit these five
scalars jointly per head on the same 800 training records as experiment 1;
no per-word delta from experiment 1 is used. Coefficient and bias scalars stay
between 0.5 and 1.5. Apply rounding and int8 saturation to exported coefficients;
measure quantized artifacts, never floating surrogate results.

Screen three L2 penalties around unit gains: 0.01, 0.1, 1.0. Choose by
equal-domain mean next-word / two-letter completion top-1 on the next 200
selection prompts per domain (positions 201-400 in the pre-existing deterministic
source pools). Ties choose stronger regularization. Confirm once on the next
200 confirmation prompts per domain, also positions 201-400. The source group
split is unchanged; these are fresh prompt blocks from the same split's
conversations/documents, rather than a wholly new corpus. Exact normalized
prompt-target identities must be disjoint from all experiment-1 prompts.

The original success and no-regression gates and source-group bootstrap apply:
>=1 percentage point primary gain, paired 95% lower bound >0, no domain
primary or individual task top-1 regression over 2 points. Report actual counts,
all candidate gains, artifact hashes and per-prompt outcomes. Do not tune using
the new confirmation block. Passing only establishes performance on these
additional domains; the original benchmark remains required for release.
