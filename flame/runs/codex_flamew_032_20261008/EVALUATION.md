# Flame-W 0.3.2 decision record

The original integer-rule B-740 cellular automaton SHA `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d` was frozen. The published short memory SHA `5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7` remains active through 64 fed symbols. The learned long integer memory SHA `f2abb7913dd626fb75cfeb96dd97c17c7d7ecbaed88f54c95fdb986734469f3b` is active thereafter. The `w-sent-bos.5f` beam decoder changes **only** its BOS anti-generic coefficient from 0.5 to 0.25 when a sentence prompt has more than 64 symbols. The proposal and its acceptance gate were frozen before the final clean sentence block was generated.

## Word loss

The exact exported integer engine (not trainer logits), bits per word symbol plus EOS from BOS:

| Set | 0.3.1 | 0.3.2 | Paired delta, 95% CI |
|---|---:|---:|---:|
| Clean short500, 7,320 targets | 5.208428 | 5.208428 | Exactly 0 |
| Clean long500, 100,618 targets | 8.305251 | 8.082371 | -0.222880 [-0.241879,-0.203769] |
| New long dev128, 27,126 targets | 8.580365 | 8.329670 | -0.250696 [-0.285911,-0.213699] |

The long dev128 set (SHA `ed3c9e7501f57dcc160cb025de325996eff9126f5ba4c39e926ec93c5778979f`) was excluded from the new memory fit and separate from the clean set, though it came from the original automaton training split. The clean loss sets were inspected during multiple iterations; treat them as repeated reporting evidence.

## Sentence check

One previously fixed decoder change was tried on a new 20-record development block (SHA `b3832c517af0571a0a117c935140313633e3d7c33f58f2c8949355538beaf6e9`), using the first 80 symbols of each record as a prompt. A single blind rater scored contextual relevance, grammar/completeness, looping and overall usefulness on 0/1/2 scales before A/B identities were revealed. The new version won 4, lost 0 and tied 16, with no additional severe failures. Its ratings file SHA is `97729676d645074fe6ab0221b04242c39990c32835780afd5f398fc3a1d355dd`.

The decisive clean block was long clean500 prompt indices 160:180, not previously used for sentence selection. The same blind rater saved ratings SHA `c7da9bd369466f432eae6e198b00cfac6975772a4686f80c78d07bb51f8c6d32` before reading the key. The new version won 5, lost 0 and tied 15; there were no additional severe failures. This passed the predeclared gate of at least as many wins as losses and at most two extra severe failures. The other 15 pairs were rated unusable for both. This is a small, subjective quality signal, not proof of generally coherent sentences.

The development and clean prompts, A/B outputs, hidden keys, locked ratings and gate JSON are preserved in the local research run `r1/runs/codex_flamew_032_sentence_20261008`. Portable package conformance reproduced three raw clean outputs and confirmed source integer logits through symbol64; 148 repository tests plus 11 subtests and the historical 0.3.1 byte-for-byte replay passed. The website API returned the exact clean candidate continuation on a 417-character prompt. Tiny ToM and leaderboard have **not** been measured for 0.3.2.
