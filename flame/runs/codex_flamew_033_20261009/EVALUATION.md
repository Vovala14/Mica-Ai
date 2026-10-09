# Flame-W 0.3.3 native integer short-memory blend

The selected candidate passed every frozen development, clean, blind sentence
no-regression, and public Tiny ToM no-regression gate in `PROTOCOL.md`.
This supports a small **equal-domain word-efficiency** update, with a measured
long-record cost. It does not establish better sentence quality or ToM ability.

## Mechanism and hashes

The original B-740 learned-integer-rule Minimal Inference Cellular Automaton
is unchanged: `model.mica` SHA-256
`1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`.
The released short integer memory SHA is
`5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7`;
the previously trained short bank SHA is
`5564ba558e7c1a6e594e324575333a626df16cc8605c9a23534d880acebab4ac`;
the released long bank SHA is
`f2abb7913dd626fb75cfeb96dd97c17c7d7ecbaed88f54c95fdb986734469f3b`.
For at most 64 fed word symbols, the integer bonus is
`floor((3 * released_short + fitted_short + 2) / 4)` elementwise. After 64,
the original long bank is used exactly. The fitted bank is an integer recall
and association readout over the automaton's word history. The cellular rules,
tokenizer, decoder, and answer-mode procedure are unchanged. The extra bank
is 673,650 bytes. No external language model or retrieval generator is used.

## Paired exact-integer language scores

Bits per word symbol plus EOS from BOS. Candidate minus released 0.3.2;
negative is better. The fourth development blocks are newly frozen and
disjoint from memory fitting and the first three development blocks.

| Set | 0.3.2 | Candidate | Delta, paired 95% CI |
|---|---:|---:|---:|
| Fresh short development, 1,000 records / 14,773 targets | 5.379107 | 5.358054 | -0.021054 [-0.022820,-0.019287] |
| Fresh long development, 128 records / 27,126 targets | 8.329670 | 8.334585 | +0.004915 [+0.002938,+0.007085] |
| Equal-domain development | | | -0.008069 [-0.009427,-0.006711] |
| Repeated clean short500, 7,320 targets | 5.208428 | 5.189430 | -0.018998 [-0.021376,-0.016611] |
| Repeated clean long500, 100,618 targets | 8.082371 | 8.086188 | +0.003818 [+0.002786,+0.004814] |
| Equal-domain clean | | | -0.007590 [-0.008872,-0.006321] |

Both frozen blends q4 and q8 passed development; q4 was chosen by lower
equal-domain loss. The long regression is real, and below the predeclared
+0.005 limit. The clean sets were inspected in prior research, so they are
confirmation of the new development decision rather than fresh selection.
`preflight.json`, `dev_results.json`, `clean_results.json` and gates retain
record SHA-256 values and per-record loss rows. Live-wrapper conformance
passed 572 positions; at 155 positions after the 64-symbol switch, source
and candidate integer scores were exactly equal.

## Blind sentences and Tiny Theory of Mind

On 20 previously unused short and 20 previously unused long clean prompts,
only four output pairs differed. The shuffled A/B ratings were written before
opening the key (ratings SHA-256
`e08115b2e8ab63de0521e3732152ef20f16ec2452af44ab400a8968838bffcc5`).
The candidate had **1 win, 1 loss, 38 ties**, with one new unusable output.
This passes the frozen no-regression gate but shows no sentence improvement.
Unedited short-prompt source → candidate examples:

- Prompt `I miss my old friends.`: ` But I really wanted to go to the park with my friends and friends.` → ` I don't know what I would do without you.`
- Prompt `I want to make her jealous`: ` of his friends.` → ` of the new York.`

Exact integer public Tiny ToM, 2,000 items, unchanged answer mode: word
normalization 632/2000 (31.60%) → 636/2000 (31.80%), paired +0.20 percentage
points [-0.25,+0.65], uncertain; character normalization 629/2000
(31.45%) → 633/2000 (31.65%), +0.20 [-0.25,+0.70]. This passes the frozen
no-point-regression >0.5-point gate, but does not prove improved ToM.

The release should state these limits explicitly. The sealed test set was
never accessed. Full model and memory hashes, exact records, blind samples,
ratings, ToM per-item rows and conformance are retained in this run folder.

## Independent packaged ToM rerun (2026-10-09)

The distributed 0.3.3 package's `run_tom.py` was run again on all 2,000 public
Tiny Theory of Mind records (dataset SHA-256
`58827a0361597ae4d5ffa79ddbab0936d1509cafcfc02cbb731e4d7a7f3584b8`).
The loaded `model.mica` SHA-256 was
`1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`;
all three memory-file hashes match those above. The rerun returned **636/2,000
(31.80%) word-normalized** and **633/2,000 (31.65%) character-normalized**.
Every item ID, label and both predictions matched the prior 0.3.3 evaluation;
all four ending log-likelihoods per item matched exactly (maximum difference
zero). The rerun output SHA-256 is
`ceecf482a8c88e444717cc14d324bde25d2a41bd0a70ffea0cdd4d0825065830`.
The packaged script has a stale `model` string of `MICA Flame-W 0.3.2` in its
output JSON; its loaded 0.3.3 `package.json` and checkpoint/memory hashes
identify the evaluated model. This metadata typo does not affect scoring.
