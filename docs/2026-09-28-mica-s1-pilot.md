# S1 native state-conditioned bucket pilot: predeclared protocol

Status: designed 2026-09-28 10:05 UTC; no S1 code or training result yet. Shared training code is in use by the 40-round Flame mix-A run, so implementation follows when it finishes. This is a bounded pilot, not a new architecture or an external model.

Flame mix-A training: `r1/runs/codex_flame_balanced_v02a_20260928`, hidden runner PID 34060, 40 rounds. Its exact integer four-set evaluator is prepared in `r1/checks/run_mica_flame_balanced_eval.py`; run only after `FINISHED` and release the GPU claim promptly. This mature candidate is a separate experiment from the one-round S1 arms.

## Question

Can a MICA rule winner depend jointly on recent bytes and an earlier word, while retaining useful short-context rules? The existing lexical2 pilot proved that an integer-rule word state survives beyond eight bytes but lost clean prediction quality when four phases were devoted to state and its separate features. S1 spends only phases 0/1 on the two required word-state tracks and lets four existing local-rule phases read the previous-word track inside their **hard candidate scores**. The other local phases stay as the original byte-tape rules. Inference remains the exported cellular automaton and integer readout.

## Matched one-round arms

All arms: F1 Flame geometry, 1,024 candidates, same seed, selected balanced mix A train and val, same 40,000 fit records / 1,500 fit steps / one round, native fit mode. Identical data order and training budget; only integer rule topology differs.

1. **F1 control:** original structured tape-only rule book, no state.
2. **State-only control:** lexical previous/current-word transitions in phases 0/1; phases 2–15 retain ordinary local tape scoring and VSET features. No separate decoder phases. The two state VSET groups are excluded from ordinary readout probes. Freeze transition immediates and fit remaining VSET/readout parameters.
3. **S1 candidate:** state-only control plus two of six score terms in each of phases 7, 11, 14 and 15 read the head cell's own previous-word VSET state, already written by phase 0. Force those two terms onto state channels rather than relying on a random channel draw. Spread five ternary channels across the four phases: 7:(0,1), 11:(2,3), 14:(4,0), 15:(1,3). The other four terms keep their matched local-byte predicates. The hard winner therefore selects an ordinary learned integer VSET immediate through a state/local-byte interaction. Balance buckets by simulation, as recurrent state affects later score rows; the tape-only row method cannot score these pages.

The state-only arm isolates the cost of retaining the tracks; S1 versus state-only isolates whether state-conditioned bucket selection helps. The unchanged F1 arm is the absolute loss comparator. Before training, direct exported-engine tests must verify word/separator transitions, that two prefixes with the same last eight bytes but different prior words produce different state-dependent winners, and that untouched phases retain their original tape scoring layout. The only change to the shared trainer occurs after the active 40-round Flame fit completes.

## Acceptance evidence

Use only `r1/data/mix/v02a/train.jsonl` to learn. Development selection uses exact exported integer scores on separate `r1/data/chat/dev1000.jsonl` and `r1/runs/codex_flame_scaling_20260928/dev_fresh1000.jsonl`; bytes plus EOS from BOS, 1,000 records in each, paired 5,000-resample record intervals. The absolute loss gate is S1 minus unchanged F1 **equal-weight mean ≤ +0.005 bits/target with paired 95% CI upper ≤ +0.005**. To establish a useful interaction beyond merely carrying state, require S1 minus state-only equal-weight mean < 0 with paired 95% CI upper < 0. State-only and S1 per-domain results are reported even if a gate fails.

Separately, on disjoint long validation probes `r1/data/ctxprobe/chat_long600.jsonl` and `everyday_long600.jsonl`, require **use(8) ≥ 0.010 bits/target in both domains**, computed with the same exact integer prefix-replacement protocol as the reference. Report use(16) but do not gate on it: Claude's analysis of `chat_long600` found the previous word starts over 16 bytes back at only about 1% of positions, so a one-word state has too little opportunity to move use(16). A functional state transition alone does not pass the use(8) gate. This correction was made before any S1 implementation or training; baseline use from older than eight bytes is near zero. Report the S1-minus-state-only context-use difference as an interaction diagnostic.

Only after both development gates pass, report exact integer clean chat/everyday val1000, paired intervals, two unedited greedy English samples, and independent whole-word suggestion quality if available. No claim of coherent sentences from loss or use(k) alone. Retain original checkpoints; roll back/reject S1 if either gate fails. Never use the sealed completion test or deploy from this pilot.
