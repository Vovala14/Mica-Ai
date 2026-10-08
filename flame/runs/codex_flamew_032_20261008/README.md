# Flame-W 0.3.2

This release keeps the B-740 learned integer-rule Minimal Inference Cellular Automaton unchanged (SHA-256 `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`). The automaton is in `flame/runs/claude_flamew_b740_20261001/train/best.mica`. This folder contains the released 0.3.1 integer memory as `memory.npz` and a second integer memory for contexts beyond 64 fed word symbols as `memory_long.npz`. `dual_memory.py` switches between them. Long-prompt sentence decoding uses the original BOS beam with anti-generic coefficient 0.25; short prompts retain 0.5.

| Artifact | SHA-256 |
|---|---|
| `memory.npz` | `5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7` |
| `memory_long.npz` | `f2abb7913dd626fb75cfeb96dd97c17c7d7ecbaed88f54c95fdb986734469f3b` |

Exact integer long clean500 loss changed from 8.305251 to 8.082371 bits per word symbol plus EOS, paired delta -0.222880 [95% CI -0.241879,-0.203769] on 100,618 targets. Short clean500 is exactly unchanged at 5.208428 on 7,320 targets. On 20 new long clean prompts, a single blinded rater preferred 0.3.2 on 5, 0.3.1 on 0, and tied 15; no additional severe failures. Most outputs were still unusable. See the [Hugging Face model card](https://huggingface.co/vynly/mica-flame-w-0.3.2) for examples, limitations and a self-contained package.

The site backend `webapp/flame_backend.py` loads these files. The 0.3.1 Tiny Theory-of-Mind and leaderboard scores have **not** been repeated for 0.3.2. The non-commercial project license applies.
