---
license: other
license_name: mica-ai-non-commercial
license_link: https://huggingface.co/vynly/mica-flame-w-0.3.3/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
library_name: mica
tags:
- cellular-automaton
- learned-integer-rules
- non-transformer
- integer-inference
- word-level
---

# MICA Flame-W 0.3.3

Experimental English word-symbol model built around the original B-740 **Minimal Inference Cellular Automaton** with learned integer rules. This update leaves the cellular rules, vocabulary, long-memory bank and sentence decoder unchanged. It blends two integer short-memory readouts for the first 64 fed symbols: `floor((3 * released + fitted + 2) / 4)`. After 64 symbols, inference uses the unchanged 0.3.2 long bank. No transformer, GRU, external language model or retrieval generator runs inside MICA.

The release is a **small word-efficiency improvement with a long-context tradeoff**, not a claim of better conversation. The model checkpoint SHA-256 remains `1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d`. Memory SHAs: released short `5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7`, fitted short `5564ba558e7c1a6e594e324575333a626df16cc8605c9a23534d880acebab4ac`, long `f2abb7913dd626fb75cfeb96dd97c17c7d7ecbaed88f54c95fdb986734469f3b`. Checkpoint plus three memory files total **36,132,566 bytes** (36.13 MB decimal). The additional bank adds 673,650 bytes over 0.3.2.

## Measured results versus 0.3.2

Exact exported-integer bits per word symbol plus EOS from BOS, lower is better. The candidate was chosen on a new fourth development block disjoint from memory fitting and previous development blocks.

| Set | 0.3.2 | 0.3.3 | Candidate minus 0.3.2, paired 95% CI |
|---|---:|---:|---:|
| Fresh short development, 1,000 records | 5.379107 | 5.358054 | −0.021054 [−0.022820,−0.019287] |
| Fresh long development, 128 records | 8.329670 | 8.334585 | +0.004915 [+0.002938,+0.007085] |
| Equal-weight development domains | | | −0.008069 [−0.009427,−0.006711] |
| Repeated clean short500, 7,320 targets | 5.208428 | 5.189430 | −0.018998 [−0.021376,−0.016611] |
| Repeated clean long500, 100,618 targets | 8.082371 | 8.086188 | +0.003818 [+0.002786,+0.004814] |
| Equal-weight clean domains | | | −0.007590 [−0.008872,−0.006321] |

The long-record regression is statistically visible and was below the predeclared +0.005-bits cap. The clean sets had been inspected in earlier research, so the fresh development block is the main model-selection evidence.

A blinded single-rater sentence check on 20 untouched short and 20 untouched long prompts found **1 win, 1 loss, 38 ties** and one new unusable output. It passed a no-regression gate but shows **no demonstrated sentence-quality gain**. Raw outputs remain often repetitive or incoherent. Unedited changed examples from short prompts:

- `I miss my old friends.` — 0.3.2: ` But I really wanted to go to the park with my friends and friends.`; 0.3.3: ` I don't know what I would do without you.`
- `I want to make her jealous` — 0.3.2: ` of his friends.`; 0.3.3: ` of the new York.`

On the public 2,000-item Tiny Theory of Mind benchmark with unchanged answer mode, word-normalized accuracy was 632/2000 (31.60%) → 636/2000 (31.80%), paired +0.20 percentage points [−0.25,+0.65]. That interval includes no improvement. The five-task leaderboard was not rerun; no improved ToM or leaderboard claim is made. The sealed test set was not accessed.

## Use

```bash
pip install numpy
python generate.py "I owe my life to you"
python generate.py --mode suggest "Can you send me"
```

`run_tom.py` reproduces the public Tiny ToM protocol. `package.json` lists the cellular geometry and every memory route. Full frozen protocol, per-record integer scores, blind ratings and conformance checks are in [the MICA project](https://github.com/Vovala14/Mica-Ai). See the [MICA model collection](https://huggingface.co/collections/vynly/mica-minimal-inference-cellular-automaton-6ac3f2f24567a92325cb45f2). `LICENSE` contains the non-commercial terms.
