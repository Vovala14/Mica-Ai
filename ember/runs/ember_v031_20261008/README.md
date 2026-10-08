# MICA Ember v0.3.1

Ember v0.3.1 is the byte-model/readout update for the native Minimal Inference Cellular Automaton. It retains the original integer-rule cellular architecture and the v0.3a next-word and two-letter completion interface.

This release improves exact byte-level loss on both clean evaluation domains. It does **not** improve the word-suggestion heads: the word heads and probe wiring are unchanged from v0.3a, so word top-1 and top-5 results are identical. Ember remains a word-assistance model, not a sentence generator.

## What is included

- `model.mica`: 4,467,396-byte learned integer MICA checkpoint (SHA-256 `e35bad48e1db529ccc07f7bfbd5070ba01ec1d0dfbaeafaf07398c7bae63cedd`). The learned readout was updated; cellular rules and probe wiring were preserved.
- `memory.npz`: 14,019-byte integer byte-memory sidecar (SHA-256 `5ac91b7a9e110a22546083a9279b06725f48aeb7b1de441c80d1ef92b2c2f44a`). It improves byte scoring when used with `memory.py` and the MICA byte-scoring engine.
- `word_heads.npz`: the v0.3a next-word and two-letter-completion readouts over the same 240 integer cellular probes.
- `word_model.py` and `mica_r1/`: CPU-only exact integer runtime for word suggestions.
- `memory.py`: validates and applies the integer memory tables.

The inference path uses integer cellular rules and integer score tables. There is no transformer or neural-network component.

## Evaluation

All reported likelihoods use the exported exact integer engine and are in bits per target byte, including EOS. The clean sets are held-out reporting data; model choice used disjoint development data.

| Clean set | v0.2A source | v0.3.1 | Paired change (95% CI) |
|---|---:|---:|---:|
| Chat val1000 | 1.860907 | 1.827462 | -0.033445 [-0.036189, -0.030813] |
| Everyday val1000 | 1.875736 | 1.836078 | -0.039658 [-0.043026, -0.036368] |
| Equal-domain mean | 1.868322 | 1.831770 | -0.036552 [-0.038721, -0.034442] |

On an independent last-500-per-domain development confirmation, the equal-domain paired change was -0.036589 bits/target [-0.039841, -0.033329]. The model was selected using the first disjoint development split, not these confirmation or clean results.

The unchanged word heads score 14.25% next-word top-1 and 48.00% top-1 for completion after two letters on the held-out word benchmark, with the same top-5 lists as v0.3a on all 400 prompts. These word metrics are not an improvement over v0.3a.

## Reproduce the packaged checks

```bash
python -m pip install -r requirements.txt
python test_word_model.py
python test_memory.py
python word_model.py "Thank you for "
python word_model.py --complete "Are you coming to the pa"
```

The tests verify the model, word-head and memory hashes, exact probe output, known suggestions, and integer memory-table behavior. The sealed test set was not accessed.
