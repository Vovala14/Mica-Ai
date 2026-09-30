# MICA AI

MICA is a learned integer-rule cellular automaton language model. Its reference
implementation follows the *MICA exact mechanism and learning specification*,
revision R1.

| Part | Where | Status |
|------|-------|--------|
| Engine and training code (R1) | [`r1/`](r1/) | added |
| MICA Flame (word-level) | [`flame/`](flame/) | checkpoint added |
| MICA Ember (letter-level) | [`ember/`](ember/) | not added yet |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest r1/tests/test_conformance.py -q    # engine conformance (§7 worked example)
```

Load the Flame checkpoint:

```python
import json, os, sys
run = "flame/runs/claude_flame_word_20260930/a_full40_lr030"
info = json.load(open(f"{run}/run_info.json"))
os.environ.update({k: str(v) for k, v in info["env"].items()})  # set geometry before import
sys.path.insert(0, "r1")
from mica_r1 import serialize
model = serialize.load(f"{run}/best.mica")
```

Flame works on word IDs, so turning its output into text needs the vocabulary
file `r1/data/word/vocab.json`. That file is not in the repo yet.

See [`r1/README.md`](r1/README.md) for the engine layout and training commands.
`r1/mica_r1/score.dll` is a Windows build of `r1/mica_r1/score_kernel.c`.

## License

Free for **personal use** and **research use**.
**Commercial use is forbidden.** See [LICENSE](LICENSE) for the full terms.
For commercial licensing, contact the repository owner.
