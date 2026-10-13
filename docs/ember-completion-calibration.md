# Ember037 completion decoder integration

Native model.mica, word_heads.npz, memory.npz, vocabulary, cellular geometry
and learned integer rules remain byte-identical to released Ember0.3.7.
The separate completion_policy.json opts this checkout into the validated
decoder; missing policy falls back to original alpha0 scoring. Next-word and
one-letter rankings are unchanged. No0.3.8 model is promoted.

For two or more typed letters, score legal words with9*prefix2+7*next integer
scores, keeping highest-vocabulary-ID ties. Additional letters filter words;
features use only the first two typed letters. For probability output, scale
the mixed logits by12/(16*16*1024) then normalize over ALL legal words.
This uses ordinary probability normalization, with no neural inference/state.

Both decoder parameters were calibrated on original training text only. The
9:7 ranking passed fresh4000 word confirmation (+0.75pp,95%[+0.225,+1.275]).
The gain12/16 probability calibration passed a DIFFERENT fresh4000 confirmation:
conditional loss -0.455753449bits/word,95%[-0.502142388,-0.410193507], with
top1/top5 exactly unchanged. Source validation population was historical BYTE
validation, so fresh word-stage text is not globally unseen byte text.

| Clean two-letter reporting | Original prefix head |9:7 decoder|Calibrated probabilities|
|---|---:|---:|---:|
| Chat conditional bits/word |3.228379771|2.729717550|2.260658576|
| Everyday conditional bits/word |4.302340884|3.695189433|3.061645928|

Clean top1 stays71.280992%chat and62.433298%everyday after probability calibration.
The prior ranking gain over original037 was+2.041805pp[+1.101800,+2.998985].
These reused clean sets are reporting, not further independent confirmation.
Loss excludes OOV and is conditional on typed prefix; accuracy includes OOV.
No byte-loss, unconditional next-word, ToM or sentence-generation gain claimed.

Native source package SHA256:

- model.mica: c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506
- word_heads.npz: 40f3bfa6f9692f4424d05cc3666afdca7ecbd648244df059ce2d181f66561e67
- memory.npz: 3827c2a38981347bff17f9666543c333178c5a89fdd089d216e5c6f8ab8847c8

Paired confirmation and clean JSON evidence accompanies this document. Full
local provenance: codex_ember_head_consensus_20261013 and
codex_ember_probability_calibration_20261013 research folders.

The stateless website backend reports the decoder policy separately from
the unchanged model version, and includes conditional suggestion probabilities.
Optional bounded typing sessions store at most two vectors for one caller;
they are not shared across website requests. Unique suggestions are casefolded
for display; current037 vocabulary already has32768 unique casefolded words.

No publishing or deployment is part of this local integration.
