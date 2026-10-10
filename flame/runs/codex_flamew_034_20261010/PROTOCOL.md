# Broader independently trained English/option grounding

Second separately frozen attempt: normalize location articles induced from
training prefixes so shorter/longer prefix matches cannot create duplicate
location identities. Regenerate training and fresh development, preserve
original gates. No new cellular update predicate is trained in this attempt.

Continue the accepted private v6 prototype (public word638/2000,31.90%).
Original B-740 and released memories remain unchanged. Freeze v5 learned CA
update rules9a0ad7b1 and completion query rules initially; train a new bounded
integer location/option grounding adapter on independently generated stories
and option paraphrases. No public benchmark labels, prompt copies or templates
enter training or selection. No sealed access or deployment.

Training induces place-prefix/boundary patterns and a finite integer predicate
linking a mentioned place to a candidate ending. Options include articles,
prepositions, whole-sentence endings, unrelated names/objects, negated and
multi-place ambiguous distractors. Ambiguous links must abstain. Text binding
remains outside the separate radius-one integer-rule cellular memory plane;
this is not native B-740 integration.

Freeze all new parameters and source before fresh1200-item,600-pair development
with disjoint names/objects/places and reserved prompt/option shapes. Gate:
>=85% supported answers, >=70% complete supported pairs, <=2% unsupported
definite answers, option permutation invariance, and positive paired gain
against old prototype's memory-only answers. Then compare both hybrid systems
with identical exact packaged source fallback on the same development inputs:
net>=6 correct, paired scenario lower95% CI>0, source-correct lost<=5%, negative
fallback unchanged. Original v5 development is retention-only, not fresh gate.

Only a frozen independently passing candidate receives public2000 evaluation.
Use correct unmodified full ctx and endings, prediction pass before labels,
verified exact source cache alignment/audits. Report both word/char scores,
paired intervals vs v6 and release, hashes, coverage and limitations. Public
score alone does not authorize selection or a release. Preserve failed runs.
