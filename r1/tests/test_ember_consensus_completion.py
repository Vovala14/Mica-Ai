"""Check integer mixture against full scoring and probability normalization."""
from test_ember_exact_completion import Model, module
import numpy as np
import pytest


@pytest.mark.parametrize('context', ['he', 'hel', 'word HE', 'café hel', 'word\the'])
def test_consensus_matches_independent_full_vocabulary_scores(context):
    m=Model();engine=module.CompletionEngine(m,alpha=7,probability_gain=12)
    typed=context.rsplit(maxsplit=1)[-1];raw=context.encode();p=len(raw)-len(typed)
    P=m.probes(raw);prefix=m._input('prefix2',raw,P,[],p+2,p);start=m._input('next',raw,P,[],p,p)
    sp=m.Wf['prefix2'].astype(np.int64)@prefix+m.B['prefix2']
    sn=m.Wf['next'].astype(np.int64)@start+m.B['next']
    sc=9*sp+7*sn;ids=m.candidates(typed)
    order=ids[np.argsort(sc[ids],kind='stable')[::-1]]
    assert engine.complete(context)==[m.words[i] for i in order[:5]]
    zz=sc[ids].astype(np.float64)*12/(16*16*1024);pp=np.exp(zz-zz.max());pp/=pp.sum()
    probs=engine.complete_with_probabilities(context,len(m.words))
    assert dict(probs)==pytest.approx({m.words[i]:p for i,p in zip(ids,pp)},abs=1e-12)
    assert sum(p for _,p in probs)==pytest.approx(1.,abs=1e-12)
    assert [w for w,p in probs]==[m.words[i] for i in order]


def test_probability_scale_preserves_ranking_and_one_letter_policy():
    m=Model();base=module.CompletionEngine(m,alpha=7);cal=module.CompletionEngine(m,alpha=7,probability_gain=12)
    for context in ['word h','word he','word hel','word H','word HE']:
        assert base.complete(context)==cal.complete(context)
        assert [w for w,p in cal.complete_with_probabilities(context)]==cal.complete(context)
    assert base.complete_with_probabilities('h')==cal.complete_with_probabilities('h')


def test_consensus_cache_reuses_both_features_and_invalidates_on_edit():
    m=Model();engine=module.CompletionEngine(m,alpha=7,probability_gain=12);session=engine.new_session()
    session.complete('word he');b=session.complete('word hel');assert m.calls==1
    assert len(session._x)==2 and all(not v.flags.writeable for v in session._x)
    assert b==engine.complete('word hel')
    engine.alpha=0;session.complete('word hel');assert session._key[-1] is False
    assert session._x[1] is None


def test_alias_suggestions_are_unique_and_probabilities_are_aggregated():
    m=Model();m.words=['being','being','BEING','best','better','before','below']
    m.candidates=lambda prefix:np.asarray([i for i,w in enumerate(m.words) if w.casefold().startswith(prefix.casefold())])
    for task in m.Wf:m.Wf[task][:]=0;m.B[task]=np.arange(7,0,-1)
    engine=module.CompletionEngine(m)
    assert engine.complete('be')==['being','best','better','before','below']
    probs=engine.complete_with_probabilities('be',20)
    assert len(probs)==5 and sum(p for _,p in probs)==pytest.approx(1.)


def test_invalid_configuration_rejected():
    for kw in [dict(alpha=-1),dict(alpha=17),dict(alpha=.5),dict(probability_gain=0)]:
        with pytest.raises(ValueError):module.CompletionEngine(Model(),**kw)


def test_backend_keeps_model_version_and_completion_suffix(monkeypatch):
    from webapp import ember_backend as backend
    class Stub:
        completion_alpha=7
        completion_probability_gain=12
        def complete_with_probabilities(self,prompt,top_k=5):return [('being',.7),('beings',.3)]
        def next_words(self,context,top_k=5):return ['your']
    monkeypatch.setattr(backend,'_model',Stub())
    out=backend.run(dict(prompt='Thank you for bei',mode='complete-word'))
    assert out['output']=='ng' and out['suggestions']==['being','beings']
    assert out['model']=='ember-v0.3.7' and out['completion_alpha']==7
    assert out['suggestion_probabilities']==[.7,.3] and out['probability_scope']=='conditional-on-typed-prefix'
    single=backend.run(dict(prompt='Thank you for b',mode='complete-word'))
    assert single['completion_alpha']==0 and single['completion_probability_gain']==16
    assert single['decoder']=='integer-completion-a0-gain16-v1'
    nxt=backend.run(dict(prompt='Thank you for',mode='next-word'))
    assert nxt['output']==' your' and 'suggestion_probabilities' not in nxt
