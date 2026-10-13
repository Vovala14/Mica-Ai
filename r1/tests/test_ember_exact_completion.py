"""Exact scoring and session invalidation without loading release weights."""
import importlib.util
from pathlib import Path
import re
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[2]/'ember/runs/ember_v038_20261013/fast_completion.py'
spec=importlib.util.spec_from_file_location('ember_completion_test',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Model:
    def __init__(self, seed=1):
        rng=np.random.default_rng(seed)
        self.words=['help','hello','helmet','being','become','sunny','sunset']
        self.Wf={t:rng.integers(-127,128,(7,4)).astype(np.float32) for t in ['next','prefix2']}
        self.B={t:rng.integers(-1024,1024,7) for t in self.Wf}
        self.calls=0
    def candidates(self,prefix):
        return np.asarray([i for i,w in enumerate(self.words) if w.startswith(prefix.lower())],int)
    def probes(self,data):
        self.calls+=1
        rows=np.zeros((len(data)+1,4),np.int64)
        for i,b in enumerate(data):rows[i+1]=(rows[i]+np.arange(1,5)*b)%127
        return rows
    def _input(self,task,data,P,starts,pos,word_start):return P[pos].copy()
    def reference(self,text,k=5):
        match=re.search(r'(?:^|\s)([A-Za-z]+)$',text)
        if not match:return []
        typed=match.group(1);idx=self.candidates(typed)
        if not len(idx):return []
        data=text.encode('utf8');start=len(data)-len(typed)
        task='next' if len(typed)==1 else 'prefix2'
        x=self.probes(data)[start if task=='next' else start+2].astype(np.float32)
        score=np.rint(self.Wf[task]@x).astype(np.int64)+self.B[task]
        order=idx[np.argsort(score[idx],kind='stable')[::-1][:k]]
        return [self.words[i] for i in order]


@pytest.mark.parametrize('context',['','hello ','hello!','h','he','hel','hello','Hello HE',
                                    'café he','猫 hel','word\the','word\nhe','word\u00a0he','zxq'])
def test_exact_prefix_rows_and_causal_truncation(context):
    m=Model();fast=module.CompletionEngine(m)
    for k in [1,5,20]:assert fast.complete(context,k)==m.reference(context,k)
    for k in [0,-1]:
        with pytest.raises(ValueError):fast.complete(context,k)


def test_ties_match_native_highest_word_index():
    m=Model()
    for t in m.Wf:m.Wf[t][:]=0;m.B[t][:]=0
    assert module.CompletionEngine(m).complete('he')==['helmet','hello','help']


def test_edit_task_case_ttl_and_model_invalidate():
    m=Model();fast=module.CompletionEngine(m);session=fast.new_session()
    session.complete('word he');session.complete('word hel');assert m.calls==1
    session.complete('word HEL');assert m.calls==2
    session.complete('other HEL');assert m.calls==3
    session.complete('other H');assert m.calls==4
    session._expires=0;session.complete('other H');assert m.calls==5
    replacement=Model(seed=2);fast.model=replacement
    assert session.complete('other H')==replacement.reference('other H')
    assert replacement.calls==2
    session.clear();assert session._key is None and session._x is None


def test_limits_and_separate_users():
    fast=module.CompletionEngine(Model());a=fast.new_session(max_context_bytes=2);b=fast.new_session()
    a.complete('long he');assert a._key is None
    b.complete('long he');assert b._key is not None
    a.clear();assert b._key is not None
    b.complete('zxq');assert b._key is None
    with pytest.raises(ValueError):fast.new_session(ttl_seconds=0)
    with pytest.raises(ValueError):fast.new_session(max_context_bytes=-1)
