"""Native MICA Flame-W adapter for lm-eval 0.4.13 (multiple-choice only).

Runs the released word LM + its frozen answer-mode calibration. The English
belief adapter is deliberately excluded: no choice overrides. Optional cache
contains already published exact native likelihoods, keyed by the complete
context and continuation and bound to all package file hashes. Greedy flags
are computed live. --verify-every also re-scores a systematic subset live.
No model is downloaded. No inference or weight file is modified.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval import evaluator
import lm_eval
import yaml

HERE=Path(__file__).resolve().parent
G={}


def key(context, continuation):
    return hashlib.sha256(json.dumps([context,continuation],ensure_ascii=False).encode('utf8')).hexdigest()


def initialize(package, cache_path, verify_every):
    package=Path(package)
    manifest=json.loads((package/'MANIFEST.json').read_text())
    for filename,expected in manifest['files'].items():
        assert hashlib.sha256((package/filename).read_bytes()).hexdigest()==expected, filename
    if cache_path:
        with gzip.open(cache_path,'rt',encoding='utf8') as f:cache=json.load(f)
        assert cache['manifest_sha256']==hashlib.sha256((package/'MANIFEST.json').read_bytes()).hexdigest()
        scores=cache['scores']
    else:scores={}
    sys.path.insert(0,str(package))
    import word_eval as E
    import context_banks
    base=E.load_model(f'mica:{package}')
    model=context_banks.load(package,base,answer_mode=True)
    vocab=E.W.Vocab.load(package/'vocab.json')
    G.update(model=model,W=E.W,vocab=vocab,scores=scores,verify_every=verify_every,
             last_context=None,last_state=None)


def ids(text):
    W,v=G['W'],G['vocab']
    return [W.SEP if t=='<sep>' else v.index.get(t,W.oov_id(t)) for t in W.tokenize(text)]


def score(job):
    index,context,continuation=job
    model=G['model']
    if context!=G['last_context']:
        st=model.start()
        for t in ids(context):st=model.feed(st,t)
        G['last_context'],G['last_state']=context,st
    state=model.fork(G['last_state'])
    cache=G['scores'].get(key(context,continuation))
    verify=cache is None or index%G['verify_every']==0
    ll,greedy=0.0,True
    for t in ids(continuation):
        z=model.scores(state).astype(np.float64)/model.div
        z=np.where(model.elig,z,-math.inf)
        greedy=greedy and int(z.argmax())==t
        if not verify and not greedy:
            break  # likelihood is cached; this proves is_greedy=False
        mx=z.max()
        ll+=float(z[t]-mx-np.log(np.exp(z-mx).sum()))
        state=model.feed(state,t)
    if verify and cache is not None:
        assert abs(ll-cache)<1e-9,(index,ll,cache)
    return (float(ll if cache is None else cache),bool(greedy),bool(verify))


def chunk(jobs):
    return [score(j) for j in jobs]


class MicaFlameLM(LM):
    def __init__(self,package,workers=8,cache=None,verify_every=1000):
        super().__init__()
        self.package,self.workers,self.cache,self.verify_every=package,workers,cache,verify_every
        self.verifications=0

    def loglikelihood(self,requests):
        jobs=[(i,*r.args) for i,r in enumerate(requests)]
        out=[];t0=time.time()
        with ProcessPoolExecutor(max_workers=self.workers,initializer=initialize,
                                 initargs=(self.package,self.cache,self.verify_every),
                                 mp_context=multiprocessing.get_context('spawn')) as ex:
            for part in ex.map(chunk,[jobs[i:i+32] for i in range(0,len(jobs),32)]):
                out.extend(part)
                if len(out)%5000<32:print(f'Flame requests {len(out)}/{len(jobs)} {time.time()-t0:.1f}s',flush=True)
        self.verifications=sum(r[2] for r in out)
        return [(r[0],r[1]) for r in out]

    def loglikelihood_rolling(self,requests):
        raise NotImplementedError('This adapter is for conditional multiple-choice scoring only')

    def generate_until(self,requests):
        raise NotImplementedError('Use the released generate.py for generation; no chat API claimed')


def arith_results(doc,results):
    # Official ArithMark-3 script normalizes by model-token count.
    from build_word_corpus import tokenize
    sums=[r[0] for r in results]
    counts=[len(tokenize(s)) for s in doc['endings']]
    norm=[s/max(n,1) for s,n in zip(sums,counts)]
    return {'acc':int(np.argmax(sums)==int(doc['label'])),
            'acc_norm':int(np.argmax(norm)==int(doc['label']))}


def make_tasks(data):
    root=Path(lm_eval.__file__).parent/'tasks'
    tasks={}
    class Loader(yaml.SafeLoader):pass
    Loader.add_constructor('!function',lambda loader,node:loader.construct_scalar(node))
    for name in ('arc_easy','arc_challenge','hellaswag','piqa'):
        path=root/('arc' if name.startswith('arc') else name)/(name+'.yaml')
        config=yaml.load(path.read_text(),Loader=Loader)
        if 'include' in config:
            parent=yaml.load((path.parent/config.pop('include')).read_text(),Loader=Loader)
            config={**parent,**config}
        config.update(dataset_path='json',dataset_name=None,
                      dataset_kwargs={'data_files':{'test':str(data/(name+'.jsonl'))}},
                      training_split=None,validation_split=None,test_split='test',num_fewshot=0)
        if name=='hellaswag':
            spec=importlib.util.spec_from_file_location('canonical_hs_utils',root/'hellaswag/utils.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            config['process_docs']=module.process_docs
        tasks[name]=ConfigurableTask(config=config)
    config=dict(task='arithmark3',dataset_path='json',dataset_name=None,
                dataset_kwargs={'data_files':{'test':str(data/'arithmark3.jsonl')}},
                test_split='test',num_fewshot=0,output_type='multiple_choice',
                doc_to_text='{{ctx}}',doc_to_target='label',doc_to_choice='{{endings}}',
                target_delimiter='',process_results=arith_results,
                metric_list=[dict(metric=n,aggregation='mean',higher_is_better=True) for n in ('acc','acc_norm')])
    tasks['arithmark3']=ConfigurableTask(config=config)
    return tasks


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--package',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--cache')
    ap.add_argument('--verify-every',type=int,default=1000);ap.add_argument('--limit',type=int)
    ap.add_argument('--tasks',default='arc_easy,arc_challenge,hellaswag,piqa,arithmark3')
    ap.add_argument('--out',default=str(HERE/'flame_harness_results.json'))
    a=ap.parse_args()
    assert a.verify_every>=1
    # arith_results needs tokenizer code but must not initialize word geometry
    # in the parent; all VM calls remain in independent worker processes.
    sys.path.insert(0,a.package)
    lm=MicaFlameLM(a.package,a.workers,a.cache,a.verify_every)
    task_dict=make_tasks(Path(a.data))
    task_dict={name:task_dict[name] for name in a.tasks.split(',')}
    result=evaluator.evaluate(lm,task_dict,limit=a.limit,bootstrap_iters=1000,log_samples=True)
    result['native_mica']={'package_manifest_sha256':hashlib.sha256((Path(a.package)/'MANIFEST.json').read_bytes()).hexdigest(),
                          'config':'word LM + frozen answer-mode calibration; NO belief adapter',
                          'live_likelihood_verifications':lm.verifications,
                          'cache_used':bool(a.cache),'harness_version':lm_eval.__version__}
    Path(a.out).write_text(json.dumps(result,default=str),encoding='utf8')
    print(json.dumps(result['results'],indent=2),flush=True)
