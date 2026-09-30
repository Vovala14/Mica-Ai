#!/usr/bin/env python3
"""Matched end-to-end comparison including stack compilation. Synthetic by default."""
import argparse, json, platform, statistics, time
from pathlib import Path
import numpy as np
from mica_r1 import spec, serialize
from mica_r1.engine import random_model
from mica_r1.batch import ModelStack, evaluate
from mica_r1.search import make_child, XorShift32


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--children',type=int,default=16)
    ap.add_argument('--batch-records',type=int,default=32)
    ap.add_argument('--record-bytes',type=int,default=256)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--routing',choices=['r1','positive','pairdiff','parity'],default='r1')
    ap.add_argument('--records')
    ap.add_argument('--model')
    ap.add_argument('--out',default='r1/runs/scoring_benchmark.json')
    a=ap.parse_args()
    if min(a.batch_records,a.record_bytes,a.repeats)<1 or a.children<0:
        ap.error('sizes/repeats must be positive and children nonnegative')
    rng=np.random.default_rng(20260920)
    if a.records:
        from data.make_records import load_records
        records=[r[:a.record_bytes] for r in load_records(a.records) if len(r)>=a.record_bytes][:a.batch_records]
        if len(records)!=a.batch_records: ap.error('not enough full-length records')
    else:
        records=[rng.integers(0,256,a.record_bytes,dtype=np.uint8).tobytes() for _ in range(a.batch_records)]
    m=serialize.load(a.model) if a.model else random_model(1)
    mut=XorShift32(1)
    models=[m]+[make_child(m,mut) for _ in range(a.children)]
    # One-time native build separately; timed runs include each new model stack.
    t=time.perf_counter(); ModelStack([m],score_backend='c'); cold=time.perf_counter()-t
    timings={'numpy':[],'c':[]}; outputs={}
    for repeat in range(a.repeats):
        for backend in (('numpy','c') if repeat%2==0 else ('c','numpy')):
            start=time.perf_counter()
            output=evaluate(ModelStack(models,routing=a.routing,score_backend=backend),records)
            elapsed=time.perf_counter()-start
            timings[backend].append(elapsed); outputs[backend]=output
            print(f'{backend} repeat {repeat+1}: {elapsed:.6f}s',flush=True)
        for x,y in zip(outputs['numpy'],outputs['c']): np.testing.assert_array_equal(x,y)
    med={k:statistics.median(v) for k,v in timings.items()}
    report={'host':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
            'geometry':spec.describe(),'configuration':vars(a),
            'data':'real records' if a.records else 'synthetic uniform random bytes; not a language-quality test',
            'cold_build_and_one_stack_seconds':cold,'seconds_including_stack':timings,
            'median_seconds':med,'speedup':med['numpy']/med['c'],
            'loss_updates_targets_bit_exact':True,
            'loss_nats':outputs['c'][0].tolist(),'updates_per_target':outputs['c'][1].tolist(),
            'targets':outputs['c'][2].tolist()}
    path=Path(a.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'median_seconds':med,'speedup':report['speedup'],'out':str(path)},indent=2))

if __name__=='__main__': main()
