"""Frozen Ember 0.3.7 byte LM, adapted to Glint-1.3 benchmark.py.

No BOS, no EOS target, first byte unscored; sentence inputs first 256 bytes;
WikiText chunks 257 bytes with stride 256. All logits use the native integer VM
and released byte memory. The optional word heads are NOT used.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
G = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(package):
    package = Path(package)
    cfg = json.loads((package / 'package.json').read_text())
    assert sha(package / cfg['model']) == cfg['model_sha256']
    for k in tuple(os.environ):
        if k.startswith('MICA_'):
            del os.environ[k]
    os.environ.update(cfg['geometry_env'])
    sys.path.insert(0, str(package))
    from mica_r1 import engine, serialize, spec
    from memory import Memory
    m = serialize.load(package / cfg['model'])
    assert spec.N_SYMBOLS == 258
    assert np.array_equal(m.pr_cell, np.broadcast_to(m.pr_cell[:1], m.pr_cell.shape))
    assert np.array_equal(m.pr_chan, np.broadcast_to(m.pr_chan[:1], m.pr_chan.shape))
    G.update(engine=engine, spec=spec, m=m, mem=Memory(package / 'memory.npz'),
             cells=m.pr_cell[0].astype(np.int32), chans=m.pr_chan[0].astype(np.int32),
             co=m.pr_co.astype(np.int32), bias=m.pr_bias.astype(np.int32))


def start():
    return G['engine'].Session(), []  # intentionally no BOS, as in Glint-1.3


def fork(state):
    s = G['engine'].Session()
    s.restore(state[0].snapshot())
    return s, list(state[1])


def feed(state, token):
    G['engine'].ingest(G['m'], state[0], token)
    state[1].append(token)
    del state[1][:-64]


def logp(state, token):
    s, hist = state
    cells = (G['cells'] + s.position) % G['spec'].N_CELLS
    f = s.F[cells, G['chans']].astype(np.int32)
    z = (G['bias'] + G['co'] @ f).astype(np.int64) + G['mem'].bonus(hist)
    z = z.astype(np.float64) / G['spec'].LOGIT_DIVISOR
    z[256] = -math.inf  # BOS excluded; all 256 bytes and EOS retain probability
    mx = z.max()
    return float(z[token] - mx - np.log(np.exp(z - mx).sum()))


def score_alternatives(sequences):
    """Share exact common prefix; equivalent to scoring each sequence alone."""
    common = 0
    while common < min(map(len, sequences)) and len({s[common] for s in sequences}) == 1:
        common += 1
    st = start()
    prefix_lp = 0.0
    for i, b in enumerate(sequences[0][:common]):
        if i:
            prefix_lp += logp(st, b)
        feed(st, b)
    values = []
    for seq in sequences:
        ss, lp = fork(st), prefix_lp
        for i in range(common, len(seq)):
            if i:
                lp += logp(ss, seq[i])
            feed(ss, seq[i])
        values.append(lp if len(seq) >= 2 else -math.inf)
    return values


def work(job):
    typ, idx, payload = job
    if typ == 'wiki':
        b = bytes.fromhex(payload)
        lp = score_alternatives([b])[0]
        return {'task': typ, 'index': idx, 'sum_lnp': lp, 'targets': len(b)-1}
    if typ == 'blimp':
        seqs = [payload[k].encode('utf8')[:256] for k in ('sentence_good', 'sentence_bad')]
        vals = score_alternatives(seqs)
        return {'task': typ, 'index': idx, 'uid': payload.get('UID'),
                'sums': vals, 'correct': vals[0] > vals[1]}
    # Glint compares LP(question + " " + choice) - LP(question), without
    # length normalization. The subtracted question LP is constant per item.
    seqs = [(payload['question'] + ' ' + t).encode('utf8')[:256]
            for t in payload['choices']['text']]
    vals = score_alternatives(seqs)
    pred = max(range(len(vals)), key=lambda k: vals[k])
    gold = payload['choices']['label'].index(payload['answerKey'])
    return {'task': typ, 'index': idx, 'sums': vals, 'pred': pred,
            'gold': gold, 'correct': pred == gold,
            'truncated': any(len((payload['question']+' '+t).encode('utf8')) > 256
                             for t in payload['choices']['text'])}


def chunk(jobs):
    return [work(j) for j in jobs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='ember_glint_results.json')
    a = ap.parse_args()
    data = HERE / 'data'
    blimp = [json.loads(l) for l in (data/'blimp.jsonl').read_text(encoding='utf8').splitlines()]
    arc = [json.loads(l) for l in (data/'arc_easy.jsonl').read_text(encoding='utf8').splitlines()]
    wiki = (data/'wiki_test.txt').read_bytes()
    jobs = [('wiki', i, wiki[i:i+257].hex()) for i in range(0, len(wiki)-1, 256)]
    jobs += [('blimp', i, r) for i,r in enumerate(blimp)]
    jobs += [('arc', i, r) for i,r in enumerate(arc)]
    if a.limit:
        jobs = [j for t in ('wiki','blimp','arc') for j in [x for x in jobs if x[0]==t][:a.limit]]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize,
                             initargs=(a.package,), mp_context=multiprocessing.get_context('spawn')) as ex:
        batches = [jobs[i:i+20] for i in range(0,len(jobs),20)]
        for part in ex.map(chunk, batches):
            rows.extend(part)
            if len(rows)%1000 < 20:
                print(f'{len(rows)}/{len(jobs)} {time.time()-t0:.1f}s', flush=True)
                (HERE/'ember_status.json').write_text(json.dumps({'done':len(rows),'total':len(jobs),'seconds':time.time()-t0}))
    wr = [r for r in rows if r['task']=='wiki']
    nll = -sum(r['sum_lnp'] for r in wr)
    nbytes = sum(r['targets'] for r in wr)
    summary = {'wikitext2_byte_ppl': math.exp(nll/nbytes), 'wikitext2_bits_per_byte': nll/nbytes/math.log(2),
               'wikitext2_scored_bytes': nbytes, 'wikitext2_total_bytes':len(wiki),
               'seconds':time.time()-t0,'limited':a.limit}
    for task in ('blimp','arc'):
        rs=[r for r in rows if r['task']==task]
        summary[task]={'n':len(rs),'correct':sum(r['correct'] for r in rs),
                       'accuracy':sum(r['correct'] for r in rs)/len(rs)}
    summary['arc_truncated_items']=sum(r.get('truncated',False) for r in rows)
    result={'model':'MICA Ember 0.3.7 byte LM', 'package':str(Path(a.package).resolve()),
            'model_sha256':sha(Path(a.package)/'model.mica'),
            'memory_sha256':sha(Path(a.package)/'memory.npz'),
            'protocol_sha256':sha(HERE/'Glint-protocol/benchmark.py'),
            'data_sha256':{p.name:sha(p) for p in data.iterdir() if p.is_file()},
            'summary':summary,'rows':rows}
    (HERE/a.out).write_text(json.dumps(result),encoding='utf8')
    (HERE/'ember_status.json').write_text(json.dumps({'finished':True,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
