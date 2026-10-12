"""Fetch benchmark data only; no model or config requests."""
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
DATA=HERE/'data';DATA.mkdir(exist_ok=True)
reference=HERE/'Glint-protocol/benchmark.py'
if not reference.exists():
    reference.parent.mkdir(exist_ok=True)
    reference.write_bytes(urlopen('https://huggingface.co/Glint-Research/Glint-1.3/resolve/c3f99e246aa2c64382f9668dc864533617482d0a/benchmark.py').read())
tree=ast.parse((HERE/'Glint-protocol/benchmark.py').read_text())
fn=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='evaluate_blimp')
configs=ast.literal_eval(next(x.value for x in fn.body if isinstance(x,ast.Assign) and x.targets[0].id=='blimp_configs'))
assert len(configs)==67
REV='877fba0801ffb7cbd8c39c1ff314a46f053f6036'


def fetch(config):
    path=DATA/(config+'.parquet')
    if not path.exists():
        path.write_bytes(urlopen(f'https://huggingface.co/datasets/nyu-mll/blimp/resolve/{REV}/{config}/train-00000-of-00001.parquet').read())
    rows=pq.read_table(path).to_pylist()
    assert len(rows)==1000
    return rows


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=8) as pool:
        groups=list(pool.map(fetch,configs))
    with (DATA/'blimp.jsonl').open('w',encoding='utf8') as f:
        for group in groups:
            for r in group:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rev='b08601e04326c79dfdd32d625aee71d232d685c3'
    path=DATA/'wikitext2_test.parquet'
    if not path.exists():
        path.write_bytes(urlopen(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{rev}/wikitext-2-raw-v1/test-00000-of-00001.parquet').read())
    text=' '.join(pq.read_table(path).column('text').to_pylist()).strip()
    (DATA/'wiki_test.txt').write_bytes(text.encode('utf8'))
    if not (DATA/'arc_easy.jsonl').exists():
        raise RuntimeError('Run fetch_slm_data.py first to obtain ARC-Easy')
    print('Data ready:',len(configs)*1000,'BLiMP pairs,',len(text.encode('utf8')),'WikiText UTF-8 bytes')
