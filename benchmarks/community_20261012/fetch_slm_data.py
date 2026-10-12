"""Download public evaluation datasets at pinned revisions. No model requests."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'


if __name__=='__main__':
    DATA.mkdir(exist_ok=True)
    sources=json.loads((HERE/'slm_data_manifest.json').read_text())
    for task,d in sources.items():
        path=DATA/(task+('.parquet' if d['file'].endswith('.parquet') else '.source.jsonl'))
        if not path.exists():
            url=f"https://huggingface.co/datasets/{d['repo']}/resolve/{d['revision']}/{d['file']}"
            path.write_bytes(urlopen(url).read())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==d['sha256'],task
        rows=pq.read_table(path).to_pylist() if path.suffix=='.parquet' else [json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]
        assert len(rows)==d['rows']
        out=DATA/(task+'.jsonl')
        with out.open('w',encoding='utf8',newline='\n') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
        # JSON newline convention is immaterial to the records; the pinned raw
        # source hash above is the authoritative dataset identity.
        print(task,len(rows),flush=True)
