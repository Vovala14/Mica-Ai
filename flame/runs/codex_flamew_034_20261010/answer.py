"""Flame-W 0.3.4 repository CLI, using the identical website answer backend."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from webapp.flame_backend import run
if __name__=='__main__':
    r=json.load(sys.stdin)
    print(json.dumps(run({'mode':'answer','prompt':r['context'],
        'question':r.get('question',''),'choices':r['choices']}),indent=2))
