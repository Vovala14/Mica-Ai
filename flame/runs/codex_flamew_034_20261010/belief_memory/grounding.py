"""Trained integer option grounding around the frozen private v6 CA."""
import json,re,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
from .completion import Adapter as Original,split_completion
from .helpers import tokens,sha
from .rules import branch

def features(place,choice,words):
    p=tokens(place);c=tokens(choice)
    exact=any(c[i:i+len(p)]==p for i in range(len(c)-len(p)+1))
    x=np.zeros(6+len(words),np.int16)
    x[:6]=[int(exact),int(all(w in c for w in p)),int(p[0] in c),int(p[-1] in c),
           int(any(w in c for w in p)),int(len(p)>1)]
    present=set(c)
    for i,w in enumerate(words):x[6+i]=int(w in present)
    return x

def places(story,cfg):
    seq=tokens(story);out=set();boundaries=set(cfg['boundaries'])
    for i in range(len(seq)):
        for prefix in cfg['prefixes']:
            if seq[i:i+len(prefix)]!=prefix:continue
            j=i+len(prefix);part=[]
            for w in seq[j:j+6]:
                if w in boundaries or not w.isalpha():break
                part.append(w)
            while part and part[0] in cfg['articles']:part.pop(0)
            if part:out.add(' '.join(part))
    return sorted(out)

class Adapter:
    def __init__(self):
        self.original=Original()
        self.cfg=json.loads((HERE/'grounding_rules.json').read_text())
    def predict(self,story,prompt,choices):
        old=self.original.predict(story,prompt,choices)
        if old['answer'] is not None:return {**old,'grounding_route':'original'}
        if not prompt.strip():
            parts=split_completion(story)
            if parts is None:return {**old,'grounding_route':'abstain'}
            story,prompt=parts
        candidates=places(story,self.cfg)
        cores=[];matched=0
        for index,choice in enumerate(choices):
            linked=[p for p in candidates if branch(self.cfg['link_rule'],features(p,choice,self.cfg['choice_words']))]
            if len(linked)>1:return {**old,'grounding_route':'ambiguous_ending'}
            if linked:cores.append('the '+linked[0]+'.');matched+=1
            else:cores.append('the unmentionedoption'+str(index)+'.')
        if matched<2 or len(set(cores))!=len(cores):return {**old,'grounding_route':'insufficient_distinct_places'}
        new=self.original.frozen.predict(story,prompt,cores)
        if new['answer'] is None:return {**old,'grounding_route':'no_memory_answer'}
        return {**new,'grounding_route':'trained_grounder','grounded_choices':cores}
