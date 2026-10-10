"""Private completion frontend and learned integer cellular belief memory.

Text chunking is an external adapter. This is NOT compiled into native B-740.
All learned arrays, noun boundaries and capitalized non-name exclusions come
from this run's annotated training set, not public Tiny ToM.
"""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
from .helpers import tokens,locate,canonical,sha
from .cellular_memory import BeliefCA,Parcel,identity
PDIM=2048

def pvector(prompt,agent,obj):
    # Split possessive suffixes before replacing a previously unseen name.
    seq=tokens(re.sub(r"\b([a-zA-Z]+)'s\b",r"\1 's",prompt))
    for value,label in [(agent,'AGENT'),(obj,'OBJECT')]:
        if not value:continue
        part=tokens(value);out=[];i=0
        while i<len(seq):
            if seq[i:i+len(part)]==part:out.append(label);i+=len(part)
            else:out.append(seq[i]);i+=1
        seq=out
    x=np.zeros(PDIM,np.int16)
    for n in (1,2):
        for i in range(len(seq)-n+1):
            j=int.from_bytes(hashlib.blake2s(' '.join(seq[i:i+n]).encode(),digest_size=4).digest(),'little')%PDIM
            x[j]=min(3,int(x[j])+1)
    return x

def bind(story,prompt,choices,cfg):
    seq=tokens(story);q=tokens(prompt)
    # Orthographic candidate extraction; learned exclusions prevent sentence
    # connectives and articles from becoming person names.
    names={x.lower() for x in re.findall(r'\b[A-Z][a-z]+\b',story)
           if x.lower() not in cfg['non_names']}
    agents={x for x in names if re.search(r'\b'+re.escape(x)+r'\b',normal(prompt))}
    if len(agents)!=1:return None
    agent=next(iter(agents));objects=set()
    option_words={w for c in choices for w in tokens(c)}
    for i,w in enumerate(seq):
        if w not in cfg['object_prefixes']:continue
        part=[]
        for v in seq[i+1:i+6]:
            if v in cfg['object_boundaries'] or not v.isalpha():break
            part.append(v)
        # A grounded option remains a place even if a later English clause
        # contains a word unseen in the learned object-boundary list.
        is_place=any(part[:len(cs)]==cs for c in choices
                     for cs in [tokens(c)[1:-1] if tokens(c)[0]=='the' else tokens(c)])
        if part and not is_place and not all(v in option_words for v in part):objects.add(' '.join(part))
    explicit={o for o in objects if re.search(r'\b'+re.escape(o)+r'\b',normal(prompt))}
    if len(explicit)==1:obj=next(iter(explicit))
    elif not explicit:
        repeated={o for o in objects if len(re.findall(r'\b'+re.escape(o)+r'\b',normal(story)))>=2}
        if len(repeated)!=1:return None
        obj=next(iter(repeated))
    else:return None
    return agent,obj

def normal(s):return ' '.join(tokens(s))

class Adapter:
    def __init__(self):
        self.cfg=json.loads((HERE/'binding.json').read_text())
        self.rules=json.loads((HERE/'integer_rules.json').read_text())
    def predict(self,story,prompt,choices):
        from .rules import prompt_features,event_features,branch,TreeCells
        slots=bind(story,prompt,choices,self.cfg)
        if slots is None:return {'answer':None,'recognized':False,'reason':'ambiguous_entities'}
        agent,obj=slots
        pm=branch(self.rules['prompt'],prompt_features(prompt,agent,obj,self.rules['prompt_words']))
        if not pm:return {'answer':None,'recognized':False,'reason':'unsupported_prompt','prompt_margin':pm}
        located=locate(story,choices)
        if located is None:return {'answer':None,'recognized':True,'reason':'ungrounded_options'}
        seq,mentions=located;canon,mapping=canonical(seq,agent,obj,mentions)
        dimension=5*len(self.rules['event_words'])+5
        cell=TreeCells(self.rules['memory'],dimension,len(choices))
        key=identity(agent,obj);topic=key[1]%cell.topics
        trace=[]
        for start,end,choice in mentions:
            x=event_features(canon,mapping[start],self.rules['event_words'])
            margin=branch(self.rules['memory'],x)
            if not cell.send(topic,Parcel(key,choice,x)):return {'answer':None,'recognized':True,'reason':'overflow'}
            trace.append({'choice':choice,'margin':margin})
        return {'answer':cell.answer(topic,key),'recognized':True,'agent':agent,'object':obj,'prompt_margin':pm,'trace':trace}
