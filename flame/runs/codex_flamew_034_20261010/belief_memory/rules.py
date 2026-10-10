"""Learned bounded integer branch rules; no sklearn or float at inference."""
import json
import numpy as np
from .adapter import HERE,tokens

def branch(rule,x):
    node=0
    for _ in range(32):
        feature=rule['feature'][node]
        if feature<0:return int(rule['value'][node])
        node=rule['left'][node] if int(x[feature])<=rule['threshold'][node] else rule['right'][node]
    raise RuntimeError('Bounded branch depth exceeded')

def event_features(seq,at,words):
    left=at
    while left>0 and seq[left-1] not in ('.','!','?'):left-=1
    right=at+1
    while right<len(seq) and seq[right] not in ('.','!','?'):right+=1
    clause=seq[left:right];place=at-left
    ai=[i for i,w in enumerate(clause) if w=='AGENT']
    oi=[i for i,w in enumerate(clause) if w=='OBJECT']
    x=np.zeros(5*len(words)+5,np.int16);index={w:i for i,w in enumerate(words)}
    for i,w in enumerate(clause):
        if w not in index:continue
        j=index[w];x[j]=1
        if ai:
            if i<ai[0]:x[len(words)+j]=1
            if i>ai[0]:x[2*len(words)+j]=1
        if i<place:x[3*len(words)+j]=1
        if i>place:x[4*len(words)+j]=1
    x[-5:]=[int(bool(ai)),int(bool(oi)),int(left==0),int(bool(ai) and ai[0]<place),int(bool(oi) and oi[0]<place)]
    return x

def prompt_features(prompt,agent,obj,words):
    text=prompt.replace('’',"'")
    import re
    for val,label in [(agent,'AGENT'),(obj,'OBJECT')]:
        text=re.sub(r'\b'+re.escape(val)+r'\b',label,text,flags=re.I)
    seq=tokens(text);index={w:i for i,w in enumerate(words)};x=np.zeros(len(words),np.int16)
    for w in seq:
        if w in index:x[index[w]]=1
    return x

class TreeCells:
    """Exact radius-one integer rule interpreter using the prior CA topology."""
    def __new__(cls,rule,dimension,choices):
        from .cellular_memory import BeliefCA,local_rule
        class CA(BeliefCA):
            def _site(self,topic,site,incoming):
                if incoming is None:return None
                oldkey=tuple(int(v) for v in self.keys[topic,site])
                if self.occupied[topic,site] and oldkey!=incoming.key:return incoming
                if not self.occupied[topic,site]:
                    self.occupied[topic,site]=True;self.keys[topic,site]=incoming.key
                    self.masks[topic,site]=self.unknown_bit
                # Homogeneous learned predicate at every cell; features and
                # branch thresholds are integers, update is COPY or SKIP.
                if (incoming.features[-5] and incoming.features[-4]
                        and branch(rule,incoming.features)):
                    self.masks[topic,site]=1<<incoming.choice
                return None
        return CA(np.zeros(dimension,np.int8),0,32,choices=choices)
