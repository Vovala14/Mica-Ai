import hashlib
from pathlib import Path
import build_word_corpus as W
def tokens(text):return W.tokenize(text.replace('’',"'"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normal(text):return ' '.join(tokens(text))

def locate(story,choices):
    seq=tokens(story); cs=[tokens(c) for c in choices]
    prefix=0;suffix=0
    while all(len(c)>prefix for c in cs) and len({c[prefix] for c in cs})==1:prefix+=1
    while all(len(c)-prefix>suffix for c in cs) and len({c[-1-suffix] for c in cs})==1:suffix+=1
    cores=[c[prefix:len(c)-suffix if suffix else None] for c in cs]
    if any(not c for c in cores) or len({tuple(c) for c in cores})!=len(cores):return None
    found=[]
    for j,core in enumerate(cores):
        for at in range(len(seq)-len(core)+1):
            if seq[at:at+len(core)]==core:found.append((at,at+len(core),j))
    found.sort()
    if len({x[2] for x in found})<2:return None
    if any(b[0]<a[1] for a,b in zip(found,found[1:])):return None
    return seq,found

def canonical(seq,agent,obj,mentions):
    replacements={a:(b,'PLACE') for a,b,_ in mentions}
    for label,part in [('AGENT',tokens(agent)),('OBJECT',tokens(obj))]:
        for i in range(len(seq)-len(part)+1):
            if seq[i:i+len(part)]==part:replacements.setdefault(i,(i+len(part),label))
    out=[];index={};i=0
    while i<len(seq):
        index[i]=len(out)
        if i in replacements:
            end,label=replacements[i];out.append(label);i=end
        else:out.append(seq[i]);i+=1
    return out,index

