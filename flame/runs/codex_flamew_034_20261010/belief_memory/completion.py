"""Generic full-context completion adapter around immutable v5 rules."""
import re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
from .adapter import Adapter as FrozenAdapter

def split_completion(full_context):
    boundaries=list(re.finditer(r'(?<=[.!?])\s+',full_context))
    if not boundaries:return None
    b=boundaries[-1]
    return full_context[:b.start()],full_context[b.end():]

class Adapter:
    def __init__(self):self.frozen=FrozenAdapter()
    def predict(self,story,prompt,choices):
        if not prompt.strip():
            parts=split_completion(story)
            if parts is None:return {'answer':None,'recognized':False,'reason':'no_complete_story_boundary'}
            story,prompt=parts
        return self.frozen.predict(story,prompt,choices)
