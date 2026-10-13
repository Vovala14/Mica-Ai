"""Score JSONL {prompt,target} rows with this native package; no training."""
import argparse,json,math
from word_model import EmberWord

def main():
    p=argparse.ArgumentParser();p.add_argument('jsonl');args=p.parse_args()
    model=EmberWord();total=correct=top5=0;loss=[]
    with open(args.jsonl,encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);target=row['target'].lower()
            ranked=model.complete_with_probabilities(row['prompt'],len(model.words))
            words=[w for w,pr in ranked];total+=1
            correct+=bool(words and words[0]==target);top5+=target in words[:5]
            probabilities=dict(ranked)
            if target in probabilities:loss.append(-math.log2(max(probabilities[target],1e-300)))
    print(json.dumps(dict(prompts=total,correct=correct,top1_percent=100*correct/total if total else None,
        top5_percent=100*top5/total if total else None,conditional_bits=sum(loss)/len(loss) if loss else None,
        in_vocabulary_loss_records=len(loss),scope='Conditional on typed prefix; not bits/byte'),indent=2))

if __name__=='__main__':main()
