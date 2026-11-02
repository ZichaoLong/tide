"""Candidate-independent distinct token samples and a separate varied-text workload."""
import hashlib
import random


def create(task, config, tokenizer):
    import torch
    lengths=[task['input_tokens']]*task['batch']
    values=[]
    topics=["Explain a river's seasonal changes", "Compare two sorting algorithms", "Describe a city's public transport", "给出一个节约用水的方案", "Explain how a battery stores energy", "Write a short account of a mountain walk", "Compare the areas of two rectangles", "解释植物如何获得能量"]
    for index in range(task['batch']):
        rng=random.Random(task['seed']+index*104729)
        if task['input_kind']=='synthetic_tokens':
            row=[rng.randrange(256,config['vocab_size']-512) for _ in range(lengths[index])]
        else:
            lengths[index]=max(1,task['input_tokens']//[4,2,1][index%3])
            text=f"{index:08d}: Sample {index}. {topics[index%len(topics)]}. Values: {rng.randrange(10000)}, {rng.randrange(10000)}. "
            row=tokenizer.encode(text,add_special_tokens=False)
            row=(row*((lengths[index]+len(row)-1)//len(row)))[:lengths[index]]
        values.append(row)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded=[[pad]*(task['input_tokens']-len(row))+row for row in values]
    masks=[[0]*(task['input_tokens']-n)+[1]*n for n in lengths]
    ids=torch.tensor(padded,dtype=torch.int64)
    mask=torch.tensor(masks,dtype=torch.int64)
    digest=hashlib.sha256(ids.numpy().tobytes()+mask.numpy().tobytes()).hexdigest()
    return ids,mask,dict(sha256=digest,lengths=lengths,logical_batch=task['batch'],kind=task['input_kind'],
        tokenizer_revision='same pinned model snapshot',input_tokens=sum(lengths),
        physical_prefill_tokens=task['batch']*task['input_tokens'], distinct_samples=len({tuple(x) for x in values}),
        corpus='project-generated controlled tokens/text; not an external natural-language dataset')
