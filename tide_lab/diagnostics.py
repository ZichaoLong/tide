"""Task-side paired interventions; no execution-core implementation."""
import torch


def swap_query(task, tokens):
    """Keep each mapping fixed; select the next stored key with a different value."""
    indices, inputs, targets = [], [], []
    for index, row in enumerate(tokens):
        pairs, query, answer = task.decode(row)
        choices = [(key,value) for key,value in pairs if key!=query and value!=answer]
        if not choices:
            continue
        changed = row.clone()
        changed[-1] = task.query_offset+choices[0][0]
        task.decode(changed)  # fail on malformed episodes
        indices.append(index); inputs.append(changed); targets.append(choices[0][1])
    if not inputs:
        raise ValueError('no different-answer query intervention available')
    return torch.tensor(indices,dtype=torch.long), torch.stack(inputs), torch.tensor(targets,dtype=torch.long)


def extend_delay(task, tokens, delay):
    if type(delay) is not int or not task.config.delay <= delay <= 128:
        raise ValueError('delay must retain or extend the original bounded delay')
    filler = tokens.new_full((tokens.shape[0],delay),task.filler_token)
    return torch.cat((tokens[:,:task.config.pairs], filler, tokens[:,-1:]),dim=1)
