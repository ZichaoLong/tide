"""Project-owned, unique held-out mappings; no checkpoint-dependent selection."""
import hashlib
import json

from .tasks import RetrievalTask


def mapping_key(task, tokens):
    pairs, _, _ = task.decode(tokens)
    return tuple(sorted(pairs))


def exclusions(task, count=256, seed=20261107):
    x, _ = task.batch('test', count, 0, seed)
    return {mapping_key(task, row) for row in x}


def make_dataset(task_config, count=2048, seed=20261108):
    task = RetrievalTask(task_config)
    excluded = exclusions(task)
    seen, examples = set(), []
    for cursor in range(200):
        x, y = task.batch('test', 256, cursor, seed)
        for tokens, target in zip(x.tolist(), y.tolist()):
            key = mapping_key(task, tokens)
            if key in excluded or key in seen:
                continue
            seen.add(key)
            examples.append(dict(index=len(examples), tokens=tokens, target=target))
            if len(examples) == count:
                result = dict(schema='tide-lab-unique-test-v1', task=task_config, seed=seed,
                    samples=count, old_test=dict(samples=256, seed=20261107, step=0,
                    unique_excluded_mappings=len(excluded)), examples=examples,
                    selection='first distinct test mappings from deterministic batches, excluding old test256; one query per mapping')
                validate_dataset(result)
                return result
    raise RuntimeError('finite unique-mapping generation exhausted')


def validate_dataset(data):
    if data['schema'] != 'tide-lab-unique-test-v1':
        raise ValueError('unsupported held-out data')
    task = RetrievalTask(data['task'])
    old = data['old_test']
    excluded = exclusions(task, old['samples'], old['seed'])
    seen = set()
    for i, row in enumerate(data['examples']):
        pairs, _, target = task.decode(row['tokens'])
        key = mapping_key(task, row['tokens'])
        if (row['index'] != i or target != row['target'] or task.mapping_split(pairs) != 'test'
            or key in seen or key in excluded):
            raise ValueError('held-out label/split/uniqueness/exclusion invariant failed')
        seen.add(key)
    if len(seen) != data['samples'] or not seen:
        raise ValueError('held-out count mismatch')
    return dict(samples=len(seen), unique_mappings=len(seen), old_test_overlap=0,
                mapping_set_sha256=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest())
