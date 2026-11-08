"""Native Qwen capability fixtures and one local shadow computation.

Only tasks, readouts, loss preparation and an HF module hook live here.
Core5Encoder continues to use the installed public foundation execution API.
"""
import hashlib
import json
import random
from contextlib import contextmanager

import torch

COLORS = ('red', 'blue', 'green', 'yellow', 'white', 'black', 'pink', 'brown')
NAMES = ('Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry')


def content_key(family, content):
    canonical = sorted(content) if family == 'lookup' else content
    return hashlib.sha256(json.dumps([family, canonical], separators=(',', ':')).encode()).hexdigest()


def split_for(key):
    bucket = int(key[:16], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def dataset(plan):
    result = {}
    for family in plan['tasks']:
        if family not in ('lookup', 'copy'):
            raise ValueError('only two preregistered tasks are supported')
        for split in ('train', 'validation', 'test'):
            count = plan['data'][split+'_per_task']
            rng = random.Random(f"{plan['data']['seed']}:{family}:{split}")
            seen, rows = set(), []
            for attempt in range(200000):
                if family == 'lookup':
                    pairs = list(zip(rng.sample(NAMES, 4), [rng.choice(COLORS) for _ in range(4)]))
                    key = content_key(family, pairs)
                    query, answer = rng.choice(pairs)
                    prompt = ('Read the table.\n' + '\n'.join(f'{k}: {v}' for k,v in pairs)
                              + f'\nWhat is the value for {query}? Reply with only the value, one word.')
                    content = dict(pairs=pairs, query=query)
                else:
                    words = [rng.choice(COLORS) for _ in range(4)]
                    key = content_key(family, words)
                    answer = words[-1]
                    prompt = 'Words: ' + ' '.join(words) + '\nRepeat the last word exactly. Reply with only that word.'
                    content = dict(words=words)
                if split_for(key) != split or key in seen:
                    continue
                seen.add(key)
                rows.append(dict(id=f'{family}-{split}-{len(rows):04d}', family=family,
                                 split=split, content_key=key, content=content,
                                 prompt=prompt, answer=answer))
                if len(rows) == count:
                    break
            if len(rows) != count:
                raise RuntimeError('finite unique fixture generation exhausted')
            result[family+'/'+split] = rows
    return result


class ShadowBranch(torch.nn.Module):
    def __init__(self, kind, dimension, interface, seed):
        super().__init__()
        if kind not in ('tide', 'mlp'):
            raise ValueError('unsupported shadow branch')
        self.kind = kind
        width = interface['width']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = torch.nn.Linear(dimension, width, bias=False)
            if kind == 'tide':
                from .core5 import Core5Encoder
                self.core = Core5Encoder(width=width, depth=interface['tide_depth'],
                                         region_width=interface['region_width'], seed=seed)
            else:
                self.core = torch.nn.Sequential(torch.nn.Linear(width, interface['mlp_hidden']),
                    torch.nn.GELU(), torch.nn.Linear(interface['mlp_hidden'], width))
            self.output = torch.nn.Linear(width, dimension, bias=False)
            torch.nn.init.zeros_(self.output.weight)
        self.last_metrics = {}

    def forward(self, x):
        value = torch.tanh(self.input(x))
        if self.kind == 'tide':
            value = self.core(value[:, None, :])
            self.last_metrics = dict(self.core.last_metrics)
        else:
            value = self.core(value)
            self.last_metrics = {'work/branch_samples': x.shape[0]}
        return self.output(value)


def statistics(x, y):
    mx, my = x.mean(0), y.mean(0)
    sx = (x-mx).square().mean().sqrt()
    sy = (y-my).square().mean().sqrt()
    if not torch.isfinite(sx+sy) or min(sx.item(), sy.item()) <= 1e-8:
        raise ValueError('teacher input/target is nonfinite or degenerate')
    return dict(x_mean=mx, y_mean=my, x_scale=sx, y_scale=sy)


def normalize(x, y, stats):
    return ((x-stats['x_mean'])/stats['x_scale'],
            (y-stats['y_mean'])/stats['y_scale'])


def blend_last(output, prediction, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    if alpha == 0:
        return output
    if prediction.shape != output[:, -1].shape:
        raise ValueError('local teacher/prediction shape mismatch')
    changed = output.clone()
    changed[:, -1] = output[:, -1] + alpha*(prediction-output[:, -1])
    return changed


@contextmanager
def intervention(lm, predictor=None, alpha=0., capture=None):
    """Modify only final decoder MLP, final position of this call.

    Original MLP still executes; no compute saving or full-layer replacement
    is claimed. The branch has independent one-position episodes.
    """
    calls = []
    def hook(module, args, output):
        x, y = args[0][:, -1], output[:, -1]
        if capture is not None:
            capture['x'], capture['y'] = x.detach().cpu(), y.detach().cpu()
        calls.append(1)
        if alpha == 0:
            return output
        pred = predictor(x)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('nonfinite local prediction')
        if capture is not None:
            capture['delta_rms'] = float((alpha*(pred-y)).square().mean().sqrt().item())
        return blend_last(output, pred, alpha)
    handle = lm.model.layers[-1].mlp.register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


def paired_summary(rows):
    n = len(rows)
    if not n:
        raise ValueError('empty paired evaluation')
    return dict(samples=n, baseline_accuracy=sum(r['base_correct'] for r in rows)/n,
                accuracy=sum(r['correct'] for r in rows)/n,
                accuracy_drop=sum(int(r['base_correct'])-int(r['correct']) for r in rows)/n,
                newly_wrong=sum(r['base_correct'] and not r['correct'] for r in rows),
                gained=sum(not r['base_correct'] and r['correct'] for r in rows),
                prediction_changes=sum(r['base_prediction']!=r['prediction'] for r in rows),
                answer_nll_increase=sum(r['nll']-r['base_nll'] for r in rows)/n,
                kl=sum(r['kl'] for r in rows)/n)


def preservation_gate(summary, delta, gates):
    return bool(summary['accuracy_drop'] <= gates['maximum_accuracy_drop'] + 1e-12
        and summary['newly_wrong']/summary['samples'] <= gates['maximum_new_error_fraction'] + 1e-12
        and summary['answer_nll_increase'] <= gates['maximum_answer_nll_increase']
        and delta > gates['minimum_nonzero_delta_rms'])
