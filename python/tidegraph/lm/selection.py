"""Explicit finite queue manifests, validated without allocating a model."""
from copy import deepcopy
import re
from .config import resolve
from ..experiment.records import fingerprint, read_json

ACTIONS = {'run', 'check', 'training-check', 'diagnose', 'memory-check'}


def single(config):
    """A public single configuration uses the same bounded queue as a matrix."""
    c = resolve(config)['request']
    cases = [dict(name=c['name'], kind='single', action='run', config=c)]
    return validate(dict(schema='tide-lm-matrix-v1', identity=fingerprint(cases), cases=cases))


def validate(value):
    value = read_json(value) if not isinstance(value, dict) else deepcopy(value)
    if value.get('schema') != 'tide-lm-matrix-v1' or not isinstance(value.get('cases'), list):
        raise ValueError('manifest: expected tide-lm-matrix-v1 cases list')
    names=set()
    for item in value['cases']:
        name=item.get('name','')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) or name in names:
            raise ValueError('manifest.cases.name: unique safe lowercase name required')
        names.add(name)
        if item.get('action') not in ACTIONS:
            raise ValueError(f'manifest case {name}: unsupported action')
        if not isinstance(item.get('kind'),str):
            raise ValueError(f'manifest case {name}: kind required')
        # Do not silently re-resolve an effective configuration to different values.
        c=resolve(item['config'])['request']
        if c != item['config'] or c['name'] != name:
            raise ValueError(f'manifest case {name}: provide the complete expanded configuration')
    if not names:
        raise ValueError('manifest: no cases selected')
    identity=fingerprint(value['cases'])
    if value.get('identity') != identity:
        raise ValueError('manifest.identity: differs from the selected cases')
    return value
