"""Finite v2 workload selection; v1 definitions and variant lists remain frozen."""
from foundation_policy import FIELDS, resolve
from foundation_workloads import large_config


def configurations(args, suite):
    overrides = {name: getattr(args, name) for name in FIELDS if getattr(args, name, None) is not None}
    # --workers is a limit; Python defaults to serial, not an explicit node-thread request.
    overrides.pop('workers', None)
    if args.tier == 'large':
        if args.variants or args.modes:
            raise ValueError('large assessment uses declared variants and modes')
        selected = [p for p in suite['large_presets'] if not args.ids or p['id'] in args.ids]
        if not selected or (args.ids and set(args.ids) != {p['id'] for p in selected}):
            raise ValueError('unknown v2 large preset')
        result = []
        for preset in selected:
            for family in args.families or ['timed-dag', 'settle']:
                if family == 'pdg':
                    raise ValueError('v2 large PDG reuses reviewed historical evidence; use the PDG scale entry for a new question')
                target = large_config(preset, family)
                denom = preset['nominal_selection_denominator']
                for nodes in sorted({4*denom, target['body_nodes']}):
                    c = large_config(preset, family, nodes)
                    c.update(execution_schema='v2', execution_options=overrides, execution_pattern='prefill-stream',
                             prefill_positions=4, target_body_nodes=target['body_nodes'],
                             evaluation_stage='target' if nodes == target['body_nodes'] else 'resource-stage')
                    variants = ['native-settle-optimized' if family == 'settle' else 'native-frontier-optimized']
                    for v in variants: resolve(c, v, args.workers, overrides)
                    result.append((c, variants))
        return result
    selected = [c for c in suite['configurations'] if not args.ids or c['id'] in args.ids]
    if not selected or (args.ids and set(args.ids) != {c['id'] for c in selected}):
        raise ValueError('unknown v2 workload id')
    result = []
    for saved in selected:
        c = dict(saved)
        if args.families and c['graph'] not in args.families:
            continue
        variants = c.pop('variants')
        c.update(execution_schema='v2', execution_options=overrides)
        if args.tier == 'smoke':
            c.update(width=16, batch=4, sequence=6)
            if c.get('training_window'): c['training_window'] = 3
            if c.get('prefill_positions'): c['prefill_positions'] = 4
        if args.variants:
            variants = [v for v in variants if v in args.variants]
        if not variants: continue
        if args.modes:
            if not set(args.modes) <= set(c['modes']):
                raise ValueError('requested mode is not defined for selected v2 workload')
            c['modes'] = args.modes
        for v in variants: resolve(c, v, args.workers, overrides)
        result.append((c, variants))
    if not result or (args.variants and set(args.variants)-{v for _, vs in result for v in vs}):
        raise ValueError('no applicable v2 configuration/variant')
    return result


def add_arguments(parser):
    import argparse
    for name in ('full_autograd', 'aggregate_autograd'):
        parser.add_argument('--'+name.replace('_','-'), choices=['replay','batched'])
    for name in ('packed', 'prefill', 'packed_sources', 'batch_next', 'parallel_regions', 'compact_events', 'defer_state_release'):
        parser.add_argument('--'+name.replace('_','-'), action=argparse.BooleanOptionalAction, default=None)
    for name, choices in [('attention_packing',['exact','single']), ('fiber_pooling',['event','csr']),
                          ('fiber_cache',['cloned','owned']), ('attention_layout',['event','head']),
                          ('projection_layout',['input','linear'])]:
        parser.add_argument('--'+name.replace('_','-'), choices=choices)
