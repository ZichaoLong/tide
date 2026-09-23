"""Finite six-class trajectories, independent schedules and actual native owners."""
from dataclasses import fields, is_dataclass, replace
import torch
import _tide_native as core
from tidegraph.compare import objective
from tidegraph.native import Native
from tidegraph.native_records import to_continuation, from_continuation
from tidegraph.reference import run
from tidegraph.settle import run as settle
from tidegraph.specialized import run as specialized, settle_layered
from test_native_settle import native, projected
from test_specialized_topologies import fixture, layered_fixture


def frozen(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if is_dataclass(value): return replace(value, **{f.name:frozen(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict): return {k:frozen(v) for k,v in value.items()}
    if isinstance(value, list): return [frozen(v) for v in value]
    if isinstance(value, tuple): return tuple(frozen(v) for v in value)
    return value


def optimizer(model, kind, native_optimizer=False):
    registry = core.ParameterRegistry()
    for name, p in model.named_parameters(remove_duplicate=False): registry.add(name, p)
    owners = registry.owners(); parameters = [v.value for v in owners]
    values = dict(lr=.0002, weight_decay=.01)
    if kind == 'adamw': values.update(eps=1e-5, betas=(.9, .99), amsgrad=True)
    else: values['momentum'] = .8 if kind == 'momentum' else 0.
    if not native_optimizer:
        opt = torch.optim.AdamW(parameters, **values) if kind == 'adamw' else torch.optim.SGD(parameters, **values)
        return opt, registry
    group = core.OptimizerGroup(); group.parameters = [v.canonical for v in owners]
    for key, value in values.items():
        if key == 'betas': group.beta1, group.beta2 = value
        else: setattr(group, key, value)
    return (core.AdamW if kind == 'adamw' else core.SGD)(registry, [group]), registry


def optimizer_state(opt, registry, native_optimizer):
    if native_optimizer:
        return {name: {'step':s.step, **{k: frozen(getattr(s, k)) for k in ('momentum_buffer', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')}}
                for name, s in opt.state().items()}
    result = {}
    for owner in registry.owners():
        state = opt.state.get(owner.value, {})
        if state:
            result[owner.canonical] = {'step':int(state.get('step',0)), **{k:frozen(state.get(k)) for k in ('momentum_buffer', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')}}
    return result


class TrainingCase:
    def __init__(self, dtype, family, implementation, memory):
        self.family, self.implementation = family, implementation
        if family == 'settle':
            self.spec, self.model, self.q, self.values = layered_fixture(dtype, memory, False)
            self.graph = self.spec.graph; self.leaves = {'input':self.values}
            # Trainable independent initial memories, including history roots.
            for b in range(2):
                for n,w in enumerate(self.model.nodes):
                    s = w.initial(); s.value = s.value.detach().requires_grad_()
                    s.slots = {k:v.detach().requires_grad_() for k,v in s.slots.items()}
                    self.q.states[b,n] = s; self.leaves[f'initial.{b}.{n}.value'] = s.value
                    self.leaves.update({f'initial.{b}.{n}.{k}':v for k,v in s.slots.items()})
            self.stops = (1, 2, 4)
        else:
            self.topology = 'ring' if family == 'pdg' else 'diamond'
            self.graph, self.model, self.q, self.inputs, self.leaves = fixture(
                dtype, 'ring' if family == 'pdg' else 'diamond-region', kind=memory, selector='tensor-history-v1')
            self.stops = (3, 6, 9)
        m = self.model; extra = m.nodes[0].extra
        extra['application_head'] = torch.nn.Parameter(torch.arange(28, dtype=dtype).reshape(7,4)/90-.1)
        extra['head_bias'] = torch.nn.Parameter(torch.zeros(7, dtype=dtype))
        extra['unused'] = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        extra['zero'] = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        m.nodes[-1].extra['shared_zero'] = extra['zero']
        self.leaves = dict(m.named_parameters()) | self.leaves
        self.head = core.DenseLinear(3) if implementation.startswith('native') else None
        self.engine = None
        if implementation.startswith('native'):
            if family == 'settle':
                self.compiled, self.engine, self.encoded_q = native(self.spec, m, self.q, mode='hst',
                    packed=implementation != 'native-serial', workers=1 if implementation == 'native-serial' else 3)
                # Used only for test record conversion; native constructs its own encoding.
                self.eg, _ = self.spec.embed(m)
            else:
                algorithm = self.topology if implementation == 'native-specialized' else 'streaming' if family == 'pdg' else 'frontier'
                self.engine = Native(self.graph, m, algorithm=algorithm, mode='hst',
                    workers=1 if implementation == 'native-serial' else 3,
                    packed=implementation not in {'native-serial','native-parallel'})

    def advance(self, stop):
        if self.family == 'settle':
            x = self.values[:,self.q.cut:stop]
            if self.engine:
                result = self.engine.run(self.encoded_q, x); self.encoded_q = result.continuation
                result = projected(self.spec, self.compiled, result)
            else:
                fn = settle_layered if self.implementation == 'specialized' else settle
                result = fn(self.spec, self.model, self.q, x, mode='hst')
        else:
            xs = [x for x in self.inputs if self.q.cut <= x.time < stop]
            if self.engine: result = self.engine.run(self.q, xs, stop, sealed_until=stop)
            elif self.implementation == 'specialized':
                result = specialized(self.graph, self.model, self.q, xs, stop, sealed_until=stop, topology=self.topology, mode='hst')
            else: result = run(self.graph, self.model, self.q, xs, stop, sealed_until=stop, mode='hst')
        self.q = result.continuation
        return result

    def loss(self, result, step):
        loss = objective(result); extra = self.model.nodes[0].extra
        if result.outputs:
            x = torch.stack([x for _,_,_,x in result.outputs])
            logits = self.head.run(x, extra['application_head'], extra['head_bias']) if self.head else torch.nn.functional.linear(
                x, extra['application_head'], extra['head_bias'])
            loss = loss + logits.square().sum()*.2
        # First unused, then connected zero: optimizer ownership must distinguish.
        if step: loss = loss + extra['zero'].sum()*0
        return loss

    def detach(self):
        self.q = self.q.detach()
        if self.family == 'settle' and self.engine:
            q = from_continuation(self.eg, self.encoded_q).detach()
            self.encoded_q = to_continuation(core, self.eg, self.compiled.encoded_graph, q)


def trajectory(dtype, family, implementation, memory, kind, native_optimizer=False):
    c = TrainingCase(dtype, family, implementation, memory)
    opt, registry = optimizer(c.model, kind, native_optimizer)
    owner_ids = {n:id(p) for n,p in c.model.named_parameters(remove_duplicate=False)}
    records = []
    for i, stop in enumerate(c.stops):
        opt.zero_grad(set_to_none=True)
        for p in c.leaves.values(): p.grad = None
        result = c.advance(stop); loss = c.loss(result,i)
        # Isolated roots include state, tensor history and pending whenever present.
        roots = {}
        for name in ('output','state','history','pending'):
            try: root = objective(result,name)
            except ValueError: continue
            if root.requires_grad:
                roots[name] = dict(zip(c.leaves, torch.autograd.grad(root,tuple(c.leaves.values()),allow_unused=True,retain_graph=True)))
        loss.backward()
        gradients = {n:frozen(p.grad) for n,p in c.leaves.items()}
        assert gradients['nodes.0.extra.unused'] is None
        zero = gradients['nodes.0.extra.zero']
        assert zero is None if i == 0 else zero is not None and not zero.count_nonzero()
        c.detach(); opt.step()
        records.append((frozen(result), frozen(roots), gradients, frozen(c.model.state_dict()),
                        optimizer_state(opt,registry,native_optimizer), registry.alias_partitions()))
        assert owner_ids == {n:id(p) for n,p in c.model.named_parameters(remove_duplicate=False)}
    return records
