"""Each finite recurrent module against its scalar equations and raw cache."""
import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.records import State
from tidegraph.weighted import bind, RecurrentConfig, RegionalConfig
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.column_modules import module_for


@pytest.mark.parametrize("family", ["affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("read_mode", ["0", "-", "+"])
def test_matrix_column_snapshots_raw_convolution_and_vjp(family, read_mode):
    dtype = torch.float64
    extra = {} if family in {"affine", "delta"} else dict(heads=2, conv_size=3, conv_stride=2)
    binding = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)),
                   memories=RecurrentConfig(memory_family=family, memory_read="state", **extra),
                   regions=RegionalConfig(read_mode=read_mode), dtype=dtype)
    node = binding.model.nodes[0]
    module = module_for(node)
    old = []
    roots = [p for p in node.parameters() if p.requires_grad]
    for b in range(2):
        s = node.kernel.step(node, node.initial(), Content(torch.tensor([.3, .2, -.1, .4, .3], dtype=dtype)), 2**54+1+b)
        slots = {n: x.detach().requires_grad_() if x.is_floating_point() else x for n, x in s.slots.items()}
        roots.extend(x for x in slots.values() if x.is_floating_point())
        old.append(State(s.value, s.last_time, s.observations, slots))
    layout = SequenceLayout(torch.tensor([0, 4, 7]), torch.tensor([2**54+t for t in (4, 6, 11, 12, 5, 9, 15)]))
    mass = torch.tensor([.3, 0., .7, .2, .8, .9, 0.], dtype=dtype, requires_grad=True)
    u = torch.linspace(-.3, .4, 28, dtype=dtype).reshape(7, 4).requires_grad_()
    roots += [mass, u]
    initial = module.state_type.from_states([node.initial() for _ in old])
    window = module.prepare(node.kernel, layout, mass, u, module.state_type.from_states(old))
    actual = module.read(node, window, read_mode=read_mode, initial=initial)
    desc, proposal, terminal = [], [], []
    for b, (a, z) in enumerate(layout.spans):
        state = old[b]
        for j in range(a, z):
            h = Content(torch.cat((mass[j:j+1], u[j])))
            fresh = node.kernel.step(node, state, h, int(layout.times[j]))
            read = node.initial() if read_mode == "0" else state if read_mode == "-" else fresh
            desc.append(node.read_program.step(node, ReadInput(read, int(layout.times[j]), h)))
            proposal.append(fresh);state=fresh
        terminal.append(state)
    expected = {n: torch.stack([r[n] for r in desc]) for n in desc[0]}
    equivalent(actual, expected)
    equivalent(window.terminal.states(), terminal)
    controls = torch.tensor([.2, 0., .4, .8, .2, .6, 0.], dtype=dtype, requires_grad=True)
    roots.append(controls)
    full = module.full(node, window, controls)
    chosen = full.rows.tolist()
    value = torch.stack([node.full_program.content(node.kernel, proposal[j], int(layout.times[j]), u[j]) for j in chosen])
    equivalent(full.content, value)
    def loss(fields, y, states):
        return (sum((v*torch.linspace(-.3, .6, v.numel(), dtype=dtype).reshape(v.shape)).sum() for v in fields.values())
                +y.square().sum()+sum(x.square().sum() for s in states for x in s.slots.values() if x.is_floating_point()))
    a, e = loss(actual, full.content, window.terminal.states()), loss(expected, value, terminal)
    for factor in (1., 0.):
        equivalent(torch.autograd.grad(a*factor, roots, allow_unused=True, retain_graph=True),
                   torch.autograd.grad(e*factor, roots, allow_unused=True, retain_graph=True))
