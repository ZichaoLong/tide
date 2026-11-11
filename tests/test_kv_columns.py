"""KV column interface: actual B/L ragged state, snapshots and selected queries.

Scalar local formulas own the oracle schedule. Column roots and exact decoded
terminal state are checked here; graph event-root adapters have separate tests.
"""
import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.records import State
from tidegraph.compare import equivalent
from tidegraph.weighted import bind, KVConfig, RegionalConfig
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.kv_columns_state import KVStateColumns, prepare_kv_columns
from tidegraph.weighted.kv_columns import read_kv_columns, full_kv_columns
from isolated_cases import vjp
from dense_checks import Checks
from column_numeric_witnesses import kv as witness


def fixture(dtype, read="qry", window=0, material_dtype=None):
    material_dtype = material_dtype or dtype
    generator = torch.Generator().manual_seed(271)
    binding = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)),
        memories=KVConfig(width=4, summary_width=2, heads=2, memory_read=read, window=window),
        regions=RegionalConfig(), dtype=material_dtype)
    binding.model.to(dtype=dtype)
    node = binding.model.nodes[0]
    if read == "qry":
        node.kernel.read_v = node.kernel.read_k
    base = 2**54
    roots = dict(node.named_parameters())
    def leaf(name, shape):
        value = (torch.rand(shape, dtype=material_dtype, generator=generator)*.2+.1).to(dtype).requires_grad_()
        roots[name] = value
        return value
    states = []
    for owner, times in enumerate(([base+1, base+4], [base+3], [])):
        n = len(times)
        slots = dict(times=torch.tensor(times, dtype=torch.int64),
            keys=leaf(f"{owner}.keys", (n, 4)), values=leaf(f"{owner}.values", (n, 4)),
            summaries=leaf(f"{owner}.summaries", (n, 2)), weights=leaf(f"{owner}.weights", (n,)))
        states.append(State(torch.zeros_like(node.bias), times[-1] if times else -1, n, slots))
    mass = torch.tensor([.3, 0., .6, 0., .5, .8, .4], dtype=material_dtype).to(dtype).requires_grad_()
    value = leaf("content", (7, 4)); roots["mass"] = mass
    layout = SequenceLayout(torch.tensor([0, 3, 6, 7]), torch.tensor([base+t for t in (10, 13, 29, 11, 18, 35, 22)]))
    return node, states, mass, value, layout, roots


@pytest.mark.parametrize("read,window", [("sum", 0), ("qry", 0), ("qry", 7)])
@pytest.mark.parametrize("mode", ["0", "-", "+"])
def test_kv_column_snapshots_terminal_and_full_vjp(dtype, read, window, mode, material_dtype=None):
    node, old, mass, content, layout, roots = fixture(dtype, read, window, material_dtype)
    initial_states = [node.initial() for _ in old]
    initial = KVStateColumns.from_states(initial_states)
    prepared = prepare_kv_columns(node.kernel, layout, mass, content, KVStateColumns.from_states(old))
    actual = read_kv_columns(node, prepared, read_mode=mode, initial=initial)
    descriptors, proposed, terminal = [], [], []
    for b, (a, z) in enumerate(layout.spans):
        previous = old[b]
        for j in range(a, z):
            h = Content(torch.cat((mass[j:j+1], content[j])))
            state = node.kernel.step(node, previous, h, int(layout.times[j]))
            snapshot = initial_states[b] if mode == "0" else previous if mode == "-" else state
            descriptors.append(node.read_program.step(node, ReadInput(snapshot, int(layout.times[j]), h)))
            proposed.append(state); previous = state
        terminal.append(previous)
    expected = {name: torch.stack([row[name] for row in descriptors]) for name in descriptors[0]}
    equivalent(actual, expected)
    equivalent(prepared.terminal.states(), terminal)
    controls = torch.tensor([.2, 0., .4, 0., .5, .6, .3], dtype=material_dtype or dtype).to(dtype).requires_grad_()
    roots["control"] = controls
    full = full_kv_columns(node, prepared, controls)
    chosen = [0, 2, 4, 5, 6]
    assert full.rows.tolist() == chosen
    values = torch.stack([node.full_program.content(node.kernel, proposed[j], int(layout.times[j]), content[j]) for j in chosen])
    equivalent(full.content, values)
    equivalent(full.strength, controls[chosen])
    def loss(desc, fresh, strength, states):
        # Independent nonsymmetric cotangents avoid reducing this whole check
        # to the nearly constant squared norm of normalized evidence.
        result = sum((v*torch.linspace(.2, .8, v.numel(), dtype=v.dtype).reshape(v.shape)).sum()
                     for v in desc.values())
        result = result+fresh.square().sum()+strength.square().sum()
        return result+sum(v.square().sum() for state in states for name, v in state.slots.items() if name != "times")
    a = loss(actual, full.content, full.strength, prepared.terminal.states())
    e = loss(expected, values, controls[chosen], terminal)
    check = Checks(f"kv-columns:{dtype}:{mode}:{read}:{window}",
                   known_numeric=witness(dtype, mode, read, window),
                   failure_reason="column FP32 reduction witness; same material FP64 check is separate")
    check(vjp(a, roots), vjp(e, roots), "gradient")
    equivalent(vjp(a, roots, zero=True), vjp(e, roots, zero=True))
    check.finish()


@pytest.mark.parametrize("read,window", [("sum", 0), ("qry", 0), ("qry", 7)])
@pytest.mark.parametrize("mode", ["0", "-", "+"])
def test_same_fp32_material_promoted(read, window, mode):
    test_kv_column_snapshots_terminal_and_full_vjp(torch.float64, read, window, mode, torch.float32)


@pytest.mark.parametrize("read", ["sum", "qry"])
def test_raw_expiry_zero_receive_and_exact_next_window(dtype, read):
    node, old, mass, content, layout, roots = fixture(dtype, read, window=7)
    first = prepare_kv_columns(node.kernel, layout, mass, content, KVStateColumns.from_states(old))
    expected = []
    for b, (a, z) in enumerate(layout.spans):
        state = old[b]
        for j in range(a, z):
            state = node.kernel.step(node, state, Content(torch.cat((mass[j:j+1], content[j]))), int(layout.times[j]))
        expected.append(state)
    equivalent(first.terminal.states(), expected)
    # A later all-zero receive must preserve expired raw entries and clocks.
    later = SequenceLayout(torch.tensor([0, 1, 2, 3]), torch.tensor([2**54+100]*3))
    zero = torch.zeros(3, dtype=dtype, requires_grad=True)
    second = prepare_kv_columns(node.kernel, later, zero, content[:3], first.terminal)
    assert second.terminal is first.terminal
    equivalent(second.terminal.states(), first.terminal.states(), atol=0, rtol=0)
    restored = KVStateColumns.from_states(first.terminal.states())
    resumed = prepare_kv_columns(node.kernel, later, mass[:3], content[:3], restored)
    direct = prepare_kv_columns(node.kernel, later, mass[:3], content[:3], first.terminal)
    equivalent(resumed.terminal.states(), direct.terminal.states(), atol=0, rtol=0)


def test_kv_read0_is_bound_and_clocks_are_never_float():
    node, old, mass, content, layout, _ = fixture(torch.float64)
    prepared = prepare_kv_columns(node.kernel, layout, mass, content, KVStateColumns.from_states(old))
    with pytest.raises(ValueError, match="bound initial"):
        read_kv_columns(node, prepared, read_mode="0")
    with pytest.raises(ValueError, match="int64"):
        SequenceLayout(layout.offsets, layout.times.double())
    with pytest.raises(ValueError, match="strictly increase"):
        SequenceLayout(torch.tensor([0, 2]), torch.tensor([17, 17]))
