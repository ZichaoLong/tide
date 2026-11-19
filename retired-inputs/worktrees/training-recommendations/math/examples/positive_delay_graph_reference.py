#!/usr/bin/env python3
"""Finite-cut reference for a positive-delay directed graph.

This standard-library-only executable witnesses the concrete semantics in
``positive-delay-graph-finite-cut-learning-note.md``.  It deliberately keeps
node kernels tiny.  The checks concern the outer mathematical contract:

* directed cycles are allowed, but every message edge has positive delay;
* a cut stores node state, selector-history, and crossing messages;
* selector controls reach Full, while Next updates persistent state without
  reading Full; the comparison snapshot survives an activation-time reset;
* one-shot and arbitrarily partitioned execution have identical recorded
  trace projections and continuations;
* dropping a crossing message changes later fibers.

This is executable evidence, not a replacement for the proofs in the note.
It requires Python 3.7 or newer and otherwise uses only the standard library.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Sequence, Set, Tuple, Union


Value = Union[int, Fraction]
NodeState = Value
HistoryState = Tuple[Tuple[str, int], ...]
Content = Value
Descriptor = Value
Control = Fraction


@dataclass(frozen=True, order=True)
class Edge:
    name: str
    source: str
    target: str
    delay: int


@dataclass(frozen=True, order=True)
class External:
    port: str
    position: int
    time: int
    value: Value


@dataclass(frozen=True, order=True)
class Message:
    send: int
    edge: str
    value: Value


@dataclass(frozen=True, order=True)
class Output:
    time: int
    port: str
    value: Value


Atom = Union[External, Message]


AggFn = Callable[[str, int, Tuple[Atom, ...]], Content]
UpdateFn = Callable[[str, NodeState, int, Content], NodeState]
ReadFn = Callable[[str, NodeState, NodeState, int, Content], Descriptor]
SelectFn = Callable[
    [str, HistoryState, int, Mapping[str, Descriptor]],
    Tuple[Tuple[str, ...], Mapping[str, Control], HistoryState],
]
NextFn = Callable[
    [str, NodeState, NodeState, int, Content, bool, Control], NodeState
]
FullFn = Callable[
    [str, NodeState, int, Content, Control],
    Tuple[Mapping[str, Value], Mapping[str, Value]],
]


def keep_comparison_state(
    _node: str,
    _old: NodeState,
    comparison: NodeState,
    _theta: int,
    _content: Content,
    _active: bool,
    _control: Control,
) -> NodeState:
    """Default Next: retain the state chosen by the candidate-adoption rule."""
    return comparison


@dataclass(frozen=True)
class Spec:
    nodes: Tuple[str, ...]
    edges: Tuple[Edge, ...]
    input_target: Mapping[str, str]
    output_source: Mapping[str, str]
    region_nodes: Mapping[str, Tuple[str, ...]]
    budget: Mapping[str, int]
    observe_all: Mapping[str, bool]
    initial_state: Mapping[str, NodeState]
    initial_history: Mapping[str, HistoryState]
    agg: AggFn
    update: UpdateFn
    read: ReadFn
    select: SelectFn
    full: FullFn
    next_state: NextFn = keep_comparison_state

    def edge_map(self) -> Dict[str, Edge]:
        return {edge.name: edge for edge in self.edges}

    def region_of(self) -> Dict[str, str]:
        return {
            node: region
            for region, nodes in self.region_nodes.items()
            for node in nodes
        }

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("the node set must be nonempty")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("node names must be unique")
        if len({edge.name for edge in self.edges}) != len(self.edges):
            raise ValueError("edge identities must be unique")
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge endpoint is not a node: {edge}")
            if edge.delay <= 0:
                raise ValueError(f"edge delay must be positive: {edge}")
        for target in self.input_target.values():
            if target not in self.nodes:
                raise ValueError(f"input target is not a node: {target}")
        for source in self.output_source.values():
            if source not in self.nodes:
                raise ValueError(f"output source is not a node: {source}")

        assigned = [
            node for nodes in self.region_nodes.values() for node in nodes
        ]
        if sorted(assigned) != sorted(self.nodes):
            raise ValueError("regions must partition the node set")
        for region, nodes in self.region_nodes.items():
            if not nodes:
                raise ValueError("regions must be nonempty")
            if not 1 <= self.budget[region] <= len(nodes):
                raise ValueError("region budget is outside its candidate range")
            if region not in self.observe_all:
                raise ValueError("every region needs a state-adoption mode")
        if set(self.initial_history) != set(self.region_nodes):
            raise ValueError("initial histories must match the region set")
        if set(self.initial_state) != set(self.nodes):
            raise ValueError("every node needs exactly one initial state")


@dataclass
class Continuation:
    cut: int
    node_state: Dict[str, NodeState]
    selector_history: Dict[str, HistoryState]
    pending: Tuple[Message, ...]


@dataclass
class SegmentTrace:
    """Complete finite-cut coordinates, including control and read snapshots."""

    start: int
    stop: int
    buckets: Dict[Tuple[str, int], Tuple[Atom, ...]] = field(
        default_factory=dict
    )
    candidates: Dict[Tuple[str, int], Tuple[str, ...]] = field(
        default_factory=dict
    )
    active: Dict[Tuple[str, int], Tuple[str, ...]] = field(
        default_factory=dict
    )
    contents: Dict[Tuple[str, int], Content] = field(default_factory=dict)
    proposals: Dict[Tuple[str, int], NodeState] = field(default_factory=dict)
    descriptors: Dict[Tuple[str, int], Descriptor] = field(default_factory=dict)
    controls: Dict[Tuple[str, int], Control] = field(default_factory=dict)
    comparison_states: Dict[Tuple[str, int], NodeState] = field(
        default_factory=dict
    )
    full_values: Dict[
        Tuple[str, int], Tuple[Dict[str, Value], Dict[str, Value]]
    ] = field(default_factory=dict)
    states: Dict[Tuple[str, int], NodeState] = field(default_factory=dict)
    histories: Dict[Tuple[str, int], HistoryState] = field(
        default_factory=dict
    )
    messages: Tuple[Message, ...] = ()
    outputs: Tuple[Output, ...] = ()


def history_map(history: HistoryState) -> Dict[str, int]:
    return dict(history)


def freeze_history(history: Mapping[str, int]) -> HistoryState:
    return tuple(sorted(history.items()))


def atom_key(atom: Atom) -> Tuple[object, ...]:
    if isinstance(atom, External):
        return ("ext", atom.port, atom.position, atom.time)
    return ("msg", atom.edge, atom.send)


def message_time(spec: Spec, message: Message) -> int:
    return message.send + spec.edge_map()[message.edge].delay


def message_target(spec: Spec, message: Message) -> str:
    return spec.edge_map()[message.edge].target


def initial_continuation(spec: Spec) -> Continuation:
    spec.validate()
    return Continuation(
        cut=0,
        node_state=dict(spec.initial_state),
        selector_history=dict(spec.initial_history),
        pending=(),
    )


def validate_external_history(
    spec: Spec,
    records: Sequence[External],
    *,
    require_contiguous: bool = False,
    require_initial_prefix: bool = False,
) -> None:
    identities = set()
    port_times = set()
    by_port: Dict[str, List[External]] = {}
    for record in records:
        if record.port not in spec.input_target:
            raise ValueError(f"unknown input port: {record.port}")
        if record.position < 0 or record.time < 0:
            raise ValueError("external positions and times must be nonnegative")
        identity = (record.port, record.position)
        if identity in identities:
            raise ValueError(f"duplicate external identity: {identity}")
        identities.add(identity)
        port_time = (record.port, record.time)
        if port_time in port_times:
            raise ValueError(f"two records share one port-time: {port_time}")
        port_times.add(port_time)
        by_port.setdefault(record.port, []).append(record)

    for port, port_records in by_port.items():
        ordered = sorted(port_records, key=lambda record: record.position)
        if require_initial_prefix and ordered[0].position != 0:
            raise ValueError(
                f"complete input on {port} must start at position zero"
            )
        if require_contiguous and any(
            later.position != earlier.position + 1
            for earlier, later in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                f"complete input positions must be contiguous on {port}"
            )
        if any(
            earlier.time >= later.time
            for earlier, later in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                f"input times must increase strictly with position on {port}"
            )


def validate_external_segment(
    spec: Spec,
    records: Sequence[External],
    start: int,
    stop: int,
) -> None:
    validate_external_history(
        spec,
        records,
        require_contiguous=True,
        require_initial_prefix=(start == 0),
    )
    if any(not start <= record.time < stop for record in records):
        raise ValueError("external record is outside the requested interval")


def run_window(
    spec: Spec,
    continuation: Continuation,
    external: Sequence[External],
    stop: int,
) -> Tuple[SegmentTrace, Continuation]:
    """Evaluate exactly the half-open logical-time interval [cut, stop)."""

    spec.validate()
    start = continuation.cut
    if stop < start:
        raise ValueError("stop precedes the continuation cut")
    validate_external_segment(spec, external, start, stop)

    edge_by_name = spec.edge_map()
    region_of = spec.region_of()
    node_state = copy.deepcopy(continuation.node_state)
    selector_history = copy.deepcopy(continuation.selector_history)
    available_messages = list(continuation.pending)
    new_messages = []
    outputs = []
    trace = SegmentTrace(start=start, stop=stop)

    for node in spec.nodes:
        trace.states[(node, start)] = copy.deepcopy(node_state[node])
    for region in spec.region_nodes:
        trace.histories[(region, start)] = copy.deepcopy(
            selector_history[region]
        )

    for theta in range(start, stop):
        content: Dict[str, Content] = {}
        proposal: Dict[str, NodeState] = {}
        descriptor: Dict[str, Descriptor] = {}

        for node in spec.nodes:
            atoms = [
                record
                for record in external
                if record.time == theta
                and spec.input_target[record.port] == node
            ]
            atoms.extend(
                message
                for message in available_messages
                if message_time(spec, message) == theta
                and message_target(spec, message) == node
            )
            bucket = tuple(sorted(atoms, key=atom_key))
            trace.buckets[(node, theta)] = bucket
            if bucket:
                content[node] = spec.agg(node, theta, bucket)
                proposal[node] = spec.update(
                    node, node_state[node], theta, content[node]
                )
                descriptor[node] = spec.read(
                    node,
                    node_state[node],
                    proposal[node],
                    theta,
                    content[node],
                )
                trace.contents[(node, theta)] = copy.deepcopy(content[node])
                trace.proposals[(node, theta)] = copy.deepcopy(proposal[node])
                trace.descriptors[(node, theta)] = copy.deepcopy(descriptor[node])

        active_by_region: Dict[str, Tuple[str, ...]] = {}
        for region, region_nodes in spec.region_nodes.items():
            candidates = tuple(node for node in region_nodes if node in content)
            trace.candidates[(region, theta)] = candidates
            old_history = selector_history[region]
            if candidates:
                active, controls, new_history = spec.select(
                    region,
                    old_history,
                    theta,
                    {node: descriptor[node] for node in candidates},
                )
            else:
                active, controls, new_history = (), {}, old_history

            if len(set(active)) != len(active):
                raise ValueError("selector returned a duplicate active node")
            if not set(active).issubset(candidates):
                raise ValueError("selector returned a non-candidate")
            if len(active) > spec.budget[region]:
                raise ValueError("selector exceeded the region budget")
            if set(controls) != set(candidates):
                raise ValueError("selector controls must match all candidates")
            for node in candidates:
                trace.controls[(node, theta)] = copy.deepcopy(controls[node])
            active_by_region[region] = tuple(active)
            trace.active[(region, theta)] = tuple(active)
            selector_history[region] = copy.deepcopy(new_history)

        for node in spec.nodes:
            if node not in proposal:
                continue
            region = region_of[node]
            adopted = (
                spec.observe_all[region]
                or node in active_by_region[region]
            )
            comparison = proposal[node] if adopted else node_state[node]
            trace.comparison_states[(node, theta)] = copy.deepcopy(comparison)
            node_state[node] = copy.deepcopy(spec.next_state(
                node,
                node_state[node],
                comparison,
                theta,
                content[node],
                node in active_by_region[region],
                trace.controls[(node, theta)],
            ))

        for node in spec.nodes:
            region = region_of[node]
            if node not in active_by_region[region]:
                continue
            edge_values, output_values = spec.full(
                node,
                trace.comparison_states[(node, theta)],
                theta,
                content[node],
                trace.controls[(node, theta)],
            )
            trace.full_values[(node, theta)] = (
                dict(edge_values), dict(output_values)
            )
            for edge_name, value in edge_values.items():
                edge = edge_by_name.get(edge_name)
                if edge is None or edge.source != node:
                    raise ValueError("Full emitted on an invalid outgoing edge")
                message = Message(send=theta, edge=edge_name, value=value)
                if any(
                    prior.send == theta and prior.edge == edge_name
                    for prior in new_messages
                ):
                    raise ValueError("two messages share one source-event edge")
                new_messages.append(message)
                available_messages.append(message)
            for port, value in output_values.items():
                if spec.output_source.get(port) != node:
                    raise ValueError("Full emitted on an invalid output port")
                outputs.append(Output(time=theta, port=port, value=value))

        for node in spec.nodes:
            trace.states[(node, theta + 1)] = copy.deepcopy(node_state[node])
        for region in spec.region_nodes:
            trace.histories[(region, theta + 1)] = copy.deepcopy(
                selector_history[region]
            )

    pending = tuple(
        sorted(
            (
                message
                for message in available_messages
                if message_time(spec, message) >= stop
            ),
            key=lambda message: (
                message_time(spec, message),
                message.edge,
                message.send,
            ),
        )
    )
    trace.messages = tuple(sorted(new_messages))
    trace.outputs = tuple(sorted(outputs))
    next_continuation = Continuation(
        cut=stop,
        node_state=copy.deepcopy(node_state),
        selector_history=copy.deepcopy(selector_history),
        pending=pending,
    )
    return trace, next_continuation


def merge_maps(
    target: Dict[object, object], source: Mapping[object, object]
) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise AssertionError(f"segment boundary disagrees at {key!r}")
        target[key] = copy.deepcopy(value)


def merge_traces(traces: Sequence[SegmentTrace]) -> SegmentTrace:
    if not traces:
        raise ValueError("at least one trace is required")
    merged = SegmentTrace(start=traces[0].start, stop=traces[-1].stop)
    messages = []
    outputs = []
    previous_stop = traces[0].start
    for trace in traces:
        if trace.start != previous_stop:
            raise ValueError("traces do not form adjacent intervals")
        for target, source in (
            (merged.buckets, trace.buckets),
            (merged.candidates, trace.candidates),
            (merged.active, trace.active),
            (merged.contents, trace.contents),
            (merged.proposals, trace.proposals),
            (merged.descriptors, trace.descriptors),
            (merged.controls, trace.controls),
            (merged.comparison_states, trace.comparison_states),
            (merged.full_values, trace.full_values),
            (merged.states, trace.states),
            (merged.histories, trace.histories),
        ):
            merge_maps(target, source)
        messages.extend(trace.messages)
        outputs.extend(trace.outputs)
        previous_stop = trace.stop
    merged.messages = tuple(sorted(messages))
    merged.outputs = tuple(sorted(outputs))
    return merged


def run_partitioned(
    spec: Spec,
    external: Sequence[External],
    cuts: Sequence[int],
) -> Tuple[SegmentTrace, Continuation]:
    if len(cuts) < 2 or cuts[0] != 0:
        raise ValueError("cuts need a start and stop and must begin at zero")
    if any(left > right for left, right in zip(cuts, cuts[1:])):
        raise ValueError("cuts must be nondecreasing")
    validate_external_history(
        spec,
        external,
        require_contiguous=True,
        require_initial_prefix=True,
    )
    continuation = initial_continuation(spec)
    traces = []
    for start, stop in zip(cuts, cuts[1:]):
        segment_external = tuple(
            record for record in external if start <= record.time < stop
        )
        trace, continuation = run_window(
            spec, continuation, segment_external, stop
        )
        traces.append(trace)
    return merge_traces(traces), continuation


def sum_payloads(_node: str, _theta: int, atoms: Tuple[Atom, ...]) -> Content:
    return sum(atom.value for atom in atoms)


def ordered_update(
    _node: str, old: NodeState, theta: int, content: Content
) -> NodeState:
    # Deliberately noncommutative across logical times, so a missing state
    # dependency is unlikely to be hidden by the conformance examples.
    return 3 * old + content + theta


def proposal_read(
    _node: str,
    _old: NodeState,
    proposal: NodeState,
    _theta: int,
    _content: Content,
) -> Descriptor:
    return proposal


def counted_top_one(
    _region: str,
    history: HistoryState,
    _theta: int,
    descriptors: Mapping[str, Descriptor],
) -> Tuple[Tuple[str, ...], Mapping[str, Control], HistoryState]:
    counts = history_map(history)
    selected = min(
        descriptors,
        key=lambda node: (-descriptors[node], counts.get(node, 0), node),
    )
    counts[selected] = counts.get(selected, 0) + 1
    return (
        (selected,),
        {node: Fraction(1) for node in descriptors},
        freeze_history(counts),
    )


def self_loop_spec(delay: int) -> Spec:
    def full(
        node: str,
        state: NodeState,
        _theta: int,
        content: Content,
        _control: Control,
    ) -> Tuple[Mapping[str, Value], Mapping[str, Value]]:
        assert node == "v"
        return {"a": content}, {"out": state}

    return Spec(
        nodes=("v",),
        edges=(Edge("a", "v", "v", delay),),
        input_target={"in": "v"},
        output_source={"out": "v"},
        region_nodes={"r": ("v",)},
        budget={"r": 1},
        observe_all={"r": True},
        initial_state={"v": 0},
        initial_history={"r": (("v", 0),)},
        agg=sum_payloads,
        update=ordered_update,
        read=proposal_read,
        select=counted_top_one,
        full=full,
    )


def two_node_ring_spec() -> Spec:
    def full(
        node: str,
        state: NodeState,
        theta: int,
        content: Content,
        _control: Control,
    ) -> Tuple[Mapping[str, Value], Mapping[str, Value]]:
        if node == "u":
            return {"uv": content + 1}, {"out_u": state + theta}
        return {"vu": content - 1}, {"out_v": state - theta}

    return Spec(
        nodes=("u", "v"),
        edges=(
            Edge("uv", "u", "v", 1),
            Edge("vu", "v", "u", 2),
        ),
        input_target={"in_u": "u", "in_v": "v"},
        output_source={"out_u": "u", "out_v": "v"},
        region_nodes={"r": ("u", "v")},
        budget={"r": 1},
        observe_all={"r": True},
        initial_state={"u": 0, "v": 0},
        initial_history={"r": (("u", 0), ("v", 0))},
        agg=sum_payloads,
        update=ordered_update,
        read=proposal_read,
        select=counted_top_one,
        full=full,
    )


def gated_reset_spec(observe_all: bool = True) -> Spec:
    """A soft gate and an activation reset with distinct cmp/next states."""

    def update(
        _node: str, old: NodeState, _theta: int, content: Content
    ) -> NodeState:
        return old + content

    def select(
        region: str,
        history: HistoryState,
        theta: int,
        descriptors: Mapping[str, Descriptor],
    ) -> Tuple[Tuple[str, ...], Mapping[str, Control], HistoryState]:
        active, _unit_controls, new_history = counted_top_one(
            region, history, theta, descriptors
        )
        controls = {
            node: Fraction(1, 2) if node in active else Fraction(0)
            for node in descriptors
        }
        return active, controls, new_history

    def next_state(
        _node: str,
        _old: NodeState,
        comparison: NodeState,
        _theta: int,
        _content: Content,
        active: bool,
        control: Control,
    ) -> NodeState:
        # The reset reads a known selection control, never a Full result.
        return 0 if active and control > 0 else comparison

    def full(
        node: str,
        comparison: NodeState,
        theta: int,
        _content: Content,
        control: Control,
    ) -> Tuple[Mapping[str, Value], Mapping[str, Value]]:
        output = control * comparison
        outgoing = {"uv": output} if node == "u" and theta == 0 else {}
        return outgoing, {"out_" + node: output}

    return Spec(
        nodes=("u", "v"),
        edges=(Edge("uv", "u", "v", 2),),
        input_target={"in_u": "u", "in_v": "v"},
        output_source={"out_u": "u", "out_v": "v"},
        region_nodes={"r": ("u", "v")},
        budget={"r": 1},
        observe_all={"r": observe_all},
        initial_state={"u": 0, "v": 0},
        initial_history={"r": (("u", 0), ("v", 0))},
        agg=sum_payloads,
        update=update,
        read=proposal_read,
        select=select,
        full=full,
        next_state=next_state,
    )


def random_cuts(stop: int, rng: random.Random) -> Tuple[int, ...]:
    interior = [theta for theta in range(1, stop) if rng.choice((False, True))]
    return tuple([0, *interior, stop])


def assert_composition(
    spec: Spec,
    external: Sequence[External],
    stop: int,
    trials: int,
    seed: int,
) -> None:
    direct, direct_continuation = run_partitioned(spec, external, (0, stop))
    rng = random.Random(seed)
    for _ in range(trials):
        cuts = random_cuts(stop, rng)
        split, split_continuation = run_partitioned(spec, external, cuts)
        if split != direct:
            raise AssertionError(f"trace differs for cuts {cuts}")
        if split_continuation != direct_continuation:
            raise AssertionError(f"continuation differs for cuts {cuts}")


Event = Tuple[str, str, int]


def event_dependencies(
    spec: Spec,
    trace: SegmentTrace,
) -> Dict[Event, Set[Event]]:
    """Build the finite-cut P/S/U/F dependency DAG from an oracle trace."""

    dependencies: Dict[Event, Set[Event]] = {}
    region_of = spec.region_of()

    for (node, theta), bucket in trace.buckets.items():
        if not bucket:
            continue
        prepare = ("P", node, theta)
        adopt = ("U", node, theta)
        dependencies.setdefault(prepare, set())
        dependencies.setdefault(adopt, set())
        region = region_of[node]
        select = ("S", region, theta)
        dependencies.setdefault(select, set()).add(prepare)
        dependencies[adopt].add(select)
        if node in trace.active[(region, theta)]:
            full = ("F", node, theta)
            dependencies.setdefault(full, set()).add(adopt)
            dependencies[full].add(select)

    for node in spec.nodes:
        times = sorted(
            theta
            for candidate, theta in trace.buckets
            if candidate == node and trace.buckets[(candidate, theta)]
        )
        for earlier, later in zip(times, times[1:]):
            dependencies[("P", node, later)].add(("U", node, earlier))

    for region in spec.region_nodes:
        times = sorted(
            theta
            for candidate, theta in trace.candidates
            if candidate == region and trace.candidates[(candidate, theta)]
        )
        for earlier, later in zip(times, times[1:]):
            dependencies[("S", region, later)].add(("S", region, earlier))

    edge_by_name = spec.edge_map()
    for message in trace.messages:
        edge = edge_by_name[message.edge]
        arrival = message_time(spec, message)
        if arrival >= trace.stop:
            continue
        dependencies[("P", edge.target, arrival)].add(
            ("F", edge.source, message.send)
        )
    return dependencies


def run_random_ready_schedule(
    spec: Spec,
    external: Sequence[External],
    oracle: SegmentTrace,
    seed: int,
) -> Tuple[SegmentTrace, Continuation]:
    """Re-evaluate a sealed cut in a random ready-event topological order."""

    if oracle.start != 0:
        raise ValueError("the randomized witness currently starts at cut zero")
    dependencies = event_dependencies(spec, oracle)
    remaining = set(dependencies)
    completed: Set[Event] = set()
    rng = random.Random(seed)
    node_state = copy.deepcopy(spec.initial_state)
    selector_history = copy.deepcopy(spec.initial_history)
    content: Dict[Tuple[str, int], Content] = {}
    proposal: Dict[Tuple[str, int], NodeState] = {}
    descriptor: Dict[Tuple[str, int], Descriptor] = {}
    old_state: Dict[Tuple[str, int], NodeState] = {}
    committed_state: Dict[Tuple[str, int], NodeState] = {}
    updated_history: Dict[Tuple[str, int], HistoryState] = {}
    produced_messages = []
    produced_outputs = []
    result = SegmentTrace(start=0, stop=oracle.stop)
    for node in spec.nodes:
        result.states[(node, 0)] = copy.deepcopy(node_state[node])
    for region in spec.region_nodes:
        result.histories[(region, 0)] = copy.deepcopy(
            selector_history[region]
        )

    while remaining:
        ready = sorted(
            event
            for event in remaining
            if dependencies[event].issubset(completed)
        )
        if not ready:
            raise AssertionError("finite-cut dependency graph contains a cycle")
        event = rng.choice(ready)
        kind, owner, theta = event

        if kind == "P":
            node = owner
            atoms = [
                record
                for record in external
                if record.time == theta
                and spec.input_target[record.port] == node
            ]
            atoms.extend(
                message
                for message in produced_messages
                if message_time(spec, message) == theta
                and message_target(spec, message) == node
            )
            bucket = tuple(sorted(atoms, key=atom_key))
            if bucket != oracle.buckets[(node, theta)]:
                raise AssertionError("random schedule constructed a wrong fiber")
            result.buckets[(node, theta)] = bucket
            old_state[(node, theta)] = copy.deepcopy(node_state[node])
            content[(node, theta)] = spec.agg(node, theta, bucket)
            proposal[(node, theta)] = spec.update(
                node, node_state[node], theta, content[(node, theta)]
            )
            descriptor[(node, theta)] = spec.read(
                node,
                node_state[node],
                proposal[(node, theta)],
                theta,
                content[(node, theta)],
            )
            result.contents[(node, theta)] = copy.deepcopy(content[(node, theta)])
            result.proposals[(node, theta)] = copy.deepcopy(proposal[(node, theta)])
            result.descriptors[(node, theta)] = copy.deepcopy(
                descriptor[(node, theta)]
            )

        elif kind == "S":
            region = owner
            candidates = tuple(
                node
                for node in spec.region_nodes[region]
                if (node, theta) in descriptor
            )
            if candidates != oracle.candidates[(region, theta)]:
                raise AssertionError("random schedule constructed wrong candidates")
            result.candidates[(region, theta)] = candidates
            active, controls, new_history = spec.select(
                region,
                selector_history[region],
                theta,
                {
                    node: descriptor[(node, theta)]
                    for node in candidates
                },
            )
            active = tuple(active)
            if len(set(active)) != len(active):
                raise ValueError("selector returned a duplicate active node")
            if not set(active).issubset(candidates):
                raise ValueError("selector returned a non-candidate")
            if len(active) > spec.budget[region]:
                raise ValueError("selector exceeded the region budget")
            if set(controls) != set(candidates):
                raise ValueError("selector controls must match all candidates")
            for node in candidates:
                result.controls[(node, theta)] = copy.deepcopy(controls[node])
            if active != oracle.active[(region, theta)]:
                raise AssertionError("random schedule changed an active set")
            result.active[(region, theta)] = active
            selector_history[region] = copy.deepcopy(new_history)
            updated_history[(region, theta)] = copy.deepcopy(new_history)

        elif kind == "U":
            node = owner
            region = spec.region_of()[node]
            active = result.active[(region, theta)]
            comparison = (
                proposal[(node, theta)]
                if spec.observe_all[region] or node in active
                else old_state[(node, theta)]
            )
            result.comparison_states[(node, theta)] = copy.deepcopy(comparison)
            node_state[node] = copy.deepcopy(spec.next_state(
                node,
                old_state[(node, theta)],
                comparison,
                theta,
                content[(node, theta)],
                node in active,
                result.controls[(node, theta)],
            ))
            committed_state[(node, theta)] = copy.deepcopy(node_state[node])

        elif kind == "F":
            node = owner
            edge_values, output_values = spec.full(
                node,
                result.comparison_states[(node, theta)],
                theta,
                content[(node, theta)],
                result.controls[(node, theta)],
            )
            result.full_values[(node, theta)] = (
                dict(edge_values), dict(output_values)
            )
            produced_messages.extend(
                Message(send=theta, edge=edge, value=value)
                for edge, value in edge_values.items()
            )
            produced_outputs.extend(
                Output(time=theta, port=port, value=value)
                for port, value in output_values.items()
            )
        else:
            raise AssertionError(f"unknown event kind: {kind}")

        remaining.remove(event)
        completed.add(event)

    # Reconstruct every semantic coordinate from the events actually run.
    # Copying idle coordinates from the oracle would make their comparison
    # tautological and could hide a broken persistence rule.
    for theta in range(oracle.stop):
        for node in spec.nodes:
            atoms = [
                record
                for record in external
                if record.time == theta
                and spec.input_target[record.port] == node
            ]
            atoms.extend(
                message
                for message in produced_messages
                if message_time(spec, message) == theta
                and message_target(spec, message) == node
            )
            bucket = tuple(sorted(atoms, key=atom_key))
            key = (node, theta)
            if key in result.buckets and result.buckets[key] != bucket:
                raise AssertionError("random schedule changed a computed fiber")
            if key not in result.buckets and bucket:
                raise AssertionError("oracle event set omitted a nonempty fiber")
            result.buckets[key] = bucket

        for region, region_nodes in spec.region_nodes.items():
            candidates = tuple(
                node for node in region_nodes if result.buckets[(node, theta)]
            )
            key = (region, theta)
            if key in result.candidates:
                if result.candidates[key] != candidates:
                    raise AssertionError("random schedule changed candidates")
            elif candidates:
                raise AssertionError("oracle event set omitted a selector event")
            else:
                result.candidates[key] = ()

            if key not in result.active:
                if candidates:
                    raise AssertionError("a selector event produced no active set")
                result.active[key] = ()

    for node in spec.nodes:
        state = copy.deepcopy(spec.initial_state[node])
        result.states[(node, 0)] = copy.deepcopy(state)
        for theta in range(oracle.stop):
            state = copy.deepcopy(committed_state.get((node, theta), state))
            result.states[(node, theta + 1)] = copy.deepcopy(state)

    for region in spec.region_nodes:
        history = copy.deepcopy(spec.initial_history[region])
        result.histories[(region, 0)] = copy.deepcopy(history)
        for theta in range(oracle.stop):
            history = copy.deepcopy(
                updated_history.get((region, theta), history)
            )
            result.histories[(region, theta + 1)] = copy.deepcopy(history)
    result.messages = tuple(sorted(produced_messages))
    result.outputs = tuple(sorted(produced_outputs))

    pending = tuple(
        sorted(
            (
                message
                for message in produced_messages
                if message_time(spec, message) >= oracle.stop
            ),
            key=lambda message: (
                message_time(spec, message),
                message.edge,
                message.send,
            ),
        )
    )
    continuation = Continuation(
        cut=oracle.stop,
        node_state=dict(node_state),
        selector_history=dict(selector_history),
        pending=pending,
    )
    return result, continuation


def assert_random_ready_schedules(
    spec: Spec,
    external: Sequence[External],
    stop: int,
    trials: int,
    seed: int,
) -> None:
    oracle, oracle_continuation = run_partitioned(spec, external, (0, stop))
    for trial in range(trials):
        trace, continuation = run_random_ready_schedule(
            spec, external, oracle, seed + trial
        )
        if trace != oracle:
            raise AssertionError(
                f"ready-event trace projection differs at trial {trial}"
            )
        if continuation != oracle_continuation:
            raise AssertionError(
                f"ready-event continuation differs at trial {trial}"
            )


def check_crossing_message_is_necessary() -> None:
    spec = self_loop_spec(delay=2)
    external = (External("in", 0, 0, 7),)
    first, continuation = run_window(spec, initial_continuation(spec), external, 1)
    if len(continuation.pending) != 1:
        raise AssertionError("delay-2 self-loop should cross cut 1")
    if message_time(spec, continuation.pending[0]) != 2:
        raise AssertionError("crossing message has the wrong arrival time")

    correct, _ = run_window(spec, continuation, (), 3)
    broken_continuation = copy.deepcopy(continuation)
    broken_continuation.pending = ()
    broken, _ = run_window(spec, broken_continuation, (), 3)
    if correct.buckets[("v", 2)] == broken.buckets[("v", 2)]:
        raise AssertionError("dropping W_b should change the time-2 fiber")
    if not first.messages:
        raise AssertionError("the first segment should record the sent message")


def check_zero_delay_is_rejected() -> None:
    try:
        self_loop_spec(delay=0).validate()
    except ValueError:
        return
    raise AssertionError("a zero-delay edge entered the positive-delay profile")


def check_invalid_input_history_is_rejected() -> None:
    spec = self_loop_spec(delay=1)
    nonmonotone = (
        External("in", 0, 2, 1),
        External("in", 1, 1, 1),
    )
    try:
        run_partitioned(spec, nonmonotone, (0, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("a nonmonotone input history was accepted")

    skipped_position = (
        External("in", 0, 0, 1),
        External("in", 2, 2, 1),
    )
    try:
        run_partitioned(spec, skipped_position, (0, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("a noncontiguous input history was accepted")

    missing_prefix = (External("in", 2, 2, 1),)
    try:
        run_partitioned(spec, missing_prefix, (0, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("an input history missing its prefix was accepted")

    extra_history = dict(spec.initial_history)
    extra_history["ghost"] = (("v", 99),)
    try:
        replace(spec, initial_history=extra_history).validate()
    except ValueError:
        return
    raise AssertionError("an unknown selector-history owner was accepted")


def check_ready_schedule_reconstructs_idle_coordinates() -> None:
    spec = self_loop_spec(delay=2)
    external = (External("in", 0, 0, 7),)
    oracle, _ = run_partitioned(spec, external, (0, 6))
    corrupted = copy.deepcopy(oracle)
    corrupted.states[("v", 2)] += 1
    replayed, _ = run_random_ready_schedule(spec, external, corrupted, seed=71)
    if replayed == corrupted:
        raise AssertionError("ready-event replay copied an idle oracle state")


def check_empty_window_is_identity() -> None:
    spec = self_loop_spec(delay=2)
    external = (External("in", 0, 0, 7),)
    _, continuation = run_window(
        spec, initial_continuation(spec), external, 1
    )
    if not continuation.pending:
        raise AssertionError("identity test needs a crossing message")

    empty, unchanged = run_window(spec, continuation, (), continuation.cut)
    if unchanged != continuation:
        raise AssertionError("an empty window changed its continuation")
    if (
        empty.buckets
        or empty.candidates
        or empty.active
        or empty.contents
        or empty.proposals
        or empty.descriptors
        or empty.controls
        or empty.comparison_states
        or empty.full_values
        or empty.messages
        or empty.outputs
    ):
        raise AssertionError("an empty window produced an event artifact")

    direct, direct_continuation = run_partitioned(spec, external, (0, 4))
    split, split_continuation = run_partitioned(
        spec, external, (0, 1, 1, 4)
    )
    if split != direct or split_continuation != direct_continuation:
        raise AssertionError("an empty cut changed trace composition")


def check_controls_and_state_snapshots(trials: int) -> None:
    external = (
        External("in_u", 0, 0, 8),
        External("in_v", 0, 0, 3),
        External("in_u", 1, 2, 2),
        External("in_v", 1, 2, 1),
        External("in_u", 2, 4, 5),
        External("in_v", 2, 4, 1),
    )
    spec = gated_reset_spec()
    trace, _ = run_partitioned(spec, external, (0, 6))
    expected_outputs = (
        Output(0, "out_u", Fraction(4)),
        Output(2, "out_v", Fraction(4)),
        Output(4, "out_u", Fraction(7, 2)),
    )
    if trace.outputs != expected_outputs:
        raise AssertionError("Full did not read its gated pre-reset snapshot")
    if trace.comparison_states[("u", 0)] != 8 or trace.states[("u", 1)] != 0:
        raise AssertionError("comparison snapshot and persistent reset were mixed")
    if trace.states[("v", 1)] != 3 or trace.comparison_states[("u", 4)] != 7:
        raise AssertionError("all-candidate adoption lost an inactive candidate's state")
    if trace.controls[("u", 0)] != Fraction(1, 2):
        raise AssertionError("the soft gate was not recorded")
    if trace.controls[("v", 0)] != 0 or ("v", 0) in trace.full_values:
        raise AssertionError("an inactive candidate lost its control or ran Full")

    # Empty fibers have no preparation, selection controls, Next, or Full.
    for theta in (1, 3, 5):
        if trace.candidates[("r", theta)] or trace.active[("r", theta)]:
            raise AssertionError("an empty region produced candidates or activity")
        if trace.histories[("r", theta)] != trace.histories[("r", theta + 1)]:
            raise AssertionError("an empty region changed selector-history")
        for node in spec.nodes:
            if trace.states[(node, theta)] != trace.states[(node, theta + 1)]:
                raise AssertionError("an empty fiber changed persistent state")
            if (
                (node, theta) in trace.controls
                or (node, theta) in trace.comparison_states
            ):
                raise AssertionError("an empty fiber acquired a control or snapshot")

    _, cut_one = run_partitioned(spec, external[:2], (0, 1))
    if cut_one.pending != (Message(0, "uv", Fraction(4)),):
        raise AssertionError("the soft-gated message did not survive cut 1")
    if cut_one.node_state != {"u": 0, "v": 3}:
        raise AssertionError("continuation saved comparison rather than next state")

    default_next = replace(spec, next_state=keep_comparison_state)
    old_behavior, _ = run_partitioned(default_next, external[:2], (0, 1))
    if (
        old_behavior.states[("u", 1)] != 8
        or old_behavior.states[("v", 1)] != 3
    ):
        raise AssertionError("default Next changed all-candidate adoption")

    for observe_all in (True, False):
        variant = gated_reset_spec(observe_all)
        assert_composition(variant, external, 6, trials, seed=901 + observe_all)
        assert_random_ready_schedules(
            variant, external, 6, trials, seed=1301 + observe_all
        )
    selected_only_trace, _ = run_partitioned(gated_reset_spec(False), external, (0, 6))
    if (
        selected_only_trace.comparison_states[("v", 0)] != 0
        or selected_only_trace.states[("v", 1)] != 0
    ):
        raise AssertionError("selected-only adoption changed an inactive candidate's state")


def run_checks(trials: int) -> None:
    self_external = (
        External("in", 0, 0, 3),
        External("in", 1, 5, 2),
    )
    assert_composition(
        self_loop_spec(delay=1),
        self_external,
        stop=12,
        trials=trials,
        seed=17,
    )
    assert_random_ready_schedules(
        self_loop_spec(delay=1),
        self_external,
        stop=12,
        trials=trials,
        seed=101,
    )

    ring_external = (
        External("in_u", 0, 0, 4),
        External("in_v", 0, 0, 5),
        External("in_u", 1, 7, 2),
    )
    assert_composition(
        two_node_ring_spec(),
        ring_external,
        stop=16,
        trials=trials,
        seed=29,
    )
    assert_random_ready_schedules(
        two_node_ring_spec(),
        ring_external,
        stop=16,
        trials=trials,
        seed=503,
    )
    check_crossing_message_is_necessary()
    check_zero_delay_is_rejected()
    check_invalid_input_history_is_rejected()
    check_ready_schedule_reconstructs_idle_coordinates()
    check_empty_window_is_identity()
    check_controls_and_state_snapshots(trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="random cut partitions and ready schedules per semantic example",
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    run_checks(args.trials)
    print(
        "PASS: positive-delay cycles preserve recorded trace projections "
        "and continuations "
        f"across {4 * args.trials} random cut partitions and "
        f"{4 * args.trials} random ready-event schedules; "
        "soft controls, pre-reset snapshots, both adoption modes, and empty fibers agree"
    )


if __name__ == "__main__":
    main()
