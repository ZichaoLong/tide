"""Actual core-5 general-port material and identity-preserving graph views."""
from copy import deepcopy
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.planning.hierarchy import (execution_view, input_origins, non_aligned,
                                         with_feedback, general_ports_plan)
from tidegraph.planning.materialize import materialize
from tidegraph.planning.model import resolve_model, workload, fingerprint
from tidegraph.planning.workload import validate_plan
from tidegraph.reference import run as reference
from tidegraph.greedy import run as greedy
from tidegraph.settle import SettleGraph
from tidegraph.settle_ports import run_ports


def resolved_from(plan, graph):
    return resolve_model(graph, plan["memories"], plan["regions"],
                         source_binding=plan["source_binding"],
                         source_hidden_width=plan["source_hidden_width"],
                         alpha_agg=plan["alpha_agg"])[0]


def test_views_keep_model_workload_physical_edges_and_original_input_clock():
    plan, _ = general_ports_plan(width=4)
    assert plan["schema"] == "tide-weighted-workload-v3"
    before = deepcopy(plan)
    views = [execution_view(plan, name) for name in ("settle", "timed-dag", "pdg")]
    assert plan == before
    assert len({v["view_id"] for v in views}) == 3
    assert {v["model_id"] for v in views} == {plan["model_id"]}
    assert {v["workload_id"] for v in views} == {plan["sha256"]}
    assert all(v["input_clock"]["origins"] == [1, 1, 3, 3] for v in views)
    assert len({v["physical_graph_id"] for v in views}) == 1
    assert plan["graph"]["edges"][0] == plan["graph"]["edges"][1]
    bound, xs, _ = materialize(plan, max_bytes=1000000)
    assert bound.graph.inputs == (0, 1, 2, 2) and bound.graph.outputs == (0, 2, 2)
    assert all(x.time % 5 == views[0]["input_clock"]["origins"][x.port] for x in xs)
    assert len({sum(x.batch == b for x in xs) for b in range(2)}) == 2


def test_explicit_clocks_cannot_be_silently_read_as_legacy_workload():
    plan, _ = general_ports_plan(width=4)
    plan["schema"] = "tide-weighted-workload-v2"
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="schema v3"):
        validate_plan(plan)


def test_delay_and_feedback_are_new_models_with_separate_legal_views():
    plan, _ = general_ports_plan(width=4)
    graph = deepcopy(plan["graph"])
    delayed = non_aligned(graph, edge_index=1, extra_delay=1)
    feedback = with_feedback(graph, source=2, target=0, delay=1)
    models = [resolved_from(plan, g) for g in (graph, delayed, feedback)]
    assert len({m["model_id"] for m in models}) == 3
    assert graph == plan["graph"]
    p, _ = workload(models[1], positions=2, input_origins=[1, 1, 3, 3], stride=5)
    assert execution_view(p, "timed-dag")["family"] == "timed-dag"
    assert input_origins(p) == (1, 1, 3, 3)
    with pytest.raises(ValueError, match="aligned"):
        execution_view(p, "settle")
    q, _ = workload(models[2], positions=2)
    assert execution_view(q, "pdg")["family"] == "pdg"
    with pytest.raises(ValueError, match="cycle"):
        execution_view(q, "timed-dag")
    assert feedback["edges"][-1] == [2, 0, 1]


@pytest.mark.parametrize("origins", [[1], [True, 1, 3, 3], [1, 1, 3, 5], [-1, 1, 3, 3]])
def test_invalid_explicit_port_clocks_fail_static_validation(origins):
    plan, _ = general_ports_plan(width=4)
    plan["input_origins"] = origins
    plan["sha256"] = fingerprint({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="input_origins"):
        validate_plan(plan)


def test_core5_general_ports_independent_schedule_roots_and_pending_continuation():
    plan, _ = general_ports_plan(width=4)
    a, ax, stop = materialize(plan, max_bytes=1000000, dtype=torch.float64)
    b, bx, _ = materialize(plan, max_bytes=1000000, dtype=torch.float64)
    spec = SettleGraph(b.graph, (1, 1, 3), binding="ports", time_stride=5)
    aq, bq = Continuation(a.graph.identity, 2), Continuation(b.graph.identity, 2)
    for cut in (2, stop):
        ar = reference(a.graph, a.model, aq, [x for x in ax if aq.cut <= x.time < cut],
                       cut, sealed_until=cut)
        br = run_ports(spec, b.model, bq, [x for x in bx if bq.cut <= x.time < cut],
                       cut, sealed_until=cut)
        equivalent(ar, br)
        left = {k: v for k, v in a.model.named_parameters() if v.requires_grad}
        right = {k: v for k, v in b.model.named_parameters() if v.requires_grad}
        for root in ("all", "state", "pending" if cut == 2 else "output"):
            ag = torch.autograd.grad(objective(ar, root), tuple(left.values()), allow_unused=True, retain_graph=True)
            bg = torch.autograd.grad(objective(br, root), tuple(right.values()), allow_unused=True, retain_graph=True)
            equivalent(dict(zip(left, ag)), dict(zip(right, bg)))
        if cut == 2:
            assert ar.continuation.pending and ar.outputs
        aq, bq = ar.continuation, br.continuation
    # Equal-rank regions and the same graph's TimedDAG/PDG inclusion have an
    # independent generic online candidate, without editing any model clocks.
    c, cx, _ = materialize(plan, max_bytes=1000000, dtype=torch.float64)
    whole = greedy(c.graph, c.model, Continuation(c.graph.identity, 2), cx,
                   stop, sealed_until=stop)
    expected = reference(a.graph, a.model, Continuation(a.graph.identity, 2), ax,
                         stop, sealed_until=stop)
    equivalent(whole, expected)
    assert {e["node"] for e in whole.trace if e["time"] == 1} >= {0, 1}
