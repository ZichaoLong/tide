from __future__ import annotations

import copy
import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from tide.builders import (
    build_mixed_regions,
    build_single_layer,
    build_small_hb,
)
from tide.engine import SettleGraph, StateStore
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    TraceInvariantError,
    validate_trace_invariants,
)
from tide.semantics import (
    DEFAULT_NEXT_FORMULA_ID,
    TRACE_PROJECTION_ID,
    TRACE_SCHEMA_VERSION,
    SemanticContextError,
    logical_schedule_record,
    semantic_context_for_plan,
    validate_repository_semantic_sources,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _stateful_sd_content_plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=1)
    nodes = tuple(
        dataclasses.replace(
            node,
            state_shape=(1,),
            state_owner=node.node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 1,
                "decay": 0.5,
                "state_shape": [1],
            },
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
                "output_shape": [1],
            },
            node_compute={
                "type": "identity",
                "formula_id": "node.identity.v1",
                "output_shape": [1],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0], profile="SD", selector_timing="content"
    )
    return dataclasses.replace(plan, nodes=nodes, regions=(region,)).validate()


def _run_stateful_micro_golden():
    plan = _stateful_sd_content_plan()
    model = SettleGraph(plan).double()
    for node in plan.nodes:
        receiver = model.receiver(node.node_id)
        receiver.ema_observe.weight.data.zero_()
        receiver.ema_observe.bias.data.zero_()
    state = StateStore(
        values={
            ("sequence", "node.0000"): torch.tensor(
                [8.0], dtype=torch.float64
            ),
            ("sequence", "node.0001"): torch.tensor(
                [6.0], dtype=torch.float64
            ),
        },
        next_position={"sequence": 4},
    )
    result = model.prefill(
        torch.tensor([[[2.0]]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
        ["sequence"],
        torch.tensor([[4]], dtype=torch.int64),
        state=state,
        detach_at_end=False,
        record_trace=True,
    )
    assert result.trace is not None
    return plan, result


class SemanticAuthorityTests(unittest.TestCase):
    def test_repository_authority_lock_and_document_bytes_are_bound(self) -> None:
        record = validate_repository_semantic_sources(_REPOSITORY_ROOT)
        self.assertEqual(record["upstream_lock"]["semantic_version"], "tide-core-3")
        self.assertEqual(
            {document["role"] for document in record["documents"]},
            {"upstream-lock", "upstream-adoption", "local-instance-semantics"},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for document in record["documents"]:
                source = _REPOSITORY_ROOT / document["path"]
                target = root / document["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            adoption = root / "docs" / "upstream-semantics.md"
            adoption.write_text(
                adoption.read_text(encoding="utf-8") + "\nsemantic drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SemanticContextError, "semantic authority hash mismatch"
            ):
                validate_repository_semantic_sources(root)

    def test_general_schedule_includes_explicit_control_dependencies(self) -> None:
        base = build_mixed_regions(d_model=2)
        regions = tuple(
            dataclasses.replace(
                region, control_dependencies=("region.optional",)
            )
            if region.region_id == "region.backbone"
            else region
            for region in base.regions
        )
        plan = dataclasses.replace(base, regions=regions).validate()
        schedule = logical_schedule_record(plan)
        ranks = {
            row["region_id"]: row["rank"]
            for row in schedule["region_ranks"]
        }
        self.assertEqual(
            ranks,
            {
                "region.root": 1,
                "region.optional": 2,
                "region.backbone": 3,
                "region.out": 4,
            },
        )
        self.assertEqual(
            plan.topological_region_ids,
            (
                "region.root",
                "region.optional",
                "region.backbone",
                "region.out",
            ),
        )
        self.assertEqual(schedule["output_rank"], 5)
        self.assertEqual(schedule["step_stride"], 6)
        edge_delays = {
            row["edge_id"]: row["delay"] for row in schedule["edge_delays"]
        }
        self.assertEqual(edge_delays["edge.root-backbone"], 2)
        self.assertEqual(edge_delays["edge.a-out"], 2)

    def test_hb_schedule_uses_line_rank_and_positive_edge_delays(self) -> None:
        plan = build_small_hb(d_model=2)
        schedule = logical_schedule_record(plan)
        region_ranks = {
            row["region_id"]: row["rank"]
            for row in schedule["region_ranks"]
        }
        for region in plan.regions:
            self.assertEqual(region_ranks[region.region_id], region.line + 1)
        self.assertTrue(
            all(row["delay"] > 0 for row in schedule["edge_delays"])
        )
        maximum_line = max(region.line for region in plan.regions)
        self.assertEqual(schedule["output_rank"], maximum_line + 2)
        self.assertEqual(schedule["step_stride"], maximum_line + 3)

    def test_sd_lazy_proposal_and_default_next_have_an_explicit_golden(self) -> None:
        plan, result = _run_stateful_micro_golden()
        trace = result.trace
        assert trace is not None
        self.assertEqual(trace.schema_version, TRACE_SCHEMA_VERSION)
        self.assertEqual(
            trace.semantic_context, semantic_context_for_plan(plan)
        )
        self.assertEqual(
            trace.semantic_context["trace_projection_id"], TRACE_PROJECTION_ID
        )
        self.assertEqual(
            {
                row["node_id"]: row["formula_id"]
                for row in trace.semantic_context["next_formula_by_node"]
            },
            {
                "node.0000": DEFAULT_NEXT_FORMULA_ID,
                "node.0001": DEFAULT_NEXT_FORMULA_ID,
            },
        )

        events = {event.node_id: event for event in trace.node_events}
        inactive = events["node.0000"]
        active = events["node.0001"]
        self.assertTrue(inactive.reached)
        self.assertFalse(inactive.observed)
        self.assertFalse(inactive.active)
        self.assertIsNone(inactive.proposal)
        torch.testing.assert_close(
            inactive.state_before, torch.tensor([8.0], dtype=torch.float64)
        )
        torch.testing.assert_close(inactive.state_for_compute, inactive.state_before)
        torch.testing.assert_close(inactive.state_after, inactive.state_before)

        self.assertTrue(active.reached)
        self.assertTrue(active.observed)
        self.assertTrue(active.active)
        torch.testing.assert_close(
            active.state_before, torch.tensor([6.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            active.proposal, torch.tensor([3.0], dtype=torch.float64)
        )
        torch.testing.assert_close(active.state_for_compute, active.proposal)
        torch.testing.assert_close(active.state_after, active.proposal)
        torch.testing.assert_close(
            active.computed, torch.tensor([2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.state.values[("sequence", "node.0000")],
            torch.tensor([8.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.state.values[("sequence", "node.0001")],
            torch.tensor([3.0], dtype=torch.float64),
        )
        self.assertEqual(result.state.next_position, {"sequence": 5})
        self.assertEqual(len(trace.state_writes), 1)
        self.assertEqual(trace.state_writes[0].owner_id, "node.0001")
        self.assertTrue(
            all(
                event.selector_history_before is None
                and event.selector_history_after is None
                for event in trace.region_events
            )
        )

        schedule = trace.semantic_context["logical_schedule"]
        node_rank = schedule["node_ranks"][0]["rank"]
        self.assertEqual(schedule["step_stride"] * 4 + node_rank, 13)
        self.assertEqual(
            schedule["step_stride"] * 4 + schedule["output_rank"], 14
        )
        validate_trace_invariants(
            plan,
            trace,
            (("sequence", 4),),
            tolerance=CPU_FLOAT64_TOLERANCE,
        )

    def test_trace_invariants_reject_next_or_schedule_mutation(self) -> None:
        plan, result = _run_stateful_micro_golden()
        trace = result.trace
        assert trace is not None

        node_events = list(trace.node_events)
        active_index = next(
            index
            for index, event in enumerate(node_events)
            if event.node_id == "node.0001"
        )
        node_events[active_index] = dataclasses.replace(
            node_events[active_index],
            state_after=node_events[active_index].state_after + 1.0,
        )
        with self.assertRaisesRegex(TraceInvariantError, "state_after"):
            validate_trace_invariants(
                plan,
                dataclasses.replace(trace, node_events=tuple(node_events)),
                (("sequence", 4),),
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        context = copy.deepcopy(trace.semantic_context)
        context["logical_schedule"]["output_rank"] += 1
        with self.assertRaisesRegex(
            TraceInvariantError, "semantic_context does not match"
        ):
            validate_trace_invariants(
                plan,
                dataclasses.replace(trace, semantic_context=context),
                (("sequence", 4),),
                tolerance=CPU_FLOAT64_TOLERANCE,
            )


if __name__ == "__main__":
    unittest.main()
