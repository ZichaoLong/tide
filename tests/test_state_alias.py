from __future__ import annotations

import dataclasses
import unittest

import torch

from tide.builders import build_single_layer
from tide.engine import ExecutionContractError, SettleGraph, StateStore


def _two_receiver_ema_plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=2)
    nodes = tuple(
        dataclasses.replace(
            node,
            state_shape=(2,),
            state_owner=node.node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 2,
                "decay": 0.5,
                "state_shape": [2],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0], profile="BO", selector_timing="content"
    )
    return dataclasses.replace(
        plan, nodes=nodes, regions=(region,)
    ).validate()


class StateStorageOwnershipTests(unittest.TestCase):
    def test_distinct_views_of_one_storage_cannot_back_different_receivers(self):
        plan = _two_receiver_ema_plan()
        model = SettleGraph(plan).double()
        backing = torch.zeros(4, dtype=torch.float64)
        state = StateStore(
            values={
                ("s", plan.nodes[0].node_id): backing[:2],
                ("s", plan.nodes[1].node_id): backing[2:],
            },
            next_position={"s": 1},
        )

        with self.assertRaisesRegex(
            ExecutionContractError, "mutable state storage is shared"
        ):
            model.interpret_token(
                torch.zeros((1, 2), dtype=torch.float64),
                torch.tensor([True]),
                ["s"],
                torch.tensor([1]),
                state=state,
            )

    def test_overlapping_external_buffer_storages_are_also_rejected(self):
        plan = _two_receiver_ema_plan()
        model = SettleGraph(plan).double()
        backing = bytearray(24)
        first = torch.frombuffer(backing, dtype=torch.float64, count=2, offset=0)
        second = torch.frombuffer(
            backing, dtype=torch.float64, count=2, offset=8
        )
        self.assertNotEqual(
            first.untyped_storage()._cdata, second.untyped_storage()._cdata
        )
        state = StateStore(
            values={
                ("s", plan.nodes[0].node_id): first,
                ("s", plan.nodes[1].node_id): second,
            },
            next_position={"s": 1},
        )

        with self.assertRaisesRegex(
            ExecutionContractError, "mutable state storage is shared"
        ):
            model.interpret_token(
                torch.zeros((1, 2), dtype=torch.float64),
                torch.tensor([True]),
                ["s"],
                torch.tensor([1]),
                state=state,
            )


if __name__ == "__main__":
    unittest.main()
