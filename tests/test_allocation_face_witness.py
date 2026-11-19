import json

import torch

from scripts.check_allocation_face_witness import main


def test_device_witness_uses_independent_vjp_and_preserves_other_rows(tmp_path):
    # Independent regions/samples/times must not receive a selected row's VJP.
    score = torch.full((2, 3, 2, 4), .75)
    score[:, 2] = -2
    payload = dict(score=score, received=torch.ones_like(score), selected_index=1)
    case = dict(coordinate=dict(sample=1, time_index=2, local_node=1),
                roles=[False]*3, budget=1, profile="quad", partition="group", paths=[payload])
    fixture, output = tmp_path/"witness.pt", tmp_path/"result.json"
    torch.save([case], fixture)
    assert main(["--witness", str(fixture), "--output", str(output), "--device", "cpu", "--threads", "1"]) == 0
    record = json.loads(output.read_text())
    row = record["cases"][0]
    assert row["submission_shape"] == [3, 4, 4]
    assert row["score_vjp"] == [-.5, .5, 0.]
    assert row["support_matches"] and row["unchanged_other_rows"]
