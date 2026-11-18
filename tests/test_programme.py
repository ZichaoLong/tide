"""Finite-driver lifecycle and non-Cartesian plan checks."""
from argparse import Namespace
import json
from pathlib import Path
import sys

from benchmarks.programme import build_plan, run_child


def _arguments(**updates):
    values = dict(configs=["configs/fc-k.json", "configs/fc-s.json"], suite="representative", batch=1,
                  length=2, precision=["fp32", "bf16"], threads=4, warmup=0, steps=1,
                  modes=["inference", "train"], subjects=None, device="cpu", child_timeout=10,
                  continuous_transformers=None, checkpoints=False, prefill_chunk=2, decode=1,
                  activities=["natural", "sparse03", "sparse10", "high"])
    values.update(updates)
    return Namespace(**values)


def test_representative_preserves_modes_and_adds_fc_s_reference(tmp_path):
    plan = build_plan(_arguments(), tmp_path)
    assert len(plan) == 4
    assert all(row["command"][0] == sys.executable for row in plan)
    for row in plan:
        assert ("ssd_parameters" in row["command"]) == ("fc-s" in row["config"])
        assert row["command"].count("--modes") == 1
        assert all("transformer_"+name in row["command"] for name in "ABCD")
    assert len({row["output"] for row in plan}) == len(plan)


def test_shape_changes_one_dimension_and_generated_inputs_are_recorded(tmp_path):
    plan = build_plan(_arguments(configs=["configs/fc-k.json"], suite="shape", precision=["fp32"]), tmp_path)
    assert len(plan) == 4
    shapes = []
    for row in plan:
        args = row["command"]
        config = json.loads(Path(row["config"]).read_text())
        shapes.append((int(args[args.index("--batch")+1]), int(args[args.index("--length")+1]), config["topology"]["n"]))
    assert shapes == [(1, 2, 8), (4, 2, 8), (1, 8, 8), (1, 2, 16)]


def test_zero_process_exit_cannot_hide_inner_failure(tmp_path):
    output = tmp_path/"result.json"
    script = "import json,sys;json.dump({'rows':[{'finite':False}]},open(sys.argv[1],'w'))"
    row = dict(command=[sys.executable, "-c", script, str(output)], log=str(tmp_path/"child.log"),
               output=str(output), timeout_seconds=10)
    run_child(row, lambda: None)
    assert row["exit_code"] == 0
    assert row["state"] == "inner_failed"


def test_child_timeout_records_termination(tmp_path):
    row = dict(command=[sys.executable, "-c", "import time;time.sleep(10)"], log=str(tmp_path/"child.log"),
               output=str(tmp_path/"missing.json"), timeout_seconds=.1)
    run_child(row, lambda: None)
    assert row["state"] == "timed_out"
    assert row["descendants_gone"]
