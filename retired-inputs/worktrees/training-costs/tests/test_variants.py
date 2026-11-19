"""Finite candidate plans and real independent normal/diagnostic children."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.variants import VARIANTS, build_plan


def test_variants_preserve_mathematical_config_and_separate_ledger_children(tmp_path):
    args = Namespace(config="configs/fc-k.json", variants=list(VARIANTS), ledger_variants=["baseline", "combined"],
                     device="cpu", precision="fp32", batch=1, length=2, threads=4, seed=1107, warmup=2,
                     activity="natural", optimizer_foreach=True, device_budget_gib=None, mode="train", steps=5,
                     ledger_segments=1, child_timeout=30, rss_gib=4)
    plan = build_plan(args, tmp_path)
    assert [row["kind"] for row in plan] == ["normal"]*len(VARIANTS)+["ledger"]*2
    original = json.loads(Path(args.config).read_text())
    for row in plan:
        resolved = json.loads(Path(row["config_file"]["path"]).read_text())
        assert {k: resolved[k] for k in original if k not in VARIANTS["baseline"]} == original
        assert all(resolved[k] == value for k, value in VARIANTS[row["variant"]].items())
        if row["kind"] == "normal":
            command = row["command"]
            assert command[command.index("--steps")+1] == "5"
            assert command[command.index("--subjects")+1] == "tide"


def test_small_normal_variants_use_five_samples_and_a_separate_ledger_process(tmp_path):
    config = tmp_path/"small.json"
    config.write_text(json.dumps(dict(topology=dict(kind="fc", n=2, layers=2), d=64, sharing="region_shared")))
    output = tmp_path/"variants"
    child = subprocess.run([sys.executable, "-m", "benchmarks.variants", "--config", str(config), "--mode", "train",
                            "--precision", "fp32", "--batch", "1", "--length", "2", "--threads", "4", "--warmup", "1",
                            "--ledger-variants", "combined", "--rss-gib", "4", "--seconds", "150", "--child-timeout", "40",
                            "--output-dir", str(output)], text=True, capture_output=True, timeout=160)
    assert child.returncode == 0, child.stderr+child.stdout
    record = json.loads((output/"variants.json").read_text())
    assert record["state"] == "completed"
    assert len(record["children"]) == len({row["pid"] for row in record["children"]}) == 6
    assert len(record["summaries"]) == 5
    assert all(row["initial_parameter_input_equal"] and len(row["seconds"]) == 5 for row in record["summaries"])
    assert all(not row["numerical_equivalence_qualification"] for row in record["summaries"])
    ledger = json.loads(Path(record["children"][-1]["output"]).read_text())
    assert ledger["observations"][0]["actual_work"] is None
