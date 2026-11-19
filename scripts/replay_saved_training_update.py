"""Replay only original CPU/FP32 update index 2 from its own retained update 1.

No tensor outputs or deletion. Exact replay permits a separate retirement review;
it never changes the original cross-path failure or proves arbitrary RNG recovery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmarks.inference import source_identity
from benchmarks.workloads import tensor_hash
from scripts.check_training_paths import build_candidate, compare, observe, training_data
from scripts.diagnose_saved_training import Inputs
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS
from tide import ModelConfig

# These four executed helpers are byte-identical to original a5853660.
HELPERS = {
    "scripts/check_training_paths.py": "a1c28e84f7b7f1ded10402923daee2f59598540cce9f6fd6db12603c11a45b88",
    "scripts/qualify_precision.py": "cfb0226a7aaafdcf94cb5de737d227863dd77ca3c0543f1ea64ceea46544062d",
    "scripts/check_placement.py": "5942b5eedb4d388219d91015c242826b7ba2426258878c873e5366394a30271d",
    "benchmarks/workloads.py": "d899b06a16ad205ce8100e60e3feff5e47a9c7ba288fae94c266a731fae44f0b"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(directory, verified_path, owner):
    directory, verified_path = Path(directory).resolve(), Path(verified_path).resolve()
    original = json.loads((directory / "result.json").read_text())
    verified = json.loads(verified_path.read_text())
    verified_sha = sha(verified_path)
    require(owner in {"baseline", "weight_access"} and original["device"] == "cpu" and
            original["precision"] == "fp32" and original["parameters"] <= 200_000_000, "unsupported replay scope")
    require(verified["state"] == "diagnosis_complete" and verified["torch"] == torch.__version__ and
            verified["source_record"]["sha256"] == sha(directory / "result.json"), "verified record identity mismatch")
    root = Path(__file__).resolve().parents[1]
    helpers = {name: sha(root / name) for name in HELPERS}
    source = source_identity()
    require(helpers == HELPERS and source["tide_content_sha256"] == original["source"]["tide_content_sha256"],
            "production or executed helper source changed")
    require(original["tolerance"] == THRESHOLDS["fp32"], "original threshold changed")
    names = ["anchor.pt", owner + "/update-1.pt", owner + "/update-2.pt"]
    manifests = {row["path"]: row for row in original["artifacts"]}
    verified_inputs = {row["path"]: row for row in verified["input_files"]}
    def guard():
        for name in names:
            path = directory / name
            row, item = verified_inputs[str(path)], path.stat()
            require(row["content_sha256_verified"] and row["manifest_sha256"] == manifests[name]["sha256"] and
                    (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) ==
                    (row["device"], row["inode"], row["bytes"], row["mtime_ns"]), "verified input stat guard changed: " + name)
    guard()
    inputs = Inputs(directory, original, 12 * 2**30, False, guard)
    anchor, before = inputs.load(names[0]), inputs.load(names[1])
    require(anchor["config"] == original["base_config"] and tensor_hash(anchor["inputs"][3]) == original["input_hashes"][3],
            "original configuration/input index identity mismatch")
    torch.set_num_threads(original["threads"])
    config = ModelConfig.from_dict(anchor["config"])
    model, optimizer, state = build_candidate(config, owner, before, [torch.device("cpu")], "fp32", "auto",
                                             overrides=original["resolved_candidates"][owner])
    start = compare(training_data(model, optimizer, state), {k: before[k] for k in ("parameters", "optimizer", "state")}, original["tolerance"])
    require(start["comparison"]["exact"], "saved training start not exact")
    position = state.position
    # observe clears old .grad; no production random operators execute after construction.
    torch.set_rng_state(anchor["extra"]["rng"]["cpu"])
    rng = torch.get_rng_state().clone()
    updated, following, work = observe(model, optimizer, state, anchor["inputs"][3], "fp32", [torch.device("cpu")],
                                      original["workload"]["chunk"], update=True)
    rng_unchanged = torch.equal(rng, torch.get_rng_state())
    steps = updated["optimizer_update"]
    require(set(steps["before"].values()) == {3} and set(steps["after"].values()) == {4} and steps["checked"] and
            position == 3 * original["workload"]["length"] and following.position == 4 * original["workload"]["length"],
            "expected real steps 3->4 and positions 3L->4L")
    expected = inputs.load(names[2])
    checked = compare(updated, expected, original["tolerance"])
    guard()
    require(sha(verified_path) == verified_sha and sha(directory / "result.json") == verified["source_record"]["sha256"],
            "input provenance records changed during replay")
    return dict(schema="tide-saved-update-replay-v1", state="exact" if checked["comparison"]["exact"] and rng_unchanged else "not_exact",
        owner=owner, index=2, original_state=original["state"], original_source=original["source"], source=source,
        tool_sha256=sha(__file__), helper_sha256=helpers, verified_record_sha256=verified_sha,
        input_files=[verified_inputs[str(directory / name)] for name in names], input_sha_policy="prior real SHA256 verification plus current pre/post stat guards; no new hash claim",
        config=model.config.to_dict(), torch=torch.__version__, threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        original_threads=original["threads"], original_cpu_affinity=original.get("cpu_affinity"),
        start=start, comparison=checked, optimizer_update=steps, positions=[position, following.position], work=work,
        rng_unchanged=rng_unchanged, gradient_policy="observe zero_grad(set_to_none=True) discards prior gradients before real backward",
        rng_policy="update1 has no RNG sidecar; reset retained common-anchor CPU RNG; unchanged-state check covers this no-random-operator path only",
        retirement_policy="external review only after exact replay; original failure manifest and both update1/anchor/minimal witnesses remain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--verified-inputs", type=Path, required=True)
    parser.add_argument("--owner", choices=("baseline", "weight_access"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    record = run(args.qualification_dir, args.verified_inputs, args.owner)
    require(len(json.dumps(record).encode()) < 64 * 2**20, "result exceeds 64 MiB")
    atomic(args.output, record)
    return 0 if record["state"] == "exact" else 2


if __name__ == "__main__":
    raise SystemExit(main())
