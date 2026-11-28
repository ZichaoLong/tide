"""Create the two-task ModelArts qualification using committed project and foundation sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from tide_portable import pack, sha, write, require


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def freeze_runq(payload, manifest=None):
    """Select the account release (or explicit candidate) at development time."""
    cmd = ["maflow", "admin", "runq", "freeze", "--directory", str(payload), "--json"]
    if manifest is not None:
        cmd.extend(["--manifest", str(manifest.expanduser().resolve(strict=True))])
    selection = json.loads(subprocess.check_output(cmd, text=True))
    path = payload / ".maflow/runq/release.json"
    marker = json.loads(path.read_bytes())
    commit, checksum = marker["commit"], marker["sha256"]
    require(
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) and re.fullmatch(r"[0-9a-f]{64}", checksum),
        "invalid frozen runq identity",
    )
    archive = path.with_name(f"runq-{commit}.tar.gz")
    size = archive.stat().st_size
    require(
        selection.get("commit") == commit and selection.get("sha256") == checksum
        and selection.get("bytes") == size and sha(archive) == checksum,
        "maflow runq release differs from the frozen archive",
    )
    return {"commit": commit, "sha256": checksum, "bytes": size}


def extract(repo, revision, paths, target):
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.SpooledTemporaryFile() as archive:
        subprocess.run(
            ["git", "-C", str(repo), "archive", revision, "--", *paths], stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            for item in tar:
                require(item.isdir() or item.isfile(), "expected plain git members")
                if item.isfile():
                    path = target / item.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(item) as source, path.open("xb") as dest:
                        shutil.copyfileobj(source, dest)


def corpus(root):
    import numpy as np

    root.mkdir(mode=0o700)
    splits = {}
    train_ids = []
    for split, labels in (("train", range(4)), ("validation", range(4, 10))):
        rows = []
        tokens = []
        for i, label in enumerate(labels):
            doc_id = hashlib.sha256(("tide-transfer-v1/" + str(label)).encode()).hexdigest()
            tokens.append(((np.arange(64) * 3 + label) % 12).astype("<u2"))
            rows.append({"document_id": doc_id, "start": 64 * i, "stop": 64 * (i + 1)})
            if split == "train":
                train_ids.append(doc_id)
        np.concatenate(tokens).tofile(root / (split + ".bin"))
        np.save(root / (split + ".npy"), np.asarray([(r["start"], r["stop"]) for r in rows], dtype=np.int64))
        (root / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        splits[split] = {
            "tokens": split + ".bin",
            "documents": split + ".npy",
            "document_ids": split + ".jsonl",
            "token_count": len(rows) * 64,
        }
    (root / "tokenizer-training-document-ids.txt").write_text("\n".join(train_ids) + "\n")
    write(root / "tokenizer.json", {"kind": "qualification integer-token vocabulary", "vocab_size": 12})
    files = {p.name: {"size_bytes": p.stat().st_size, "sha256": sha(p)} for p in root.iterdir()}
    write(
        root / "manifest.json",
        {
            "schema_version": "tide-pilot-data-v1",
            "dataset_id": "tide-location-qualification-v1",
            "vocab_size": 12,
            "token_dtype": "<u2",
            "tokenizer": {"file": "tokenizer.json"},
            "files": files,
            "splits": splits,
        },
    )


def qualification_inputs(originals, repo, base, repos):
    extract(repo, repos[0]["commit"], ["pilotlab"], originals / "experiment")
    extract(base, repos[1]["commit"], ["src/tide"], originals / "foundation")
    sys.path.insert(0, str(originals / "experiment"))
    from pilotlab.data3 import prepare

    corpus(originals / "corpus")
    data = originals / "view"
    prepare(originals / "corpus", data)
    files = {
        str(p): sha(p)
        for directory in (originals / "experiment/pilotlab", originals / "foundation/src/tide")
        for p in sorted(directory.rglob("*.py"))
    }
    identity = {
        "schema": "pilot-r3-source-v1",
        "source_files": files,
        "data_manifest_sha256": sha(data / "manifest.json"),
    }
    bits = {
        "phase": "a1",
        "source_identity": identity,
        "init_recipe": "default",
        "batch_size": 1,
        "sequence_length": 4,
        "stream_chunks": 3,
        "cpu_threads": 1,
        "precision": "fp32",
        "target_tokens": 12,
        "warmup_tokens": 4,
        "schedule_tokens": 64,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "eval_tokens": 4,
        "health_interval_tokens": 4,
        "eval_interval_tokens": 0,
        "closing_reserve_seconds": 15,
        "checkpoint_interval_steps": 1,
        "record_step_fingerprints": True,
        "capture_failure": False,
    }
    configs = [
        (
            "tide-resume",
            {
                "family": "tide",
                "config": {
                    "topology": {"kind": "fc", "n": 2, "layers": 2},
                    "d": 64,
                    "memory": "K",
                    "attention_backend": "math",
                    "allocation": "quad",
                    "budget": 1,
                },
            },
            1,
        ),
        (
            "transformer-peer",
            {"family": "transformer", "d": 16, "heads": 2, "layers": 1, "ffn_width": 32, "window": 32},
            None,
        ),
    ]
    paths = []
    for rank, (key, model, interrupt) in enumerate(configs):
        source = originals / (key + ".json")
        write(source, {**bits, "id": key, "model_id": key, "model": model, "seed": 7141 + rank})
        paths.append(source)
    return paths, data


def generate(a):
    output = a.output.expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir(mode=0o700)
    originals = output / "source-records"
    repo = Path(__file__).resolve().parents[1]
    base = a.foundation.expanduser().resolve()
    repos = []
    adapter_commit = git(repo, "rev-parse", "HEAD")
    for role, path, revision in (
        ("experiment", repo, a.experiment_revision),
        ("foundation", base, a.foundation_revision),
    ):
        branch = git(path, "symbolic-ref", "--short", "HEAD")
        rev = git(path, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}")
        repos.append(
            {"role": role, "url": git(path, "remote", "get-url", "origin"), "branch": branch, "commit": rev}
        )
    qualification = not a.task
    if qualification:
        require(a.data is None, "--data requires at least one original --task")
        paths, data = qualification_inputs(originals, repo, base, repos)
    else:
        require(a.data is not None, "--task requires the original R3 --data directory")
        paths, data = a.task, a.data
    nodes = a.nodes or (1 if a.site == "local" else min(2, len(paths)))
    require(nodes >= 1 and (a.site != "local" or nodes == 1), "local groups use one existing runq slot")
    require(
        a.worker_seconds > 0 and a.run_seconds > a.worker_seconds, "worker budget must fit the group deadline"
    )
    tasks = []
    for rank, path in enumerate(paths):
        task = json.loads(path.read_text())
        request = {"threads": task["cpu_threads"], "ram_mib": a.ram_mib, "backend": a.backend}
        if a.backend == "npu":
            request["hbm_mib"] = a.hbm_mib
        item = {
            "id": task["id"],
            "fingerprint": sha(path),
            "worker": rank % nodes,
            "resources": request,
            "worker_seconds": a.worker_seconds,
            "inspection_seconds": a.inspection_seconds if qualification else 0,
        }
        if qualification and task["id"] == "tide-resume":
            item["interrupt_after_step"] = 1
        tasks.append(item)
    pack(payload / "frozen", paths, data, a.evidence)
    runtime = freeze_runq(payload, a.runq_manifest)
    for name in ("tide_portable.py", "tide_batch.py"):
        (payload / name).write_bytes(
            subprocess.check_output(["git", "-C", str(repo), "show", adapter_commit + ":scripts/" + name])
        )
    write(
        payload / "release.json",
        {
            "repositories": repos,
            "runq_commit": runtime["commit"],
            "runq": runtime,
            "adapter_commit": adapter_commit,
            "run_seconds": a.run_seconds,
            "resume_run_seconds": a.resume_run_seconds or a.run_seconds,
            "frozen_sha256": sha(payload / "frozen/freeze.json"),
            "adapter_sha256": {name: sha(payload / name) for name in ("tide_portable.py", "tide_batch.py")},
        },
    )
    plan = {
        "schema": "maflow-batch/v1",
        "batch": a.batch,
        "project": a.project,
        "repositories": repos,
        "inputs": [
            {
                "role": "frozen-tasks-data-and-gates",
                "path": "frozen/freeze.json",
                "sha256": sha(payload / "frozen/freeze.json"),
            },
            {"role": "runtime-releases", "path": "release.json", "sha256": sha(payload / "release.json")},
        ],
        "groups": [
            {
                "id": "tide-workers",
                "site": a.site,
                "nodes": nodes,
                "resume_nodes": 1,
                "run_seconds": a.run_seconds,
                "cloud_cards_per_node": 1,
                "payload": "payload",
                "command": [
                    a.python_command or (sys.executable if a.site == "local" else "python3"),
                    "-B",
                    "-u",
                    "tide_batch.py",
                    "cluster",
                ],
                "tasks": tasks,
            }
        ],
    }
    if a.site == "local":
        plan["groups"][0]["resources"] = {
            **tasks[0]["resources"],
            "threads": max(t["resources"]["threads"] for t in tasks),
        }
    group = plan["groups"][0]
    if a.authority:
        plan["authority"] = a.authority
    if a.target or a.spec or a.interactive:
        require(a.site == "modelarts", "a cloud target, spec or interactive window needs a ModelArts group")
        if a.target:
            group["target"] = a.target
        if a.spec:
            group["spec"] = a.spec
        if a.interactive:
            group["interactive"] = True
    if a.queue_seconds is not None:
        require(a.queue_seconds > 0, "queue budget must be positive")
        group["queue_seconds"] = a.queue_seconds
    if a.resume_run_seconds is not None:
        require(a.resume_run_seconds > a.worker_seconds, "resume budget must fit a worker")
        group["resume_run_seconds"] = a.resume_run_seconds
    write(output / "plan.json", plan)
    return output / "plan.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--foundation", required=True, type=Path)
    p.add_argument("--experiment-revision", default="HEAD")
    p.add_argument("--foundation-revision", default="HEAD")
    p.add_argument("--runq-manifest", type=Path, help="use a specific maflow runq release.json instead of the installed release")
    p.add_argument("--batch", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    p.add_argument(
        "--task",
        type=Path,
        action="append",
        help="existing frozen R3 task (repeatable); omit for the transfer qualification",
    )
    p.add_argument("--data", type=Path)
    p.add_argument("--evidence", type=Path, action="append", default=[])
    p.add_argument("--site", choices=("local", "modelarts"), default="modelarts")
    p.add_argument("--authority", help="the batch's authoritative OBS alias, or local")
    p.add_argument("--target", help="configured ModelArts compute target")
    p.add_argument("--spec", help="configured ModelArts resource specification")
    p.add_argument("--interactive", action="store_true", help="keep the job available after task receipts until finish or deadline")
    p.add_argument("--queue-seconds", type=int)
    p.add_argument("--resume-run-seconds", type=int)
    p.add_argument("--nodes", type=int)
    p.add_argument("--run-seconds", type=int, default=900)
    p.add_argument("--worker-seconds", type=int, default=420)
    p.add_argument("--ram-mib", type=int, default=8192)
    p.add_argument("--hbm-mib", type=int, default=8192)
    p.add_argument("--python-command")
    p.add_argument("--inspection-seconds", type=int, default=300)
    print(generate(p.parse_args()))
