"""Verify the fixed main menu without importing Torch or allocating a model.

Run with Python >=3.10. JSON distinguishes configuration expression from
execution, numerical and performance qualification. It is not a training gate.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def catalog(root=ROOT):
    root = Path(root)
    _package("_catalog_tide", root/"src/tide")
    topology = _module("_catalog_tide.topology", root/"src/tide/topology.py")
    config = _module("_catalog_tide.config", root/"src/tide/config.py")
    _package("_catalog_math", root/"math/reference/backbones")
    capacity = _module("_catalog_math.capacity", root/"math/reference/backbones/capacity.py")
    profiles, geometry = [], []
    parameters = allocations = budget_checks = 0
    families = Counter()
    for name, profile in capacity.BOUND_PROFILES.items():
        families[profile["variant"]] += 1
        profiles.append(dict(name=name, spatial_profile=profile["spatial_profile"],
                             family=profile["family"], timing=profile["timing"],
                             widths=profile["widths"], input_period=profile["input_period"]))
        for n in capacity.NODE_SIZES:
            for hubs in ((0,) if profile["family"] in ("fc", "lattice") else (0, 4)):
                arguments = dict(kind=profile["family"], n=n, hubs=hubs, timing=profile["timing"])
                if profile["family"] == "multiscale":
                    arguments["profile"] = profile["spatial_profile"].removeprefix("multiscale-")
                elif profile["family"] == "exchange":
                    arguments.update(depth=profile["expansion"], rounds=profile["rounds"])
                else:
                    arguments["layers"] = profile["depth"]
                requested = topology.TopologyConfig(**arguments)
                resolved, widths = topology._resolve(requested)
                if tuple(profile["widths"]) != widths or resolved.period != profile["input_period"]:
                    raise AssertionError(f"width/period mismatch: {name}/{n}/{hubs}")
                if profile["family"] == "multiscale" and hubs:
                    if resolved.probes != tuple(capacity.ms.probes_for(n)):
                        raise AssertionError(f"probe mismatch: {name}/{n}/{hubs}")
                geometry.append(asdict(resolved))
                for d in capacity.DIMENSIONS:
                    for memory in (("K", "S") if not hubs else ("K", "S", "X")):
                        for sharing, rank in capacity.SHARING:
                            for aggregation in capacity.AGG_PROFILES:
                                parameters += 1
                                for allocation in capacity.ALLOCATION_PROFILES:
                                    config.ModelConfig(topology=requested, d=d, memory=memory, sharing=sharing,
                                                       adapter_rank=8 if sharing == "independent" else rank,
                                                       aggregation=aggregation, allocation=allocation)
                                    allocations += 1
                for budget in capacity.BUDGETS:
                    if budget <= n:
                        for allocation in ("quad", "shannon"):
                            config.ModelConfig(topology=requested, budget=budget, allocation=allocation)
                            budget_checks += 1
    counts = dict(spatial_profiles=len(capacity.PROFILES), bound_profiles=len(profiles),
                  geometry_resolutions=len(geometry), parameter_configs=parameters, allocation_rows=allocations)
    expected = dict(spatial_profiles=41, bound_profiles=55, geometry_resolutions=475,
                    parameter_configs=225400, allocation_rows=676200)
    if counts != expected:
        raise AssertionError(f"fixed catalog counts changed: {counts}")
    sources = [*sorted((root/"src/tide").glob("*.py")), root/"scripts/support_catalog.py"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(
        schema="tide-support-catalog-1", mathematics="4f5b9f20a4030d68ee66ae2d26e34659a5e37234",
        implementation=dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                            source_sha256=identity, files=hashes,
                            source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--",
                                                                       "src/tide", "scripts/support_catalog.py"], cwd=root, text=True))),
        main_menu=dict(**counts, by_variant=dict(families), n=list(capacity.NODE_SIZES), d=list(capacity.DIMENSIONS),
                       sharing=list(capacity.SHARING), reception=list(capacity.AGG_PROFILES),
                       allocation=list(capacity.ALLOCATION_PROFILES), budgets=list(capacity.BUDGETS),
                       profiles=profiles, independent_rank="catalog rank=0 is inactive; API rank=8 is ignored",
                       no_hub_memory="X equals K; the fixed catalog deduplicates X"),
        optional_bindings=[
            dict(name="group_hub_fixed_pass", implemented=True, catalog_member=False,
                 interface="ModelConfig.hub_allocation=fixed_pass", restriction="HB group with hubs; omit H scoring parameters"),
            dict(name="explicit_local_clocks", implemented=True, catalog_member=False,
                 interface="ClockConfig and edge_delay_pair", restriction="strict pair requires all local clocks explicit and common input period > depth"),
            dict(name="terminal_hubs", implemented=True, catalog_member=False,
                 interface="TopologyConfig.terminal_hubs", restriction="Exchange/Multiscale; existing port binding still returns D only"),
            dict(name="reverse_memory", implemented=True, catalog_member=False,
                 interface="ModelConfig.memory=X_reverse", restriction="D uses SSD, H uses KV; explicit instance extension"),
            dict(name="narrow_kv", implemented=True, catalog_member=False,
                 interface="ModelConfig.kv_width", restriction="positive multiple of 32 <= d; total Q/K/V width together, not GQA; graph needs K memory"),
            dict(name="memory_region_sharing", implemented=True, catalog_member=False,
                 interface="ModelConfig.memory_sharing=region_shared", restriction="memory only; FFN and other modules retain sharing; node state private"),
            dict(name="full_ffn_width", implemented=True, catalog_member=False,
                 interface="ModelConfig.ffn_width", restriction="positive multiple of 16; explicit shape change, not equivalent module ratio"),
        ],
        qualification=dict(
            metadata=dict(state="passed", geometry_sha256=hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
                          budget_checks=budget_checks, torch_imported="torch" in sys.modules,
                          scope="configuration constructors and width/period/probe metadata only; no model tensors"),
            execution=dict(state="not_run_by_catalog", checks=["tests/test_optional_bindings.py", "tests/test_models.py"],
                           scope="selected independent witnesses; never the full catalog product"),
            numerical=dict(state="not_run_by_catalog", checks=["tests/test_kv_memory.py", "tests/test_ssd_memory.py", "tests/test_timed_reference.py"],
                           retained_evidence="docs/evidence/roadmap-numerics-followup-20261118.md",
                           scope="existing evidence keeps its recorded source; optional bindings require their own qualification"),
            performance=dict(state="not_run_by_catalog", retained_evidence="docs/evidence/README.md",
                             scope="specific workload/device/precision records only; no inferred optional-binding qualification")),
        excluded=["old teaching bindings", "LH-DAG/PDG", "arbitrary graph/local-function programs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional NEW JSON path; stdout otherwise")
    args = parser.parse_args()
    record = catalog()
    if args.output:
        if args.output.exists():
            parser.error("output already exists; use a new path")
        from scripts.job import atomic
        atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
