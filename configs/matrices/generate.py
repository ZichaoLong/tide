#!/usr/bin/env python3
"""Generate this one finite reviewed plan using installed public planning APIs.

No Torch import, tensors, measurements or trajectory prepass. Generated JSON
contains every case; this generator is not imported by the experiment runtime.
Existing files are refused. Run from the fixed source installation.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
from tidegraph.experiment.records import write_json
from tidegraph.experiment.config import resolve
from tidegraph.planning.hierarchy import general_ports_plan

GIB = 1024**3
SITES = dict(cpu="local-cpu.json", npu="local-npu.json", cuda="gpu3090.json")
LARGE = dict(max_materialize_bytes=32*GIB, checkpoint_max_bytes=16*GIB)


def recipe(topology="fc", *, width=16, depth=2, target=None, family="kv", profile="baseline"):
    result = dict(topology=topology, family=family, depth=depth, region_width=2,
                  profile=profile, history=8, sparse_denominator=8)
    result["width" if target is None else "target_parameters"] = width if target is None else target
    return result


def configuration(platform, name, model, *, batch=2, positions=3, family="auto", mode="inference",
                  client="python-native", schedule="prefill", packing="packed", return_packing="scalar",
                  placement=None, validation="strict", dtype="float32", threads=None, peers=None,
                  communication="ordinary", resources=None, question=""):
    execution = dict(family=family, client=client, device="cpu" if platform == "cpu" else platform+":0",
        placement=placement or ("cpu" if platform == "cpu" else "mixed-c"), dtype=dtype,
        scoring_dtype="float64" if dtype == "float64" else "float32", schedule=schedule, operator="aten",
        packing=packing, return_packing=return_packing, return_mode="complete", workers=1,
        intraop_threads=threads if threads is not None else 4 if platform == "cpu" else 1,
        interop_threads=1, communication=communication)
    if peers is not None:
        execution.update(peer_devices=[platform+":"+str(i) for i in range(peers)],
                         placement_strategy="balanced-shared-groups")
    return dict(schema="tide-experiment-v1", name=name, description=question, model=deepcopy(model),
        task=dict(samples=batch, positions=positions, seed=7, input_pattern="staggered-v1",
            objective=dict(kind="weighted-output-square", scale=.001),
            optimizer=dict(kind="sgd", lr=.001, momentum=.3, weight_decay=.01)), execution=execution,
        measurement=dict(mode=mode, warmup=1, steps=2, validation=validation, checkpoint_check=True,
                         timeout_seconds=1800 if "scale" in name or "capacity" in name else 900),
        resources=dict(max_materialize_bytes=2*GIB, checkpoint_max_bytes=2*GIB, live_results=1) | (resources or {}))


def requests():
    groups = {}

    def add(platform, group, label, selected=None, *, model=None, **options):
        name = platform+"-"+label
        value = configuration(platform, name, model or dict(recipe=selected or recipe()), **options)
        groups.setdefault((platform, group), []).append(dict(name=name, config=value))
        return value

    general, _ = general_ports_plan(width=8)
    definition = {k: general[k] for k in ("graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")}
    medium, many = recipe(width=128), recipe(depth=8, target=1_000_000_000)
    for platform in SITES:
        for topology in ("fc", "hb", "lh"):
            local = recipe(topology, depth=1 if topology == "hb" else 2)
            add(platform, "coverage", topology+"-settle", local, family="settle",
                question="Aligned physical topology; complete control/state/ports and exact continuation.")
            add(platform, "coverage", topology+"-nonaligned", family="timed-dag",
                model=dict(recipe=local, transforms=[dict(kind="non-aligned", edge_index=0, extra_delay=1)]),
                question="New model with an actual changed physical delay, not a renamed layered instance.")
            add(platform, "coverage", topology+"-feedback", family="pdg",
                model=dict(recipe=local, transforms=[dict(kind="feedback", delay=2)]),
                question="Actual positive-delay feedback cycle, pending feedback and original logical time.")
        for family in ("timed-dag", "pdg"):
            add(platform, "coverage", "fc-view-"+family, family=family,
                question="Same physical/model/workload identity as fc-settle, embedded in another legal level.")
        config = add(platform, "coverage", "general-settle", family="settle", model=dict(definition=definition),
                     question="Same-rank independent regions, parallel edges, interior and repeated physical ports.")
        config["task"]["input_origins"] = general["input_origins"]
        for batch in (1, 8, 32, 128, 512):
            add(platform, "batch", f"fc-b{batch}", medium, batch=batch, positions=2, validation="explore",
                question="Fixed d128/4-node model, real staggered samples. Record strict FP32 comparison even when numerical exploration is accepted.")
        add(platform, "batch", "fc-stream-b8", medium, batch=8, positions=2, schedule="streaming", validation="explore",
            question="Same model/input as prefill B8; compare online control and actual time-batch sizes.")
        add(platform, "batch", "fc-prefill-p8-b8", medium, batch=8, positions=8, validation="explore",
            question="Longer online prefill/cache at unchanged original logical times; new workload identity.")
        for label, packing, returns in (("a", "scalar", "scalar"), ("c", "packed", "packed")):
            add(platform, "packing", "packing-"+label+"-b128", medium, batch=128, positions=2,
                packing=packing, return_packing=returns, validation="explore",
                question="A scalar/scalar or C packed/packed. B is the existing batch/fc-b128 packed/scalar case; no duplicate measurement or third auto algorithm.")
        for policy in ("scalar", "packed"):
            config = add(platform, "finite", "finite-b128-"+policy, medium, batch=128, positions=2,
                validation="explore", question="Same B128 material and full return; only bounded runtime finite diagnostic packing differs. Original numerical tolerance and nonfinite refusal remain.")
            config["execution"]["finite_check_packing"] = policy
            config = add(platform, "finite", "finite-training-"+policy, recipe(width=8), mode="training",
                question="Same small shared-parameter training, complete gradients/slots and exact resume; finite diagnostics do not change mathematical leaves.")
            config["execution"]["finite_check_packing"] = policy
        config = add(platform, "finite", "finite-b512-packed", medium, batch=512, positions=2,
            validation="explore", question="Actual B512 after bounded diagnostic packing; report per-node unique active samples and real operator batch, without changing logical batch.")
        config["execution"]["finite_check_packing"] = "packed"
        config = add(platform, "finite", "finite-b128-profile", medium, batch=128, positions=2,
            validation="explore", question="Separate observed window, not throughput. Attribute runtime finite-check region and actual kernels without inferring exclusive critical path from inclusive time.")
        config["execution"]["finite_check_packing"] = "packed"
        config["measurement"].update(observation="profile", steps=1)
        for client in ("pytorch", "libtorch"):
            add(platform, "clients", "common-"+client, client=client,
                question="Common complete-return material; native baseline is coverage/fc-settle. Client/build identities remain separate.")
        for client in ("pytorch", "python-native", "libtorch"):
            add(platform, "clients", "train-"+client, recipe(width=8), client=client, mode="training",
                question="Small first-order SGD, independent roots/parameters/slots and exact resume; preserve strict failures.")
        add(platform, "modules", "hb-qry-position-att", recipe("hb", depth=1, profile="textbook"),
            question="qry, relative rotation and full-region Attention representative; no candidate Cartesian product.")
        add(platform, "modules", "lh-heterogeneous", recipe("lh", family="mixed"),
            question="KV opening/closing with private GDN and SSD middle regions and exact short convolution.")
        add(platform, "scale", "scale-fc-100m", recipe(target=100_000_000), batch=1, positions=2,
            validation="explore", resources=LARGE, question="Fixed B1 curve: d128 -> natural 100M -> approximately1B. FFN remains4d.")
        for label, selected in (("fc-few", recipe(target=1_000_000_000)), ("fc-many", many),
                                ("hb", recipe("hb", target=1_000_000_000)), ("lh", recipe("lh", target=1_000_000_000))):
            add(platform, "scale", "scale-"+label+"-1b", selected, batch=1, positions=2,
                validation="explore", resources=LARGE,
                question="Natural approximately1B:4 large-local FC nodes,16 smaller-local FC nodes,20-node HB/LH. Distinct models/workloads; no equivalent-speedup claim.")
        for batch in (128, 512):
            add(platform, "scale", f"scale-fc-few-1b-b{batch}", recipe(target=1_000_000_000), batch=batch, positions=2,
                validation="explore", resources=LARGE,
                question="Explicit mandatory approximately1B large-batch admission candidate. Current per-card/host/disk budget and finite calibration first; persist refusal rather than OOM-search or shrinking batch.")
        observed = add(platform, "scale-observation", "scale-profile-fc-few-1b-b128",
            recipe(target=1_000_000_000), batch=128, positions=2, validation="none", resources=LARGE,
            question="Separate actual1B/B128 operator-shape, bucket and packing observation after the matching formal process; no throughput or new independent qualification. Same seed/model/first two windows, no reference prepass.")
        observed["measurement"].update(observation="profile", steps=1, checkpoint_check=False)
        for batch in (128, 512):
            add(platform, "resident-match", f"resident-match-b{batch}", recipe(width=16),
                batch=batch, positions=2, validation="explore",
                question="Same d16 FC4 model, staggered samples and P2 as resident-bounded; eager packed/scalar complete return. Separately qualify CPU/mixed-c before comparing device layout costs; d128 batch rows are not this workload.")
        if platform == "cpu":
            for threads in (1, 16):
                add(platform, "threads", f"threads-{threads}", medium, batch=128, positions=2,
                    threads=threads, validation="explore", question="ATen/BLAS comparison;4-thread baseline is batch/fc-b128, scheduler workers stays1.")
            workers = add(platform, "threads", "scheduler-workers-2", medium, batch=8, positions=2,
                validation="explore", question="Scheduler workers separated from4 ATen threads; old workers2 strict boundary remains.")
            workers["execution"]["workers"] = 2
            add(platform, "clients", "fp64-reference", recipe(width=8), dtype="float64", client="pytorch",
                question="Small independent numeric/formula anchor; does not replace FP32/FP16 performance.")
            continue
        for placement in ("mixed-a", "mixed-b"):
            add(platform, "placement", placement+"-b128", medium, batch=128, positions=2,
                placement=placement, validation="explore", question="Same material as mixed-c batch/fc-b128; control placement and synchronization cost.")
        for batch in (1, 128, 512):
            add(platform, "resident", f"resident-b{batch}", batch=batch, positions=2, placement="resident", validation="explore",
                question="Device-owned row/time batching; static layout refusal is a resource fact. Reopen only with explicit reviewed limits/profile, never silently alter batch.")
        for batch in (128, 512):
            add(platform, "resident-bounded", f"resident-bounded-b{batch}", batch=batch, positions=2,
                placement="resident", validation="explore",
                resources=dict(resident_limits=dict(queue=12*batch, stages=32, cache=33,
                    tensor_bytes=(2 if batch == 128 else 4)*GIB, operator_bytes=128*1024**2,
                    timeout_ms=60000, prefill=True)),
                question="Aligned FC4/P2 static atom bound: (2 input ports+4 physical edges)*P2*B=12B. Positive delays and sealed independent samples bound stages without multiplying B. Preserve default-layout refusal; no reference trajectory or logical microbatch change.")
        add(platform, "resident", "resident-training", recipe(width=8), placement="resident", mode="training",
            question="Single-device resident first-order SGD/root/checkpoint boundary; inference cannot certify training.")
        add(platform, "precision", "fp16-inference-b128", medium, batch=128, positions=2, dtype="float16", validation="explore",
            question="Explicit FP16 payload/FP32 scoring, independent dtype tolerance and strict discrete checks.")
        add(platform, "precision", "fp16-training", recipe(width=8), mode="training", dtype="float16",
            question="Small FP16 training with FP32 master/slot policy and exact resume.")
        maximum = 4 if platform == "cuda" else 8
        for count in (2, 3, maximum):
            add(platform, "multicard", f"model-1b-cards-{count}", many, batch=1, positions=2, peers=count,
                validation="explore", resources=LARGE, question="Same16-node model as scale/fc-many-1b, without parameter de-sharing. Verify all cards' parameter/state/compute owners.")
        for communication in ("ordinary", "blocking", "cpu_bridge"):
            add(platform, "multicard", "train-two-"+communication, recipe(width=8, depth=4), mode="training", peers=2,
                communication=communication, question="Same small two-card training model; actual transport, shared gradients, stream ownership and exact resume.")
        for count in (1, 2, maximum):
            add(platform, "capacity", f"capacity-train-1b-cards-{count}", many, batch=1, positions=2, mode="training",
                peers=None if count == 1 else count, validation="explore", resources=LARGE,
                question="Same approximately1B training model over1/2/maximum cards; gradients/slots/journal/restore budget. Refusal is recorded, not called hardware unsupported.")
        add(platform, "capacity", "capacity-fc-near-2b", recipe(target=2_000_000_000), batch=1, positions=2,
            validation="explore", resources=LARGE, question="One bounded approximately2B FP32 capacity candidate after1B calibration; any changed scale is a new explicit reviewed model.")
        if platform == "cuda":
            for count in (1, 2, 3, 4):
                add(platform, "capacity-calibrated", f"capacity-4561m-cards-{count}",
                    recipe(width=4352, depth=8), batch=1, positions=2,
                    peers=None if count == 1 else count, validation="explore",
                    resources=LARGE | dict(checkpoint_max_bytes=32*GIB, device_reserve_bytes=3*GIB,
                        workspace_bytes={f"cuda:{i}":GIB for i in range(count)},
                        activation_bytes={f"cuda:{i}":256*1024**2 for i in range(count)},
                        communication_bytes={f"cuda:{i}":64*1024**2 for i in range(count)}),
                    question="One calibrated FC16 model:4.561B trainable,8 indivisible groups. Same model on1/2/3/4 cards.1B construction measured4.788GB; reserve1GiB workspace+256MiB activation+64MiB communication per card plus3GiB unused. Two checkpoint generations require42.49GB free disk; review/retire completed inputs first, never add nominal card memory or OOM-search.")
    return groups


def generate(output, sites):
    output, sites = Path(output).resolve(), Path(sites).resolve()
    documents, rows = {}, []
    for (platform, group), cases in sorted(requests().items()):
        name = platform+"-"+group+".json"
        site = sites/SITES[platform]
        documents[name] = dict(schema="tide-experiment-matrix-v1", name=platform+"-"+group,
            description="Finite named stage; preview/current resource admission before execution. No implicit Cartesian run or qualification.",
            site=os.path.relpath(site, output), cases=cases)
        for case in cases:
            r = resolve(case["config"], site=site)
            p, e, b = r["plan"], r["execution"]["effective"], r["capacity_budget"]
            rows.append(dict(case=case["name"], matrix=name, status="planned", question=case["config"]["description"],
                **{k:r[k] for k in ("model_id", "workload_id", "task_id", "execution_id")},
                parameters=r["inventory"]["totals"]["trainable"], unique_parameters=r["inventory"]["totals"]["unique"],
                nodes=p["graph"]["nodes"], regions=p["graph"]["regions"], physical_edges=len(p["graph"]["edges"]),
                topology=(p.get("recipe") or {}).get("requested", {}).get("topology", "general-ports"),
                graph=r["graph_view"]["family"], width=p["width"], logical_batch=p["samples"], positions=p["positions"], stride=p["stride"], span=p["span"],
                client=e["client"], placement=e["placement"], devices=list(e["peer_devices"] or [e["device"]]),
                node_owners={d:len(v["nodes"]) for d,v in r["payload_placement"]["per_device"].items()},
                mode=r["measurement"]["mode"], dtype=e["dtype"], scoring_dtype=e["scoring_dtype"], packing=e["packing"], return_packing=e["return_packing"],
                timeout_seconds=r["measurement"]["timeout_seconds"], validation_policy=r["measurement"]["validation"],
                static_budget_status=b["status"], static_rejections=b["static_rejections"],
                estimated_peak_bytes={d:v["estimated_peak_bytes"] for d,v in b["per_device"].items()},
                unknown_components={d:v["unknown_components"] for d,v in b["per_device"].items()},
                host_estimated_bytes=b["host"]["estimated_tensor_reservation_bytes"], disk_estimated_bytes=b["disk"]["estimated_new_bytes"]))
    counts = {p:sum(r["case"].startswith(p+"-") for r in rows) for p in SITES}
    documents["index.json"] = dict(schema="tide-finite-matrix-plan-v1", upstream="0770593838a37c790a568226930567ed60f99b83",
        status="planned-not-runtime-evidence", site_paths="templates; actual builds/current free resources are independent run inputs", counts=counts, cases=rows)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in documents):
        raise FileExistsError("choose a new output directory; generated files are not overwritten")
    for name, document in documents.items():
        write_json(output/name, document, exclusive=True)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sites-dir", default=str(Path(__file__).resolve().parents[1]/"sites"))
    arguments = parser.parse_args()
    print(generate(arguments.output_dir, arguments.sites_dir))
