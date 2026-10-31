"""The finite table must retain the requested comparison axes and real graphs."""
import json
from pathlib import Path
import pytest
from tidegraph.experiment.config import resolve

ROOT = Path(__file__).resolve().parents[1]/"configs"/"matrices"


def cases(platform, group):
    return {row["name"]: row["config"] for row in json.loads((ROOT/f"{platform}-{group}.json").read_text())["cases"]}


@pytest.mark.parametrize("platform", ["cpu", "npu", "cuda"])
def test_large_batches_and_abc_use_same_model_task_and_effective_workload(platform):
    batch, packing = cases(platform, "batch"), cases(platform, "packing")
    curve = [resolve(batch[f"{platform}-fc-b{n}"]) for n in (1, 8, 32, 128, 512)]
    assert len({row["model_id"] for row in curve}) == 1
    assert [row["plan"]["samples"] for row in curve] == [1, 8, 32, 128, 512]
    assert all(row["plan"]["input_pattern"] == "staggered-v1" for row in curve)
    a, b, c = resolve(packing[f"{platform}-packing-a-b128"]), curve[3], resolve(packing[f"{platform}-packing-c-b128"])
    assert len({row["task_id"] for row in (a,b,c)}) == 1
    keys = ("device", "dtype", "scoring_dtype", "client", "placement", "intraop_threads", "schedule")
    assert all({key:row["execution"]["effective"][key] for key in keys} ==
               {key:b["execution"]["effective"][key] for key in keys} for row in (a,c))
    assert all(row["execution"]["effective"].get("return_mode", "complete") == "complete" for row in (a,b,c))
    assert [(row["execution"]["effective"]["packing"], row["execution"]["effective"]["return_packing"])
            for row in (a,b,c)] == [("scalar","scalar"),("packed","scalar"),("packed","packed")]


def test_graph_level_embeddings_and_actual_changed_models_are_distinct():
    coverage = cases("cpu", "coverage")
    base = resolve(coverage["cpu-fc-settle"])
    for family in ("timed-dag", "pdg"):
        embedded = resolve(coverage["cpu-fc-view-"+family])
        assert embedded["model_id"] == base["model_id"]
        assert embedded["workload_id"] == base["workload_id"]
        assert embedded["graph_view"]["family"] == family
    for topology in ("fc", "hb", "lh"):
        original = resolve(coverage[f"cpu-{topology}-settle"])
        delayed = resolve(coverage[f"cpu-{topology}-nonaligned"])
        feedback = resolve(coverage[f"cpu-{topology}-feedback"])
        assert len({row["model_id"] for row in (original,delayed,feedback)}) == 3
        assert delayed["plan"]["graph"]["edges"][0][2] == original["plan"]["graph"]["edges"][0][2]+1
        assert feedback["plan"]["graph"]["kind"] == "positive-delay"
        assert len(feedback["plan"]["graph"]["edges"]) == len(original["plan"]["graph"]["edges"])+1
    general = resolve(coverage["cpu-general-settle"])
    graph = general["plan"]["graph"]
    assert len(set(graph["ranks"])) < len(graph["ranks"])
    assert graph["inputs"][-2:] == [2,2] and graph["outputs"][-2:] == [2,2]
    assert len(set(map(tuple,graph["edges"]))) < len(graph["edges"])


@pytest.mark.parametrize("platform,maximum", [("cuda",4),("npu",8)])
def test_multicard_is_one_shared_model_and_every_requested_card_owns_nodes(platform, maximum):
    single = resolve(cases(platform,"scale")[f"{platform}-scale-fc-many-1b"])
    group = cases(platform,"multicard")
    for count in ((2,3,7,maximum) if platform == "npu" else (2,3,maximum)):
        current = resolve(group[f"{platform}-model-1b-cards-{count}"])
        assert current["model_id"] == single["model_id"]
        assert current["workload_id"] == single["workload_id"]
        owners = current["payload_placement"]["per_device"]
        assert len(owners) == count and all(row["nodes"] for row in owners.values())
        assert sum(row["parameter_elements"] for row in owners.values()) == current["inventory"]["totals"]["unique"]


def test_all_platforms_include_four_real_1b_geometries_and_large_batch_admission():
    for platform in ("cpu","npu","cuda"):
        scale = cases(platform,"scale")
        geometries = [resolve(scale[f"{platform}-scale-{name}-1b"]) for name in ("fc-few","fc-many","hb","lh")]
        assert [r["plan"]["graph"]["nodes"] for r in geometries] == [4,16,20,20]
        assert all(900_000_000 < r["inventory"]["totals"]["trainable"] < 1_100_000_000 for r in geometries)
        for batch in (128,512):
            candidate = resolve(scale[f"{platform}-scale-fc-few-1b-b{batch}"])
            assert candidate["model_id"] == geometries[0]["model_id"]
            assert candidate["plan"]["samples"] == batch
            assert candidate["measurement"]["validation"] == "explore"
            assert candidate["capacity_budget"]["automatic_admission"] is False


def test_plan_index_is_not_runtime_evidence_and_counts_are_finite():
    index = json.loads((ROOT/"index.json").read_text())
    assert index["status"] == "planned-not-runtime-evidence"
    observed = {platform:sum(len(cases(platform, path.stem[len(platform)+1:]))
                           for path in ROOT.glob(platform+"-*.json")) for platform in ("cpu","npu","cuda")}
    assert observed == index["counts"] == dict(cpu=48,npu=72,cuda=69)
    assert all(row["status"] == "planned" and row["timeout_seconds"] in (900,1800) for row in index["cases"])


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_bounded_resident_keeps_default_refusal_and_same_workload(platform):
    default, bounded = cases(platform, "resident"), cases(platform, "resident-bounded")
    for batch in (128, 512):
        old = resolve(default[f"{platform}-resident-b{batch}"])
        new = resolve(bounded[f"{platform}-resident-bounded-b{batch}"])
        assert old["capacity_budget"]["static_rejections"]
        assert old["workload_id"] == new["workload_id"]
        assert old["model_id"] == new["model_id"]
        for eager_platform in (platform, "cpu"):
            eager = resolve(cases(eager_platform, "resident-match")[f"{eager_platform}-resident-match-b{batch}"])
            assert eager["model_id"] == new["model_id"]
            assert eager["workload_id"] == new["workload_id"]
            assert eager["task_id"] == new["task_id"]
        limits = new["resources"]["resident_limits"]
        assert limits["queue"] == 12*batch and limits["stages"] == 32
        assert new["plan"]["samples"] == batch
