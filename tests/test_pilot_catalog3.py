"""Subject identity, exact capacity and finite-match acceptance contracts."""
import copy
from collections import Counter, defaultdict
import hashlib
import json

import pytest
import torch

from pilotlab.catalog3 import (CATALOG_VERSION, COMBINED_CATALOG_VERSION,
                              COMBINED_WHOLE_PARAMETER_REFERENCE, MAX_AUTHORIZED_SUBJECTS,
                              SUBJECT_BACKBONE_PARAMETER_LIMIT, SUBJECT_QUOTAS,
                              SUBJECT_WHOLE_PARAMETER_LIMIT, SUPPLEMENTAL_ALLOCATION_BUDGETS,
                              SUPPLEMENTAL_CATALOG_VERSION, SUPPLEMENTAL_PARAMETER_REFERENCE,
                              SUPPLEMENTAL_SUBJECT_QUOTAS, control_parameter_counts,
                              exact_parameter_counts, generate_combined_subjects,
                              generate_subjects, generate_supplemental_subjects,
                              match_control_catalog, match_controls, model_key,
                              subject_catalog_contract)
from pilotlab.models3 import ACTIVITY_VERSION, TIMED_LAST_NONEMPTY, build_model, parameter_counts


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def subjects():
    return generate_subjects()


@pytest.fixture(scope="module")
def supplemental_subjects():
    return generate_supplemental_subjects()


def _flatten_spec(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten_spec(item, name))
        else:
            result[name] = item
    return result


def test_supplemental_catalog_is_32_templates_with_four_real_allocation_budgets(subjects, supplemental_subjects):
    rows = supplemental_subjects
    assert len(rows) == len({row["id"] for row in rows}) == len({row["model_key"] for row in rows}) == 128
    assert Counter(row["group"] for row in rows) == Counter(SUPPLEMENTAL_SUBJECT_QUOTAS)
    assert {row["model_key"] for row in rows}.isdisjoint(row["model_key"] for row in subjects)
    assert {row["catalog_version"] for row in rows} == {SUPPLEMENTAL_CATALOG_VERSION}
    templates = defaultdict(list)
    for row in rows:
        config = row["model"]["config"]
        assert (config["topology"]["n"], config["d"]) == (16, 128)
        assert row["model_key"] == row["effective_key"] == model_key(row["model"])
        assert row["exposure"] == {"maximum_tuning_loss_tokens": 8*2**20,
                                   "formal_loss_tokens_per_trajectory": 12*2**20}
        assert row["control_matching"] == {"kinds": ["dense_total", "dense_active", "moe_joint"],
                                           "parameter_tolerance": .10, "ratio_tolerance": .02}
        templates[row["template_id"]].append(row)
    assert len(templates) == 32
    for template_rows in templates.values():
        assert tuple(row["model"]["config"]["budget"] for row in template_rows) == SUPPLEMENTAL_ALLOCATION_BUDGETS
        reference = _flatten_spec(template_rows[0]["model"])
        by_budget = {str(row["model"]["config"]["budget"]): row["id"] for row in template_rows}
        for row in template_rows:
            current = _flatten_spec(row["model"])
            changed = {key for key in reference.keys() | current.keys() if reference.get(key) != current.get(key)}
            assert changed == (set() if row is template_rows[0] else {"config.budget"})
            sweep = row["allocation_budget_sweep"]
            assert sweep["axis"] == "config.budget"
            assert sweep["values"] == list(SUPPLEMENTAL_ALLOCATION_BUDGETS)
            assert sweep["subject_ids_by_budget"] == by_budget


def test_supplemental_single_axis_controls_and_primary_baselines(subjects, supplemental_subjects):
    by_id = {row["id"]: row for row in supplemental_subjects}
    primary = {row["id"]: row for row in subjects}
    expected_axes = {
        "fc": {"base": None, "memory_s": "config.memory", "depth3": "config.topology.layers",
               "region_shared": "config.sharing"},
        "lattice": {"base": None, "memory_s": "config.memory", "depth2": "config.topology.depth",
                    "memory_shared": "config.memory_sharing", "receive64": "config.aggregation",
                    "ffn2": "config.ffn_width"},
        "exchange": {"base": None, "memory_k": "config.memory", "memory_s": "config.memory",
                     "memory_x_reverse": "config.memory", "independent": "config.sharing",
                     "shannon": "config.allocation", "kv64": "config.kv_width", "hub_fixed": "config.hub_allocation"},
    }
    multiscale_axes = {"base": None, "memory_k": "config.memory", "memory_s": "config.memory",
                       "singleton": "config.topology.partition", "mixed": "config.topology.partition",
                       "checkerboard": "config.topology.hub_mask", "shannon": "config.allocation"}
    expected_axes.update(multiscale_settle=multiscale_axes, multiscale_timed=multiscale_axes)
    assert {(row["group"], row["coverage_variant"]) for row in supplemental_subjects} == {
        (group, variant) for group, axes in expected_axes.items() for variant in axes}
    for row in supplemental_subjects:
        comparison = row["single_axis_control"]
        axis = expected_axes[row["group"]][row["coverage_variant"]]
        assert comparison["axis"] == axis
        current = _flatten_spec(row["model"])
        if axis is None:
            baseline = copy.deepcopy(primary[row["primary_baseline_id"]]["model"])
            baseline["config"]["topology"]["n"] = 16
            baseline["config"]["budget"] = row["model"]["config"]["budget"]
            assert row["model"] == baseline
            assert comparison["reference_subject_id"] is None
            continue
        reference = by_id[comparison["reference_subject_id"]]
        assert reference["group"] == row["group"] and reference["coverage_variant"] == "base"
        assert reference["model"]["config"]["budget"] == row["model"]["config"]["budget"]
        before = _flatten_spec(reference["model"])
        expected = {axis}
        if current.get("config.allocation") == "shannon" and axis == "config.allocation":
            assert current["config.shannon_support_policy"] == "weak-log-v1"
            expected.add("config.shannon_support_policy")
        assert {key for key in before.keys() | current.keys() if before.get(key) != current.get(key)} == expected
        assert (comparison["reference_value"], comparison["variant_value"]) == (before[axis], current[axis])


def test_supplemental_multiscale_pairs_preserve_period_clocks_and_timed_readout(supplemental_subjects):
    from tide import ModelConfig
    from tide.topology import build_topology
    pairs = defaultdict(list)
    for row in supplemental_subjects:
        if row["edge_delay_pair"]:
            pairs[row["edge_delay_pair"]].append(row)
    assert len(pairs) == 28
    for pair in pairs.values():
        assert len(pair) == 2
        assert {row["group"] for row in pair} == {"multiscale_settle", "multiscale_timed"}
        settle = next(row for row in pair if row["group"] == "multiscale_settle")
        timed = next(row for row in pair if row["group"] == "multiscale_timed")
        a, b = (ModelConfig.from_dict(row["model"]["config"]) for row in (settle, timed))
        assert a.clock == b.clock
        assert a.clock.position_origin == "absolute"
        assert (a.clock.position_period, a.clock.kv_window, a.clock.ssd_stride) == (6, 128*6, 6)
        assert a.clock.short_decay is not None
        assert a.topology.period == b.topology.period == 6
        a_dict, b_dict = a.to_dict(), b.to_dict()
        a_dict["topology"]["timing"] = b_dict["topology"]["timing"]
        assert a_dict == b_dict
        ga, gb = build_topology(a.topology), build_topology(b.topology)
        assert ga.nodes == gb.nodes and ga.regions == gb.regions
        assert [(e.src, e.dst, e.source_class) for e in ga.edges] == [(e.src, e.dst, e.source_class) for e in gb.edges]
        assert any(e.delay > 1 for e in ga.edges) and all(e.delay == 1 for e in gb.edges)
        assert timed["model"]["readout_policy"] == TIMED_LAST_NONEMPTY
        assert "readout_policy" not in settle["model"]
        for row, partner in ((settle, timed), (timed, settle)):
            assert row["timing_pair"]["partner_subject_id"] == partner["id"]
            assert row["timing_pair"]["input_period"] == 6
            assert row["timing_pair"]["local_clocks_identical"] is True


def test_supplemental_ordinary_budget_does_not_substitute_for_hub_or_measured_activity(supplemental_subjects):
    from tide import ModelConfig
    from tide.topology import build_topology
    for row in supplemental_subjects:
        config = ModelConfig.from_dict(row["model"]["config"])
        graph = build_topology(config.topology)
        allocation = row["allocation"]
        assert allocation["nominal_ordinary_budget_ratio"] == config.budget/16
        assert allocation["ordinary_allocation_budget"] == config.budget
        assert allocation["present_hub_nodes"] == sum(node.role == "H" for node in graph.nodes)
        assert sum(allocation["region_counts"].values()) == len(graph.regions)
        assert "not Full fraction" in allocation["nominal_ratio_scope"]
        assert "requires qualified per-token measurements" in allocation["actual_parameter_activity"]
        if row["group"] in ("fc", "lattice"):
            assert allocation["hub_policy"] == "no_hubs"
            assert allocation["present_hub_nodes"] == 0
        elif row["coverage_variant"] == "mixed":
            assert allocation["region_counts"]["mixed"] > 0
            assert allocation["region_counts"]["hub_only"] == 0
            assert allocation["hub_policy"] == "bounded_hubs_in_mixed_regions"
            assert allocation["mixed_region_nominal_total_budget"] == config.budget+1
            assert allocation["mixed_hub_bounds"] == "mu/4 <= hub control <= mu"
        elif row["coverage_variant"] == "singleton":
            assert allocation["hub_policy"] == "fixed_pass_on_singleton_hubs"
            assert allocation["separate_hub_nominal_budget"] is None
        elif row["coverage_variant"] == "hub_fixed":
            assert allocation["hub_policy"] == "fixed_pass_on_hub_only_regions"
            assert allocation["separate_hub_nominal_budget"] is None
        else:
            assert allocation["hub_policy"] == "separate_hub_regions_use_quad_allocation"
            assert allocation["separate_hub_nominal_budget"] == 1
        if row["coverage_variant"] != "mixed":
            assert allocation["mixed_region_nominal_total_budget"] is None


def test_supplemental_exact_capacity_requires_a_separate_combined_admission(supplemental_subjects):
    counts = [exact_parameter_counts(row["model"], 8192) for row in supplemental_subjects]
    whole = sum(row["whole_model_parameters"] for row in counts)
    assert whole == SUPPLEMENTAL_PARAMETER_REFERENCE["whole_parameters"] == 860_924_228
    assert sum(row["backbone_total_parameters"] for row in counts) == SUPPLEMENTAL_PARAMETER_REFERENCE["backbone_parameters"] == 726_706_500
    assert max(row["backbone_total_parameters"] for row in counts) == SUPPLEMENTAL_PARAMETER_REFERENCE["max_backbone_parameters"] == 22_682_912
    assert all(row["backbone_total_parameters"] <= SUBJECT_BACKBONE_PARAMETER_LIMIT for row in counts)
    assert whole+982_581_878 == COMBINED_WHOLE_PARAMETER_REFERENCE == 1_843_506_106
    assert COMBINED_WHOLE_PARAMETER_REFERENCE > SUBJECT_WHOLE_PARAMETER_LIMIT
    for enabled, size, version in ((False, 216, CATALOG_VERSION), (True, 344, COMBINED_CATALOG_VERSION)):
        contract = subject_catalog_contract(include_supplemental=enabled)
        assert contract["catalog_version"] == version
        assert contract["subject_count_limit"] == size
        assert contract["control_count_limit"] == 3*size
        assert contract["total_model_count_limit"] == 4*size
        assert contract["primary_subject_count"] == 216
        assert contract["supplemental_subject_count"] == (128 if enabled else 0)
        assert contract["primary_whole_parameter_limit"] == 1_000_000_000
        assert contract["primary_parameter_limit_applies_to_combined_catalog"] is False
        assert contract["parameter_reference_is_storage_admission"] is False
        assert contract["resource_admission"]["required"] is True
        assert contract["resource_admission"]["status"] == "requires_independent_measured_gate"


def test_combined_catalog_is_explicit_and_matcher_has_a_finite_344_subject_ceiling(subjects, supplemental_subjects):
    combined = generate_combined_subjects()
    assert len(combined) == len({row["model_key"] for row in combined}) == MAX_AUTHORIZED_SUBJECTS == 344
    assert combined[:216] == subjects and combined[216:] == supplemental_subjects
    assert generate_subjects() == subjects  # Calling the combined API never expands the default.
    report = match_control_catalog(combined, {})
    assert report["reasons"] == ["A3 activity mapping must cover exactly the requested subject IDs"]
    assert report["controls"] == []
    assert report["parameter_tolerance"] == .10 and report["ratio_tolerance"] == .02
    extra = dict(combined[-1], id="unauthorized-extra-subject")
    rejected = match_control_catalog(combined+[extra], {})
    assert rejected["reasons"] == ["requires 1..344 distinct subject identities"]
    assert rejected["controls"] == [] and "candidate_counts" not in rejected


def test_216_distinct_effective_models_and_legal_paired_clocks(subjects):
    from tide import ModelConfig
    from tide.topology import build_topology
    assert len(subjects) == len({row["id"] for row in subjects}) == len({row["model_key"] for row in subjects}) == 216
    assert Counter(row["group"] for row in subjects) == Counter(SUBJECT_QUOTAS)
    assert {row["model"]["config"]["d"] for row in subjects} == {64, 128, 256}
    pairs = defaultdict(list)
    for row in subjects:
        config = ModelConfig.from_dict(row["model"]["config"])
        build_topology(config.topology)
        if row["edge_delay_pair"]:
            pairs[row["edge_delay_pair"]].append(config)
    assert len(pairs) == 36
    for pair in pairs.values():
        assert len(pair) == 2
        assert {c.topology.timing for c in pair} == {"settle", "timed"}
        assert pair[0].clock == pair[1].clock
        assert pair[0].topology.period == pair[1].topology.period
        assert pair[0].clock.position_origin == "absolute"
        assert all(getattr(pair[0].clock, name) is not None for name in
                   ("position_period", "kv_window", "short_decay", "ssd_stride"))


def test_identity_deduplicates_aliases_defaults_and_execution_settings(subjects):
    spec = copy.deepcopy(subjects[0]["model"])
    same = copy.deepcopy(spec)
    same["config"].update(memory="X", adapter_rank=16, checkpoint_regions=True,
                         ffn_packing="power2", ffn_weight_access="grouped",
                         region_batch_size=8, attention_backend="math", parameter_indexing="static_view")
    assert model_key(spec) == model_key(same)  # no H nodes; no adapters
    same["config"]["kv_width"] = spec["config"]["d"]
    assert model_key(spec) == model_key(same)
    default_ffn = copy.deepcopy(spec)
    default_ffn["config"].pop("ffn_width")
    explicit_ffn = copy.deepcopy(default_ffn)
    explicit_ffn["config"]["ffn_width"] = spec["config"]["d"]*4
    assert model_key(default_ffn) == model_key(explicit_ffn)
    shared = copy.deepcopy(spec)
    shared["config"]["sharing"] = "region_shared"
    layer_shared = copy.deepcopy(shared)
    layer_shared["config"]["sharing"] = "layer_shared"
    assert model_key(shared) == model_key(layer_shared)  # FC one region per layer
    assert model_key(spec) != model_key(shared)
    pool = copy.deepcopy(spec)
    pool["config"]["allocation"] = "pool"
    pool2 = copy.deepcopy(pool)
    pool2["config"]["budget"] = 2
    assert model_key(pool) == model_key(pool2)
    no_hubs = dict(family="tide", config=dict(topology=dict(kind="multiscale", profile="compact-5-dd4", n=2), d=64))
    singleton = copy.deepcopy(no_hubs)
    singleton["config"]["topology"]["partition"] = "singleton"
    assert model_key(no_hubs) == model_key(singleton)
    new_timed = next(row["model"] for row in subjects if row["group"] == "multiscale_timed")
    old_timed = copy.deepcopy(new_timed)
    old_timed.pop("readout_policy")
    assert model_key(new_timed) != model_key(old_timed)


def test_capacity_envelope_counts_real_modules_and_keeps_content_widths(subjects):
    whole = 0
    for row in subjects:
        counts = exact_parameter_counts(row["model"], 8192)
        whole += counts["whole_model_parameters"]
        assert counts["backbone_total_parameters"] <= SUBJECT_BACKBONE_PARAMETER_LIMIT
    assert whole <= SUBJECT_WHOLE_PARAMETER_LIMIT
    assert Counter(row["model"]["config"]["d"] for row in subjects) == {64: 72, 128: 72, 256: 72}


def test_capacity_changes_preserve_declared_single_axis_comparisons(subjects):
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            else:
                result[name] = item
        return result

    variants = {
        "fc": {"depth3": "topology.layers", "shared": "sharing", "ffn1": "ffn_width"},
        "lattice": {"depth2": "topology.depth", "ffn2": "ffn_width", "nodes4": "topology.n",
                    "receive64": "aggregation", "receive128": "aggregation",
                    "memoryshared": "memory_sharing", "layershared": "sharing"},
        "exchange": {"shannon": "allocation", "pool": "allocation", "independent": "sharing"},
    }
    for family, comparisons in variants.items():
        memories = ("K", "S", "X", "X_reverse") if family == "exchange" else ("K", "S")
        for width in (64, 128, 256):
            for memory in memories:
                rows = {row["coverage_variant"]: row for row in subjects if row["group"] == family
                        and row["model"]["config"]["d"] == width and row["model"]["config"]["memory"] == memory}
                baseline = flatten(rows["base"]["model"]["config"])
                for variant, axis in comparisons.items():
                    changed = flatten(rows[variant]["model"]["config"])
                    if variant == "shannon":
                        assert changed["shannon_support_policy"] == "weak-log-v1"
                    assert {key for key in baseline if baseline[key] != changed[key]} == {axis}


@pytest.mark.parametrize("family", ["transformer", "moe"])
@pytest.mark.parametrize("width,layers,hidden", [(32, 1, 16), (64, 2, 96), (128, 3, 192)])
def test_control_formulas_count_actual_unique_parameters(family, width, layers, hidden):
    spec = dict(family=family, d=width, layers=layers, ffn_width=hidden, heads=2,
                window=8, experts=5, top_k=2)
    formula = control_parameter_counts(spec, 31)
    actual = parameter_counts(build_model(spec, 31))
    meta = exact_parameter_counts(spec, 31)
    assert actual == meta
    for key in ("backbone_total_parameters", "vocabulary_parameters", "whole_model_parameters"):
        assert formula[key] == actual[key]
    output = build_model(spec, 31)(torch.tensor([[1, 2]]), record_work=True)
    assert output.work["activity"]["mean_active_backbone_parameters"] == formula["mean_active_backbone_parameters"]


@pytest.mark.parametrize("memory,sharing", [("K", "independent"), ("S", "region_shared"), ("X", "layer_shared")])
def test_tide_meta_count_equals_materialized_count(memory, sharing):
    spec = dict(family="tide", config=dict(topology=dict(kind="exchange", n=2, hubs=1, depth=1, rounds=1),
                d=64, memory=memory, sharing=sharing, memory_sharing="region_shared"))
    assert exact_parameter_counts(spec, 31) == parameter_counts(build_model(spec, 31))


def test_three_controls_match_without_dummy_parameters(subjects):
    subject = subjects[0]
    model = build_model(subject["model"], 31)
    activity = model(torch.tensor([[1, 2, 3, 4]]), record_work=True).work["activity"]
    report = match_controls(subject, activity, 31)
    assert report["qualified"], report
    assert report["controls"]["dense_total"]["relative_total_error"] <= .1
    assert report["controls"]["dense_active"]["relative_active_error"] <= .1
    joint = report["controls"]["moe_joint"]
    assert joint["relative_total_error"] <= .1 and joint["relative_active_error"] <= .1
    assert joint["absolute_active_ratio_error"] <= .02
    assert joint["model"]["top_k"] < joint["model"]["experts"]
    for row in report["controls"].values():
        measured = build_model(row["model"], 31)(torch.tensor([[1, 2]]), record_work=True).work["activity"]
        assert measured["mean_active_backbone_parameters"] == row["counts"]["mean_active_backbone_parameters"]


def test_missing_activity_and_unmatchable_ratio_block_without_relaxation(subjects):
    subject = subjects[0]
    assert not match_controls(subject, {"qualified": False}, 31)["qualified"]
    total = exact_parameter_counts(subject["model"], 31)["backbone_total_parameters"]
    report = match_controls(subject, {"qualified": True, "backbone_total_parameters": total,
                                    "mean_active_backbone_parameters": 1}, 31)
    assert not report["qualified"]
    assert "moe_joint" in report["reasons"]
    assert report["controls"]["moe_joint"]["reason"]
    with pytest.raises(ValueError, match="frozen"):
        match_controls(subject, {}, 31, parameter_tolerance=.2)


@pytest.fixture(scope="module")
def global_inputs(subjects):
    """Synthetic contract fixtures, never experiment activity or admission evidence."""
    selected = [row for row in subjects if row["group"] in ("fc", "lattice")
                and row["coverage_variant"] == "base" and row["model"]["config"]["d"] == 64]
    activities = {}
    for row in selected:
        total = exact_parameter_counts(row["model"])["backbone_total_parameters"]
        activities[row["id"]] = {"qualified": True, "version": ACTIVITY_VERSION,
            "method": "actual-finish-controls-with-explicit-parameter-coverage",
            "aggregation": "last prescribed sample of each fixed A3 seed",
            "backbone_total_parameters": total, "mean_active_backbone_parameters": .9*total,
            "measured_external_tokens": 256}
    return selected, activities


@pytest.fixture(scope="module")
def global_catalog(global_inputs):
    return match_control_catalog(*global_inputs)


def test_new_shannon_recipes_preserve_all_other_preparation_fields(subjects):
    from tide import ModelConfig

    payload = [{"id": row["id"], "model": copy.deepcopy(row["model"]), "model_key": row["model_key"]} for row in subjects]
    for row in payload:
        config = row["model"]["config"]
        if config["allocation"] == "shannon":
            assert config.pop("shannon_support_policy") == "weak-log-v1"
        # The original preparation fingerprint predates the foundation's
        # independent default change for the reception padding kernel.
        assert config["reception_mask_padding"] == ModelConfig().reception_mask_padding
        config["reception_mask_padding"] = False
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert digest == "0ab7d9a027ac25590c72ed98c75f69c36adb32078c656d32d92b321e4d2e5371"


def test_global_catalog_reuses_actual_structures_and_independently_meets_every_threshold(global_inputs, global_catalog):
    subjects, activities = global_inputs
    report = global_catalog
    assert report["qualified"], report["reasons"]
    assert report["relationship_count"] == 3*len(subjects)
    assert report["control_count"] < report["relationship_count"]
    controls = {row["id"]: row for row in report["controls"]}
    assert len(controls) == len({row["model_key"] for row in controls.values()})
    relationships = {(row["subject_id"], row["kind"]) for row in report["matches"]}
    assert relationships == {(subject["id"], kind) for subject in subjects
                              for kind in ("dense_total", "dense_active", "moe_joint")}
    use = Counter(row["control_id"] for row in report["matches"])
    assert max(use.values()) > 1
    for match in report["matches"]:
        control = controls[match["control_id"]]
        spec = control["model"]
        exact = exact_parameter_counts(spec)
        assert exact["whole_model_parameters"] == control["counts"]["whole_model_parameters"]
        d, layers, hidden = spec["d"], spec["layers"], spec["ffn_width"]
        # Independent decoder arithmetic includes every shared norm, attention
        # and router matrix. It is not top-k/E used as an activity substitute.
        shared = 4*d*d+2*d
        if spec["family"] == "moe":
            shared += d*spec["experts"]
            expected_total = layers*(shared+3*d*hidden*spec["experts"])+d
            expected_active = layers*(shared+3*d*hidden*spec["top_k"])+d
            assert spec["top_k"] < spec["experts"]
        else:
            expected_total = expected_active = layers*(shared+3*d*hidden)+d
        assert expected_total == exact["backbone_total_parameters"]
        activity = activities[match["subject_id"]]
        target_total, target_active = activity["backbone_total_parameters"], activity["mean_active_backbone_parameters"]
        total_error = abs(expected_total-target_total)/target_total
        active_error = abs(expected_active-target_active)/target_active
        ratio_error = abs(expected_active*target_total-target_active*expected_total)/(target_total*expected_total)
        assert match["total_relative_error"] == pytest.approx(total_error, abs=1e-14)
        assert match["active_relative_error"] == pytest.approx(active_error, abs=1e-14)
        assert match["ratio_absolute_error"] == pytest.approx(ratio_error, abs=1e-14)
        if match["kind"] in ("dense_total", "moe_joint"):
            assert total_error <= .10
        if match["kind"] in ("dense_active", "moe_joint"):
            assert active_error <= .10
        if match["kind"] == "moe_joint":
            assert ratio_error <= .02
        assert match["activity_requires_runtime_confirmation"] is True
        local = report["per_subject"][match["subject_id"]]["controls"][match["kind"]]
        assert local["model"] == spec and local["control_id"] == control["id"]


def test_global_catalog_is_order_invariant_deterministic_and_never_uses_nll(global_inputs, global_catalog):
    subjects, activity = global_inputs
    untouched = copy.deepcopy((subjects, activity))
    reversed_report = match_control_catalog(list(reversed(subjects)), dict(reversed(list(activity.items()))))
    assert reversed_report["input_sha256"] == global_catalog["input_sha256"]
    assert reversed_report["candidate_library_sha256"] == global_catalog["candidate_library_sha256"]
    assert reversed_report["resolved_catalog_sha256"] == global_catalog["resolved_catalog_sha256"]
    altered = copy.deepcopy(activity)
    for index, row in enumerate(altered.values()):
        row["irrelevant_selection_nll"] = 99-index
    changed_nll = match_control_catalog(subjects, altered)
    assert changed_nll["resolved_catalog_sha256"] == global_catalog["resolved_catalog_sha256"]
    assert (subjects, activity) == untouched


@pytest.mark.parametrize("damage", ["unqualified", "missing_subject", "probe", "unversioned", "wrong_count"])
def test_global_catalog_rejects_estimates_or_non_a3_activity_without_new_search(global_inputs, damage):
    subjects, activity = copy.deepcopy(global_inputs)
    first = subjects[0]["id"]
    if damage == "unqualified":
        activity[first]["qualified"] = False
    elif damage == "missing_subject":
        activity.pop(first)
    elif damage == "probe":
        activity[first]["aggregation"] = "single cold initialization sample"
    elif damage == "unversioned":
        activity[first].pop("version")
    else:
        activity[first]["backbone_total_parameters"] += 1
    report = match_control_catalog(subjects, activity)
    assert not report["qualified"]
    assert report["reasons"]
    assert report["controls"] == []
    assert "candidate_counts" not in report


def test_global_catalog_keeps_unmatchable_relationships_explicit(global_inputs):
    subjects, activity = copy.deepcopy(global_inputs)
    subjects = subjects[:1]
    activity = {subjects[0]["id"]: activity[subjects[0]["id"]]}
    activity[subjects[0]["id"]]["mean_active_backbone_parameters"] = 1
    report = match_control_catalog(subjects, activity)
    assert not report["qualified"]
    assert report["relationship_count"] == 3
    assert any(row["kind"] == "moe_joint" and not row["qualified"] for row in report["matches"])
    assert report["parameter_tolerance"] == .10 and report["ratio_tolerance"] == .02


def test_global_catalog_prioritizes_an_available_eligible_control(global_inputs, global_catalog):
    subjects, activity = global_inputs
    available = copy.deepcopy(next(row for row in global_catalog["controls"] if row["model"]["family"] == "moe"))
    available["id"] = "existing-qualified-moe"
    report = match_control_catalog(subjects, activity, available_controls=[available])
    assert report["qualified"]
    retained = next(row for row in report["controls"] if row["id"] == available["id"])
    assert retained["reused_available_control"] is True
    assert report["input_sha256"] != global_catalog["input_sha256"]
    assert any(row["control_id"] == available["id"] for row in report["matches"])


def test_global_catalog_does_not_expand_its_finite_library_for_available_candidates(global_inputs):
    subjects, activity = global_inputs
    outside = {"family": "moe", "d": 80, "heads": 2, "layers": 1, "ffn_width": 16, "experts": 2, "top_k": 1}
    report = match_control_catalog(subjects, activity, available_controls=[outside])
    assert not report["qualified"]
    assert report["reasons"] == ["available control is outside this finite candidate union"]
