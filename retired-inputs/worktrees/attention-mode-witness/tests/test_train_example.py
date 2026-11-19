"""Public example settings, real update boundaries and fresh-data continuation."""
from dataclasses import replace
import json

import pytest
import torch

from examples import train
from tide import ModelConfig, TopologyConfig, TideModel, load_checkpoint, save_checkpoint


def model_config(timed=False, **options):
    topology = (TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4", timing="timed")
                if timed else TopologyConfig(n=2, layers=2))
    return ModelConfig(topology=topology, d=64, memory="X" if timed else "K",
                       sharing="region_shared", attention_backend="math", **options)


def arguments(tmp_path, *options, config=None):
    path = tmp_path/"model.json"
    path.write_text(json.dumps((config or model_config()).to_dict()))
    return train.parser().parse_args(["--config", str(path), "--batch", "1", "--length", "2",
                                      "--steps", "2", *map(str, options)])


def records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def steps(optimizer):
    return {int(entry["step"]) for entry in optimizer.state.values()}


def test_preset_model_replacement_and_explicit_false_flags(tmp_path):
    candidate = json.loads((train.PRESETS/"hb-timed.json").read_text())
    candidate["execution"].update(batch=7, precision="bf16", foreach=True)
    candidate["model"].update(region_batch_size=4, checkpoint_regions=True)
    preset = tmp_path/"preset.json"
    preset.write_text(json.dumps(candidate))
    args = arguments(tmp_path, "--preset", preset, "--batch", "3", "--memory", "S",
                     "--no-foreach", "--no-checkpoint-regions")
    settings = train.resolve(args)
    assert settings["config"] == replace(model_config(), memory="S")
    assert settings["execution"]["batch"] == 3
    assert settings["execution"]["precision"] == "bf16"
    assert settings["execution"]["foreach"] is False
    assert settings["source"]["preset"]["path"] == str(preset)
    assert len(settings["source"]["config"]["sha256"]) == 64
    assert settings["source"]["explicit_overrides"]["checkpoint_regions"] is False
    for name in ("default", "fc-k", "hb-settle", "hb-timed"):
        resolved = train.resolve(train.parser().parse_args(["--preset", name]))
        assert resolved["execution"]["device"] == "cpu"


@pytest.mark.parametrize("options,message", [
    (["--batch", "0"], "batch"), (["--length", "-1"], "length"),
    (["--steps", "0"], "steps"), (["--threads", "0"], "threads"),
    (["--seed", "-1"], "seed"), (["--lr", "nan"], "learning_rate"),
    (["--device", "automatic"], "explicit"),
    (["--mode", "grad_forward", "--lr", "0.001"], "require --mode train"),
    (["--mode", "inference", "--no-foreach"], "require --mode train"),
    (["--mode", "inference", "--checkpoint", "new.pt"], "require --mode train"),
])
def test_invalid_runtime_settings(options, message):
    with pytest.raises(ValueError, match=message):
        train.resolve(train.parser().parse_args(options))


@pytest.mark.parametrize("change", [
    {"schema": "unknown"}, {"execution": {"batch": True}},
    {"execution": {"foreach": 1}}, {"execution": {"hidden_optimizer": True}},
    {"unexpected": True},
])
def test_invalid_preset_fields_are_rejected(tmp_path, change):
    preset = json.loads((train.PRESETS/"default.json").read_text())
    preset.update(change)
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(preset))
    with pytest.raises(ValueError):
        train.resolve(train.parser().parse_args(["--preset", str(path)]))


def test_list_and_npu_dry_run_do_not_construct_models_or_import_vendor(monkeypatch, capsys):
    import builtins
    import tide
    original = builtins.__import__
    def observed(name, *args, **kwargs):
        assert not name.startswith("torch_npu")
        return original(name, *args, **kwargs)
    def unexpected(*args, **kwargs):
        raise AssertionError("discovery must not allocate model weights")
    monkeypatch.setattr(builtins, "__import__", observed)
    monkeypatch.setattr(tide, "TideModel", unexpected)
    assert train.main(["--list-presets"]) == 0
    assert {row["name"] for row in records(capsys)[0]["presets"]} >= {"default", "fc-k", "hb-settle", "hb-timed"}
    assert train.main(["--preset", "hb-timed", "--device", "npu:3", "--dry-run"]) == 0
    plan = records(capsys)[0]
    assert plan["device_initialized"] is False and plan["execution"]["device"] == "npu:3"


@pytest.mark.parametrize("mode", ["train", "grad_forward", "inference"])
def test_modes_own_graph_and_optimizer_boundaries(tmp_path, capsys, mode):
    args = arguments(tmp_path, "--mode", mode)
    result = train.run(args, train.resolve(args))
    rows = records(capsys)
    assert all(row["forward_requires_grad"] == (mode != "inference") for row in rows[1:])
    assert all(row["optimizer_updated"] == (mode == "train") for row in rows[1:])
    assert result["state"].position == 4
    if mode == "train":
        assert steps(result["optimizer"]) == {2}
    else:
        assert result["optimizer"] is None
        assert all(parameter.grad is None for parameter in result["model"].parameters())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_timed_training_flushes_and_restores_actual_updates(tmp_path, capsys, monkeypatch, precision):
    path = tmp_path/"saved.pt"
    config = model_config(True, region_batch_size=4, checkpoint_regions=True, local_attention_backend="math")
    args = arguments(tmp_path, "--checkpoint", path, "--precision", precision, config=config)
    original = TideModel.flush
    gradient_norms = []
    def observed(model, state):
        tail = original(model, state)
        if torch.is_grad_enabled():
            tail.values.register_hook(lambda value: gradient_norms.append(float(value.abs().sum())))
        return tail
    monkeypatch.setattr(TideModel, "flush", observed)
    result = train.run(args, train.resolve(args))
    rows = records(capsys)
    saved, state, slots = load_checkpoint(path)
    assert saved.config == config
    assert state.position == 6 and result["state"].position == 9
    assert {int(slot["step"]) for slot in slots["state"].values()} == {2}
    assert steps(result["optimizer"]) == {3}
    assert len(gradient_norms) == 4 and all(value > 0 for value in gradient_norms)
    assert [row["state_position"] for row in rows if row["kind"] == "step"] == [3, 6]
    report = rows[-1]
    assert report["checkpoint_data"] == report["next_forward"] == "exact"
    assert report["paired_real_updates"] == 2
    assert report["update_threshold"] == {"atol": 1e-6, "rtol": 2e-3}
    assert report["next_optimizer_update"] in {"within_threshold", "finite_deviation"}


def test_separate_resume_preserves_saved_model_and_optimizer_and_uses_current_runtime(tmp_path, capsys):
    path = tmp_path/"saved.pt"
    config = model_config(region_batch_size=4, checkpoint_regions=True)
    args = arguments(tmp_path, "--lr", "0.0003", "--no-foreach", config=config)
    initial = train.run(args, train.resolve(args))
    save_checkpoint(path, initial["model"], state=initial["state"], optimizer=initial["optimizer"])
    records(capsys)
    def resume(*extra):
        cli = train.parser().parse_args(["--resume", str(path), "--batch", "1", "--length", "1",
                                         "--steps", "1", "--memory", "K", *extra])
        return train.run(cli, train.resolve(cli))
    resumed = resume()
    rows = records(capsys)
    assert resumed["model"].config == config  # No assertion against default-preset geometry.
    assert resumed["state"].position == 5 and steps(resumed["optimizer"]) == {3}
    assert resumed["execution"]["learning_rate"] == 0.0003 and resumed["execution"]["foreach"] is False
    assert "fresh synthetic" in rows[0]["data"] and rows[0]["state_position"] == 4
    overridden = resume("--lr", "0.0002", "--foreach")
    records(capsys)
    assert overridden["execution"]["learning_rate"] == 0.0002 and overridden["execution"]["foreach"] is True
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        resume("--memory", "S")
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        resume("--preset", "default")


def test_resume_rejects_missing_or_wrong_optimizer_and_existing_save(tmp_path):
    path = tmp_path/"without-optimizer.pt"
    model = TideModel(model_config())
    save_checkpoint(path, model)
    args = train.parser().parse_args(["--resume", str(path), "--steps", "1"])
    with pytest.raises(ValueError, match="single-group AdamW"):
        train.run(args, train.resolve(args))
    save_checkpoint(path, model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
    with pytest.raises(ValueError, match="single-group AdamW"):
        train.run(args, train.resolve(args))
    with pytest.raises(ValueError, match="new checkpoint"):
        train.resolve(train.parser().parse_args(["--checkpoint", str(path)]))
    with pytest.raises(ValueError, match="does not exist"):
        train.resolve(train.parser().parse_args(["--resume", str(tmp_path/"missing.pt")]))


def test_restore_keeps_finite_update_deviation_and_rejects_nonfinite(tmp_path, monkeypatch):
    import tide
    torch.manual_seed(17)
    model = TideModel(model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, foreach=True)
    x = torch.randn(1, 2, 64)
    *_, state = train.update(model, optimizer, x, None, "fp32")
    path = tmp_path/"paired.pt"
    save_checkpoint(path, model, state=state, optimizer=optimizer)
    original = tide.load_checkpoint
    def changed_gradient(*args, **kwargs):
        restored, carry, slots = original(*args, **kwargs)
        for parameter in restored.parameters():
            parameter.register_hook(lambda gradient: -gradient)
        return restored, carry, slots
    monkeypatch.setattr(tide, "load_checkpoint", changed_gradient)
    report, _ = train.verify_restore(path, model, optimizer, state, x, "fp32")
    assert report["next_optimizer_update"] == "finite_deviation"
    assert report["parameter_threshold_failures"] and report["parameter_max_abs"] > 1e-6
    for parameter in model.parameters():
        parameter.register_hook(lambda gradient: gradient*float("nan"))
    with pytest.raises(ArithmeticError, match="nonfinite objective or gradient"):
        train.update(model, optimizer, x, state, "fp32")
