import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import pytest
from tidegraph.experiment.cli import main


def example():
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", depth=1,
        region_width=1, width=8)), execution=dict(device="cpu"), measurement=dict(validation="explore"))


@pytest.mark.parametrize("arguments", [["--help"], ["qualify", "--help"], ["inspect", "--help"],
                                     ["job", "--help"], ["results", "--help"]])
def test_help_needs_no_torch_or_vendor_plugin(arguments):
    source = str(Path(__file__).resolve().parents[1]/"python")
    code = "import sys; sys.path.insert(0,sys.argv[1]); from tidegraph.__main__ import main; main(sys.argv[2:])"
    result = subprocess.run([sys.executable, "-S", "-c", code, source, *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_run_forwards_public_engine_resolved_config_and_resume(tmp_path, monkeypatch, capsys):
    request = tmp_path/"run.json"
    request.write_text(json.dumps(example()))
    called = []
    engine = ModuleType("tidegraph.experiment.engine")
    def run(config, output_dir, *, resume, validation_override):
        called.append((config, output_dir, resume, validation_override))
        return dict(state="completed", execution_status="completed", validation=dict(numeric="failed", mode="explore"))
    engine.run = run
    monkeypatch.setitem(sys.modules, "tidegraph.experiment.engine", engine)
    import tidegraph.experiment as package
    monkeypatch.setattr(package, "engine", engine, raising=False)
    monkeypatch.setattr("tidegraph.experiment.cli._runtime_environment", lambda config, argv: None)
    assert main(["run", str(request), "--output-dir", str(tmp_path/"new"), "--resume", "checkpoint",
                 "--validation", "none"]) == 0
    config, output, resume, policy = called[0]
    assert config["schema"] == "tide-experiment-resolved-v1"
    assert config["measurement"]["validation"] == "explore"
    assert resume == "checkpoint" and policy == "none"
    assert '"numeric": "failed"' in capsys.readouterr().out


def test_invalid_config_exits_without_loading_engine(tmp_path, capsys):
    request = tmp_path/"invalid.json"
    value = example()
    value["execution"]["client"] = "invented-client"
    request.write_text(json.dumps(value))
    with pytest.raises(SystemExit) as error:
        main(["plan", str(request)])
    assert error.value.code == 1
    assert "execution/resources" in capsys.readouterr().err


def test_foreground_library_environment_reexecs_before_runtime_import(monkeypatch):
    from tidegraph.experiment.cli import _runtime_environment
    from tidegraph.experiment.config import resolve
    request = example()
    request["site"] = dict(environment={"LD_LIBRARY_PATH": "/chosen/python-stack"},
                           standalone_environment={"LD_LIBRARY_PATH": "/different/cpp-sdk"})
    config = resolve(request)
    calls = []
    monkeypatch.setattr("os.execve", lambda executable, command, environment: calls.append((command, environment)))
    _runtime_environment(config, ["run", "run.json", "--output-dir", "new"])
    command, environment = calls[0]
    assert command[1:4] == ["-m", "tidegraph", "run"]
    assert environment["LD_LIBRARY_PATH"] == "/chosen/python-stack"
    assert "/different/cpp-sdk" not in environment["LD_LIBRARY_PATH"]


@pytest.mark.parametrize("matrix", [False, True])
def test_explicit_site_is_relative_to_invocation_directory(tmp_path, monkeypatch, capsys, matrix):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"configs").mkdir()
    (tmp_path/"site.json").write_text(json.dumps(dict(name="invocation-site")))
    value = dict(schema="tide-experiment-matrix-v1", cases=[dict(name="one", config=example())]) if matrix else example()
    (tmp_path/"configs/run.json").write_text(json.dumps(value))
    assert main(["plan", "configs/run.json", "--site", "site.json", "--format", "json", *(["--matrix"] if matrix else [])]) == 0
    assert json.loads(capsys.readouterr().out)["site"]["name"] == "invocation-site"
