"""Build resource policy: affinity, explicit overrides and CLI propagation."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_resources", ROOT/"scripts/build_resources.py")
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def test_affinity_defaults_half_and_uncapped_explicit_count(monkeypatch):
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: {2, 4, 9, 15, 21})
    parser = argparse.ArgumentParser()
    resources.add_build_jobs(parser)
    assert parser.parse_args([]).jobs == 5
    assert parser.parse_args(["--jobs", "half"]).jobs == 3
    assert parser.parse_args(["--jobs", "320"]).jobs == 320
    for invalid in ("0", "-1", "1.5", "True", "auto"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--jobs", invalid])


def test_portable_cpu_count_fallback(monkeypatch):
    monkeypatch.delattr(resources.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: None)
    assert resources.resolve_jobs("all") == resources.resolve_jobs("half") == 1


@pytest.mark.parametrize("name", ["build", "build_device_control", "build_weighted_benchmark",
                                   "build_resident_consumer", "qualify"])
def test_entrypoints_expose_shared_policy_without_torch(name):
    result = subprocess.run([sys.executable, "-S", str(ROOT/f"scripts/{name}.py"), "--help"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "all|half|N" in result.stdout
