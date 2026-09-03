"""Tests for the dependency-light TIDE runtime boundary."""

import argparse
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
sys.path.insert(0, str(_SOURCE_ROOT))

from tide.runtime import (  # noqa: E402
    RuntimeConfigurationError,
    RuntimeRequest,
    RuntimeUnavailableError,
    _isolated_availability,
    _select_backend_with_reason,
    _seed_selected_runtime,
    add_runtime_arguments,
    normalize_device,
    resolve_runtime,
)


_TEST_BACKEND = os.environ.get("TEST_BACKEND", "").strip().lower()
_TEST_DTYPE = os.environ.get("TEST_DTYPE", "float32").strip().lower()
if _TEST_BACKEND not in {"", "cpu", "cuda", "npu"}:
    print(
        "invalid TEST_BACKEND: expected cpu, cuda, npu, or unset",
        file=sys.stderr,
    )
    raise SystemExit(2)
if _TEST_DTYPE not in {"float64", "float32", "float16", "bfloat16"}:
    print(
        "invalid TEST_DTYPE: expected float64, float32, float16, or bfloat16",
        file=sys.stderr,
    )
    raise SystemExit(2)
if _TEST_BACKEND in {"cuda", "npu"} and _TEST_DTYPE == "float64":
    print("float64 is a CPU-only oracle dtype", file=sys.stderr)
    raise SystemExit(2)


class _FakeAccelerator:
    def __init__(self, *, available=True, count=1):
        self.available = available
        self.count = count
        self.selected = None
        self.seed = None
        self.synchronized = None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def set_device(self, index):
        self.selected = index

    def get_device_name(self, index):
        return f"fake-device-{index}"

    def manual_seed(self, seed):
        self.seed = seed

    def synchronize(self, index):
        self.synchronized = index


def _fake_torch(cuda=None, npu=None):
    generator = types.SimpleNamespace(manual_seed=mock.Mock())
    torch = types.SimpleNamespace(
        __version__="fake-torch",
        device=lambda value: value,
        random=types.SimpleNamespace(default_generator=generator),
        float64=object(),
        float32=object(),
        float16=object(),
        bfloat16=object(),
    )
    if cuda is not None:
        torch.cuda = cuda
    if npu is not None:
        torch.npu = npu
    return torch


class RuntimeSyntaxTests(unittest.TestCase):
    def test_normalize_device_suffix(self):
        self.assertEqual(normalize_device("cuda:2", None), ("cuda", 2))
        self.assertEqual(normalize_device(" NPU:0 ", None), ("npu", 0))

    def test_reject_conflicting_index(self):
        with self.assertRaises(RuntimeConfigurationError):
            normalize_device("cuda:1", 2)

    def test_reject_auto_index(self):
        with self.assertRaises(RuntimeConfigurationError):
            normalize_device("auto", 0)

    def test_reject_cpu_nonzero_index(self):
        with self.assertRaises(RuntimeConfigurationError):
            normalize_device("cpu", 1)

    def test_reject_negative_device_index(self):
        with self.assertRaises(RuntimeConfigurationError):
            normalize_device("npu", -1)

    def test_reject_negative_seed_before_importing_torch(self):
        with mock.patch("tide.runtime.importlib.import_module") as imported:
            with self.assertRaises(RuntimeConfigurationError):
                resolve_runtime(RuntimeRequest("cpu", 0, "float32", -1))
        imported.assert_not_called()

    def test_parser_exposes_all_project_dtypes(self):
        parser = argparse.ArgumentParser()
        add_runtime_arguments(parser)
        for dtype in ("float64", "float32", "float16", "bfloat16"):
            args = parser.parse_args(["--device", "cpu", "--dtype", dtype])
            self.assertEqual(args.dtype, dtype)

    def test_training_style_parser_requires_explicit_device(self):
        parser = argparse.ArgumentParser()
        add_runtime_arguments(parser, require_explicit_device=True)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        self.assertEqual(parser.parse_args(["--device", "auto"]).device, "auto")

    def test_float64_accelerator_request_fails_without_import(self):
        with mock.patch("tide.runtime.importlib.import_module") as imported:
            with self.assertRaisesRegex(
                RuntimeConfigurationError, "CPU oracle"
            ):
                resolve_runtime(RuntimeRequest("cuda", 0, "float64", 0))
        imported.assert_not_called()

    def test_isolated_probe_forces_vendor_autoload_off(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                'TIDE_RUNTIME_PROBE={"available": false, "count": 0, '
                '"error": null}\n'
            ),
            stderr="",
        )
        with mock.patch(
            "tide.runtime.subprocess.run", return_value=completed
        ) as run:
            result = _isolated_availability("npu")
        probe_env = run.call_args.kwargs["env"]
        self.assertEqual(probe_env["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
        self.assertEqual(probe_env["TIDE_RUNTIME_PROBE_BACKEND"], "npu")
        self.assertFalse(result["available"])

    def test_auto_selection_exposes_single_accelerator_reason(self):
        with mock.patch(
            "tide.runtime._isolated_availability",
            side_effect=[
                {"available": False, "count": 0, "fatal": False},
                {"available": True, "count": 1, "fatal": False},
            ],
        ):
            self.assertEqual(
                _select_backend_with_reason("auto"),
                ("npu", "auto:single-visible-npu"),
            )

    def test_auto_without_accelerator_exposes_cpu_reason(self):
        with mock.patch(
            "tide.runtime._isolated_availability",
            return_value={"available": False, "count": 0, "fatal": False},
        ):
            self.assertEqual(
                _select_backend_with_reason("auto"),
                ("cpu", "auto:no-accelerator-selected-cpu"),
            )

    def test_auto_rejects_ambiguous_accelerator_families(self):
        with mock.patch(
            "tide.runtime._isolated_availability",
            return_value={"available": True, "count": 1, "fatal": False},
        ):
            with self.assertRaisesRegex(RuntimeUnavailableError, "both CUDA and NPU"):
                _select_backend_with_reason("auto")

    def test_cpu_resolution_does_not_import_torch_npu(self):
        fake = _fake_torch()
        imported_names = []

        def import_module(name):
            imported_names.append(name)
            if name == "torch":
                return fake
            raise AssertionError(f"unexpected import: {name}")

        # Resolve every string patch target before replacing importlib.import_module;
        # Python 3.11's mock target resolver itself imports ``tide``.
        with mock.patch("tide.runtime._seed_selected_runtime"), mock.patch(
            "tide.runtime._minimum_operator_probe",
            return_value=((0.0, 1.0, 4.0, 9.0), (0.0, 2.0, 4.0, 6.0)),
        ), mock.patch(
            "tide.runtime.importlib.import_module", side_effect=import_module
        ):
            runtime = resolve_runtime(RuntimeRequest("cpu", 0, "float32", 7))
        self.assertEqual(imported_names, ["torch"])
        self.assertEqual(runtime.info.resolution_reason, "explicit:cpu")

    def test_npu_resolution_imports_plugin_after_torch(self):
        npu = _FakeAccelerator()
        fake = _fake_torch(npu=npu)
        plugin = types.SimpleNamespace(__version__="fake-npu")
        imported_names = []

        def import_module(name):
            imported_names.append(name)
            if name == "torch":
                return fake
            if name == "torch_npu":
                return plugin
            raise AssertionError(f"unexpected import: {name}")

        with mock.patch("tide.runtime._seed_selected_runtime"), mock.patch(
            "tide.runtime._minimum_operator_probe",
            return_value=((0.0, 1.0, 4.0, 9.0), (0.0, 2.0, 4.0, 6.0)),
        ), mock.patch(
            "tide.runtime.importlib.import_module", side_effect=import_module
        ):
            runtime = resolve_runtime(RuntimeRequest("npu", 0, "float32", 7))
        self.assertEqual(imported_names, ["torch", "torch_npu"])
        self.assertEqual(runtime.info.plugin_version, "fake-npu")
        self.assertEqual(npu.selected, 0)

    def test_explicit_unavailable_cuda_fails_without_cpu_fallback(self):
        cuda = _FakeAccelerator(available=False, count=0)
        fake = _fake_torch(cuda=cuda)
        with mock.patch(
            "tide.runtime.importlib.import_module", return_value=fake
        ):
            with self.assertRaisesRegex(
                RuntimeUnavailableError, "no CPU fallback"
            ):
                resolve_runtime(RuntimeRequest("cuda", 0, "float32", 7))

    def test_seed_targets_cpu_and_selected_family(self):
        cuda = _FakeAccelerator()
        fake = _fake_torch(cuda=cuda)
        with mock.patch("tide.runtime.random.seed") as python_seed:
            _seed_selected_runtime(fake, "cuda", 19)
        python_seed.assert_called_once_with(19)
        fake.random.default_generator.manual_seed.assert_called_once_with(19)
        self.assertEqual(cuda.seed, 19)


class RuntimeImportAndHelpTests(unittest.TestCase):
    def _subprocess_environment(self, blocker_root):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(blocker_root), str(_SOURCE_ROOT)]
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    def test_package_import_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            blocker_root = pathlib.Path(temporary_root)
            (blocker_root / "torch.py").write_text(
                "raise RuntimeError('torch must stay lazy')\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys, tide; assert 'torch' not in sys.modules",
                ],
                cwd=str(_REPOSITORY_ROOT),
                env=self._subprocess_environment(blocker_root),
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_help_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            blocker_root = pathlib.Path(temporary_root)
            (blocker_root / "torch.py").write_text(
                "raise RuntimeError('torch must stay lazy')\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, "-m", "tide.runtime", "--help"],
                cwd=str(_REPOSITORY_ROOT),
                env=self._subprocess_environment(blocker_root),
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--device", completed.stdout)
        self.assertIn("float64", completed.stdout)


class RuntimeContractTests(unittest.TestCase):
    def test_pyproject_declares_dependency_light_src_package(self):
        pyproject = (_REPOSITORY_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('requires-python = ">=3.9"', pyproject)
        self.assertIn("dependencies = []", pyproject)
        self.assertIn('tide-runtime = "tide.runtime:main"', pyproject)
        self.assertIn('package-dir = {"" = "src"}', pyproject)

    def test_contract_targets_and_commands_match_repository(self):
        contract_path = _REPOSITORY_ROOT / ".torch-portability" / "contract.json"
        with contract_path.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
        targets = {target["id"]: target for target in contract["targets"]}
        self.assertEqual(
            targets["python-cpu-aarch64"]["status"], "implemented"
        )
        self.assertEqual(targets["python-cpu-aarch64"]["policy"], "required")
        self.assertEqual(targets["python-cpu-x86_64"]["status"], "planned")
        self.assertEqual(targets["python-npu-aarch64"]["policy"], "required")
        self.assertEqual(targets["python-npu-aarch64"]["status"], "implemented")
        self.assertEqual(targets["python-cuda-x86_64"]["policy"], "optional")
        commands = contract["contract"]["validation_commands"]
        self.assertIn("unittest discover", commands["cpu_fp32"])
        self.assertIn("test_equivalence.py", commands["cpu_equivalence"])
        self.assertIn("test_backend_semantics.py", commands["npu_fp32_semantic"])
        self.assertIn("backend_parity_worker.py", commands["npu_fp32_parity"])
        self.assertIn("TEST_DTYPE=float64", commands["cpu_float64_oracle"])
        checkpoint = contract["contract"]["checkpoint"]
        self.assertTrue(checkpoint["enabled"])
        self.assertEqual(
            checkpoint["schema"], "tide.settlegraph.checkpoint.v1"
        )
        self.assertTrue((_REPOSITORY_ROOT / "tests" / "test_runtime.py").is_file())
        self.assertTrue(
            (_REPOSITORY_ROOT / "tests" / "test_backend_semantics.py").is_file()
        )
        self.assertTrue(
            (_REPOSITORY_ROOT / "tests" / "backend_parity_worker.py").is_file()
        )
        self.assertTrue((_SOURCE_ROOT / "tide" / "runtime.py").is_file())


@unittest.skipUnless(
    bool(_TEST_BACKEND),
    "set TEST_BACKEND to require a real backend operator probe",
)
class RuntimeRequiredBackendTests(unittest.TestCase):
    def test_required_backend_operator_probe(self):
        runtime = resolve_runtime(
            RuntimeRequest(_TEST_BACKEND, 0, _TEST_DTYPE, 7)
        )
        self.assertEqual(runtime.info.backend, _TEST_BACKEND)
        self.assertEqual(runtime.info.resolution_reason, f"explicit:{_TEST_BACKEND}")
        self.assertEqual(runtime.info.dtype, _TEST_DTYPE)
        self.assertEqual(runtime.info.probe_forward, (0.0, 1.0, 4.0, 9.0))
        self.assertEqual(runtime.info.probe_gradient, (0.0, 2.0, 4.0, 6.0))
        runtime.synchronize()


if __name__ == "__main__":
    unittest.main()
