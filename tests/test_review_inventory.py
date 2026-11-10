"""An automatic syntax inventory must not invent review coverage or clean source."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from review_inventory import apply_reviews, inventory, source_tree
from review_symbols import clang_symbols, digest, python_symbols


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Inventory Test", "-c",
        "user.email=inventory@example.invalid", *args], cwd=root).decode().strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--quiet")
    (tmp_path/"module.py").write_text("def original():\n    return 1\n")
    (tmp_path/"core.cpp").write_text("int answer() { return 42; }\n")
    (tmp_path/"link.py").symlink_to("module.py")
    (tmp_path/"README.md").write_text("outside implementation selection\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "fixed source")
    return tmp_path


def test_inventory_reads_committed_bytes_and_keeps_review_pending(repository):
    original = (repository/"module.py").read_bytes()
    (repository/"module.py").write_text("def dirty_change(): pass\n")
    (repository/"untracked.py").write_text("def untracked(): pass\n")
    report = inventory(repository)
    assert report == inventory(repository)
    assert report["source"] == git(repository, "rev-parse", "HEAD")
    assert report["summary"]["files"] == 2
    assert report["summary"]["symbols"] == 1
    assert report["summary"]["symbol_review"] == {"pending": 1}
    files = {row["path"]: row for row in report["files"]}
    assert files["module.py"]["sha256"] == digest(original)
    assert files["module.py"]["symbols"][0]["name"] == "original"
    assert files["module.py"]["review"]["state"] == "static_only"
    assert files["core.cpp"]["parsing"]["state"] == "not_parsed"
    assert report["omitted_nonregular_entries"] == [dict(path="link.py", mode="120000", reason="nonregular Git entry")]
    _, all_files, _ = source_tree(repository, "HEAD")
    assert all_files["README.md"] == b"outside implementation selection\n"


def test_python_ast_covers_nested_async_lambda_and_decorator_range():
    source = b"""class API:
    @decorate
    async def invoke(self):
        def nested():
            return lambda: 1
        return nested()
"""
    symbols, parsing = python_symbols(source, "api.py")
    assert parsing["state"] == "parsed"
    assert [s["name"] for s in symbols] == ["API.invoke", "API.invoke.nested", "API.invoke.nested.<lambda>"]
    assert symbols[0]["line"] == 2
    assert symbols[0]["sha256"] == digest(b"".join(source.splitlines(keepends=True)[1:6]))
    assert symbols[0]["kind"] == "AsyncFunctionDef"
    assert len({s["id"] for s in symbols}) == 3
    assert all(s["definition"] for s in symbols)
    invalid, failed = python_symbols(b"def bad(:\n", "broken.py")
    assert invalid == [] and failed["state"] == "failed" and failed["diagnostic"]


def test_explicit_reviews_are_hash_bound_and_do_not_review_other_files(repository):
    report = inventory(repository)
    file = next(r for r in report["files"] if r["path"] == "module.py")
    claim = dict(path=file["path"], sha256=file["sha256"], symbols=[file["symbols"][0]["id"]],
                 reviewer="test review", evidence="report.md#scope", scope="manual function audit")
    claims = dict(schema="tide-implementation-review-v1", source=report["source"], files=[claim])
    reviewed = inventory(repository, reviews=claims)
    assert reviewed["summary"]["symbol_review"] == {"reviewed": 1}
    assert reviewed["summary"]["file_review"] == {"pending": 1, "partially_reviewed": 1}
    for field, value, message in (("sha256", "0"*64, "source changed"),
            ("symbols", ["absent@1:0"], "symbol is missing"), ("evidence", "", "requires reviewer")):
        corrupted = copy.deepcopy(claims)
        corrupted["files"][0][field] = value
        with pytest.raises(ValueError, match=message):
            inventory(repository, reviews=corrupted)
    with pytest.raises(ValueError, match="schema/source"):
        inventory(repository, reviews=claims | {"source": "HEAD"})
    apply_reviews(report["files"], [claim | {"symbols": "*"}, claim])
    assert file["review"]["state"] == "reviewed"


@pytest.mark.skipif(shutil.which("clang++") is None, reason="explicit C++ parser unavailable")
def test_clang_enumerates_requested_file_and_keeps_partial_failures(tmp_path):
    (tmp_path/"helper.hpp").write_text("inline int helper() { return 1; }\n")
    source = b'''#include "helper.hpp"
namespace api { struct X {
    int f(int);
    int f(double) { return helper(); }
}; int X::f(int x) { return x; } }
template <typename T> T twice(T x) { return x+x; }
int main() { return api::X{}.f(1); }
'''
    (tmp_path/"source.cpp").write_bytes(source)
    symbols, parsing = clang_symbols(source, "source.cpp", tmp_path)
    assert parsing["state"] == "parsed", parsing
    assert [s["name"] for s in symbols] == ["api::X::f"]*3+["twice", "main"]
    assert [s["definition"] for s in symbols] == [False, True, True, True, True]
    assert len({s["id"] for s in symbols}) == len(symbols)
    assert parsing["command"][-1] == "source.cpp"
    assert "clang" in parsing["parser_version"].lower()
    malformed = b"int valid() { return 1; }\nint broken( ;\n"
    (tmp_path/"broken.cpp").write_bytes(malformed)
    partial, failed = clang_symbols(malformed, "broken.cpp", tmp_path)
    assert failed["state"] == "partial" and failed["exit_code"] != 0 and failed["diagnostic"]
    assert any(s["name"] == "valid" for s in partial)


def test_missing_cpp_parser_is_explicit():
    symbols, parsing = clang_symbols(b"int f();", "unused.cpp", ROOT, driver="tide-no-such-clang")
    assert symbols == [] and parsing["state"] == "unavailable"


def test_cli_is_torch_free_and_refuses_overwriting_a_report(repository, tmp_path):
    output = tmp_path/"inventory.json"
    command = [sys.executable, "-S", str(ROOT/"scripts/review_inventory.py"),
               "--root", str(repository), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    before = output.read_bytes()
    assert json.loads(before)["summary"]["files"] == 2
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "already exists" in second.stderr
    assert output.read_bytes() == before
