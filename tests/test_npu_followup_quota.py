"""The site follow-up fails closed when actual user quota cannot be inspected."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("npu_followup", Path(__file__).resolve().parents[1] / "scripts/run_npu_eight_followup.py")
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


def test_quota_uses_account_blocks_including_soft_limit_marker(monkeypatch):
    text = "Filesystem blocks quota limit grace files quota limit grace\n/dev/test 666494184* 419430400 838860800 6days 563706 0 0\n"
    monkeypatch.setattr(followup.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=text, stderr=""))
    assert followup.quota_blocks() == [dict(filesystem="/dev/test", used_kib=666494184,
                                           soft_kib=419430400, hard_kib=838860800)]


@pytest.mark.parametrize("code,text", [(0, ""), (2, "/dev/test 1 2 3")])
def test_unreadable_quota_never_substitutes_filesystem_free(monkeypatch, code, text):
    monkeypatch.setattr(followup.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stdout=text, stderr="unavailable"))
    with pytest.raises(RuntimeError, match="quota"):
        followup.quota_blocks()
