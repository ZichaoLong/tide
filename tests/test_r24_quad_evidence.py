"""Resolve the r24 consumer evidence from this checkout's original bytes."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
IMPLEMENTATION = "6e14a6cc90e3ea1972611ee98b9de9bfea3b14fe"


def check_reference(reference, local):
    assert Path(reference["path"]).name == local.name
    assert reference["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()


def test_original_diagnosis_and_delivered_reference():
    original = EVIDENCE / "pilot-r3-r24-quad-total-constraint.json"
    diagnosis = json.loads(original.read_bytes())
    row = diagnosis["observed"]["cases"][0]
    weak = Fraction(row["target"][0]) - Fraction(row["upper"][0])
    assert weak == Fraction(diagnosis["mechanism"]["exact_positive_remainder"]) == Fraction(3, 2**19)
    assert row["c"][1] == 0
    acceptance = json.loads((EVIDENCE / "pilot-r3-r24-foundation-quad-total-acceptance.json").read_bytes())
    check_reference(acceptance["diagnosis"], original)
    check_reference(acceptance["reference"], ROOT / "scripts/r24_quad_reference.py")
    check_reference(acceptance["driver"], ROOT / "scripts/accept_r24_quad.py")
    assert acceptance["authority"]["commit"] == IMPLEMENTATION


def test_consumer_acceptance_and_batch_adoption():
    record = EVIDENCE / "pilot-r3-r24-foundation-quad-total-acceptance.json"
    acceptance = json.loads(record.read_bytes())
    for backend in ("cpu", "npu"):
        trajectory = acceptance["qualification"][backend]
        assert trajectory["capture"]["passed"] and trajectory["capture"]["allocation"]["rows"] == 40_960
        assert trajectory["restored"]["passed"] and trajectory["restored"]["loaded_state_exact"]
        assert trajectory["direct"]["pid"] != trajectory["restored"]["pid"]
    r25 = json.loads((EVIDENCE / "pilot-r3-main-r25-handoff.json").read_bytes())
    r26 = json.loads((EVIDENCE / "pilot-r3-main-r26-handoff.json").read_bytes())
    check_reference(r25["acceptance"], record)
    assert r25["foundation_commit"] == r26["foundation_commit"] == IMPLEMENTATION
    assert r25["fault"]["post_fault"]["step"] > 921
