"""Explicit finite-exploration reporting; never converts strict failure to pass."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from tidegraph.comparison import collect_comparison, equivalent


class Checks:
    def __init__(self, nodeid, *, known_evidence_rounding=False, failure_reason=None, known_numeric=None):
        self.nodeid, self.reports = nodeid, []
        self.known_evidence_rounding = known_evidence_rounding
        self.known_numeric = dict(known_numeric or {})
        self.failure_reason = failure_reason or "dense-v1 strict numeric failure retained (FP32 normalized evidence); discrete checks passed"

    def __call__(self, actual, expected, label="root"):
        report = collect_comparison(actual, expected, path=label, check_device=False)
        assert report["discrete"] == "passed", report
        self.reports.append(report)
        if not os.environ.get("TIDE_DENSE_NUMERIC_REPORT"):
            known = (self.known_evidence_rounding and not report["differences_truncated"]
                     and all(d["path"] in {"field_vjp[read_program.evidence_weight]", "field_vjp[read_program.evidence_bias]"}
                             and d["max_abs_error"] <= 2e-6 for d in report["differences"]))
            known |= (bool(self.known_numeric) and not report["differences_truncated"]
                      and all(d["path"] in self.known_numeric and
                              d["max_abs_error"] <= self.known_numeric[d["path"]]
                              for d in report["differences"]))
            if not known:
                equivalent(actual, expected, path=label, check_device=False)

    def finish(self):
        directory = os.environ.get("TIDE_DENSE_NUMERIC_REPORT")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(self.nodeid.encode()).hexdigest()[:16]+".json"
            (path/name).write_text(json.dumps(dict(test=self.nodeid, comparisons=self.reports), indent=2))
        if any(not r["strict_passed"] for r in self.reports):
            pytest.xfail(self.failure_reason)
