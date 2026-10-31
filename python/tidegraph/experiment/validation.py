"""Independent scalar replay after candidate timing; never supplies a trace."""
import torch
from ..comparison import collect_comparison, NumericMismatch
from .eager import Eager


def _merge(reports):
    numeric = any(x["numeric"] == "failed" for x in reports)
    discrete = any(x["discrete"] == "failed" for x in reports)
    return dict(finite="passed", numeric="not_comparable" if discrete else "failed" if numeric else "passed",
                discrete="failed" if discrete else "passed", strict_passed=not (numeric or discrete),
                reports=reports)


def independent(resolved, actual, parameters, training, start, *, task_state=None):
    reference = Eager(resolved, reference=True)
    from ..planning.precision import tolerances
    dtype = getattr(torch, resolved["execution"]["effective"]["dtype"])
    atol, rtol = tolerances(dtype)
    def compare(a, b, path):
        return collect_comparison(a, b, path=path, check_device=False, atol=atol, rtol=rtol)
    reports = []
    try:
        for index in range(start+len(actual)):
            expected, loss = reference.step(index, return_mode="complete")
            if index >= start:
                value, actual_loss = actual[index-start]
                if reference.storage_reference:
                    from ..qualification_weighted_precision import assert_records, widen
                    try:
                        assert_records(value, expected, dtype)
                    except NumericMismatch:
                        pass  # Only the complete collecting comparison decides.
                    value = widen(value)
                reports.append(compare(value, expected, f"window[{index}]"))
                reports.append(compare(torch.tensor(actual_loss, dtype=dtype), torch.tensor(loss, dtype=dtype), f"loss[{index}]"))
        if resolved["measurement"]["mode"] == "training":
            reports.append(compare(parameters, reference.parameters(), "parameters"))
            reports.append(compare(training, reference.training_state(), "training"))
        if task_state is not None:
            reports.append(compare(task_state, reference.task_state(), "task-state"))
    finally:
        reference.close()
    return dict(_merge(reports), complete="checked", reference="independent Python scalar CPU schedule; seeded prefix replay")


def outcome(report, policy):
    if policy == "none":
        return "not_checked"
    if report["strict_passed"]:
        return "passed"
    if policy == "explore" and report["numeric"] == "failed" and report["discrete"] == "passed":
        return "exploratory_accepted"
    return "failed"
