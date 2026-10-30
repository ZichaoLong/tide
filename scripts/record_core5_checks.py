#!/usr/bin/env python3
"""Extract per-configuration correctness evidence from actual pytest JUnit records.

No benchmark time is synthesized from test duration. Failed/skipped cases stay
failed/skipped; missing complete checks cannot become qualified by exit status.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from core5_study_records import complete_checks
from durable_records import write_json
from source_identity import digest


def extract(path, source):
    cases, models = [], {}
    for test in ET.parse(path).getroot().iter("testcase"):
        properties = {p.attrib["name"]: p.attrib["value"] for p in test.findall("./properties/property")}
        if "core5_case" not in properties:
            continue
        row = json.loads(properties["core5_case"])
        facts = row.pop("facts")
        key = facts["workload_id"]
        if key in models and models[key] != facts:
            raise ValueError("same workload has inconsistent configuration facts")
        models[key] = facts
        row["workload_id"] = key
        state = "failed" if test.find("failure") is not None or test.find("error") is not None else "skipped" if test.find("skipped") is not None else "passed"
        checks = json.loads(properties.get("core5_checks", "{}"))
        row.update(state=state, checks=checks, test=test.attrib["classname"]+"::"+test.attrib["name"])
        row["correctness_qualified"] = complete_checks(state, row["mode"], checks, checks.get("exact_resume"))
        cases.append(row)
    if not cases:
        raise ValueError("no explicit core5 configuration records in JUnit input")
    if len({c["test"] for c in cases}) != len(cases):
        raise ValueError("duplicate test identity")
    return dict(schema="tide-core5-correctness-evidence-v1", source=source, junit_sha256=digest(path),
                scope="exact listed CPU configurations; no performance or untested-backend qualification", models=models, cases=cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--junit", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(a.source) != 40 or any(c not in "0123456789abcdef" for c in a.source):
        p.error("source must be the exact tested commit")
    if a.output.exists():
        p.error("refuse overwriting an evidence file")
    write_json(a.output, extract(a.junit, a.source))


if __name__ == "__main__":
    main()
