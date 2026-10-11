"""Compose child import paths without discarding the activated vendor stack."""
import os


def with_pythonpath(environment, *paths):
    result = dict(environment)
    entries = [os.fspath(path) for path in paths]
    if result.get("PYTHONPATH"):
        entries.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(entries)
    return result
