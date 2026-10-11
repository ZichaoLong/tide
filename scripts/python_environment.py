"""Compose child import paths without discarding the activated vendor stack."""
from contextlib import contextmanager
import os
from pathlib import Path


def with_pythonpath(environment, *paths):
    result = dict(environment)
    entries = [os.fspath(path) for path in paths]
    if result.get("PYTHONPATH"):
        entries.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(entries)
    return result


@contextmanager
def working_directory(path):
    """Keep vendor-generated files with run artifacts, including on failure."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
