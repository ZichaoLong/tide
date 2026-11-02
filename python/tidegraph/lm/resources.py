"""Account quota and project disk admission, independent of model allocation."""
from pathlib import Path
import re
import shutil
import subprocess

GIB = 1024**3


def quota_used():
    result = subprocess.run(["quota", "-w"], capture_output=True, text=True, timeout=15)
    values = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("/") and len(fields) >= 4:
            if re.fullmatch(r"[0-9]+\*?", fields[1]):
                values.append(int(fields[1].rstrip("*"))*1024)
    if not values:
        raise RuntimeError("quota: cannot read account usage; inspect quota -vs before writing")
    return max(values)


def directory_bytes(root):
    result = subprocess.run(["du", "-sb", str(root)], capture_output=True, text=True, check=True, timeout=60)
    return int(result.stdout.split()[0])


def disk_admission(root, additional, *, project_gib, account_gib=300):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    used, quota = directory_bytes(root), quota_used()
    free = shutil.disk_usage(root).free
    if min(additional, project_gib, account_gib) < 0:
        raise ValueError("disk budget values must be nonnegative")
    if used+additional > project_gib*GIB:
        raise RuntimeError(f"project disk budget: {used}+{additional} > {project_gib} GiB")
    if quota+additional > account_gib*GIB:
        raise RuntimeError(f"account stop threshold: {quota}+{additional} > {account_gib} GiB")
    if free < additional+GIB:
        raise RuntimeError("filesystem lacks requested space plus 1 GiB reserve")
    return dict(project_bytes=used, account_bytes=quota, filesystem_free_bytes=free)
