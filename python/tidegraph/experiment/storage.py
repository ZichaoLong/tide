"""Linux output-filesystem and shared host/tmpfs resource observations."""
from pathlib import Path
import re
import shutil


def filesystem(path, *, mountinfo="/proc/self/mountinfo"):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    usage = shutil.disk_usage(path)
    result = dict(path=str(path), total_bytes=usage.total, used_bytes=usage.used,
                  free_bytes=usage.free, filesystem="unknown", memory_backed=None)
    table = Path(mountinfo)
    if table.is_file():
        matches = []
        for line in table.read_text().splitlines():
            before, after = line.split(" - ", 1)
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), before.split()[4]))
            if mount == path or mount in path.parents:
                matches.append((len(mount.parts), str(mount), after.split()[0]))
        if matches:
            _, mount, kind = max(matches)
            result.update(mount=mount, filesystem=kind, memory_backed=kind in {"tmpfs", "ramfs"})
    return result


def host_memory():
    values = {row.split(":")[0]: int(row.split()[1])*1024
              for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith(("MemAvailable:", "Shmem:"))}
    result = dict(available_bytes=values["MemAvailable"], system_shmem_bytes=values.get("Shmem"),
                  scope="available host memory capped by cgroup v2 remaining; shmem is already included in cgroup usage")
    root = Path("/sys/fs/cgroup")
    for name, filename in (("cgroup_limit_bytes", "memory.max"), ("cgroup_current_bytes", "memory.current"),
                           ("cgroup_peak_bytes", "memory.peak")):
        file = root/filename
        if file.is_file():
            value = file.read_text().strip()
            if value != "max":
                result[name] = int(value)
    if "cgroup_limit_bytes" in result and "cgroup_current_bytes" in result:
        result["available_bytes"] = max(0, min(result["available_bytes"],
            result["cgroup_limit_bytes"]-result["cgroup_current_bytes"]))
    return result
