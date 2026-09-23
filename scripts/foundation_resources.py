"""Linux resource discovery for one aggregate bounded foundation task.

Limits include every ancestor, not just the transient service. Memory usage can
be shared with unrelated account workloads; it is recorded as such, not as RSS.
"""
import math
import os
from pathlib import Path


def cpu_list(text):
    result = set()
    for field in text.strip().split(','):
        if not field: continue
        pair = field.split('-'); a = int(pair[0]); b = int(pair[-1])
        result.update(range(a,b+1))
    return result


def read(path):
    try: return path.read_text().strip()
    except OSError: return None


def resources():
    affinity = set(os.sched_getaffinity(0)); available = set(affinity)
    quota, memory_limits, memory_remaining, groups = [], [], [], []
    controllers = {}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, names, suffix = line.split(':',2)
        for name in names.split(','): controllers[name] = suffix
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - '); fields = left.split(); kind, _, options = right.split()[:3]
        if kind in {'cgroup','cgroup2'}: mounts.append((Path(fields[4]), fields[3], kind, options.split(',')))
    for mount, mount_root, kind, names in mounts:
        keys = [''] if kind == 'cgroup2' else [n for n in names if n in controllers]
        if not keys: continue
        relative = Path(controllers[keys[0]])
        try: relative = relative.relative_to(mount_root)
        except ValueError: continue
        current = mount/relative
        while current == mount or mount in current.parents:
            row = {'path':str(current),'version':2 if kind == 'cgroup2' else 1}
            cpus = read(current/('cpuset.cpus.effective' if kind == 'cgroup2' else 'cpuset.cpus'))
            if cpus: available &= cpu_list(cpus); row['cpuset'] = cpus
            if kind == 'cgroup2':
                limit = read(current/'cpu.max')
                if limit:
                    amount, period = limit.split()
                    if amount != 'max': quota.append(int(amount)/int(period)); row['cpu_quota'] = quota[-1]
                limit, used = read(current/'memory.max'), read(current/'memory.current')
            else:
                amount, period = read(current/'cpu.cfs_quota_us'), read(current/'cpu.cfs_period_us')
                if amount and period and int(amount) > 0:
                    quota.append(int(amount)/int(period)); row['cpu_quota'] = quota[-1]
                limit, used = read(current/'memory.limit_in_bytes'), read(current/'memory.usage_in_bytes')
            if limit and limit != 'max' and int(limit) < 2**60:
                memory_limits.append(int(limit)); row['memory_limit_bytes'] = int(limit)
                if used:
                    memory_remaining.append(max(0,int(limit)-int(used))); row['memory_used_bytes'] = int(used)
            elif used: row['memory_used_bytes'] = int(used)
            if len(row) > 2: groups.append(row)
            if current == mount: break
            current = current.parent
    if not available: raise RuntimeError('no effective CPUs')
    info = {k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    effective_cpus = min([len(available)]+quota)
    effective_memory = min([info['MemTotal'],info['MemAvailable']]+memory_limits+memory_remaining)
    physical = set()
    for cpu in available:
        base = Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        physical.add((read(base/'physical_package_id'),read(base/'core_id')))
    numa = {p.parent.name:sorted(cpu_list(p.read_text()) & available) for p in Path('/sys/devices/system/node').glob('node*/cpulist')}
    return {'affinity':sorted(affinity),'effective_cpu_ids':sorted(available),'effective_cpu_count':effective_cpus,
            'physical_cores':len(physical),'numa':numa,'cpu_budget':max(1,math.floor(effective_cpus/2)),
            'host_total_bytes':info['MemTotal'],'host_available_bytes':info['MemAvailable'],
            'effective_memory_bytes':effective_memory,'memory_budget_bytes':effective_memory//2,
            'cgroups':groups,'basis':'half min(affinity intersect cpuset, ancestor CPU quotas); half min(host total, MemAvailable, ancestor memory limits and remaining)',
            'scope':'combined task budget; cgroup memory may include other account workloads'}


def balanced_affinity(record, count=None):
    count = record['cpu_budget'] if count is None else count
    if count < 1 or count > record['cpu_budget']: raise ValueError('CPU request exceeds aggregate task budget')
    groups = [list(v) for v in record['numa'].values() if v] or [record['effective_cpu_ids'].copy()]
    selected = []
    while len(selected) < count:
        for group in groups:
            if group and len(selected) < count: selected.append(group.pop(0))
    return sorted(selected)


if __name__ == '__main__':
    import json
    print(json.dumps(resources(),indent=2))
