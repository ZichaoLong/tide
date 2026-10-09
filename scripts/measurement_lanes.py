"""Admission and observations for a bounded group of NUMA-separated workloads.

The controller applies the project's half-resource policy once to the whole
group. A CPU mask is a hard limit on this group's CPU access, not an exclusive
reservation against other users. Node memory estimates are not kernel guarantees.
"""
from pathlib import Path
import re

from foundation_resources import cpu_list, resources


def node_memory(root=Path('/sys/devices/system/node')):
    result = {}
    for path in sorted(root.glob('node[0-9]*')):
        fields = {}
        for line in (path / 'meminfo').read_text().splitlines():
            parts = line.split()
            fields[parts[2].removesuffix(':')] = int(parts[3]) * 1024
        # Do not count active file pages or reclaimable slab. Discount inactive
        # clean file pages and leave a fixed local reserve; never call this the
        # kernel's per-node MemAvailable (Linux does not provide that here).
        clean = max(0, fields['Inactive(file)'] - fields['Dirty'] - fields['Writeback'])
        estimate = max(0, fields['MemFree'] + clean * 3 // 4 - 2 * 1024**3)
        result[int(path.name[4:])] = dict(
            cpus=sorted(cpu_list((path / 'cpulist').read_text())),
            total_bytes=fields['MemTotal'], free_bytes=fields['MemFree'],
            clean_inactive_file_bytes=clean,
            available_estimate_bytes=min(estimate, fields['MemTotal']))
    return result


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def identifiers(values, name):
    if (not isinstance(values, list) or not values or
            any(type(v) is not int or v < 0 for v in values) or
            len(values) != len(set(values))):
        raise ValueError(name + ' must contain distinct nonnegative integers')
    return set(values)


def admit(lanes, *, reserve_bytes, host=None, nodes=None, controller_cpu_slots=2):
    """Admit one group before any child starts; refuse partial/overlapping masks."""
    positive(reserve_bytes, 'reserve_bytes')
    positive(controller_cpu_slots, 'controller_cpu_slots')
    if not lanes:
        raise ValueError('at least one lane is required')
    host = resources() if host is None else host
    nodes = node_memory() if nodes is None else nodes
    used_cpus, used_nodes, names = set(), set(), set()
    memory = reserve_bytes
    for lane in lanes:
        name = lane['name']
        if (not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name)
                or name in names):
            raise ValueError('invalid or duplicate lane name')
        names.add(name)
        cpus = identifiers(lane['cpus'], 'cpus')
        selected = identifiers(lane['memory_nodes'], 'memory_nodes')
        amount = positive(lane['memory_bytes'], 'memory_bytes')
        if not cpus <= set(host['effective_cpu_ids']):
            raise ValueError('CPU mask exceeds effective affinity/cpuset')
        if not selected <= nodes.keys():
            raise ValueError('unknown memory node')
        local_cpus = {cpu for n in selected for cpu in nodes[n]['cpus']}
        if not cpus <= local_cpus:
            raise ValueError('CPU mask is outside selected NUMA nodes')
        if cpus & used_cpus or selected & used_nodes:
            raise ValueError('measurement lanes overlap CPUs or memory nodes')
        if amount > sum(nodes[n]['available_estimate_bytes'] for n in selected):
            raise MemoryError('lane exceeds selected-node memory estimate: ' + name)
        memory += amount
        used_cpus |= cpus
        used_nodes |= selected
    # Count allowed cores, including runtime/compiler threads, rather than just
    # the nominal ATen thread count. All children inherit the bounded CPU mask.
    if len(used_cpus) + controller_cpu_slots > host['cpu_budget']:
        raise ValueError('combined CPU masks exceed aggregate half-resource budget')
    if memory > host['memory_budget_bytes']:
        raise MemoryError('combined lanes plus shared reserve exceed aggregate memory budget')
    return dict(host=host, nodes=nodes, lanes=lanes,
                reserved_cpu_count=len(used_cpus) + controller_cpu_slots,
                controller_cpu_slots=controller_cpu_slots, reserved_memory_bytes=memory,
                shared_reserve_bytes=reserve_bytes,
                node_estimate_policy='MemFree + 0.75 * clean Inactive(file) - 2GiB/node',
                limits='Cooperating children only; external load and shared file pages remain uncontrolled.')


def numa_command(command, lane, executable='numactl'):
    return [executable, '--physcpubind=' + ','.join(map(str, lane['cpus'])),
            '--membind=' + ','.join(map(str, lane['memory_nodes'])), *command]


def process_placement(pid, proc=Path('/proc')):
    """Observe every live thread mask and private anonymous NUMA mappings.

    Shared file-backed pages are explicitly excluded: mbind does not relocate
    an existing shared library/page-cache page. Procfs can report either ENOENT
    or ESRCH when a process/thread exits during a read. Other errors propagate.
    """
    path = proc / str(pid)
    masks = []
    try:
        tasks = list((path / 'task').iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return None
    for task in tasks:
        try:
            status = (task / 'status').read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        value = next(line.split(':', 1)[1] for line in status.splitlines()
                     if line.startswith('Cpus_allowed_list:'))
        masks.append(sorted(cpu_list(value)))
    pages, policies = {}, set()
    try:
        maps = (path / 'numa_maps').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in maps.splitlines():
        fields = line.split()
        if 'file=' in line or not any(f.startswith('anon=') for f in fields):
            continue
        policies.add(fields[1])
        for field in fields:
            if re.fullmatch(r'N\d+=\d+', field):
                key, value = field.split('=')
                node = int(key[1:])
                pages[node] = pages.get(node, 0) + int(value)
    return dict(pid=pid, thread_count=len(masks),
                thread_cpu_masks=sorted({tuple(mask) for mask in masks}),
                anonymous_pages_by_node=pages, anonymous_policies=sorted(policies))


def check_placement(observation, lane):
    if observation is None:
        return
    allowed = set(lane['cpus'])
    if any(not set(mask) <= allowed for mask in observation['thread_cpu_masks']):
        raise RuntimeError('child thread escaped CPU lane: ' + lane['name'])
    if any(count and node not in lane['memory_nodes']
           for node, count in observation['anonymous_pages_by_node'].items()):
        raise RuntimeError('private anonymous pages escaped NUMA lane: ' + lane['name'])
