"""Explicit device-memory estimates; allocation errors are never waived."""
import sys


def assess(config, budget, backend, *, observer=None):
    execution = config['execution']
    policy = execution.get('device_memory_admission', 'strict')
    rows = []
    for index in range(execution['devices'] if backend is not None else 0):
        free, total = backend.mem_get_info(index)
        reserve = max(config['resources']['device_reserve_gib'] * 1024**3, total * .10)
        estimate = budget['per_device'][index]['estimated_bytes']
        within = estimate <= free - reserve
        rows.append(dict(device=index, free=free, total=total, reserve=reserve,
                         estimated_bytes=estimate, available_after_reserve=free-reserve,
                         within_budget=within, policy=policy,
                         decision='within_budget' if within else
                         ('observe_exceeded' if policy == 'observe' else 'rejected')))
    # Persist before allocation, including when allocation later fails.
    if observer is not None:
        observer(rows)
    for row in rows:
        if row['within_budget']:
            continue
        message = (f"device {row['device']}, estimate={row['estimated_bytes']}, "
                   f"free={row['free']}, reserve={row['reserve']}")
        if policy != 'observe':
            raise RuntimeError('capacity rejected before model allocation: ' + message)
        print('device_memory_admission=observe: static budget exceeded: ' + message +
              '; attempting allocation; allocation/runtime errors remain failures', file=sys.stderr, flush=True)
    return rows
