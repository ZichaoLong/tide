"""Reconcile terminal attempts, including failures with no domain audit.

Accepted measurements come only from the domain auditor. Queue completion or a
successful process receipt alone cannot promote a measurement to accepted.
Callers retain raw artifacts and pass a separately validated over-bound flag.
"""


def failure_kind(attempt):
    if attempt['state'] == 'not-started':
        return 'not-started'
    if attempt.get('completed_over_bound'):
        if attempt.get('model_executed') is not True or attempt['state'] != 'failed':
            raise ValueError('over-bound observation must be a completed failed attempt')
        return 'completed-over-bound'
    if attempt.get('queue_status') == 'timed_out' and not attempt.get('model_executed'):
        return 'npu-admission-timeout'
    error = attempt.get('error', '')
    if (attempt.get('model_executed') is False and
            'lane exceeds selected-node memory estimate' in error):
        return 'cpu-memory-admission'
    if 'ProcessLookupError' in error and attempt.get('proc_exit_race'):
        return 'monitor-process-exit-race'
    if 'lane time bound exceeded' in error:
        return 'execution-timeout'
    if attempt.get('npu_oom'):
        return 'npu-oom'
    return 'unclassified-failure'


def reconcile(total, accepted, attempts):
    """Keep failed history even if another attempt of the cell later passes.

    Inputs refer to a single measurement series. Deduplicate repeated references
    to the same attempt, never pool series or count failed complete observations
    as formally accepted. Unknown failures remain visible for review.
    """
    if type(total) is not int or total <= 0:
        raise ValueError('positive cell count required')
    accepted = set(accepted)
    if any(type(cell) is not int or not 0 <= cell < total for cell in accepted):
        raise ValueError('invalid accepted cell')
    unique = {}
    for row in attempts:
        if (type(row['cell']) is not int or not 0 <= row['cell'] < total or
                row['state'] not in ('passed', 'failed', 'not-started')):
            raise ValueError('invalid terminal attempt')
        key = row['run'], row['cell']
        if key in unique and unique[key] != row:
            raise ValueError('conflicting terminal records for one attempt')
        unique[key] = row
    retained = [dict(row, category=failure_kind(row)) for row in unique.values()
                if row['state'] != 'passed']
    cells = []
    for cell in range(total):
        history = sorted((r for r in retained if r['cell'] == cell),
                         key=lambda r: (r['finished'], r['run']))
        category = ('accepted' if cell in accepted else
                    history[-1]['category'] if history else 'not-measured')
        cells.append(dict(cell=cell, category=category,
                          attempts=[r['run'] for r in history]))
    counts = {}
    for row in cells:
        counts[row['category']] = counts.get(row['category'], 0) + 1
    return dict(accepted_cells=sorted(accepted), category_counts=counts,
                cell_outcomes=cells, retained_failures=retained)
