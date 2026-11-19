"""Finish the existing finite search through its unchanged frozen launcher.

This sole submission owner schedules one 4-, 2-, and 1-device slot. It does not
change model execution, existing observations, memory budgets, or point limits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic
import manage

SLOTS = {
    0: [('17b', 'tide', 4), ('17b', 'tf5', 4), ('17b', 'tf32', 4), ('17b', 'tf48', 4)],
    1: [('17b', 'tf32', 2), ('17b', 'tf48', 2)],
    2: [('9b', 'tf48', 1)],
}


def read(name, default=None):
    path = BASE / name
    return json.loads(path.read_text()) if path.exists() else default


def key(series):
    scale, kind, cards = series
    return f'{scale}/{kind}/{cards}'


def snapshot(submissions):
    rows = []
    for entry in submissions:
        folder = ROOT / 'artifacts' / entry['name']
        state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        rows.append(dict(entry, lifecycle=state.get('state'), result=manage.result(folder / 'inference.json')))
    return rows


def forecast(points, batch):
    # Allocator peaks include the full request. Use the last two distinct
    # successful batch sizes, retaining the larger repeated measurement.
    by_batch = {}
    for row in points:
        if row['lifecycle'] == 'passed':
            values = by_batch.setdefault(row['batch'], {'allocated': 0, 'reserved': 0})
            for name in values:
                values[name] = max(values[name], row['result'][name])
    batches = sorted(by_batch)
    if len(batches) < 2:
        return None
    left, right = batches[-2:]
    return {name: by_batch[right][name] + max(0, (by_batch[right][name] - by_batch[left][name]) / (right - left)) * (batch - right)
            for name in ('allocated', 'reserved')}


def decision(points):
    active = [r for r in points if r['lifecycle'] not in ('passed', 'failed')]
    if active:
        return {'action': 'wait'}
    for row in points:
        if row['lifecycle'] == 'failed' and 'sampled budget' not in str((row['result'] or {}).get('error')):
            raise RuntimeError('non-budget execution failure requires review: ' + row['name'])
    good = [r for r in points if r['lifecycle'] == 'passed']
    screens = [r for r in points if not r['confirm']]
    confirmed = [r for r in good if r['confirm']]
    refused = [r for r in points if r['lifecycle'] == 'failed']
    if not good:
        if points:
            raise RuntimeError('series has no passing anchor')
        return dict(action='submit', batch=8, confirm=False, reason='Common B8 calibration for fixed-request batch search')
    best = max(good, key=lambda r: r['result']['rate'])
    unique = {}
    for row in good:
        if row['batch'] not in unique or row['confirm']:
            unique[row['batch']] = row
    increasing = [unique[b] for b in sorted(unique)]
    low = max(unique)
    upper = min((r['batch'] for r in refused if r['batch'] > low), default=None)
    adjacent = upper is not None and upper == low + 1
    plateau = len(increasing) >= 3 and all(
        increasing[i]['result']['rate'] / increasing[i-1]['result']['rate'] <= 1.02 for i in (-1, -2))
    stop = ('adjacent sampled-budget boundary' if adjacent else 'two consecutive <=2% throughput gains' if plateau
            else 'finite eight-screen limit' if len(screens) >= 8 else None)
    # Existing confirmations already imply the manually reviewed search is at
    # its final stage. Do not restart growth after those formal runs.
    if confirmed or any(r['confirm'] for r in points):
        stop = stop or 'manually selected confirmation stage'
    if stop:
        if not confirmed:
            return dict(action='submit', batch=best['batch'], confirm=True,
                        reason=stop + '; confirm highest observed passing candidate', forecast=forecast(good, best['batch']))
        winner = max(confirmed, key=lambda r: r['result']['rate'])
        alternatives = [r for r in good if r['batch'] not in {c['batch'] for c in confirmed}]
        contender = max(alternatives, key=lambda r: r['result']['rate']) if alternatives else None
        if len(confirmed) < 2 and contender and contender['result']['rate'] >= winner['result']['rate'] * .95:
            return dict(action='submit', batch=contender['batch'], confirm=True,
                        reason='Confirm nearby throughput competitor within5% of formal winner',
                        forecast=forecast(good, contender['batch']))
        return dict(action='finish', winner=winner['name'], passing_batch=low, rejected_batch=upper,
                    stop_reason=stop, tested_batches=sorted({r['batch'] for r in points}),
                    confirmations=[r['name'] for r in confirmed],
                    claim='best observed confirmed throughput within fixed request/model and48/56GiB per-card budgets; not a global or exhaustive integer-batch optimum')
    if upper is not None:
        batch = (low + upper) // 2
        reason = 'Refine measured passing/refused batch bracket without hardware OOM'
    elif len(unique) == 1:
        batch = min(4096, low * 4)
        reason = 'Geometric growth from common B8 calibration; at most4x'
    else:
        previous = sorted(unique)[-2]
        left, right = unique[previous]['result'], unique[low]['result']
        bounds = []
        for name, limit in [('allocated', 48), ('reserved', 56)]:
            slope = (right[name] - left[name]) / (low - previous)
            if slope > 0:
                bounds.append(low + (limit - right[name]) / slope)
        estimate = math.floor(min(bounds)) if bounds else low * 4
        batch = min(4096, low * 4, max(low + 1, estimate))
        reason = 'Linear peak-memory forecast from latest two batches; at most4x growth; adjacent boundary probe allowed'
    predicted = forecast(good, batch)
    if predicted and (predicted['allocated'] >= 54 or predicted['reserved'] >= 60):
        raise RuntimeError('forecast exceeds explicit boundary-probe safety bound')
    if batch in {r['batch'] for r in points}:
        raise RuntimeError('search proposed an already tested batch')
    return dict(action='submit', batch=batch, confirm=False, reason=reason, forecast=predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    preflight = read('preflight.json')
    deadline = datetime.fromisoformat(preflight['created_at_utc']).timestamp() + preflight['plan']['maximum_project_wall_seconds']
    manager_hash = hashlib.sha256((BASE / 'manage.py').read_bytes()).hexdigest()
    events = read('controller-decisions.json', [])
    while True:
        if time.time() >= deadline - 120:
            raise RuntimeError('finite project deadline reached; inspect active job identities before any further work')
        assert hashlib.sha256((BASE / 'manage.py').read_bytes()).hexdigest() == manager_hash
        rows = snapshot(read('submissions.json'))
        done = read('series-decisions.json', {})
        finished = True
        for slot, scheduled in SLOTS.items():
            pending = [s for s in scheduled if done.get(key(s), {}).get('state') != 'finished']
            if not pending:
                continue
            finished = False
            series = pending[0]
            scale, kind, cards = series
            occupied = [r for r in rows if r['slot'] == slot and r['lifecycle'] not in ('passed', 'failed')]
            if occupied:
                continue
            points = [r for r in rows if (r['scale'], r['kind'], r['cards']) == series]
            action = decision(points)
            if action['action'] == 'wait':
                continue
            event = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), series=key(series), slot=slot, **action)
            print(json.dumps(event), flush=True)
            if args.dry_run:
                continue
            events.append(event); atomic(BASE / 'controller-decisions.json', events)
            if action['action'] == 'finish':
                done[key(series)] = dict(state='finished', **event)
                atomic(BASE / 'series-decisions.json', done)
                continue
            argv = [sys.executable, str(BASE / 'manage.py'), 'submit', kind, scale, str(cards), str(action['batch']),
                    '--slot', str(slot), '--reason', action['reason']]
            if action['confirm']:
                argv.append('--confirm')
            if action.get('forecast'):
                argv += ['--allocated-forecast', str(action['forecast']['allocated']),
                         '--reserved-forecast', str(action['forecast']['reserved'])]
            subprocess.run(argv, cwd=ROOT, check=True)
        if args.dry_run or finished:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
