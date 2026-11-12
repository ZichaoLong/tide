"""Read this account's single quota filesystem, including soft-limit grace."""
import os
import subprocess
import time


def parse_quota(stdout, returncode, *, at=None):
    # quota exits 1 for an exceeded soft limit even while writes remain allowed.
    # -p gives epoch seconds, including explicit zero grace fields.
    if returncode not in (0, 1):
        raise RuntimeError(f'quota query failed with exit {returncode}')
    rows = [line.split() for line in stdout.splitlines() if line.lstrip().startswith('/dev/')]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise RuntimeError('cannot unambiguously decode account quota -w -p')
    row = rows[0]
    try:
        used, soft, hard, grace, files, file_soft, file_hard, file_grace = (
            int(value.rstrip('*')) for value in row[1:])
    except ValueError as error:
        raise RuntimeError('invalid quota counters') from error
    if min(used, soft, hard, grace, files, file_soft, file_hard, file_grace) < 0 or hard == 0:
        raise RuntimeError('invalid or unavailable hard quota')
    at = time.time() if at is None else at
    for value, limit, expires, label in ((used, soft, grace, 'blocks'), (files, file_soft, file_grace, 'files')):
        if limit and value > limit and (not expires or at >= expires):
            raise RuntimeError(f'{label} soft-quota grace expired or unknown')
    if file_hard and files >= file_hard:
        raise RuntimeError('file hard quota reached')
    return dict(filesystem=row[0],used_gib=used/2**20,soft_gib=soft/2**20,
                hard_gib=hard/2**20,hard_quota_headroom_gib=(hard-used)/2**20,
                soft_exceeded=bool(soft and used>soft),grace_expires_epoch=grace,
                files=files,file_soft=file_soft,file_hard=file_hard,
                file_grace_expires_epoch=file_grace,query_returncode=returncode)


def account_quota():
    reply = subprocess.run(['quota','-w','-p'],capture_output=True,text=True,timeout=20,
                           env={**os.environ,'LC_ALL':'C'})
    if reply.stderr.strip():
        raise RuntimeError(f'quota query diagnostic: {reply.stderr.strip()}')
    return parse_quota(reply.stdout,reply.returncode)
