"""Allocated-byte observation of live outputs, tolerant only of disappearance.

Atomic record/checkpoint replacement is not a disk-limit failure. Permission,
I/O and other scan errors still fail closed. The authoritative account quota
is checked independently; a traversal is not a filesystem snapshot.
"""
import os
import stat


def allocated_usage(paths):
    stack=list(map(os.fspath,paths));seen=set();size=vanished=entries=0
    while stack:
        path=stack.pop()
        try:
            info=os.stat(path,follow_symlinks=False)
        except FileNotFoundError:
            vanished+=1
            continue
        identity=(info.st_dev,info.st_ino)
        if identity in seen:
            continue
        seen.add(identity);size+=info.st_blocks*512;entries+=1
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as directory:
                    stack.extend(entry.path for entry in directory)
            except FileNotFoundError:
                vanished+=1
    return dict(bytes=size,gib=size/2**30,entries=entries,vanished=vanished,
                scope='allocated bytes, hardlinks once, no symlink traversal; live observation, not atomic snapshot')


def campaign_usage(root,pattern='v12-*'):
    return allocated_usage(p for base in ('runs','artifacts/jobs') for p in (root/base).glob(pattern))
