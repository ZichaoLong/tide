# Durable job records and re-entry after damaged metadata

Clean source: 3604ec002722e701c74bd13e6f681b88ada14199.
16 tests passed / 0.44s, exit 0. CPU aarch64 Linux, Python 3.11.15;
these are dependency-free control-record checks, not graph numerical tests.
The pytest session also loads the ordinary Torch 2.10.0+cpu test environment.

Unit tide-foundation-records-qualified-20260922-0115 ended inactive/dead,
MainPID 0, Result=success, exit 0. Started 2026-09-22T01:13:17Z, finished
01:13:20Z. The clean main source remained unchanged for this short job; its
tracked files were compared byte-for-byte with git archive after exit. The
concurrent graph CPU qualification uses a different, frozen source/worktree.

```sh
python scripts/job.py --output-dir artifacts/records-qualified-20260922-0115 -- \
  python -m pytest tests/test_durable_records.py -q --dtype both
```

Working directory /home/zlong/llm/graph-execution-foundation; resolved interpreter
/home/zlong/anaconda3/bin/python. Service used background.slice, Nice=10,
OMP/OpenBLAS=1 and TORCH_DEVICE_BACKEND_AUTOLOAD=0. Persistent records:
artifacts/records-qualified-20260922-0115/{status.json,task.log,post-run-audit.json}.

## Covered failures and behavior

- Eight new/existing-target cases inject serialization, partial write,
  file-fsync and rename failure. The previous record survives, no incomplete
  target is published, staging is cleaned, and retry succeeds.
- Replacement observes complete staged content and the previous published
  record. Directory-fsync failure after replacement remains an error although
  the complete new record already exists.
- Two real launcher subprocesses preserve success/exit 0 and failure/exit 7,
  capture workload logs, and reject duplicate output without overwriting it.
- Four status-command fixtures retain healthy jobs and the current handoff while
  reporting malformed JSON, a non-object, missing metadata or inconsistent
  terminal state as unknown; the status command exits nonzero.
- One explicit terminal postmortem retains failed/exit 1 and its observed time
  without inventing a workload start timestamp.

The initial strict-reader misclassification and corrected output are retained
in artifacts/durable-records-postmortem-repro/. The original disk-full launcher
failure at artifacts/single-dev-20260921-2054/status.json remains unchanged.
The shared helper is used by job.py, develop.py and verify.py for their status/
result records. It requires coordinated ownership; this gate is neither a
power-loss test nor a concurrent record-writer protocol. Linux filesystem fsync
and atomic same-directory replacement are the declared persistence boundary.
