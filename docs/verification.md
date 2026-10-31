# Development gates and immutable qualification

The complete regression gate remains CPU FP64/FP32. Accelerator gates and
installed consumers are specified in [accelerators.md](accelerators.md);
compilation alone is not accelerator execution evidence. Use the matching Torch Python for scripts
and LibTorch discovery, explicit ATen/BLAS threads and all affinity-available build workers by default. The active
source, unit, output paths and exact next commands belong in `STATUS.md`.

## Gate scopes

| Entry point | Scope |
| --- | --- |
| `check_downstream_consumer.py --build-dir MATCHING_CPU_BUILD --output-dir NEW` | Fixed-source directed CPU installed core-5 consumer: lock/checkpoint rejection, independent records/gradients/optimizer and fresh-process application resume; not a full backend gate |
| `develop.py TESTS...` | Archive dirty source, build, run explicitly selected pytest paths |
| `develop_lh.py --component iocortex --scope smoke` | Archive source, run six IOCortex configurations per dtype with original assertions on and off; no exports |
| `develop_lh.py --component iocortex` | Full IOCortex matrix, both assertion variants, complete fixtures and independent Python comparison |
| `qualify.py --lh-snapshot SNAPSHOT` | Full CPU regression and all original LH component gates, including IOCortex/Python |
| `build_device_control.py --checks CHECKS...` | Build only the named standalone device components; follow with `verify_device_control.py --checks CHECKS...` |

The device build defaults to the full backend. Its optional component subset
records only binaries actually built, runs the applicable host CTests, and does
not certify unbuilt components, public clients or the complete graph matrix.

`full` remains the default; the clean qualification entry point has no smoke
switch. The six smoke configurations collectively include all six pooling
profiles, both clear/lead settings, three original execution modes and three
native schedules. Every selected configuration still runs actual `think` and
ragged `think_single_step`, complete state/message comparisons and every-cut
single-PDG checks. It is not the Cartesian full matrix or a qualification claim.
The assertions-on FP64 diagnostic limitation is avoided in smoke by selecting
original single mode for active-softmax; full qualification retains and reports
the 24 unavailable configurations, and tests them in the assertions-off build.

`check_lh_selector.py` records scope and exact commands in its manifest. Smoke
exports no fixtures. `check_lh_iocortex_python.py` requires an explicitly full,
passed gate, correct source and binary fingerprints, and the exact hashed
fixture inventory. A partial or old unscoped manifest is rejected.

## Source and result lifecycle

1. Use short directed tests while editing. Preserve a failed reproducer and its
   source identity. Commit a coherent implementation after its directed gate.
2. Create a detached worktree at that exact commit for long qualification. Give
   it its own build directory and a new durable output directory. Pass the
   immutable original LH snapshot by absolute path. Do not share mutable build
   caches with a concurrent development or qualification job.
3. Mark tracked worktree files read-only and leave that checkout unchanged until
   termination. Run `scripts/job.py --output-dir ... -- python scripts/qualify.py
   --output-dir ... --lh-snapshot ...` through the documented detached user
   service, in `background.slice`, with `Nice=10` and explicit environment.
4. Confirm the live service's PID, slice/control group and persistent running
   record. Record its exact source/worktree/command in STATUS; development may
   continue on the main branch because the job reads only the frozen worktree.
5. Inspect terminal service status, workload exit code and every expected result.
   Check source identity, clean state, inventories and acceptance counts. Commit
   reviewed evidence separately. A submitted or live job has no pass result.

Preserve active inputs, unsaved changes and minimal reproducers for new issues.
The 2026-10-12 user authorization permits retiring historical raw artifacts
after necessary records are retained; follow [retention](artifact-retention.md).
Historical absolute paths need not remain online and do not gate new development.
Inspect cleanup dry runs; never clean reference repositories or toolchains.
Project `build`/`artifacts` may be symlinks; check their targets and disk capacity.

## Durable record failures

scripts/durable_records.py centralizes same-directory staging, file flush/fsync,
atomic replacement and directory fsync for job status, development results and
CPU verification manifests. Before replacement, a failed serialization/write/
file-fsync/rename preserves the previous complete value. A directory-fsync error
after replacement is still reported, even though the new complete file may exist.
This does not certify behavior under actual power loss; orphan staging files can
remain after a killed process. Record writers require one coordinated owner.

The status command reads current `artifacts/jobs/NAME/status.json` and legacy
`artifacts/NAME/status.json` records. It does not recursively scan frozen worktrees
or archived copies. It continues to show healthy jobs if another status.json is
unreadable or inconsistent, reports that record's state as unknown and exits
nonzero. An explicit terminal systemd postmortem can use its observed timestamp
without inventing a workload start. Historical failures remain failed. Always
cross-check unit termination, workload exit and acceptance artifacts; a running
record left by an interrupted terminal write cannot establish success.

Fault-injection and real launcher/re-entry tests: tests/test_durable_records.py;
clean-source [evidence](evidence/durable-records.md).
These dependency-free control tools do not change graph execution or checkpoint
value semantics. Qualification for this later control change is recorded
separately from earlier frozen CPU worktrees in STATUS.

## 历史清理的定向验证

[迁移契约](history-migration.md)和`verify_history_mapping.py`核验完整旧/新Git对象，
`tests/test_history_mapping.py`用独立构造的分叉/合并历史注入错误；
`tests/test_retired_record_reader.py`核验显式外部归档和校验/覆盖拒绝。
该门禁不认证数学/设备新功能。新CPU构建与安装消费者只记录实际定向范围。

## 无用户systemd的目标容器

仅在明确授权的无服务管理器容器使用 `scripts/container_job.py`。Linux subreaper收养孤儿后代，
同时覆盖Ninja自行创建的子进程组；此机制不保证跨容器/宿主重启。启动器拒绝dirty源码和已有输出，
独立session的监督进程将日志/状态持久化，超时或stop先TERM再KILL全部工作后代，核对PID开始时间、回收孤儿并检查无存活成员后记终态。
Linux zombie不是仍在运行的工作负载。监督进程丢失或PID身份不符时明确失败，不能从缺失推断通过。

```sh
python scripts/container_job.py start --cwd FROZEN_CHECKOUT --output-dir NEW_JOB --timeout 3600 -- COMMAND ARGUMENTS
python scripts/container_job.py status --output-dir NEW_JOB
python scripts/container_job.py stop --output-dir NEW_JOB
```

启动后断开SSH，再以新连接核对running记录及实际PID；停止探针再核对整个组为空。
正式资格仍检查每个内部结果，外层exit0不扩大验收范围。本机长任务仍使用background.slice。
