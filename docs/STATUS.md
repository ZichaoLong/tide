# 当前交接 — KV 查询块资格与三层诊断收尾（2026-11-11）

当前授权：默认 KV bank/Read/Full 块与块伴随、FC4/FC16 CPU/NPU 三层复测、可重复诊断设施。
本轮聚焦 FC；LH/GPU/多卡不扩展。数学固定 `0770593838a37c790a568226930567ed60f99b83`，19 份快照核验通过。

## 固定源码与已完成任务

实现 `5ff3ce2bfe72b4416814b750d5260f5c90038eca`；范围标记/诊断修正 `598b4fb5f487befe008f50afb1fe1e378504c417`。
构建及主资格：`artifacts/kv-block-20261111/`，生命周期 `artifacts/jobs/kvblock-*/status.json`。
CPU 三个定向组 370 通过/13 保留 xfail；一般组 173 通过/4 xfail/2 SDK 路径启动失败。
真实 SDK 的 Settle FP32/FP64 补验通过（`kvblock-settle-supplement`），原失败保留。
NPU 定向组 32 通过/1 xfail；Python/native 安装消费者三步及新进程恢复、SDK complete/outputs 恢复均通过。
profile 构建 598b4fb 定向 12 项通过；第一轮宏变量重名编译失败保持原记录。

`gef-kvblock-measure-{cpu,npu}`、`gef-kvblock-threads` 均已终态 passed；不是严格数值通过。
主计时 FC16 B8/L128/d128：auto/packed-v2/direct-FC 训练 CPU 5.835/11.601/1.390 秒，NPU 10.928/15.124/0.675 秒。
块命中 Read16256/16384，Full8278/8344（CPU）；严格 FC16 FP32 投影失败仍在。
小型 CPU FP64 FC4 独立公式值/梯度通过；NPU FP32 有限超差保持。
主数据 `measure-cpu/`、`measure-npu-0/`、`measure-cpu32/`；后续审阅 profile、census 与资格证据。

## 活动作业与下一步

已完成 `gef-kvblock-triage`：固定 598b4fb，冻结 `/dev/shm/zlong-gef-block-26ab27d`，
CPU80–111/ATen1、128GiB、600秒；`artifacts/kv-block-20261111/triage.sh` 是完整命令。
将原 FP32 权重提升 FP64，保持 B8/L128 和 blocks 默认路径；Python/native 值/末状态和任务梯度均通过原严格阈值，梯度最大绝对差约1.8e-10。原FP32失败不改写。
日志及终态 `artifacts/jobs/kvblock-triage/`；输出 `artifacts/kv-block-20261111/triage-promoted/`。
查 `python scripts/status.py` 与 `systemctl --user show gef-kvblock-triage`；必要停止同名 unit。

native 后备预检小修与可重复诊断收集器准备提交；Python 诊断定向10项通过（包含3项配对/计时/失败隔离检查）。
随后用该提交新固定源码构建native，并运行块/后备、选定返回、独立根与通用图定向；
完成三层配对、profile、独立根/数值/恢复证据，更新 ROADMAP/契约/经验并提交推送。
不把性能收益或作业终态改写成完整规模严格等价，也不把已定位开销说成已消除。

资源：空闲 CPU/内存/NPU 可用，共享 NPU 未授权；重型构建和计时分开。
账户153GiB/软300/硬800，文件系统约944GiB可用，shm约398GiB可用，没有存储阻塞。
开发激活 `artifacts/kv-block-env.sh`；生产构建分别有固定身份，不能混用新源码与旧库。
用户未跟踪 `docs/tide-learning-lab-TL-REQ-004-handoff.md`、`docs/tide-learning-lab-TL-REQ-005-handoff.md` 原样保留。

Historical source IDs: python scripts/resolve_history_commit.py OLD_SHA; see docs/history-migration.md.
