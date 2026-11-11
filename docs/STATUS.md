# 当前交接 — 张量窗口及 KV 位置扩展（2026-11-11）

已授权按 [ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)继续。
固定上游 `0770593838a37c790a568226930567ed60f99b83`，启动 19 快照检查通过。
分支 `graph-execution-foundation`；现有张量整链 `4b9ca54`、诊断 `01cde3b` 已提交。

## 已完成的固定任务

固定 `01cde3bd837f82ba9f5ed424f41b039f1747e908`：
- `column-chain-qualify` passed：232 passed / 13 具名 FP32 numerical xfailed；
  CPU FP32/FP64 × Python/native 安装消费者三步 AdamW、独立标量参考、新进程精确恢复通过；
  standalone SDK FP64 独立 streaming/raw state/VJP/两步 SGD 通过。
- `column-chain-cost-cpu`、`column-chain-npu` 均 passed；NPU 两安装客户端独立训练/恢复通过。
  FC4/FC16、B8/L128、FP32 三模式两客户端/直接 FC/Transformer，共50成本点；
  8个 B2/L32 profile 另列。FC16 native 完整训练 CPU1 1.2863s / NPU 0.6788s，
  直接 FC 1.3706/0.6978s。完整尺寸未新增严格投影/多步训练资格，不把小图资格放大。
  CPU80–111/ATen1，NPU host0–15/ATen2、物理2；同项目 CPU 池分离，外部争用未排除。

结果 `artifacts/columns-20261111/fixed-{cost,profile}-{cpu,npu}/`，
`installed-*`、`qualified-build-reuse.json`、`chain-review.json`；任务 `artifacts/jobs/column-chain-*/`。
待把这些原始记录整理为独立固定证据提交。原 dev1/dev2/dev4 启动失败及13数值xfail保留。

## 当前实现与动作

KV位置列 `9fa852c` 固定CPU与NPU资格均 passed；六安装客户端严格三步训练和新进程恢复通过。
standalone新构建CPU检查通过。详情见[位置证据](evidence/column-positions-20261111.md)。
FC正式记录见[整链证据](evidence/tensor-windows-20261111.md)。当前无活跃任务。

native位置构建 `artifacts/columns-20261111/position-dev1/build`，独立SDK `build-sdk`；
干净资格worktree `qualified-position`。原二进制/原失败保留。
环境 `artifacts/column-env.sh`，私有 `artifacts/additive-work/env`；CANN9.2/Torch2.12。
用户未跟踪的两个 `docs/tide-learning-lab-TL-REQ-004/005-handoff.md` 原样保留，不纳入提交。

## 下一步

1. 继续P4：native affine/delta/GDN/SSD、固定逻辑时间卷积与typed状态列，区域/消息接口保持。
2. 相关独立标量/公式、共享根、保留窗/checkpoint与实际公共分派；再固定源码资格。
3. P5—P6：代表规模与执行位置、限制和治理。GPU真机延后；当前新列CPU/mixed-c
   不等于mixed-a/b、resident、多卡或所有规模迁移完成。

长任务固定源码、background.slice、有限资源/时长。可用CPU/内存和空闲NPU可用；
共享NPU须当前授权。当前无存储阻塞。实际未提交状态见git status，作业见scripts/status.py。
