# 当前交接 — 批量张量贯穿实施中（2026-11-11）

已授权按 [ROADMAP 页首 P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)
持续实施。固定 core-5 数学为锚点；KV、区域分配、Full、消息与下游交接共同保持 B×L 张量列。
私有状态按记忆分型；API 可迁移，历史兼容不得迫使默认计算拆行重拼。

## 当前源码与已核验范围

分支 `graph-execution-foundation`，HEAD `0cbf06aa8fc954fc76f7311f01e2868aef69508c`。
上游固定 `0770593838a37c790a568226930567ed60f99b83`，19 份离线快照启动校验通过。
已提交区域列实现 `2d2f8de` 的资格见 [区域证据](evidence/regional-columns-20261111.md)。

本阶段实现准备提交：KV ragged State/Read/Full、消息与区域列、Python/C++ Settle 独立整链；
Python affine/delta/GDN/SSD 与原始逻辑时间短卷积；完整位置 cut 的 raw checkpoint 转换。
默认 eager 客户端新增 `run_positions/run_ports`，按能力选列路径，其他配置声明一般图后备。
[完整契约与迁移](weighted-tensor-windows.md)列明普通列根和旧完整独立事件根的区别。
旧事件输入 API 不自动改变观察契约；下游高效入口应传 B×L 张量，避免先拆成 External。

开发检查：122 项 FP64 新旧边界通过；新默认/五记忆/区域候选/恢复 84 passed、13 具名
FP32 strict xfailed；同材料 FP64 检查独立通过，不改严格阈值/离散/None-zero 检查。
standalone SDK 新 `tidegraph-column-check` 已通过独立 streaming、raw state、VJP、两步 SGD；
Python/native CPU FP64 消费者三步 AdamW 与新进程精确恢复通过。
这些是开发证据，正式固定提交资格、新默认硬件及成本仍待运行。
原严格报告 `artifacts/columns-20261111/strict-fp32-columns/`、`column-development-2.log` 保留。

用户未跟踪的 `docs/tide-learning-lab-TL-REQ-004-handoff.md` 和
`docs/tide-learning-lab-TL-REQ-005-handoff.md` 保留原样，不纳入本次提交。

## 当前动作与作业

`gef-column-chain-dev3` passed：最新六参数 native 构建与 122 项 FP64 定向检查。
`gef-column-chain-cost-dev1` passed：dev3 冻结输入上 CPU/NPU 有限测量，
新列 FC16 B8/L128 完整训练 Python CPU1 1.2374s、NPU 0.7070s；native 1.3531/0.7137s；
直接 FC 1.4269/0.7633s。原始 `artifacts/columns-20261111/cost-dev1/`；这些不是固定提交资格。
新公共默认 `run_positions/run_ports`、十候选补查/恢复/容量检查开发 84 passed/13 数值 xfailed，
其中 xfail 仅覆盖固定 fixture 的具名字段与幅度，原阈值、离散、None/zero 及同材料 FP64 保持。
原始数值报告 `artifacts/columns-20261111/strict-fp32-columns/` 和 `default-development.log`。

准备 `gef-column-chain-dev4`：冻结当前源码至 `artifacts/columns-20261111/chain-dev1/source`，
新 `frozen-files-4.json`；原 dev3/cost 已结束，保存旧源码。native 增量构建后检查新默认接口，
独立 SDK 构建并运行 `tidegraph-column-check`。全部 320 可用 CPU，512GiB、1800s，
`artifacts/jobs/column-chain-dev4/{status.json,task.log}`；源码冻结后不再编辑。
dev4 的 native 构建和 84 passed/13 xfailed 完成；SDK 配置因 module purge 后 Ninja 不在 PATH 失败。
修正仅在新作业 `gef-column-chain-sdk-dev1` 显式保留构建工具 PATH，SDK libraries 仍独立；
同一冻结 dev4 源码、320 workers/512GiB/1800s，记录 `artifacts/jobs/column-chain-sdk-dev1/`。
standalone 补作业已 passed。当前无活跃作业；接下来固定实现，独立资格和有限配对成本。

原 `column-chain-dev1` 编译失败和 `dev2` pytest collection 失败不改写；后续成功另列。
Python 构建 `artifacts/columns-20261111/chain-dev1/build`；环境 `artifacts/column-env.sh`，
CANN9.2/Torch2.12/TorchNPU2.12.post3，私有 `artifacts/additive-work/env`。
新 standalone 构建 `chain-dev1/build-sdk` 使用独立 SDK module，不混用 Python wheel 库。

## 接续顺序

1. `git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
2. 核对 dev3 终态；补列容量/非法输入/空样本、候选覆盖、standalone 和旧路径回归。
3. 默认公共入口的能力分派、观察根/恢复迁移；不要按 FC fixture 硬编码。
4. 固定实现后做 CPU/NPU FC4/FC16 三模式同材料列/直接 FC/Transformer 有限对照。
5. 完成相关模块契约、语义/架构/复现导航；实现先提交、固定资格后独立证据提交。
6. 按 P4—P6 推进，不把公开原型、单项检查或局部提速当整体完成。

旧 FC16 B8/L128/d128 默认/直接 FC 完整训练 CPU 5.835/1.390s、NPU 10.928/0.675s，
对应 [查询块证据](evidence/kv-query-blocks-20261111.md)。不是本轮新实现成绩。
新长任务必须固定源码、有限资源/时长；空闲 CPU/内存/NPU 可用，共享 NPU 须当前授权。
GPU 真机延后。独立存储容量检查及失败原因保留于任务日志，不重启历史矩阵。

历史源码 ID 用 `python scripts/resolve_history_commit.py OLD_SHA`；见 [history-migration](history-migration.md)。
