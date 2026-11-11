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

KV rotary/距离项的新列前向/一阶VJP已完成，原始 int64 先求差；容量计入相对旋转临时量。
`column-position-dev1` passed：新 frozen source/native 构建，79 passed / 13 xfailed；
补充非空 bound Read0、位置 checkpoint、collector 检查15 passed。固定提交资格待运行。
开发反例曾让 time-major oracle 从0扫至2**54，已停止两个进程并改从对应空 cut 开始；
这是测试范围修正，不改数学或阈值。空样本独立叶测试改按公共列根契约构造 oracle，
旧独立事件根仍由记录 API 检验，不隐去 None/zero 差异。

native 构建 `artifacts/columns-20261111/position-dev1/build`，冻结输入 `position-dev1/source`、
`frozen-files.json`。当前无活跃服务；原二进制保留。
环境 `artifacts/column-env.sh`，私有 `artifacts/additive-work/env`；CANN9.2/Torch2.12。
用户未跟踪的两个 `docs/tide-learning-lab-TL-REQ-004/005-handoff.md` 原样保留，不纳入提交。

## 下一步

1. 提交位置扩展及收集设施；新干净 worktree 核对 C++/二进制 hash 后复用匹配构建。
2. 位置项固定 CPU 检查、安装 CPU/NPU Python/native 三步及新进程恢复、独立 SDK 实跑。
3. 单独提交张量整链和位置扩展的资格/成本证据；同步导航与性能经验，不改旧证据。
4. 继续 P4—P6：native 循环列/异构、代表规模与执行位置、限制和治理；GPU真机延后。
   新列单 payload 的 CPU/mixed-c 不等于 mixed-a/b、resident、多卡迁移完成。

长任务固定源码、background.slice、有限资源/时长。可用CPU/内存和空闲NPU可用；
共享NPU须当前授权。quota已用224GiB/软300GiB，空闲内存约1.8TiB，无存储阻塞。
运行/未提交事实以 `git status --short --branch`、`python scripts/status.py` 为准。
