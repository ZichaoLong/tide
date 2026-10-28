# 当前交接 — 2026-10-28

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`。
参考仓库/公共环境只读，不使用子代理；实现、有限实验、分阶段提交和普通推送继续获授权。

## 已完成阶段

实现`1f1ec5780e054e460e3471547519ace3f38fa3af`：共享准备按首事件稳定求值，修复换owner后的梯度微差；
CPU benchmark显式FP64/真实静态存储计费，实际最终梯度/optimizer槽核对；两步精确checkpoint诊断。
[固定证据](evidence/core5-stable-preparation-precision-20261028.md)及同名JSON已作为`0e04da6`单独提交并推送。

- CPU扩大门禁`core5-stable-preparation-qualify`：**failed/1，572 passed、2 failed**。两项为FP32 GDN
  HB/default AdamW；旧1850434同样复现，第三次conv.weight[0,2]差1.411885e-6。原断言/失败保留。
- `core5-stable-preparation-diagnostics`passed：D64 FP64 Python/native各两步完整记录、梯度、参数、
  optimizer和实际checkpoint换owner后下一窗口逐值恢复通过。
- NPU SDK2.10/CANN9.0 core/consumer及FC/KV、HB/KV各4次更新/恢复通过；物理13→npu:0。
  CUDA2.10/cu128 core/consumer和CPU两dtype反例通过，无GPU真机。本次未改/重跑resident核矩阵。
- `core5-explicit-precision-study`passed/0，五项都完整核对：D64四种接口/调度FP64训练5.702/2.896/2.958/2.903s；
  同拓扑/core-3主体参数匹配的D2048 FP64推理151.416s，424候选/199 Full/946消息。
  大例约92.62亿主体参数，测量后RSS峰值约90.35GiB；原FP32失败不解除，无大型训练/跨卡吞吐资格。
  每配置一次流程、项目计时串行、外部争用未隔离。

上述unit均加`-20261028`，日志/结果位于`artifacts/jobs/同名`，全部终态。
CPU固定树`artifacts/worktrees/core5-stable-preparation-qualified`；设备固定树`core5-private-qualified`，均只读1f1ec57。
CPU开发缓存`artifacts/worktrees/core5-stable-preparation-dev/build/cpu`可在原任务终态后顺序增量复用；
其dirty快照和此前资格hash均已保留，不能把开发基提交当资格来源。

## 正在接续的增量

新发现GDN边界的`core5-gdn-fp32-witness`保留实际近零梯度、槽值和独立80位Decimal更新，failed/2。
默认AdamW会把约1e-11的梯度微差累积为参数越界；前向/梯度在容差内不能替代更新检查。
一次后续动态`core5-gdn-stable-normalization-probe`已passed/0：仅Python，把half已用的normalized
L2/scaled-RMS一阶收缩应用于FP32，原前向/输入/容差不变，三步最后参数差约7.00355e-7。
该探针尚不是公共实现，不认证native/Jacobian/checkpoint；没有回写上面的原失败。

已在主工作区接入Python/C++ FP32同一解析一阶收缩，保留原FP16/FP64及int64边界；
短检查11项通过（后又补L2版本/零根和C++FP32逐值前向反例，须在下项开发门禁核验）。
已完成`core5-fp32-normalization-dev-20261028`，冻结输入复用已终态的`core5-stable-preparation-dev`；
入口`.local/core5-fp32-normalization-dev.sh`，输出`artifacts/jobs/core5-fp32-normalization-dev`。
background.slice/Nice10、build2/算子1、64G/15min，增量重建core后183项通过，含原两项FP32 GDN AdamW及恢复、独立Jacobian、FP32逐值前向、版本/None/零。
下一步：
扩大独立literal Jacobian与版本/None/零检查，复验原两项GDN失败和相关循环/packed/多步恢复。
实现已准备提交；随后以该提交运行相关CPU与NPU/CUDA固定资格，另记证据；原1f1ec57失败保持原终态。
不重跑已经通过的D2048 KV测量（候选不改变KV公式），除非实际源码依赖或新失败要求。
本阶段原算术探针、启动失败及首个arithmetic-trace误覆盖的partial边界已纳入证据，不恢复旧队列。

## 未解除边界

D64/D2048 FP32、FP32/FP16 GDN AdamW、SDK affine、反馈PDG、默认异步双NPU严格见证保留；
CUDA真机/x86_64、大型设备完整训练与跨卡吞吐待验。广泛私有合批、resident组合矩阵/窗口反向仍有空间。
唯一待办见[ROADMAP](ROADMAP.md)，可复用经验见[performance-lessons](performance-lessons.md)。
重入：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。

下一固定资格预定`core5-fp32-normalization-{qualify,npu,cuda}-20261028`，新输出同名artifacts/jobs，
CPU使用新的固定worktree，设备checkout仅在旧任务终态后切换；background.slice/Nice10，build2/算子1，64G/20min。
