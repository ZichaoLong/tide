# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；上游固定`0770593838a37c790a568226930567ed60f99b83`。
逐项推进非GPU限制，GPU延后，无子代理；已授权修改、有限实验、提交与普通推送。
参考仓库、公共工具链只读；原输入/容差/optimizer和严格失败保持。

## 已固定与终态

实现15c3e65、固定资格证据46ecd5a已推送，见[资格](evidence/core5-resident-boundary-20261029.md)。
CPU facility10、公共NPU14、两套eager各6组、SDK组件9项通过；生产eager继承e0d6e27的CPU517 passed/2原strict xfail。
CPU study `core5-boundary-study-cpu-20261029.service` passed/0，
NPU study `core5-boundary-study-npu-20261029.service` failed/1，均已终态，无旧活跃任务。
原始记录`artifacts/jobs/core5-boundary-study-{cpu,npu}/{status,study}.json`及各result.json。
CPU四项、NPU四项mixed对照及resident中型完整通过，新版各自磁盘换owner恢复精确。
D64 native单/双NPU显式blocking自身恢复精确，CPU独立核对均在node89 keys一坐标严格失败。
resident D64在构造时报Aggregate scratch budget exceeded，尚无训练计时/恢复结果。
独立profile已终态；单次测量不宣称稳定收益，最终证据尚待整理。

## 当前推进

已定位resident按全局queue为每节点分配纤维，进而重复展开来源矩阵。
新候选按物理入边+端口的静态上界压缩局部槽；全局indices/sentinel、SourceDomain和SrcClass分开，
前向、VJP重放和peer请求一致，超界拒绝。CPU元数据9项通过；设备编译/资格待执行。
当前源码/文档修改待固定提交；不复跑未变化的CPU/mixed数值失败或历史全矩阵。

## 接续命令与有限计划

先`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
固定紧凑纤维实现后新建`artifacts/worktrees/core5-compact-fiber-qualified`；
单独background.slice构建新resident目录，不复用CANN合并对象；core仅以完整源码/二进制hash核验复用。
先CPU元数据和公共NPU完整记录/训练/跨卡及SDK相关门禁；通过后串行一次中型和一次原D64 resident，
同plan/输入/容差/容量，无真实轨迹预跑。容量/数值/运行时失败即停止该候选，记录精确边界。
新unit、driver和输出在提交后、启动前补入本页；运行中不称通过。

旧构建只读：`artifacts/worktrees/core5-independent-qualified/build/{npu-python,npu-core-v2}`；
上轮resident对照`npu-python-resident-boundary`，其dirty-origin和hash不改写。
当前环境由`docs/development.md`定位/opt/software现行规范，旧home指南失效；不修改共享环境。
新结果进入性能证据、ROADMAP逐项表、相关契约和performance-lessons，证据独立提交推送。
停止/重启条件按ROADMAP保持：D64/D2048/反馈FP32、FP16 GDN、默认异步双NPU、晚注册NPU限制未解除。
