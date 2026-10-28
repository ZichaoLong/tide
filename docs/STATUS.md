# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`。
用户授权本机非GPU工作、分阶段提交及普通推送；GPU限制延后。参考仓库/公共环境只读，不使用子代理。
本轮实现固定`f03e9bee0f069d48f5f1544d3821b7781409a27f`，对照固定`38ebf422b198d7cc041b6238fb954cf9775c02c0`。
实现已推送；本交接与新增证据/导航构成后续文档交付，生产源码没有未提交修改。

## 本轮闭环结果

[独立输入KV合批报告](evidence/core5-independent-projection-20261029.md)及同名JSON保存精确命令、
源码/二进制身份、独立公式与完整训练证据、终态和原日志hash；无需旧运行目录即可从公共入口复现。

- KV剩余同形正接收行按独立输入/基座合批；保留原计算复用、每行epsilon、None/零根及私有状态。
  新局部矩阵packing预算8MiB，不改变dtype、图v13、checkpoint v5或原数值容差。
- CPU相关固定资格：273 passed、2原FP16 GDN/default AdamW strict xfail；复用库前核对C++内容和23个binary hash。
- 纯PyTorch NPU与Python/native NPU各六组eager通过（FP32/FP16、FC/HB/一般ports、4更新、独立根、恢复/CPU交接）；
  Python/native公共KV常驻前向/训练/磁盘恢复12项通过。
- 新standalone SDK core/resident构建通过。两dtype的flow/training/node-training六项通过：
  每dtype前向21案例、单卡训练5配置/20更新、双卡节点训练5配置/20更新及换放置恢复/共同拒绝；
  SDK eager另有两dtype×3图×4次AdamW及完整原生checkpoint/CPU交接通过。
- 三对单次完整训练计时均通过全部返回、最后梯度、参数和SGD槽核对：CPU中型388.129→405.010ms，
  同参数D64 CPU显式FP64 2991.719→2996.551ms，NPU纯PyTorch mixed-c中型1825.444→1850.355ms。
  未观察到明确端到端加速。D64的34次共享+445次独立准备覆盖479次写入，State最大batch仍1。
- D64两步完整训练与两调度换owner下一窗的记录/梯度/参数/SGD槽逐值恢复通过；独立profile也通过。
  反向/finite阶段跨度最大，仍有大量Query/Aggregate VJP和字段拆装；不能只凭BMM数推断性能完成。

## 作业终态与停止边界

**当前无排队或运行任务；本轮约定的有限资格、计时和profile全部结束。**
作业前缀为`core5-independent-`，unit均追加`-20261029.service`：
`fix-dev`、`qualify`、`npu-build`、`python-npu`、`pynative-device-fixed`、`study-cpu`、`study-npu`均passed/0。
记录在`artifacts/jobs/core5-independent-*/status.json`及`task.log`；pynative-device-fixed的目录名为`pynative-device`。
CPU study七个stage、NPU study两个stage均passed/0；完整运行记录见各目录的`study.json`。
四个原失败仍failed：`dev`构建+测试1170秒到限/124；`dev-tests`235 passed/2 strict xfail/2旧断言失败；
`npu`配置缺Ninja/1；`pynative-npu`覆盖CANN PYTHONPATH导致tbe初始化失败/1。
后两项只修正任务私有环境。开发原archive、manifest及failure-explanation保留；未将旧任务回填通过。

作业输入仍是只读固定树`artifacts/worktrees/core5-independent-qualified`（f03e9be）和
`artifacts/worktrees/core5-independent-baseline`（38ebf42）；它们是运行来源，不是公共复现的依赖。
生产ABI增量已在CPU及两套NPU环境重建/核对；本提交没有新增CUDA构建或真机资格，旧构建只认证原提交。

## 仍保留的边界与接续

[非GPU边界证据](evidence/core5-nongpu-boundaries-20261029.md)及[ROADMAP](ROADMAP.md)区分已解除与未解除：
D64/D2048及原反馈FP32完整等价、eager FP16 GDN/default AdamW、默认异步双NPU仍未解除。
外层FP64汇合和既定九项分段精度探针不足，未采用为生产修复；两调度各自精确恢复不替代跨调度训练等价。
显式CPU FP64已有独立可用资格；原SDK affine mixed-a/c见证已通过，不外推整个旧矩阵。
CUDA按用户要求延后，x86_64无本机硬件，大型设备训练/跨卡吞吐未验，容量/正支持拒绝继续生效。

本轮不继续重复既定探针、旧队列或扩大性能矩阵。新的算术方案须覆盖局部贡献舍入、独立根与跨步更新；
异步问题须有新运行时/修复，并同时通过纯Torch反例和完整图恢复。性能后续按新profile优先审查
Read/Full/Aggregate私有VJP与字段拆装，再考虑resident组合矩阵/窗口反向复用；没有低成本优化耗尽的结论。
历史core-3、LH与本轮正负结果都在[性能经验](performance-lessons.md)，无需重跑旧实验恢复上下文。

接续只需先执行：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
本轮无必须恢复的长任务；按ROADMAP中的新证据触发条件开展下一项，不将未解除限制改写为通过。
