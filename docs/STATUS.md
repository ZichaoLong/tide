# 当前交接 — 2026-10-28

分支`graph-execution-foundation`；工作路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`。
固定数学`0770593838a37c790a568226930567ed60f99b83`，19份当前/5份旧快照校验通过。
前一轮限制调查与有限验证已终态；A—E 完成审计发现 D 的进一步合批尚未交付，继续本轮授权工作。
当前已实现 quad 实际响应与共享组内私有适配矩阵批，开发验证中；旧失败与旧证据保持不变。
源码、固定证据及未解除的限制分别记录；不能据本轮称所有数值域、硬件或性能优化已经完成。

## 已交付与来源

- `b0321b9`/`62073dd`/`12ad204`：历史经验、GDN首次梯度分歧和纯Torch双NPU异步反例；默认失败保留。
- 实现`2da3f2c76c13cff0c41af45fddb5f24f2e99421f`：五族真实参数/别名清单、core-3主体预算匹配、
  可扩展FC/HB/DAG/PDG、v2二进制材料；共享KV低秩计算、初始化消冗、阶段ready副本、冻结bank及发布去重。
- 设施`0503593d71a64d26f24e0a4168e16af8c1633baa`：v2→v1无损有界诊断、FP32/FP64分歧诊断、
  验证子进程scratch隔离与失败metrics保留。数学函数、图v13/checkpoint v5、原容差不变。
- [本轮固定证据](evidence/core5-performance-transfer-20261028.md)及同名JSON提供全部源码/材料/二进制/日志身份、
  实测工作量、正/负结果与公共复现入口；[经验](performance-lessons.md)包含本仓库core-3与新实现的可复用经验。

## 最新终态

以下job路径均为`artifacts/jobs/NAME`，unit为`NAME-20261028`，记录`status.json`/`task.log`。

| NAME | 终态与范围 |
| --- | --- |
| core5-performance-qualify-cpu | passed；固定2da3f2c，376 passed/2 strict xfail，24 native图各4更新及精确checkpoint |
| core5-performance-qualify-npu | passed；固定0503593，四组件×FP32/FP16，完整flow/配置/训练/双卡节点训练恢复；物理1/3 |
| core5-performance-qualify-cuda | passed；同内容hash增量sm80/90/100构建、loader/help、CPU FP32/FP64核验；无GPU真机 |
| core5-performance-study-cpu | failed；原medium前后与HB64 SSD通过，D64 KV及D2048严格失败/拒绝，CPU profile亦保留失败 |
| core5-workload-diagnostics | failed；设施14项通过；Python/native FP64两步通过，FP32及旧消费者/原PDG严格失败 |
| core5-performance-study-npu | failed；原medium前后和HB64 SSD resident通过；HB64 KV resident/mixed/profile及SDK affine严格失败 |

冻结源码：`artifacts/worktrees/core5-performance-qualified`(2da3f2c)、`core5-diagnostics-qualified`(0503593)。
复用开发二进制先验证全部core/component/binary hash；对应冻结dirty开发副本`core5-performance-dev2`未被改写。
原CUDA开发job编译完成后因并行NPU生成kernel_meta改变Git身份而failed；后续同源重验通过，原failed不重标。
早期缺头文件/Ninja的开发失败、所有数值失败仍在证据/作业记录，不恢复旧队列。

CPU原HB medium同材料391.432→365.834ms；NPU 2576.159→2795.223ms，均仅一次流程。
NPU allocated小幅下降，未测得端到端提速。92.62亿参数KV材料已实际构造/warmup，随后分配残差拒绝；
不能给出大型吞吐结论。结果和已知争用详见报告，不外推全部模型或硬件。

## 保留限制与后续入口

唯一待办是[ROADMAP](ROADMAP.md)。优先稳定quad共同偏移响应/VJP及D64训练数值边界，随后按profile
处理同形私有模块矩阵批和窗口反向构造。CUDA真机/x86_64、大规模设备训练/跨卡吞吐仍待实际硬件资格。
eager FP16 GDN/AdamW、默认异步双NPU、SDK FP32 affine、反馈PDG原严格限制没有解除。
参考仓库和公共环境只读；无子代理；阶段提交/普通推送获用户授权，无强推。

重新进入：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
查看本轮终态：`python scripts/status.py --all-jobs`；核对独立结果与日志，不能只依赖已回收unit的默认状态。
复现使用[workload入口](weighted-workloads.md)与[开发规范](development.md)，新建输出目录、固定源码和有限任务；
不依赖历史绝对运行目录、不自动恢复或扩大矩阵。当前提交只维护本交接、待办、相关契约与小型固定证据。

## 当前继续工作

以 b7c5173 为起点交付 quad scalar/batch/resident 实际响应修复与共享组内私有投影 BMM。
独立行 VJP 保持 None/零、别名贡献和版本；新增统计分开真实矩阵批与 State 回调。
`core5-quad-dev-20261028` passed：332 项 CPU FP32/FP64 分配/区域/执行回归。
`core5-private-dev-20261028` failed：新增 standalone 测试括号/include 编译错误，原记录保留。
修正后 `core5-private-dev2-20261028` passed：244 项私有投影/共享/State VJP/图/设备源适配；
全部开发源码分别在对应 job/source.tar.gz。当前开发副本为 `artifacts/worktrees/core5-quad-dev`，已终态。

接下来将固定实现提交到 `artifacts/worktrees/core5-private-qualified`，只读运行三个新 job/unit：
`core5-private-qualify-cpu-20261028`、`core5-private-qualify-npu-20261028`、
`core5-private-qualify-cuda-20261028`；输出各自 `artifacts/jobs/NAME/{status.json,task.log}`。
CPU 复用逐文件/二进制哈希相同的开发构建，定向完整资格和 standalone benchmark client；
NPU/CUDA 各自重新构建匹配 core，分配组件与完整 flow/配置/训练/双卡训练组件单列。
NPU 真机限等候180秒，CUDA只构建/loader/CPU检查，无GPU真机。
2 build workers、单线程算子，CPU32GiB/20min，设备构建64GiB/90min；scratch在job输出中。
源码固定后运行，不能将计划/运行中的任务标为通过。

三项终态后以同一材料做一次完整训练前后测量，并定向重验 D64 梯度/参数与 D2048 拒绝；
大型案例限单样本单位置推理、96GiB/20min，不扩大旧矩阵或恢复旧队列。
下一步核对 `python scripts/status.py`、每个新 job 的原始记录；停止用 `systemctl --user stop UNIT`。
