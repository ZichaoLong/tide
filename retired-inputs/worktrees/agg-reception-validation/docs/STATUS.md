# 当前交接：接收 Attention、FP32 训练与大参数推理正在实施

用户已确认 [ROADMAP](ROADMAP.md) 的七阶段计划，实际训练优先 FP32。
分支 `graph-execution-foundation`；本轮起点 `df2a0f18ec0efe321b2064f60ce0ea4d9d78c9ac`。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，本轮108文件核验通过。
持续 FC/HB 实例路线与[范围约定](project-scope.md)不变。

## 当前工作

实现已集成：placed同分区区域合批、SSD block/incremental/doubling、接收QKV融合与
结构padding候选、完整Reception微基准、新连续推理与A–D/Transformer分层放置。
`28941994`修复 inference_mode 生成静态mask后切换训练的缓存错误；`aa4a7c5e`
修复诊断工具冷导入泄漏故障注入别名（harness问题）。当前候选配置在configs/agg-inference。
默认数学和旧实现保留；代码可用与正式数值/性能资格分别记账。

[起点证据](evidence/reception-start-20261118.md)已提交，后续原始记录保持独立。
`agg-cost-doubling-fp32-v1`已完成：同B8/L256 FP32 source训练中位1.147秒（旧1.368）。
`agg-reception-sites-fp32-v1` 48行全部原阈值内。`agg-light64-variants-fp32-v1`
六候选完成，组合scan+QKV+mask+接收重算1.683秒，基线1.905秒；仍需结合容量挑选默认。

完整CPU首轮 `agg-integration-cpu-v1` 冻结b4cf1514：715 passed / 11 failed。
10项由上述诊断alias泄漏复现，已修；另1项由前序CLI测试泄漏Torch线程数引起，
`1553b301`恢复测试线程隔离，原exact断言保留。
不能称当前主分支全套通过；最终受影响/全套与隔离安装尚待完成。

`agg-train-fc-fp32-100-v1`和`agg-train-settle-fp32-100-v1`已完成100真实更新与配对
恢复，均原恢复阈值内。稳定输入位置吞吐约30333和2644/s；两者新进程实际更新也已完成。
Timed100更新已完成；0.5B容量到B4/L128约274位置/s、29.53GiB allocated。
light64容量比较已完成，B24重算与B16不重算的峰值/持续运行另核验。
三接收正式FP32路径9候选中1 within/8 finite；99比较41超差，无观察到的结构分歧，
9次新进程恢复更新均原阈值内。BF16接收微测48行完整，27 within/21 finite，不改阈值。全部根任务清单及冻结命令可查
`artifacts/agg-inference-preflight-20261118/jobs.json`（快照，实际终态仍逐查status.json）。

新推理作业：FC/HB0.5B、9B单卡A–D已完成；17B双卡仍按完整负载执行。
原始文件是对应`artifacts/agg-inference-*/inference.json`和reference.json。
先有16次增量初步对照；现已提交FC/HB约1B、9B/17B的256次主测、
Timed B4/prompt2048连续流、17B串行batch1/batch4与并发batch4对照、接收双卡资格。
每卡预算allocated48/reserved56GiB；静态容量计划在本轮preflight目录，不能替代实测。

根统一NPU和提交。Reception代理整理数值证据，SSD代理整理新成本证据，Inference
代理在隔离工作树审查静态参数gather候选。不要重交活动任务、修改冻结源或清理依赖。
下一步完成当前资格/容量检查，补模式与大尺度矩阵、连续训练/恢复、安装、推荐和最终证据。

[起步记录](../artifacts/agg-inference-preflight-20261118/preflight.json)保存实际账户、
工作区、服务、资源和四份用户文档SHA。最近配额约137GiB/300GiB软限、700GiB硬限；
320 CPU、约1.8TiB可用内存。观测时7张无进程空闲NPU，每卡64GiB；逐次准入仍需复核。
当前没有遗留Tide服务。共享NPU授权只属于本线程，当前优先协作队列独占准入。

## 可复用证据及未解决边界

此前生产 `src/tide` tree 为 `d3e8a6cb40eceaf487dc642a4283d58f90d5726f`。
[训练集成](evidence/training-validation-20261118.md)记录冻结 `ae247dee` 的625项CPU检查，
后续预设/同步边界检查及安装事实；新源码需自己的相关验证。
[训练建议](execution-recommendations.md)、[计算流程](computation-and-throughput.md)、
[连续训练](evidence/training-continuous-20261118.md)、[成本](evidence/training-costs-20261118.md)
和[四类模式](evidence/training-modes-20261118.md)保留各自实际源码和负载身份。

现有接收三种 incoming 配置已实现，有独立FP64公式/梯度及部分组合运行；本轮补完整
训练、持续推理和大参数性能交付，不把所有配置称为已同等验证。此前FC大batch连续
测点为BF16，本轮优先补FP32。

[31候选路径](evidence/training-paths-20261118.md)与原FP32/BF16有限超差均保留。
Timed BF16 prefix有支持/时间分歧，单卡FC BF16恢复有梯度/moment超差，多卡路径有
反向及恢复差异；精确保存/下一前向不意味着后续优化轨迹精确相同。CPU FP64用于
有界诊断，阈值不为通过标签修改。

完整模型仍是PyTorch，LibTorch是区域评分切片。CPU/NPU已有所列实际证据；
CUDA、peer DMA/HCCL、全模型C++、compiled/resident训练没有自动新增资格。

## 用户文件与接续

以下四份未跟踪文件保持原字节、不纳入提交：
`tide-learning-lab-TL-REQ-004-handoff.md`、`tide-learning-lab-TL-REQ-005-handoff.md`、
`tide-learning-lab-TL-REQ-006-handoff.md`、`upstream-lh-semantics-discussion-20261113.md`，
均位于docs，SHA见本轮起步记录。参考、公共环境和下游只读；没有下游发布或验收。
既有checkpoint、trace、冻结源和工作树保留，不在运行任务期间清理复现依赖。

接续先查实际HEAD/工作区、活动服务与本轮artifacts终态，再读ROADMAP；不要重复提交
仍在运行或已经完成的实验。阶段汇报后继续推进到本轮交付完成。
