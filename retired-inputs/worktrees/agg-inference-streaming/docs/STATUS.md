# 当前交接：接收 Attention、FP32 训练与大参数推理正在实施

用户已确认 [ROADMAP](ROADMAP.md) 的七阶段计划，实际训练优先 FP32。
分支 `graph-execution-foundation`；本轮起点 `df2a0f18ec0efe321b2064f60ce0ea4d9d78c9ac`。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，本轮108文件核验通过。
持续 FC/HB 实例路线与[范围约定](project-scope.md)不变。

## 当前工作

已提交并冻结：`d845674b` 放置分区内相容区域合批；`57eb12c3` SSD block/
incremental/doubling；`b4cf1514` 接收融合QKV与结构padding掩码；`d635a982`
完整Reception微基准。默认数学及旧执行候选保留。CPU开发检查已通过相关公式/VJP、
多步与恢复；正式主分支完整CPU套件正在运行，不能提前称全套通过。

[起点测量](evidence/reception-start-20261118.md)记录source/light64 FP32完整训练及
SSD微测的四个已完成冻结作业。L256矩阵scan微测有明显收益且原阈值通过；整图收益、
长期更新和推荐仍待各自结果，不从局部加速外推。

当前根代理管理的持久作业（逐次查artifacts终态，不重复提交）：
- `agg-integration-cpu-v1`：冻结 `b4cf1514` 完整CPU套件，CPU56–63。
- `agg-cost-doubling-fp32-v1`：source B8/L256新scan完整成本/trace，CPU80–83。
- `agg-light64-variants-fp32-v1`：6个接收/重算/scan组合，CPU84–87。
- `agg-fc-capacity-fp32-v1`：FC FP32 B32/64/128有限训练搜寻，CPU88–91。
- `agg-reception-sites-fp32-v1`：3种接收、真实零质量/padding、4候选，CPU92–95。

`codex/agg-inference-streaming`正在完成独立持续推理入口与实际负载A–D审计；
`codex/agg-inference-ssd`补新候选的非空状态路径资格；`codex/agg-inference-reception`
做有界有限超差解释。工作树在`artifacts/worktrees/agg-inference-*`；根统一集成与NPU调度。
下一步：接入持续推理入口、正式路径/恢复检查，根据候选数据选择100更新主测和
0.5B/1B/9B/17B推理矩阵。所有训练优先FP32。阶段汇报后继续。

[起步记录](../artifacts/agg-inference-preflight-20261118/preflight.json)保存实际账户、
工作区、服务、资源和四份用户文档SHA。配额约125GiB/300GiB软限、700GiB硬限；
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
