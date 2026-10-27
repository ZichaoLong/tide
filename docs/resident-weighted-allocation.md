# 带权常驻分配组件

固定 `0770593` [区域分配 §6/S.2](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)
的 `Alloc`：输入完整候选评分、cap、lower/upper、正temperature及可选shift；
求解带盒约束和总量区间的严格凹问题。支持quad、Bernoulli、固定alpha∈[1,2]的ent。
实现/测试/证据由[semantic-map](semantic-map.md)、[STATUS](STATUS.md)导航。

`weighted_allocation.h::append_weighted_allocation` 将设备上的完整区域表示为
`scores/caps[rows,width]`、`bounds[rows,4]`（lower/upper/temperature/shift）和
`lengths[rows]`。浮点字段显式FP32，长度/分支int64，活动根与连接标志bool。
长度零表示不存在的区域帧；填充不属于候选，不参与任何有限检查、求和或梯度。
每个非空行必须包含整个区域候选集合，不能按物理分片分别归一化。

当前设备program没有完整ragged约束分配原语。专用组件读取实际长度，在设备上
判定zero/caps/slack/lower/upper/equality分支、执行有界求解和活动面修正。
行之间独立，无主机标量决定迭代；紧凑短行由单个scalar worker拥有，避免不同行共享
cache line时的并发写。Exp/Ln使用CANN vector primitive，CUDA源有独立适配。
固定80次二分是前向数值过程，绝不是迭代导数。还没有据此声称融合矩阵算子、
全图packed常驻或性能优势；Agg/State/Describe/发送/图调度的集成分别验收。

返回实际计算的controls、multiplier/residual、分支与响应斜率tape。
三种正则分别使用其精确响应映射VJP，连接标志与数值零分开：zero无根、caps只连cap，
普通分支连接score/cap/temperature及实际选中的bound，shift按显式输入声明。
slack的upper根为连接零；等式约束的shift为连接零。Bernoulli即使前向舍入为cap，
仍保留声明的正响应斜率。反向用实际响应点，anchor居中归约保留常数根的零score梯度。
`append_weighted_allocation_vjp`接受每行独立root存在标志，输出[score,cap]、
[lower,upper,temperature,shift]及六项bool连接；调用者再与实际参数requires-grad相交。
该接口是显式一阶组件，不挂接宿主autograd，不提供二阶导数或HST。

tape属于最近一次program调用。保留多次窗口的训练owner必须在覆盖之前journal实际tape；
本组件不自动保存完整训练历史或checkpoint。长度保存在自己的设备buffer。
前向/反向scratch分别提前预算，不能用总预算推定其它队列容量。
错误先写各行status，再统一合并sticky error；上层只能在error=0时提交状态/消息。
invalid长度用既有metadata错误2；数值域20、数学正支持下溢21、残差22、VJP溢出23。
失败不代表已完成部分可返回；上层不应发布失败的proposal。
不裁剪数学正支持，FP32下溢零显式拒绝；不把real零cap误认为填充。

| 对应 | 独立验证 |
| --- | --- |
| 定义域、解与返回 | `ascendc/tide_weighted_allocate.cpp`；`test_resident_weighted_allocation.py`枚举quad可行面、80位Decimal熵响应、小型手算 |
| 精确解一阶VJP | `tide_weighted_allocation_vjp.cpp`；独立隐式公式/标量参考、光滑分支方向差分、孤立行根及None/零 |
| padding、正支持、严格失败 | CPU语义源码适配检查poison padding、empty/zero/full/singleton、Bernoulli舍入cap斜率、非有限及下溢拒绝 |
| 实际设备重放 | `weighted_allocation_check.cpp`；同program改变输入/长度，CPU独立Alloc对照、全部梯度/连接、失败不提交及容量拒绝 |

CPU语义适配器只执行同一份登记kernel源，验证其公式与内存接口；不是常驻CPU后备，
不能认证CANN/CUDA运行。独立SDK目标为`--checks weighted-allocation`，FP32 scoring。
实际设备、CUDA编译和CPU源码检查分别记录；完整weighted resident图生命周期仍在ROADMAP。
