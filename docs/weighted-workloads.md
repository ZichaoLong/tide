# 参数清单、匹配用例与二进制材料

固定数学仍为0770593；本工具构造测试输入，不增加数学函数或代替配置合法性检查。
入口：[参数形状](../scripts/weighted_shapes.py)、[共享清单](../scripts/weighted_inventory.py)、
[静态计划](../scripts/weighted_workload_plan.py)、[实际构造](../scripts/weighted_plan_materialize.py)。

| 数学绑定 | 静态角色/形状 | 独立核验 |
| --- | --- | --- |
| 架构§2来源变换与共享 | aggregate A/B、class embedding、adapted factors；按真实region/node/source规则去重 | `test_weighted_inventory`四绑定、异构记忆、异宽类别与显式同形alias |
| KV§1—4、循环各§1—3 | memory/summary/卷积、Describe、Full；sum/qry/state分别计数 | 五族十种read×七种评分配置，与实际nn.Module全部角色/shape/requires_grad/alias对照 |
| 分配§3、发送/评分 | local/head/encoder、evidence/key/gain；只计适用参数 | 同上；实际materialize先构造正式Config并验证 |
| FC/HB§1及§4.4 | 独立静态物理拓扑、原始时间、单位强度广播，parent SrcClass | `test_weighted_workload_plan`对照公共生成器、同秩区域、非对齐DAG/反馈PDG的实际前后向 |

清单输出trainable/frozen/unique、逐角色元素数、共享owner及全部别名。固定单位transport是
frozen Parameter；旧接口的零buffer不属于可训练参数。去重后参数量、角色展开量、物理副本
分别解释；同类不同原子不合并，状态私有。公共bind仍支持清单外的合法程序。

旧core-3 D2048、480节点、2208边：Add总9468053696，Attention总17521117376；
embedding/head共206045184。新用例无任务头，所以匹配主体9262008512/17315072192，
头参数明确为零。匹配通过实际Full SwiGLU宽度完成：每节点每单位宽度增加3d参数，
节点间FFN宽度至多差1，最近整数匹配误差不超过1.5d；目标低于其余固定角色的最小成本时拒绝。
参数匹配不代表函数相同、计算量相同或模型质量可比。固定拓扑同宽对照另列，不混入匹配对照。
`--match-core3 add|attention`按本次物理拓扑/宽度重新计算旧主体参数预算，可生成较小的匹配用例；
公式与`flow_topology.make_packet`的完整模型计数减去embedding/head独立对照。
`--match add-body|attention-body`始终指上述旧D2048大预算，不随新拓扑自动缩放。

```sh
python -S scripts/prepare_weighted_workload.py --topology ranked --layers 15 \
  --region-width 32 --width 2048 --samples 512 --positions 12 \
  --match add-body --output-dir NEW_PLAN
python scripts/benchmark_weighted.py --device cpu --implementation python \
  --workload NEW_PLAN/workload.json --max-materialize-bytes BYTES --output-dir NEW_RUN
python scripts/export_weighted_benchmark.py --native-library CPU_CORE \
  --workload NEW_PLAN/workload.json --max-materialize-bytes BYTES --binary --output NEW_FIXTURE.json
```

例如`--width 64 --match-core3 add`保留同一480节点/2208边拓扑，生成缩小的主体预算匹配。
第一条无需Torch或张量分配，可用于全尺寸准入规划。后两条须按有限任务预算选取样本/时间窗；
不意味着512×12完整训练必能入场。`memory_floor`是下界：参数、旧零buffer、FP32master/
梯度/optimizer槽与角色展开分别列出，**不能相加就当实际峰值**。原始状态/KV/卷积、在途队列、
journal/adjoint、通信、临时packing、workspace及allocator开销仍需执行器分区准入和实际观测。
`--max-materialize-bytes`仅拒绝已知静态存储超额，既不保证初始化峰值，也不保证训练可容纳。

计划固定schema/upstream/hash、物理拓扑、逐节点/区域配置、source分类、共享模式、宽度、
初态构造策略、输入强度和原始时间。span/stride为int64域；不把事件编号当逻辑时间。
CLI取k=区域规模或指定k，实际正支持在线决定；`regional_total_zero`的k仍为发送强度上限，不改成普通Top-k。
一般合法配置可经程序接口构造，正式runtime Config拥有最终适用性验证。

benchmark fixture v2以JSON保存结构与索引，FP32小端二进制保存实际TensorImpl去重的张量。
每张量有shape/offset/bytes/SHA256，全文件有bytes/SHA256；只对实际冻结零buffer使用零编码。
相等但独立的张量不去重；C++复核字节范围、完整文件、逐张量、别名索引及图identity后安装公共函数。
OpenSSL Crypto只用于独立benchmark材料校验，不增加公共C++核心或Python运行依赖。
材料不含CPU预跑、路由或选定批次，不是checkpoint替代物。v1与原small/medium输入继续保留。

性能计时和独立事后完整返回比较沿用[性能契约](weighted-performance.md)。初始化/材料I/O、
完整训练、参考验证分别记录；大模型不展开成JSON scalar lists。新结果连回
[性能经验](performance-lessons.md)，失败、容量拒绝和负收益同样保留。

## 分歧隔离与历史消费者

`scripts/diagnose_weighted_workload.py --workload PLAN --dtype float32|float64
--implementation python|native --output-dir NEW` 使用同一FP32种子材料；FP64由两条独立
调度在开始前显式提升。分别记录完整返回、离散结构、输出loss参数梯度及一至两步SGD参数。
native另需`--native-library CORE`。原严格容差和None/零区别保持；FP64通过不是FP32修复，
这个输出loss诊断也不替代独立字段根与checkpoint的完整训练门禁。

`scripts/decode_weighted_fixture.py --input V2.json --output V1.json` 在默认2000万元素上限内，
校验完整文件和逐张量哈希，原样保留张量表索引/共享身份及每个FP32值，只把冻结零编码展开。
它让历史独立消费者读取同一实际材料，不运行参考、不重新初始化；旧消费者仍须自己验证
其适用域。大模型继续使用v2，不能通过这个诊断入口无限展开JSON。
