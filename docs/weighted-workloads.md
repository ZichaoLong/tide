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

[固定源码运行证据](evidence/core5-performance-transfer-20261028.md)包含五族Add/Attention主体
预算清单、实际92.62亿参数KV材料与有限执行；大型拒绝和完整返回不通过不能解释为性能通过。
[07e56ca后续同材料复核](evidence/core5-private-projection-20261028.md)保留D64训练及D2048残差失败，
并记录新增真实私有矩阵批的覆盖率、完整调用成本和资源边界。
[474f9af后续复核](evidence/core5-local-allocation-factorized-kv-20261028.md)中，同一D2048材料
已执行测量窗，随后完整状态核对严格失败；D64完整训练限制仍保留，不能解释为等价吞吐。

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

两种CPU benchmark现可显式给`--dtype float64`，沿用同一FP32材料并在两条调度开始前提升；
参数数量、共享身份和拓扑不变。计划的`memory_floor`仍表示其原FP32材料，运行额外记录
`effective_static_tensor_bytes`，静态存储准入按请求dtype计费；转换过程峰值仍不包含在下界中。
二进制fixture保持原FP32编码与SHA256，standalone只在逐张量校验后转换，每个表项一次，保留别名。
CPU更高精度的完整通过不能回填到FP32失败；也不代表与旧core-3不同dtype计时可直接算加速。

诊断每步另比较optimizer完整state_dict。`--checkpoint-check --steps 2`在第一次更新后分别保存
两条调度的真实状态/参数/动量，换owner精确解码；第二窗口分别对照各自未中断执行，完整返回、
参数梯度、参数和optimizer均要求逐值相等。它不向另一调度提供参考状态或轨迹。

Python 诊断可加 `--capture-allocations` 保存两条调度实际调用的分配输入、分支、free掩码、
控制和残差到 `allocations.json`；原函数和反向连接不变，观察不进入性能计时。
记录明确标为成功调用的观察，不冒充抛出的失败。native 的此类全调用观察尚未实现。
`scripts/diagnose_weighted_projection.py --workload PLAN --output-dir NEW` 在实际标量 KV 操作数上
比较展开矩阵与低秩投影，并显式用 FP64 算术诊断局部误差；不据局部误差推断完整训练通过。

批量分配拒绝时，两种 benchmark 消费者保存候选实际失败行 `allocation-failure.json`，
包含原 dtype/device、完整分配输入、实际响应与控制、区域节点/int64时间、执行阶段与窗口。
独立 `python scripts/diagnose_allocation_failure.py --failure ROW.json --output NEW.json`
只用标准库 Decimal，以捕获的二进制浮点值为精确输入，从扩张区间独立求根；不调用生产分配器、
不重建模型。输出真实可行性、候选值误差和隐含乘子差；它认证局部分配，不能替代完整图/VJP。
`AllocationFailure.record()`、公共 C++ 异常及未覆盖的拒绝类型见[分配契约](weighted-allocation.md)。

推理测量不导出未更新的完整参数副本，事后参数比较仅用于训练。两条路径在独立核对前释放
候选执行器，再构造参考；完整候选返回仍逐项保留和比较。测量计时边界不变，验证阶段的额外峰值
与候选执行峰值分开解释，不能因少了一个参数副本就声称模型执行本身节省了同等内存。

`scripts/decode_weighted_fixture.py --input V2.json --output V1.json` 在默认2000万元素上限内，
校验完整文件和逐张量哈希，原样保留张量表索引/共享身份及每个FP32值，只把冻结零编码展开。
它让历史独立消费者读取同一实际材料，不运行参考、不重新初始化；旧消费者仍须自己验证
其适用域。大模型继续使用v2，不能通过这个诊断入口无限展开JSON。
