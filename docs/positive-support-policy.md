# Shannon 正支持的显式有限精度模式

对应固定 core-5 `0770593` 区域分配 §4.1/§6、Aggregate 与 KV 定义。
来源：[TL-REQ-003](https://github.com/ZichaoLong/tide/blob/43bc5c457ed948ad81c4fbd9e41ae61516300e16/docs/requests/TL-REQ-003.md)。
这是数值存储/求值与声明一阶 VJP 的选择，不改发送评分、正则、loss 或训练配方。
默认 `strict` 仍拒绝正支持下溢；新模式不能把旧失败改成旧模式通过。

## 公共入口与范围

`RegionalConfig(support_policy="positive-floor-v1", regularizer="ent", ent_alpha=1., ...)`，
或 `allocate/allocate_batch(..., support_policy="positive-floor-v1")`。
单设备 Python eager、KV、FP32/FP64；scalar 与 packed 使用各自调度。
NPU 仅 FP32；CUDA 尚无本模式真机资格。native/standalone/resident、FP16、多设备、
其他记忆及其他正则不在首版范围，配置/编码明确拒绝，不借旧实现静默执行。
局部分配本身不依赖图的记忆类型；KV 限制属于整图绑定。

已安装 CLI 的模型 recipe 可用 `region_overrides.support_policy`，任意公共 resolved
模型的区域配置可直接填写该字段。例：

```json
{
  "regularizer": "ent",
  "ent_alpha": 1.0,
  "region_overrides": {"support_policy": "positive-floor-v1"}
}
```

以上是 `model.recipe` 的增量，不是完整实验配置；其余模型、输入、执行字段保持原值。
`local_score=cat, region_score=ind` 无须改变。放置/scoring dtype 仍显式选择；
评分到 payload 的 cast 若再次丢失正支持，原守卫仍拒绝，不再次猜测性补值。

## 前向与拒绝边界

在原求解器和原残差检查之后，普通 Shannon 分支有数学正支持的坐标
（`caps>0`、有限 log 响应）若控制舍入为零，存为该 dtype 最小正次正规数：
FP32 `2^-149`，FP64 `2^-1074`。它不同于最小正规数 `finfo.tiny`。
只替换零；已可表示的次正规数、真实零、零 cap、全零/全 cap 单点分支均不改变。

除原残差界之外，补入总质量还须满足

`added <= 64*N*eps*max(sum(original_controls), lower)`

且新总量不超过 `upper` 加同一相对预算，所有坐标必须可存为正且不越 cap。
全行下溢、极弱总量产生实质相对变化、非有限 log 响应或后端冲零均拒绝。
新增分配拒绝继续给出 `AllocationFailure.record()` 的实际行、原始控制和响应；
它不是成功结果。其他非有限、求解器残差、非法配置异常仍失败。

实际 HB16 行的原数学尾部约 `3.105879657443601e-46`，本模式存为 `2^-149`。
总质量变化极小，但该坐标的相对误差并不小。多个不同尾部都补为同值会丢失比例；
后继聚合按**已存质量**求值，不声称保存原任意小实数的比例。需要这一性质时应另提
log/缩放载荷方案，而非继续扩大本模式的保证。

## Aggregate、KV 与一阶 VJP

- Aggregate 先按最大质量缩放并归一化，再加权内容。单消息质量为最小次正规数时，
  内容仍能保留；物理重复边仍各占一行。零总量仍走原零分支。
- Aggregate VJP 是已存质量/内容下的加权均值导数；单消息对质量导数精确为零。
  多个极小质量、不同内容可能要求不可表示的质量伴随，此时明确拒绝，不裁剪。
- Shannon VJP 身份为 `ent-positive-floor-v1`。保存未补值的实际居中响应，
  用其 log 强度求自由/饱和面及 KKT 导数；不是 floored 控制的 `c/tau`，
  也不是普通 clamp 的导数。log 收缩保留可表示的“极小斜率 × 大伴随”。
  等式分支共同 score shift 仍为连接零，None 与零根、版本检查和共享汇总保留。
- KV 存原始正写入质量、内容、int64 时间/计数。qry/Full 融合
  `min(total,1)*softmax(logits+log(weights))@values` 的一阶 VJP，
  避免中间 log 导数溢出或小伴随先下溢。sum 读取保持原式。
  原始时间衰减、窗口可见性和普通浮点舍入不变；短通道衰减仍可能舍入到零，
  不删除 raw KV 项，也不影响未衰减通道或结构上的写入。
- 仅一阶；不可表示/非有限公开伴随明确失败。该策略不保证任意大分差、
  任意长序列、任意损失根均可在有限 dtype 中求值。

本模式的反向是复合数值程序的声明 VJP：分配用未补值响应，后续函数用已存输入。
不把整条链宣称为高精度原实值程序的精确梯度；独立局部公式与完整小图分别验收。

## 身份、诊断与恢复

新策略进入区域、涉及的 memory/full/aggregate profile 及模型/图身份。
默认 strict 字段不进入旧身份，旧配置锁继续保持原义。固定数学版本、图编码和
checkpoint v5 不变；无新增持久 log 张量，raw KV/消息仍完整编码。

旧 strict checkpoint 不能直接按新图身份加载。需要从旧 checkpoint 引入权重/训练材料时，
先在原固定安装中读取，并显式记录数值模式转换为新运行；本交付不提供自动身份改写，
也不称转换为旧轨迹精确续接。新模式同源码/环境/设备的精确恢复另验，包括下一窗、
完整状态、独立梯度和 optimizer 槽。下游应用的 RNG、scheduler、数据进度仍由下游保存。

`Allocation.support_repairs` 在普通新模式行给出布尔补值坐标；
strict 及 scalar 单点分支为 None。可选公共观察：

```python
from tidegraph.weighted import observe_support
with observe_support() as observation:
    result = run_candidate(...)
    loss.backward()
summary = observation.metrics()
```

只在显式 metrics 时取回计数；包含真实求值和 autograd replay，不等于唯一事件数。
提供补值坐标/行计数以及每种设备/dtype/候选数的局部候选索引直方图
（前16组、每组前32列）；不能将局部索引当作全局节点ID。
无逐步全量日志或参考轨迹反馈。性能成本尚未测量，不宣称比 strict 或 FP64 快。

## 独立检查与复现

[实际行](../tests/data/tl_req_003_shannon_row.json)来自固定下游证据；
[局部检查](../tests/test_positive_support.py)用80位 Decimal/闭式公式及普通尺度 gradcheck，
[整图检查](../tests/test_positive_support_graph.py)使用两区域、重复边、不齐 B2 输入、
KV sum/qry、独立 CPU FP64 scalar 参考、完整根/共享参数、AdamW及跨窗。
这是小型机制见证，不是原 HB16 的第64→87步或完整学习效果验收。

```sh
# 在调用者选择的匹配私有环境中；不要求旧产物目录。
python -m pytest -q tests/test_positive_support.py tests/test_positive_support_config.py tests/test_positive_support_graph.py
# NPU：由站点设备队列设置物理可见性，逻辑设备从0开始。
TIDE_POSITIVE_SUPPORT_DEVICE=npu:0 python -m pytest -q tests/test_positive_support.py tests/test_positive_support_graph.py
# 实际行的已安装公共 API 消费者
python examples/consumer_core5/positive_support.py --row tests/data/tl_req_003_shannon_row.json
```

整图套件的 fresh-process 检查在两个新进程保存/恢复完整下一窗和 optimizer；
可独立运行 `tests/positive_support_resume.py write|resume --directory NEW --device cpu`。
正式资格需固定源码/安装并用持久任务，不能以这些命令的存在代替运行证据。
