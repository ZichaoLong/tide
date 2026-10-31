# 实验放置与逐卡容量预览

本页是公共实验入口的静态规划契约，复用固定 core-5 数学、
[payload owner](execution-placement.md)、[带权 Read/控制位置](weighted-placement.md)
及[resident owner](resident-weighted-placement.md)。图身份、参数绑定和 checkpoint
版本保持[公共语义](semantics.md)定义。规划无需导入 Torch、分配模型或运行参考轨迹；
规划完成与实际构建、执行、严格验收及容量资格分别记账。

## 放置一个模型

```python
from tidegraph.planning import ExecutionSpec, plan_payload_placement, capacity_budget
from tidegraph.planning.recipes import prepare

plan, inventory = prepare(
    dict(topology="hb", family="kv", width=64, region_width=2, depth=1),
    samples=8, positions=4,
)
devices = ("cuda:0", "cuda:1", "cuda:2")
placement = plan_payload_placement(plan, devices, strategy="balanced-shared-groups")
execution = ExecutionSpec(
    client="python-native", device=devices[0], peer_devices=devices,
    node_devices=placement["node_devices"], placement="mixed-c",
    dtype="float32", scoring_dtype="float32", communication="ordinary",
    return_mode="complete",
)
budget = capacity_budget(
    plan, execution, placement=placement, mode="inference", checkpoint=True,
    resources={"logical_windows": 5, "live_results": 1,
               "device_reserve_bytes": 4 * 1024**3},
)
```

`devices` 是当前进程可见的逻辑设备，支持任意正卡数，包括三卡和未来七卡；
CPU 使用单个 `cpu`。同一计划不能混用 CUDA/NPU/CPU 后端。设备列表不能证明设备存在，
不能把 T4 使用 `cuda:N` 的语法有效性当作 sm75 构建或实际算子通过。
`node_devices`、`region_devices` 和 `parameter_devices` 给出实际模型所需的 owner。
`cross_device_edge_ids` 保留每条物理跨卡边，包括重复边；不是活动边预测。

| 策略 | 可解释规则与拒绝 |
| --- | --- |
| `explicit` | `node_devices` 是完整列表或节点 ID 到设备的映射；遗漏、非法设备、拆开真实共享组均拒绝。首节点属于首设备。 |
| `region-round-robin` | 节点使用 `devices[region % len(devices)]`，保留既有实验映射；仍检查共享约束。 |
| `balanced-shared-groups` | 从 canonical alias 形成不可拆分组件，首节点组件固定首设备；其余按参数与 inert buffer 总字节降序、最小节点 ID 平票；选择可放入且当前负载最小的设备，设备输入顺序平票。 |

贪心权重是精确静态 payload bank，尚未包含动态状态、workspace、反向或恢复峰值。
`device_limits={device: bytes}` 可限制**已扣预留量**的静态负载；一个共享组件不能使用
多张卡空闲量的和。`largest_indivisible_group` 给出最大组件、真实参数和共享见证。
超出单卡的组件需要用户明确评估张量切分等扩展；规划不复制独立参数绕过约束。

Python `ownership.place_payloads` 和独立 C++ `tide::place_payloads` 是运行时消费者。
每节点参数/缓冲区同卡；真实共享节点必须同卡。区域所有参数也同卡，默认跟首成员，
已有共享 canonical owner 时跟随该 owner。一个区域的独立节点可跨卡，完整区域
Attention、归一化与发送控制仍在区域控制位置处理，不拆成几个独立区域。
端口 scale 保留单 canonical leaf，远端使用通过可微复制取得视图。
本规划针对公共工厂在 workload 身份中声明的绑定；运行后自行改变 alias 的模型须重新
提供相应配置和核对，不可沿用旧参数库存。

放置身份 `tide-payload-placement-v1` 与数学模型/输入身份分开。
`ExecutionSpec` 新增 `node_devices`、`communication`、`return_mode`。
三者使用默认空映射、`ordinary`、`complete` 时仍生成原
`tide-weighted-execution-v1` 字节及身份；任一非默认使用 v2 并记录全部新增字段。
历史 v1 不追改为含新字段的记录。

## 请求通信不等于实测通信

| 请求 | 需要实际运行确认的事实 |
| --- | --- |
| `ordinary` | 普通显式复制 API；复制成功不能证明底层直接 P2P、NCCL、原子或 capture。 |
| `blocking` | 显式同步的执行配置；记录生效机制、同步边界及端到端成本，不把旧环境变量观察当作所有客户端通过。 |
| `cpu_bridge` | 显式经 CPU 的传输；实现和一阶根、共享梯度、精确恢复需各客户端独立检查。 |

`capacity_budget.communication.effective` 始终为 `null`，运行报告负责记录生效方式。
resident 只接受 `ordinary` 作为现有协议请求；不能请求 eager 的 CPU 桥或 blocking
来绕过设备协议。CUDA peer gate 仍检查双向直接访问及原生跨卡原子，见
`tools/device_online/cuda/peer.cpp`。当前 3090 的普通复制证据不能授予多卡 resident 资格。
NPU resident 的服务流、SQ/CQ 和在途任务资源也不是 HBM 字节预算可以保证的能力。

## 有限范围与逐卡预算

`capacity_budget(plan, execution, placement=..., mode="inference"|"training",
optimizer="sgd"|"adamw", checkpoint=True, resources=...)` 只返回预览。
每张卡都有自己的静态 bank、分项估计、未知项和余量；不自动准入、不静默改变 batch、
窗口、缓存、模块或精度。`static-refusal` 表示已有确定约束拒绝；
`requires-measurement` 表示仍需现场检查和小规模实测校准。

| 分项 | 依据与边界 |
| --- | --- |
| 参数/物理基础存储 | canonical 去重参数按 payload dtype 精确计数，另计每节点 inert buffers；共享组不复制数学参数。resident 额外 role bank 副本单列未知，不能拿 eager bank 当 resident 全预算。 |
| 私有状态/KV/卷积 | 五种记忆的真实浮点字段与 int64 元数据；有限窗最大写入次数，保留未被零接收清除的原始过期项。resident 预留 initial/current/old 三份；训练保留窗另列快照。 |
| 消息/输入/返回 | 全候选均发送的结构上界；物理重复边和端口分开。完整返回采用保守字段模板估计，动态区域 Attention 矩阵仍是未知 workspace。 |
| 梯度/master/optimizer | 全 canonical trainable 参数的 payload 梯度；FP16 另计 FP32 master 和 master grad；SGD预留一 momentum 槽，AdamW 两槽，实际是否出现梯度由运行核对。 |
| checkpoint/恢复 | 独立 C++/resident staged bank、状态及槽；Python eager推理从CPU解码权重并复制到既有bank，不重复预留设备权重，主存副本仍计费；Python 训练另预留 optimizer deepcopy preflight，FP16 保存时的 master/槽临时副本单列；原生 codec 的单 Tensor `2^28` 元素上限继续检查。 |
| workspace/activation/communication/journal | 未给具体预算时 `null` 并列入 `unknown_components`；显式保留量不等于实测峰值。字段/评分复制、allocator、对象、vendor 和服务资源仍需观测。 |

`resources.logical_windows` 默认 **1**，调用者应计入 warmup、测量与恢复后窗口。
`live_results` 是同时保留结果数量；训练的 `retained_windows` 是保留的 autograd 窗数量。
默认从公共工厂空状态起步；导入已有 checkpoint 时须把其中历史/KV 纳入声明有限范围，
不能拿新运行的一窗预算覆盖旧缓存。
KV `window=0` 依照累计有限窗增长，不以固定 cache 截断。Settle 的对齐结构按原逻辑位置
给上界；非对齐 TimedDAG 按所有物理路径数并以有限时域封顶；PDG 反馈按每整数时间
至多一次 node/sample 作用给上界。它们都不运行数值模型，也不向候选提供真实轨迹。
这些上界可能较松，不能当作实测 candidate、Full、消息或时间批长。

resident另输出 `resident_layout`，与 `weighted_flow_preflight` 同公式读取显式
`resources.resident_limits` 的 queue/stages/cache/tensor_bytes，逐卡预留 physical raw
initial/current/old，完整前向journal、queue/control在coordinator。
FP16也沿用该C++前置守卫的FP32浮点预算，实际storage dtype单独记录。
`effective_owner_contract` 明示resident区域控制、journal及optimizer向量在coordinator，
不能把eager区域payload owner当作其实际位置。resident表中的canonical参数仍是逻辑基数，
额外physical role banks、数值tape、peer packet及服务资源为未知，须继续使用实际运行库准入。
AMSGrad额外maximum槽、optimizer表/标量和下游head/loss专属内存也须按实际配置补预算。

B128/B512没有额外的静态小batch白名单。原始时间、metadata和预算用整数；
同一模型的参数量不随batch虚增，私有state、完整返回与恢复状态分别随真实batch计费。
保守默认stages与queue可能随batch放大，物理填充journal会比逻辑活动量大很多。
这时应检查明确的配置与preflight数值，再用有限实测校准；预算默认不构成硬件支持上限，
也不能通过静默缩减batch、cache或使用CPU真实轨迹获得放行。

```json
{
  "logical_windows": 5,
  "live_results": 1,
  "retained_windows": 1,
  "devices": {
    "cuda:0": {"capacity_bytes": 25769803776, "free_bytes": 25000000000, "reserve_bytes": 4294967296},
    "cuda:1": {"capacity_bytes": 25769803776, "free_bytes": 24000000000, "reserve_bytes": 4294967296}
  },
  "workspace_bytes": {"cuda:0": 1073741824, "cuda:1": 1073741824},
  "host_reserve_bytes": 8589934592,
  "disk_reserve_bytes": 1073741824
}
```

`activation_bytes`、`communication_bytes`、`journal_bytes` 也接受逐卡映射或每卡统一数值。
现场可填 `host_free_bytes`、`disk_free_bytes`，文件/构造上限使用现有
`checkpoint_max_bytes`、`max_materialize_bytes`。示例数字只说明单位，不是现场探测结果。
预算过线先拒绝分配；正余量仍需验证实际参数、状态、消息与计算位置，分别测构造、前向、
反向、optimizer、保存和新 owner 恢复峰值。小型成功不外推大尺度或其他客户端。

`complete`、`outputs`、`metrics` 是独立返回请求，身份和计时口径分开。
规划展示返回需求大小，但在运行时证明避免完整记录分配之前，仍为完整内部记录预留；
减少 CPU 返回字节不能自动等同减少设备计算或构造成本。
完整独立记录与 checkpoint 精确解码检查不由最小输出模式替代。
resident 的实际公共调用与轻量返回字段见[按需返回契约](resident-weighted-flow.md#按需返回与完整诊断)。

实现位于 `planning/placement.py`、`planning/capacity.py`、`planning/resident_capacity.py`、`planning/execution.py`；
相关契约检查为 `tests/test_experiment_placement.py`。本页描述已实现的静态入口，
实际多卡构建、运行、精确恢复和性能证据见 STATUS 与对应审阅记录。
