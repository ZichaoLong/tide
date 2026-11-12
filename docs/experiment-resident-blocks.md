# 实验入口的 resident 张量窗口

公共语义依据固定 core-5 SettleGraph ports、带权消息和 KV 局部定义；
设备 owner、named fields、独立根与 raw archive 见 [resident-block-session](resident-block-session.md)。
实验适配层只生成输入、计算任务 loss、选择声明的 owner 和解码实际值，不实现第二套局部公式或调度。

## 默认策略与执行身份

新请求使用 `execution.resident_executor=auto`。在 resident、FP32、单设备、prefill、
内置输出任务、真实 Settle ports 视图及相容 KV sum/qry 区域下，Python-native 三种返回模式
以及 standalone LibTorch 的 outputs/metrics
选择 `block-v2`。同一区内 KV 配置必须一致；不同区域可有不同配置。
计划时不导入 Torch、不运行数值模型，实例化时再次构造真正的 SettleGraph 并检查公共 Model 能力。
standalone 使用 `--resident-executor` 与互斥的 `--resident-block-limits`/`--resident-limits`，
Python 启动器只下发已选择 owner 的容量 schema；旧记录显式下发 general。
普通 PDG/TimedDAG、其他局部记忆、任务插件、streaming、half、多卡及 standalone complete
继续由 `general` owner 处理，并记录具体静态原因。显式 `block-v2` 遇到不相容请求直接拒绝。

execution-v7 记录新策略及输入选择；`resident_dispatch` 记录实际静态选择、原因和请求/生效容量，
并参与实验执行身份。历史 execution record 缺少该策略时仍解释为 general；
已冻结 resolved 配置不重新选择。`execution.resident_executor=general` 保留旧执行 record 身份。
选中的 block owner 若发生设备、容量、求解或数值错误，直接报错，不在运行后切换到 general。

## 容量不是旧队列的改名

`resources.resident_block_limits` 接受：

- `cache`：每私有 KV 状态的条目容量，保留 int64 原始时间；不是累计逻辑时间上限。
- `forward_bytes`、`backward_bytes`、`optimizer_bytes`、`total_bytes`、`workspace_bytes`：独立数值阶段和总持有量边界。
- `windows`：同时保留用于跨窗反向的窗口数；实验内置任务仍每窗一次 optimizer/detach。
- `timeout_ms`：设备程序的有界等待。

默认 cache 按声明有限窗数与数学 KV 窗口推导；显式更小值不会被自动放大。
导入更长历史或延长运行时仍须有足够容量，不能通过丢失历史换取成功。
总预算与 workspace 分开记录；[capacity](../python/tidegraph/planning/resident_block_capacity.py)
计算实际 padded raw 初态/Read0 存储下界，使用显式总分区预留，不能把旧 queue/stages journal 预算套到新 owner。
role padding、数值 tape、allocator 和设备任务资源仍需实际 owner 守卫与测量。

显式旧 `resident_limits`（包括空字典）选择 general，逐项保留旧 queue/stages 等含义；
两类容量同时提供或 block-v2 配旧容量会拒绝。manifest 同时保存用户请求和实际传给 owner 的生效值。
不接受未知容量键，不静默忽略历史限制。

## 输入、loss、完整返回和续接

输入一次交接 `positions[B,L]`、`payload[B,L,P,d+1]` 和独立 `present[B,L,P]`。
最后位置在样例间一致；稀疏、错列和提前结束的端口使用缺席位，发生次数仍按每物理端口连续编号。
使用与独立标量参考相同的 seeded FP32 输入材料；重复边/端口不会因为共享数值而合并身份。

内置 loss 是逐物理输出的 `scale × Σ ‖cG‖²`，从 `output.strength/content` 两个字段直接求值，
其显式 cotangent 为 `2 scale c ‖G‖²` 和 `2 scale c² G`。存在性掩码和连接零根分开保留。
不生成逐事件梯度 Tensor，也不重复乘发送强度。新 block 的三种 return_mode 均计算这个任务标量；
旧 general 的 metrics 推理行为保持原执行身份。

complete 在消费窗口前复制实际字段，step/detach 后获取真实 Continuation，随后在 CPU 展开观测。
fiber 来自输入与候选实际逐边消息；proposal 来自 raw 字段及实际 int64 长度；
描述、控制、Full、端口输出均取候选字段。weighted Aggregate 的 contributions 为空，
该绑定的 history 只含最后查询时间，Next 采用 proposal，Full 广播全部已声明局部输出槽。
解码校验实际 candidate 与 fiber 数目一致，不调用参考、求解器或状态更新补写结果。
outputs 只展开端口输出；metrics 只返回 cut，工作计数来自实际设备字段的块归约。
这些返回边界各自计时，不宣称减少 CPU 返回等同于消除设备 tape。

checkpoint 使用原子 LibTorch schema-2 archive；Python checkpoint 字典观察包含所有实际字段，
包括 raw 填充、绑定初态、原始时钟、存在性、输入账本、全 canonical 参数、alias 与 optimizer 槽。
不能用有效 KV 投影替代精确恢复。SGD 解码覆盖包括 inert owner 的完整 packed offsets，
仅按已注册参数名投影梯度与 optimizer，不把不相连梯度变成零 Tensor。
推理当前复用同一 block owner，保留其空闲 optimizer 存储但不调用 backward/step；构造成本单列。

## 实现与核验入口

| 职责 | 实现/检查 |
| --- | --- |
| 无分配策略、容量身份 | [resident_policy.py](../python/tidegraph/planning/resident_policy.py)、[execution.py](../python/tidegraph/planning/execution.py)、[test_experiment_resident_blocks.py](../tests/test_experiment_resident_blocks.py) |
| 张量输入、字段 loss/VJP | [block_inputs.py](../python/tidegraph/experiment/block_inputs.py)、[test_experiment_block_records.py](../tests/test_experiment_block_records.py) |
| 默认 owner、消费与 detach 边界 | [block_resident.py](../python/tidegraph/experiment/block_resident.py)、[test_experiment_block_adapter.py](../tests/test_experiment_block_adapter.py) |
| 实际完整记录 | [block_records.py](../python/tidegraph/experiment/block_records.py)、一般端口/重复边/零接收/两窗 codec 检查 |
| 精确 archive 观察、全 owner SGD | [block_checkpoint.py](../python/tidegraph/experiment/block_checkpoint.py)、[training_layout.py](../python/tidegraph/experiment/training_layout.py)、[test_experiment_block_checkpoint.py](../tests/test_experiment_block_checkpoint.py) |

CPU 输入与解码检查验证适配边界、独立公式及调用顺序，不能冒充真实 resident 算子资格。
固定提交的安装 `tidegraph check` 仍需独立完整前向、参数根、训练和精确下一窗；
公共设备组件、standalone 消费者、实验默认路径和 FC16 尺度分别记录证据。
对应[a3c固定资格](evidence/resident-block-public-20261112.md)已有两种Memory的独立full检查、
公共新进程及standalone训练archive实测；FC16/LH176容量拒绝没有有效规模吞吐或恢复结论。
