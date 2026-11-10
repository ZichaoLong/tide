# 加法基线、默认Full矩阵与借用计费：固定资格

计算实现固定于`e6f7bd5f570b6227d7664e3be45c4f27eb5b611a`；
`0e6916dc8c74cdfe3fc6694a1423b85d191bccbd`增加局部成本探针与实际默认调用观察；
`57439d75da7d356a87639e8cba2f74ba680dcf8d`只修正half预算拒绝见证。
数学仍为`0770593838a37c790a568226930567ed60f99b83`。
[机器记录](additive-interface-qualification-20261110.json)保留构建、命令、JUnit及逐项结果身份；
[有限成本](additive-interface-performance-20261110.md)另列，不能由小型资格推定规模性能或原语言任务等价。

## 本次实现与证据边界

| 变化 | 实现与独立检查 | 通过所说明的内容 |
| --- | --- | --- |
| KV/SSD层链与FC加法基线的非零左态 | `scripts/{layer_tensor_baseline,fc_tensor_baseline,tensor_baseline_state}.py`；`test_tensor_baseline_continuation` | 独立标量调度对照输出/raw终态、共享参数与非零初态梯度、三步SGD/momentum、跨窗保留图、原始int64时间/空隙、真正新进程投影checkpoint恢复 |
| 默认KV Full共享bank矩阵 | `weighted/bank_attention.py`、`weighted_bank_attention.cpp`；`test_bank_attention`及query/snapshot/sequence检查 | 同一底层K/V的矩阵前向、不同偏移/不齐快照、独立快照VJP、零根/部分根、过期极大值、版本守卫；实际公共入口观察到新分派 |
| resident首次借用计费 | `weighted_storage_budget.h`及RowGraph/Describe/Recurrent/KV Full调用；真实设备构造/数值见证 | 固定借用集合后按storage union增量收费；真实新增量仍受限，漏报借用保守拒绝；不放宽owner/journal/workspace预算 |

加法基线只是**输出及raw状态的数值投影**。它把无用字段的None与连接零按数值比较，
不认证完整记录、全部中间根或optimizer槽的结构身份；投影checkpoint也不是公共图checkpoint。
这些义务由原公共实现和相应独立检查承担。FC基线仍有B循环，SSD基线使用有界L²矩阵，
只接受L≤256；它们是实际可达参照，不是无冗余或最优算法的证明。

默认Full在FP32/FP64、strict、无位置变换、真实共享连续bank且padding有界时采用新矩阵；
其他合法布局继续已有packed路径。按能力分派，不按FC/HB/LH名称选择。
保留完整的独立快照伴随，不把数值共享误作根共享；只声明一阶VJP。
图v13/checkpoint v5不变。契约见[查询](../weighted-dense-query.md)、
[层链](../layer-chain-baseline.md)、[FC基线](../fc16-cost-attribution.md)。

## 构建与host资格

匹配CANN9.2、Torch2.12、TorchNPU2.12.post3；Python wheel环境和独立LibTorch SDK隔离。
使用320个affinity可用CPU构建，三套构建及客户端在216.5秒完成；固定清洁源码，
`background.slice`，45分钟/512GiB上限。测试与性能计时未同重型构建重叠。

| 固定测试组 | 结果 | 范围 |
| --- | --- | --- |
| CPU anchors，FP64/FP32 | 56通过 | 新bank、三种有状态基线及原独立公式 |
| CPU一般图，FP64/FP32 | 285通过，2个启动失败 | 分量窗口/反馈与容量、一般Settle端口、区域契约；失败另列 |
| CPU query，FP64/FP32 | 152通过，7保留xfail，2严格数值失败 | 快照/布局/整序列根、weighted窗口、Describe训练 |
| NPU配置query组，FP32 | 37通过，1保留xfail，2相同数值失败 | 设备感知用例在NPU执行，也含CPU辅助检查；不能把总数全称NPU实测 |
| 新默认调用观察 | CPU 4、NPU 2通过 | Python/native公共默认确实调用bank边界；CPU补验使用相同固定probe源码/安装并留存JUnit |

CPU前三组合计493通过/7xfail/4失败。两项`test_standalone_settle_entry`误在Python
extension构建目录找standalone程序；使用真正SDK构建后FP64/FP32均通过。原JUnit不改写。

另两项是`test_sum_memory_and_other_descriptors[float32-{python,native}]`：
Read/sum evidence根`read_program.evidence.encoders.mem.out_bias`差
`1.5355180948972702e-6`，原阈值atol1e-6/rtol1e-5。上一固定554f2f5复现同一差值，
FP64通过；本次新路径只用于Full，不进入这个Read分支。原严格失败保留，不改全局容差，
也不因“有限”就认证所有误差均为舍入。NPU配置组中这两个失败自身仍是CPU辅助用例。

## 训练与恢复，按客户端分列

- 安装的纯PyTorch/Python→native，CPU和NPU共4组：独立三步训练比较、选定返回、
  参数/梯度/optimizer及新进程精确恢复均通过。
- 独立SDK，CPU FP64/NPU FP32的`complete`：独立标量调度、独立字段根与None/零、
  三步训练、raw checkpoint及新进程下一步精确恢复通过。
- 独立SDK相同设备的`outputs`：实际最小返回路径、三步训练及精确恢复通过；
  此模式的独立数值/根为`not_checked`，不借用complete结果冒充它的全部资格。
- 四个SDK流程均验证不同query策略的恢复身份拒绝；CPU SDK kernel检查通过。

## Resident真机见证

资格、补验与测量使用空闲physical11映射`npu:0`，合作队列持锁；未使用共享卡。
服务、FP32 Describe、FP32/FP16 KV Full及两dtype各21个weighted小图组合通过，
分别覆盖局部一阶VJP、实际在线推进、完整返回、raw续接及容量拒绝。
这不新增resident完整训练owner、optimizer或吞吐资格。

首次借用见证：局部额度32,768 B、已持有参数bank262,144 B；原程序可见增长271,152 B，
真正owner新增仅8,868 B。修复后可构造，额度8,867 B时明确拒绝；未声明借用仍保守收费。
Describe另以346,368 B bank、131,072 B局部额度实际执行，新增13,772 B，
两次设备owner切换的score/w/mu/evidence均通过。

原half recurrent失败来自过窄断言：它只接受最终组合守卫的拒绝，但half保守预留
可以让内层先合法拒绝。`57439d7`仅修正见证并打印实际原因，FP32仍要求最终组合守卫。
补验两dtype各24个recurrent配置的独立字段/VJP/saved-state通过。
Half真实新增39,830 B，要求39,829 B时，在`full.up`明确拒绝：该点额度29,141 B，
新增29,404 B。原失败日志与补验分列，未降低生产预算，也未解除全部历史resident拒绝。

## 从干净源码复现

按[development](../development.md)构建、安装固定实现，并在独立SDK环境构建客户端；
不要把Python NPU依赖与SDK库混载。以下`BUILD_NATIVE`等为新构建位置，输出均用新目录。

```sh
TIDE_BUILD_DIR=BUILD_NATIVE python -m pytest -q -o pythonpath= \
  tests/test_bank_attention.py tests/test_tensor_baseline_continuation.py \
  tests/test_layer_tensor_baseline.py tests/test_fc_tensor_baseline.py --dtype both
python examples/consumer_core5/dense_query.py --client native --device cpu \
  --native-library BUILD_NATIVE --no-trace --output-dir NEW_CONSUMER
python scripts/prepare_core5_study.py --recipe tests/fixtures/describe-kv-context-gain.json \
  --samples 2 --positions 3 --output-dir NEW_PLAN
python scripts/export_weighted_benchmark.py --binary --workload NEW_PLAN/workload.json \
  --native-library BUILD_NATIVE --output NEW_FIXTURE.json
python scripts/check_query_sdk.py --binary SDK_BINARY --fixture NEW_FIXTURE.json \
  --device cpu --dtype float64 --return-mode complete --output-dir NEW_SDK_CHECK
```

机器记录保留完整所选pytest清单与环境变量。NPU在队列内初始化设备后运行device-aware
测试；非零左态/投影恢复新增组目前只认证CPU。常驻具体构建/验证命令见
[预算契约](../resident-storage-accounting.md)。GPU、多卡与全部规模不从此次小型检查外推。
