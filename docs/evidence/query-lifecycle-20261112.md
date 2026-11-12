# KV 相容分组、LH 生命周期与显式异步队列资格

本报告固定实现 **`55a3074a441d77720cf00e44433bd702cd09e175`**，数学固定
`0770593838a37c790a568226930567ed60f99b83`。末节 resident KV sequence 单独固定
**`a09735f16095b2953c32a0187e06ae7a0a080014`**；后继位置策略、执行身份和整图 resident
不在这里取得资格。[机器记录](query-lifecycle-20261112.json)保留原始文件/hash、逐项终态、
比较计数、失败见证、构建、实际命令与来源。原始记录和日志已逐项核对，草稿引用的字节hash均匹配。
本轮是正确性与生命周期资格，没有新的吞吐或时延测量。

| 验证范围 | 实际结果 | 结论边界 |
| --- | --- | --- |
| Python/native CPU 定向回归 | 254 passed、9 原 FP32 xfail | JUnit 实际263个testcase；不将xfail记为通过 |
| standalone CPU 核心/owner | 两个可执行检查通过 | 五记忆、异构、端口、SGD、raw与新进程恢复；各检查范围见下文 |
| standalone benchmark | 原构建调用失败；补充构建3个CTest通过 | 原缺少JSON include的失败保留 |
| standalone 实际CLI | 五记忆各1项，共5 passed | 默认列路径三步SGD、独立record任务投影、新进程恢复 |
| LH176 CPU FP64 | 七族独立观察根数值投影通过 | None/连接零仍有20或26处差异，不认证连接等价 |
| LH176 NPU FP32 | messages/raw 两族严格数值失败 | 计算完成且有限，离散支持通过；不判为“仅舍入” |
| LH176 独立安装消费者 | CPU/NPU × Python/native，4项恢复通过 | 每项三步AdamW及新进程精确第三步；初态与上一行不同 |
| 双NPU普通队列 | stock跨卡VJP报EE1017 | 无图执行器、无自定义Function也复现；未修复该策略 |
| 显式每流队列 | 4个整链检查、4个公共消费者、SDK15组合通过 | `TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=1`，未设launch blocking |
| resident KV sequence | a097 独立FP32设备组件通过 | B2/L3/N2、非空raw、独立VJP/saved reverse；不等于整图owner |

这些集合有覆盖重叠，不能相加成一次完整矩阵或全部候选资格。

## 实现、数学与独立检查

[张量窗口契约](../weighted-tensor-windows.md)与[LH独立公式契约](../lh-tensor-baseline.md)
连接固定教材的[KV记忆](../upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
和[区域分配](../upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
55a 将相容节点的 KV Read/Full 和 Describe 按实际参数、配置、查询和活动布局合批；保留每个私有bank、
原始逻辑时间和完整输出字段。共享计算资格与参数共享、私有状态分别处理，合批不合并独立状态。

| 边界 | 代码及独立验证入口 |
| --- | --- |
| KV 相容组、Describe | Python `weighted/kv_column_groups.py`、`dense_describe_groups.py`；C++ `weighted_column_read_groups.cpp`、`weighted_column_full_groups.cpp`、`weighted_describe_groups.cpp`；`test_column_query_groups.py` 与 `cpp/test/weighted_sequence_columns.cpp` 检查异形组、非空左态、共享/分离参数、独立根、None/零 |
| 公共 standalone owner | [WindowExecutor/checkpoint](../../cpp/include/tide/weighted_column_checkpoint.h)、[实现](../../cpp/src/weighted_column_checkpoint.cpp)；[独立Streaming对照](../../cpp/test/weighted_column_checkpoint.cpp)检查一般端口、巨大int64、raw/ledger、共享参数、跨窗图、SGD及真正exec新进程恢复 |
| LH 三层 | `check_lh_tensor.py` 使用独立静态公式与候选实际列；[观察提取](../../scripts/lh_tensor_observations.py)不调用参考补写候选。七根族分别执行，数值投影与连接身份单列 |
| 公共训练/恢复 | [计划图消费者](../../examples/consumer_core5/planned_windows.py)、[实际standalone CLI](../../tests/test_standalone_columns.py)、[跨owner检查](../../tests/test_column_payload_owners.py)；生命周期自恢复不代替独立数学对照 |

相容分组的小型对照中，grouped/single局部实现可能共享数值公式；它检验组装、隔离和伴随汇总。
独立Streaming提供不同调度，LH直接公式提供独立数值/求解实现，resident组件另有CPU字面公式。
这些证据各有所指，不把两条共享局部实现的路径互比写成独立数学证明。

## 构建与 CPU

环境为 aarch64/GCC10.3.1/CANN9.2；Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post3。
Python-owned native与standalone LibTorch2.12.0/NPU SDK分别构建，均在只读固定源码及独立
`background.slice`作业中使用160个构建CPU。两者C++源码hash均为
`ba73626ac0b648232c7f5e9ea16dd9ca82f141bb71df5edbd78511ea44ac0f0b`；构建manifest、二进制和wheel身份在JSON。

原开发快照`core-dev-sdk`在构造参数与`Owner`命名冲突处失败，属于dirty开发快照。
固定55a的`core-55-sdk`完成核心构建，并实际通过`column-check`和`column-checkpoint-check`；
随后benchmark构建脚本因未提供`nlohmann/json.hpp`而失败。补充`benchmark-55`显式提供JSON include，
`weighted-comparison-check`、`weighted-run-check`、`weighted-roots-check`三个CTest全部通过。
补充成功没有改写前两个作业的failed终态。

CPU定向集覆盖位置、分组、放置、共享、训练、计划图消费者、实验分派、五记忆/区域列和core-5层级。
九个原xfail均是`test_settle_columns::test_general_ports_messages_states_and_gradients`的FP32见证：
regional_total、regional_total_zero、regional_threshold、hard_selection各Python/native两项，
proportional的Python一项。原阈值与同材料FP64检查保持。

standalone owner另外覆盖五记忆和异构FC、KV/HB、KV/SSD保留跨窗梯度；三个SGD窗口与实际exec
恢复后再次计算下一窗，核对状态、参数、梯度和动量槽。五记忆实际CLI检查以默认列执行三步，
与显式scalar Streaming比较任务loss和实际工作量，并核对分段新进程的完整checkpoint字节；
complete记录入口及错误执行身份恢复拒绝另测。任务loss投影不代替全部观察根。

## LH176 非空状态与训练恢复

展开计划为 **176个主体节点、22个区域、1058条主体物理边**，14,902,694个可训练参数。
B4/L16、d64、summary32、FFN256、KV单头，`staggered-v1`共有61个实际输入位置；stride9。
配方为`many-lh`/textbook：KV qry、Read+、cat局部分数、att区域评分、content_stats key、
context/cat控制、regional_total/quad，区域发送强度k=1。KV窗口4608，位置scale9/origin1，
Read/Full频率均启用。完整配置可由JSON中的固定recipe重建；这里不是历史原LH9B。

非空对照用每节点3条字面左记忆，起始位置`2**40`。三个层级相同参数与物理输入，均有8495个
候选、2689个selected Full、16071条访问边；直接公式计算11264个dense Full行后mask，
公共/默认计算2689行。二者真实工作不同，不能只按发送稀疏比推定计算量。

CPU FP64 `atol=1e-10/rtol=1e-8`，task、Aggregate、Describe、control、Full、messages、raw七族
分别执行VJP；public/direct和default/direct逐项比较记录相同，所有报告数值/离散字段通过。
但task/raw各20处、其他各26处是候选None对直接公式连接零；所有已连接梯度的数值确为零。
因此只通过声明的数值投影，不据此宣称带weight decay/已有动量时的优化器轨迹等价。

NPU FP32在共享物理9上执行，HF32关闭，`atol=1e-6/rtol=1e-5`。messages/raw均exit1，
`hard_failure=false/numerical_failure=true`；候选和活动支持、整数raw元数据一致，值和梯度有限。

| 失败字段数（每个public/default比较相同） | messages根 | raw根 |
| --- | ---: | ---: |
| controls | 124 | 124 |
| raw浮点字段 | 2112 | 2112 |
| 端口数值输出 | 8 | 8 |
| 所选根的前向值 | 1707 | 2112 |
| 梯度字段 | 2811 | 2795 |

这些是张量/字段计数，不是坐标数。首个端口输出见证最大绝对差`1.745224e-4`；输入梯度见证
messages为`0.0279236`、raw为`0.0518799`，对应参考最大幅度约946.89/801.62。
本报告未定位放大原因，不能把这些数值统称为已证明无害的舍入。原严格失败继续保留。

独立安装的CPU FP64与NPU FP32、Python/native四项，使用同一计划但**默认空初态与起始位置**。
各自连续三步AdamW；独立新进程执行前两步保存，再由第三个进程载入执行第三步，与连续路径
以`atol=rtol=0`比较实际末端口、raw continuation/pending/history/ledger、任务输入/参数梯度
（含None）、参数及全部AdamW槽。四项均通过，实际为`settle-columns-v1`，没有后备。
AdamW为lr1e-5、betas(0.9,0.95)、weight_decay0.01、foreach=false；线程1，deterministic=true。
这是生命周期检查，不是上述非空大时钟案例的数学复验。NPU外层作业因两个严格投影失败仍为failed。

## 双卡异步策略

三个stock探针均不使用图执行器或自定义Function，进行12轮跨卡copy/elementwise及两卡根VJP：

| TASK_QUEUE_ENABLE | PER_STREAM_QUEUE | ASCEND_LAUNCH_BLOCKING | 结果 |
| ---: | ---: | --- | --- |
| 1 | 0 | unset | 前向通过，反向EE1017 invalid event |
| 1 | 1 | unset | 前向/VJP通过 |
| 0 | 0 | unset | 前向/VJP通过 |

原失败发生在TorchNPU/CANN stock跨卡路径，事件生命周期仅为源码调查假设，尚非已证实的根因修复。
`peer-55`整体仍failed。后继整链明确选择第一种成功替代策略，即task queue1、per-stream queue1，
不是launch blocking；物理1空闲、物理9按当前对话授权共享，实际准入与双采样身份保留。

四个Python/native×KV/SSD一般端口检查对照CPU独立scalar调度，包括实际列/消息/raw、共享参数、
组合字段目标的梯度与两步AdamW，以及同进程重新创建owner恢复后第三步的精确结果。
四个公共消费者脚本另做三步任务数值对照和真正新进程精确恢复。后者继承冻结源码`PYTHONPATH`，
**不是独立wheel安装**；与上一节四项安装证据分开。其实际调度仍由主机推进，两个payload owner
放置于npu0/1，控制/算子按mixed-c执行。三个步骤的比较均通过；KV每步483个数值/162个离散叶，
SSD为511/176。

standalone SDK另实跑五记忆×mixed-a/b/c共15项，检查独立Streaming、raw、VJP和两个SGD窗口。
这些通过只适用于明确的队列/设备/配置；不解除queue0失败，不扩大为resident、一般PDG/TimedDAG
或任意规模的异步资格。本报告未测跨卡吞吐。

## a097 resident KV sequence 组件

a097组件使用已核对C++源码hash的55a SDK核心，组件与核心分别记录身份，未把旧核心标为a097重建。
CANN/Ascend910_9392构建与空闲物理1的FP32实跑均通过，设备检查不是只有编译。

用例B2/L3/N2、宽2、raw容量4、窗口8、非空左态、时钟`2**54+17`；稀疏间隔为0/3/12，
另有零写入和缺失行。实际device程序生成old/proposal/final raw；独立CPU逐行公式核对
原始int64、过期、私有owner及VJP的None/零，浮点阈值`atol=5e-6/rtol=5e-5`。
录制反向后将原输入、参数、时间和owner覆盖，再单独执行saved reverse，结果保持。

该组件证明KV状态演进与伴随可以跨B/L在设备执行；它尚不包含完整区域分配、Full/路由、整图owner、
跨窗optimizer/checkpoint、多卡或FC16原预算规模。旧FC16 task-buffer失败未由本组件解除。

## 在新检出中复现

先按[开发入口](../development.md)为固定提交建立匹配的Python/native与独立SDK；
[资格规范](../verification.md)规定只读源码、独立background.slice、资源/时限及设备准入。
JSON的`reproduction.launcher_sources`保存实际历史命令；路径是来源记录，重新执行应使用新目录，
无需恢复旧shm、旧queue或原始运行目录。55a的旧SDK query checker另有计数设施问题，
本报告使用这里列出的实际CLI/owner测试，不借后继checker为55a补资格。

生成相同计划（`PYTHONPATH`含固定源码的`python:scripts`，`TIDE_LH_PLAN`为新的输出文件）：

```python
import os
from durable_records import write_json
from prefill_regression_config import CASES
from tidegraph.planning.recipes import prepare
plan, _ = prepare(dict(CASES['many-lh'], family='kv'), samples=4, positions=16,
                  seed=7, input_pattern='staggered-v1')
write_json(os.environ['TIDE_LH_PLAN'], plan)
```

每个根使用独立output-dir；CPU七族与NPU两族分别见上文。示例：

```sh
python scripts/check_lh_tensor.py --plan "$TIDE_LH_PLAN" --client native \
  --native-library "$TIDE_NATIVE_BUILD" --device cpu --dtype float64 \
  --left-entries 3 --start-position 1099511627776 --root-family messages \
  --output-dir artifacts/NEW-lh-messages
PYTHONPATH="$TIDE_INSTALL" python examples/consumer_core5/planned_windows.py \
  --plan "$TIDE_LH_PLAN" --client native --native-library "$TIDE_NATIVE_BUILD" \
  --device cpu --dtype float64 --output-dir artifacts/NEW-lh-installed
```

双卡须在新分配的设备中显式设置`TASK_QUEUE_ENABLE=1/PER_STREAM_QUEUE=1`并unset
`ASCEND_LAUNCH_BLOCKING`，分别运行`test_column_payload_owners.py`、
`tensor_window.py --peer-device npu:1`和`TIDE_COLUMN_PEER_DEVICE=npu:1`的SDK `tidegraph-column-check`。
原普通策略用`diagnose_torch_peer.py --policy stock`复现；不让环境默认值混入策略比较。
resident组件在a097用`build_device_control.py --checks resident-block-kv-sequence`构建，
再以`verify_device_control.py --checks resident-block-kv-sequence --device npu:0`取得实际设备结果。
完整CPU定向文件清单、安装步骤、SDK构建include及每项原命令均在机器记录。
