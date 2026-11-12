# Resident 块公共接入、训练恢复及规模拒绝

固定实现 **`a3c4f62dcc3fce9bbf43e5a8f4ce5578c561efc4`**，包含`4248bde1`公共接入与
sum Memory、`d00f0d76`异宽source伴随gather修复；数学固定`0770593`。
本轮小型 **KV sum/qry、单NPU FP32** 的独立数值、训练及新进程恢复通过。
FC16和LH176规模试验均被明确容量守卫拒绝，**没有有效吞吐或规模恢复结论**。
[机器记录](resident-block-public-20261112.json)保存构建身份、逐进程命令、终态、配置、日志hash和拒绝公式。
文件名中的v3是本次构建轮次；公共算法身份仍为`settle-block-v2`。

## 实现、工具链和资源

新[Model owner](../resident-block-session.md)将真实Settle端口模型、参数别名与raw状态接入
独立设备块调度。sum使用自己的Memory tape/VJP，不合成Read Q/K/V参数，不引入当前content依赖。
`Read^0`绑定初态、old/proposal/Full快照与独立Mem/total根保留。
[实验入口](../experiment-resident-blocks.md)的execution-v7按静态能力选择`auto|general|block-v2`，
输入/loss均交接B×L列，完整返回只解码候选实际字段，checkpoint使用原子schema-2 archive。
SDK输出任务同样接入；其complete观察仍显式保留general。已选owner的容量/设备失败不运行后备算法。

Python-owned与独立SDK两套resident构建均成功，component hash相同：
`86fcf8f75c5041e89f80dc5caa9ac924748713364d15bb33163083193b4a7369`。
二者复用各自的`00bb5ac2`核心，构建入口核对当前C++源码hash
`0d01ac0ca73870abf99a31a1b004ad953a33f5e36fbf2e914d6713c96b88ad61`及真实二进制，
没有声称重新构建了a3c核心。两套runtime都是aarch64 **ABI1**，不是两个ABI编号。

Python模块为`ascend/dev-workspace-9.2.0`，Python3.12.13、Torch2.12.0+cpu、
TorchNPU2.12.0.post3；SDK为`libtorch-npu/2.12.0-cann9.2.0`，AscendC目标`Ascend910_9392`。
两套运行库分shell加载，SDK测试wrapper不加载Python TorchNPU。
设备资格进程启动时unset `ASCEND_LAUNCH_BLOCKING`、`TASK_QUEUE_ENABLE`、`PER_STREAM_QUEUE`；
这只记录请求环境，不推断厂商内部默认或推广到双卡。

两构建服务在`background.slice`，各CPU160核/512GiB/2400秒；后继benchmark构建CPU64–319、
256GiB/1800秒。设备资格各有独立CPU池/256GiB/1800秒，主机算术1线程；规模点2线程，
CPU48–63、每点600秒。native使用空闲队列分配的物理5，SDK物理10，standalone和规模物理5。
本轮无有效规模计时，因此不作设备峰值或独占吞吐声明。冻结源码与构建使用有界shm，未遇存储阻塞。

## 验证结果与原失败

| 项目 | 实际结果 | 能证明的范围 |
| --- | --- | --- |
| Python-owned CPU门禁 | 70 passed，零skip/xfail | 静态分派、输入/loss、完整解码、checkpoint布局、dispatch/sequence/flow元数据；不是NPU数值 |
| SDK Model检查 | passed，qry/sum | 原参数/别名、低秩source、异宽/异rank padding和非空raw导入 |
| 原SDK外层构建 | **failed** | resident构建及Model检查已通过，随后benchmark缺显式nlohmann JSON头路径；失败保持 |
| 后继benchmark构建 | passed；4 CTest passed | 显式提供已有项目JSON include后构建、加载及CPU adapter检查；未改变数值源码 |
| Python-native真机 | 6个进程passed | qry/sum公共消费者及各自fresh-process，再各自公共实验full检查 |
| standalone SDK真机 | 9个进程passed | Read-selection、messages、sequence、完整flow、session，qry/sum公共消费者及fresh-process |
| standalone实验客户端 | 两项pytest passed，零skip | sum/qry分别whole三步、prefix一步、新进程resumed两步；全部archive张量逐值相同，outputs/metrics任务与计数一致 |
| FC16/LH176规模 | **两项failed** | 具体子预算拒绝，未得到训练步、吞吐或精确恢复 |

不同粒度的检查不合计成一个“总测试数”。原SDK错误是构建入口材料问题，原scale错误是容量问题；
后继成功、原外层failed与每个内部检查分别保留。全部七个作业已终态，无存活服务cgroup。

设备Read-selection分别检查qry/sum的独立字面公式、完整区域、Mem/total及控制根，尤其sum没有
当前content或Read query参数的虚构梯度。messages另验Read/proposal独立快照、七种根、真实共享、
未乘强度的(c,G)、显式cG、无连接与十项拒绝。完整flow用独立标量公式检查十种根、三种Read快照、
B2/L2、重复物理边/端口、共享参数及精确int64。

session实际检查两个保留窗口的BPTT、三步AdamW与raw恢复；public Model消费者使用独立Streaming
日程，包含非零低秩source与异宽/异rank padding、实际输入/参数根和Continuation。
这覆盖d00f的非连续source伴随device gather修复。qry/sum各由第二个进程读取archive并核对实际下一步。
Python安装消费者分别调用同一公共C++ owner，以独立Python标量日程比较，亦有三AdamW及新进程。

公共实验`check --scope full`用FC两区域×两节点、d4、B2/L3、非齐输入，分别qry/sum；
独立完整前向、output/connected-zero参数根、三步SGD及精确下一窗通过。
这里的恢复是同进程新owner；不能冒充fresh-process，后者由上面的消费者和standalone测试另行提供。
standalone outputs/metrics测试不自称完成事件根或独立数学完整资格。

## 两项规模拒绝的具体含义

FC16是[原resident代表](resident-shared-bank-20261112.json)的B4/L16/d128/H2/FFN512、
4区域×4节点、KV qry、61个实际输入位置；**不是**主成本表的B8/L128任务。
保留原cache49、backward/optimizer及总预算18,522,046,464字节；
新forward预算6,375,342,080是旧tensor与retained之和，workspace512MiB。
新owner不使用旧queue/stages。构造0.95159秒，构造后进程设备分配88,811,520字节。
warmup在[查询预留](../../tools/device_online/weighted_kv_query_internal.h)拒绝：

```text
rows=256, banks=4, cache=49, d=128, ds=32
required = rows × (256 × cache × (d×ds+d+ds+1) + 256)
         = 13,670,416,384 bytes
available = 6,120,144,499 bytes
```

required是保守模型，不是测得的峰值。多bank内部确有逐输入行矩阵展开和每query历史重投影，
见[静态审阅](lh-block-static-20261112.md)；因此既不能说实测需要13.67GB，也不能只减守卫常数宣称问题解决。

LH176沿用00bb材料，B4/L16/d64，qry/sum混合，14,902,694可训练、14,904,826唯一参数元素、
5809 canonical owners。此次借用同一总预算/分区作一次有限边界检查，不代表该分区已按LH校准。
没有warmup；在[session构造](../../tools/device_online/block_settle_session.cpp)即拒绝，早于窗口执行：

```text
optimizer_bytes = 402,653,184
parameter_layout allowance = optimizer_bytes / 8 = 50,331,648
FP32 values + owner flags = 4 × 14,904,826 + 5,809 = 59,625,113
```

这是canonical layout子预算不足，不是设备OOM，也不是本次运行发现了raw拷贝瓶颈。
[DeviceOptimizer](../../tools/device_online/device_optimizer.cpp)还有按算法、槽和tiles计算的实际守卫，
本次尚未到达；仅取消`/8`不能证明所有常驻存储已正确计费。
应统一layout、optimizer私有槽/metadata、role bank及总存储的实际模型，再保留边界拒绝检查。

两项各一次，未加预算、缩尺寸或修改容差重试。由于比旧FC16 EE1019更早拒绝，
本轮既未复现也未解除那个设备任务资源失败。小图独立根与恢复通过不能替代这些规模验收。

## 从干净源码复现

按[development](../development.md)在a3c固定检出构建两套匹配core与resident，
独立SDK benchmark显式提供项目私有JSON include。机器记录保存原完整build/launch命令；
旧shm、venv和日志路径是身份引用，不是必备运行目录。
构建和设备任务按当前站点规则放有限持久服务，以下是其workload入口：

```sh
python examples/consumer_core5/block_resident.py --device npu:0 --memory-read qry \
  --native-library CORE --resident-library RESIDENT --checkpoint-path NEW.pt
python examples/consumer_core5/block_resident.py --device npu:0 --memory-read qry \
  --native-library CORE --resident-library RESIDENT --checkpoint-path NEW.pt --restore-check
```

sum使用新的checkpoint重复。SDK运行`tide-block-resident-check --device=npu:0 --dtype=float32
--memory-read qry --checkpoint-path NEW.pt`，第二个进程加`--restore-check`；sum同理。
组件检查名称与完整argv位于JSON的`device_processes`。
standalone实验资格用`tests/test_standalone_resident_blocks.py`，设置匹配的
`TIDE_WEIGHTED_BENCHMARK`、`TIDE_BUILD_DIR`与已分配的`TIDE_STANDALONE_BLOCK_DEVICE`；
Python fixture导出和SDK执行分别加载对应runtime，可沿用机器记录的wrapper方式。

机器记录的`small_configurations`可直接换site路径后交给`tidegraph check CONFIG --scope full`。
`scale[].configuration`保存FC完整配置和LH非模型配置。LH模型从固定源码重新生成：
`tidegraph.planning.recipes.prepare(dict(CASES['many-lh'], family='kv'), samples=4, positions=16,
seed=7, input_pattern='staggered-v1')`，`CASES`在`scripts/prefill_regression_config.py`。
按`model_definition_reproduction.keys`取plan字段装入`model.definition`；其canonical JSON hash随报告保存。
已用Python3.12纯planning复查：生成模型及完整resolved config身份与原运行一致。
在此步骤保留原site时应匹配原config_id；换site/build生成新执行身份。
不要把原失败配方排成默认重跑队列；出现新表示/容量实现后才做有界后继。

本报告不新增新block的循环记忆、half、多设备、一般PDG或GPU资格；旧general能力保持独立。
也不把公共接入或一次设备提交当作共享历史bank、角色分组GEMM和规模性能已经完成。
当前剩余工作与下一动作只维护于ROADMAP/STATUS。
