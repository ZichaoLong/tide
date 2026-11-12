# Resident 张量块执行器边界

数学固定为 [core-5 0770593](upstream.json)。本页负责独立调度原语与局部数值边界；
完整 Aggregate→KV→区域→消息组合见[完整图块](resident-block-flow.md)，
Model、Continuation、独立字段根、optimizer与checkpoint见[公共块owner](resident-block-session.md)。
旧 [WeightedFlow](resident-weighted-flow.md) 的资格不继承为新执行器资格。
当前实际运行证据由 STATUS 和证据报告维护；源码、CPU语义源检查、构建、真机和规模分别记录。

## 为什么重新划分执行边界

旧 WeightedFlow 的 DeviceReady 每区域、每样本一次最多消费一个时间；每节点拥有自己的
局部数值程序，TaskServices 为每个服务创建独立程序和通知。共享 workspace 与父子关闭修正
不减少服务数量，也没有让 B×L 张量贯穿整个数值链。窗口反向另按节点、字段构造复制与伴随。
原 FC16 仍发生 task-buffer 失败，见[存储与服务契约](resident-storage-accounting.md)。

新边界让调度产生**整块行索引和有效行数**。同能力组的数值阶段一次消费这一块，参数 owner、
私有状态、消息存在性分别作为张量元数据；程序体按能力组构造，而不是按实际事件展开。
主机可编译潜在依赖与上传边界输入，窗口内的就绪、分组、实际发送与推进由设备决定。

## 已实现的原语与适用域

`tools/device_online/block_dispatch.h::BlockDispatcher` 接一张有限窗口的**潜在任务 DAG**。
它独立实现就绪闭包与选择，不调用旧 DeviceReady、TaskServices 或 CPU 真实轨迹。
潜在任务由结构与模块依赖确定，包含最后可能缺席的任务；不是实际执行序列。
`append_block_settle_flow`已用它驱动完整区域的B×L数值阶段，`BlockResidentSession`另行
实现公共Model绑定、raw Continuation和schema-2 checkpoint。原语自身不负责这些解码或训练职责。
公共Python owner按显式Settle证书及局部能力分派；实验CLI/standalone消费者的实际接入和资格
仍须分别记录，不能从一个组件或公共类存在推断所有默认入口已经迁移。

| 字段 | 含义 |
| --- | --- |
| task `[group,sample,owner,time,unconditional]` | 全部 int64；group 是数值能力组，time 保持原逻辑时钟 |
| dependency `[source,target,transmit_presence]` | 完成义务与消息存在性分开；重复边分别计数 |
| seed / valid | 实际外部存在性与物理槽是否使用；有依赖的潜在任务不能通过 valid 删除 |
| state `[status,unresolved,arrived]` | 未满足的完整输入义务，以及实际到达数；与 payload 数值无关 |
| selection / metadata | 整块任务槽、行数、能力组和 generation；padding 使用独立末尾 sentinel |
| emitted | 数值阶段返回的实际发送存在性，零 payload 可以为 true |
| journal / ledger | 实际数值任务、发送存在性、批次与有效长度；旧窗口未覆盖的尾部不属于当前结果 |

一个任务必须是不可拆开的语义单元。例如 SelStep 任务包含一个完整区域帧的全部候选；
KV 的任务可以是模块证明合法的序列段。行容量只切任务集合，不能切区域归一化分母、
Attention 可见区间或独立状态更新。异形模块按能力分组，不要求所有私有状态形状一致。

设备首先在 scratch 中关闭已经确定缺席的任务：它仍履行所有出边的完成义务，但不产生消息。
依赖全部关闭且收到实际消息、外部 seed 或声明 unconditional 的任务才进入数值阶段。
设备选最早槽的能力组，按稳定槽序打包其所有就绪任务至行容量；同秩独立区域、不同 B/L
坐标可以同批。实际数值发送结果提交后才使后继就绪。

语义 lowerer 必须列全数据、私有状态、区域控制和消息存在性的依赖。对于已可静态展开的
Settle 窗口，这不需要任何真实轨迹。正延时反馈在一个有限潜在时间展开中也可成为 DAG；
CPU 见证包含两样本的正延时反馈。**一般 PDG 动态产生潜在时间/槽位的 lowerer 尚未接入**；
本原语不能据这个见证宣称已经覆盖一般 PDG、非对齐端口或完整 weighted 程序。

## 事务、容量与生命周期

reset 先检查全部描述和严格递增 generation，再发布新窗口状态。choose 仅修改 scratch；
数值阶段也必须写 staging。commit 先检查数值错误、generation、选择及 journal 余量，
再提交状态、实际 journal；错误时不提交这一批，包括 scratch 中已关闭的缺席任务。
调用者只能在成功 commit 后发布 staging payload/raw/history。元数据检查与 payload 发布
在同一设备程序的有序流中完成；这不是跨设备原子事务。

迟到或重复的 completion token 拒绝，error 保持 sticky。完整 owner 应在设备/数值失败后
失效，并从最近完整 checkpoint 新建；原语不自行修复损坏状态。外层仍负责异步借用寿命、
参数版本和保存反向所需的实际数值 tape，不能在尚未完成时复用选中输入或 workspace。

`batch_capacity` 限制一次数值调用的并行任务，`tasks` 是当前窗口潜在任务槽，
`journal_capacity` 限制本窗口保留的实际记录。显式 tensor_bytes 在设备分配前核对元数据额度，
数值 bank/tape/workspace 由完整 owner 另行预算。这些限制均不对累计逻辑时间施加上限；
完成后可以重用窗口资源。跨窗口尚需保留的状态/消息/tape 必须交接，不能通过 reset 丢弃。
当前没有实现窗口内动态释放/再插入任务槽，不把有限窗口原语称为完整持久队列。
当前元数据闭包采用单 worker 有序扫描；它提供独立正确性起点，不作大规模吞吐结论。
调度内核的捕获数量不随实际任务数增长，但完整数值能力组仍需单独核算厂商任务/流额度；
本原语没有自动解除旧 FC16 的 task-buffer 失败。

## 数值阶段与完整owner的分层

以下两个局部边界供完整组合调用；通用WeightedFlow仍承担其声明域，
[完整图块](resident-block-flow.md)与[公共owner](resident-block-session.md)承担新的Settle张量路径：

- `settle_block_plan.h::make_settle_block_plan` 和 `lower_settle_blocks` 从显式 ports Settle
  结构建立 `Local → complete Region → per-node Gate → Full → downstream Local` 潜在任务。
  每个任务代表该 owner 的整个已证明合法 B×L 窗口；rank/delay/stride 使用 int64，
  同秩独立区域、重复边及重复外部端口保留。描述符只保存首位置时钟，实际逐帧时钟由
  数值输入持有。当前拒绝非 ports 绑定、未对齐边、空区域与预算/时钟溢出。
- `block_read_selection.h::append_block_read_selection` 实际组合现有 resident KV/qry、
  Describe 和完整 SelStep 数值算子。输入为 `[R,N]` 的 mass/present/times/node owners、
  `[R,N,d]` 的 content，以及 `[R*N,raw_width]` 的 typed raw Read 快照，其中 `R=B×L`。
  raw 整数与浮点独立存储；调用者必须提供真实的绑定 Read^0/Read^-/Read^+ 快照。
  整个窗口以一个数值程序处理，不以逐事件 Tensor 交接；完整区域按 frame/region 打包，
  padding 不进入候选集合。输出为 `[R,N]` 的 controls 与 active，另保留实际数值 tape。

`append_block_node_gate` 只对选中**目标节点**的帧支持求 any，决定该节点整个 Full 块是否有工作；
它不是对整个区域求 any 后广播。Full 仍必须消费完整 `[R,N]` 支持，逐位置决定状态保留和消息存在性。
Gate 反例包含同区域内无活动节点与有活动节点，区域总活动不能替代目标支持。

组合的 `append_block_read_selection_vjp` 接受各个 control 的独立根位与 cotangent，
先汇集完整区域反向，再经 Describe 和 KV Query 返回 raw、mass、content 与各参数角色梯度。
结构连接位独立于数值零；absent 根忽略未使用的 NaN。保存的 region、Describe、Query tapes
支持独立 reverse 程序，不重读后来被覆盖的原输入/参数。`BlockReadObservations`另接
两通道Mem/total与具名Describe独立根；公共owner通过`block_settle_observations`发布这些根，
与control根分开。局部批量算子汇总同参数bank贡献，外层canonical布局汇总跨角色真实别名。

当前数值组合使用同一 KV 配置与区域配置；参数 bank 通过 owner 显式绑定。
查询支持原始时间窗口、位置和 FP32/FP16 raw，由既有局部能力决定，但本轮新增组合检查只覆盖
FP32 raw、qry/cat/ind/context/cat/quad。其 raw 输入暂按帧提供快照，尚未采用共享 KV 前缀 bank；
这会复制历史存储，不能据整块外形宣称已经取得完整 prefill 性能。其他记忆/异构组需独立接入。

当前实现按职责连接如下，资格应沿各层独立检查读取：

| 职责 | 实际入口 | 交付边界 |
| --- | --- | --- |
| 物理纤维、Aggregate/Upd、Read快照、区域选择与逐边/端口消息 | `append_block_settle_flow`、`append_block_kv_local` | 设备choose/commit选择完整区域，实际presence不由CPU轨迹提供；独立全图检查见[flow](resident-block-flow.md) |
| Full与消息的独立根 | [消息组合](resident-block-kv-messages.md)、`block_settle_observations` | 原始(c,G)、显式cG、Mem/Describe/control及old/proposal/final根分别绑定 |
| canonical共享参数、保留窗和optimizer | `BlockSettleSession` | 消费实际tape/journal，汇总全部物理使用者，None与连接零分开；session检查和公共Model检查不互相替代 |
| Model、完整位置cut、输入账本与磁盘恢复 | `BlockResidentSession`、`WeightedBlockResidentTraining` | 从候选raw缓冲解码Continuation；schema-2显式迁移，使用[独立安装消费者](../examples/consumer_core5/README.md#resident张量块的安装消费者) |

公共新owner当前是明确的Settle/KV适用域，详见[接口范围](resident-block-session.md#public-model-and-python-entry)。
其他记忆、一般PDG动态lowerer、跨设备owner、共享KV前缀存储及规模优化不能从旧general路径
继承为新块实现。已实现的源码与尚待完成的设备/规模资格分开，当前待办只维护在ROADMAP；
本页不复制阶段任务列表。原FC16 task-buffer失败须由新架构的原预算实测单独判定。

## 可运行的检查与证明边界

`tests/test_resident_block_dispatch.py` 在 CPU 执行同一设备元数据语义源，与独立递归 DAG
求值比较；包含随机置换 DAG、重复边、同秩、B/L 合批、零消息/缺席、控制依赖、巨大时钟、
窗口重用、未关闭环、迟到/重复完成与失败不提交。它不证明设备栈或 weighted 数值。

`block_dispatch_check.cpp` 以**一次设备程序提交**执行 choose → 按组 vendor BMM →
设备数值存在性 → commit 循环。改变数值输入会改变实际后继，主机不提供已选序列。
独立 C++ 递归参考核对结果/存在性/状态；另检查空窗、精确时间、真实额度不足与数值拒绝。
这是调度和数值块交接检查，尚无 weighted 全图、一阶 VJP、optimizer 或跨卡资格。

`tests/test_resident_block_selection.py` 检查静态 Settle lowerer、完整区域元数据、逐目标 Gate、
独立 control 根及连接位，CPU 执行的是同一设备元数据语义源。
`block_read_selection_check.cpp` 则执行真实 KV Read/Describe/SelStep 和整链 control VJP；
`block_selection_formula.cpp` 用手写两候选 quad 闭式解、显式 KV Attention 与网络公式作独立锚点，
不调用候选、RegionKernel 或参考递归。材料为 B2/L2/N3、两个同秩区域、两个共享参数 bank，
含大时钟、缺失/零质量、独立非零与连接零根、全部输入/参数梯度和原缓冲覆盖后的 saved reverse。
检查输出的三次 replay 时间只用于诊断这一小组合，不是模型吞吐或完整 resident 性能。
实际设备执行结果须由固定源码资格单列；源码存在和语法检查不等于数值/VJP 已在设备通过。

```sh
python -m pytest -q tests/test_resident_block_dispatch.py tests/test_resident_block_selection.py
python scripts/build_device_control.py --core-build NPU_STANDALONE_CORE \
  --build-dir NEW_BUILD --ascendc-soc Ascend910_9392 --jobs all \
  --checks resident-block resident-block-selection
python scripts/verify_device_control.py --build-dir NEW_BUILD \
  --output-dir NEW_RESULTS --device npu:0 --checks resident-block resident-block-selection
```

构建/运行按固定源码、独立持久服务与空闲设备规范执行。CUDA 从已注册的独立后端生成入口，
源码存在与 CPU 语义源通过都不构成 CUDA 构建或 GPU 真机通过。
