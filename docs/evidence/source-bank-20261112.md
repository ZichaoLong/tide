# 来源共享、区域传输与有限成本

实现固定 `72e1d22c5071da604b43079d96e90b960f919799`，数学固定 `0770593838a37c790a568226930567ed60f99b83`。
本轮交付唯一来源 bank、incidence Aggregate 与区域参数/实际操作数传输复用；定向资格有两项新的 FP32 严格失败，原阈值不变。
**LH 默认路径的完整训练成本没有明显改善**，CPU/NPU 约为 1.406/2.902 秒。
完整配置、32 点原始身份/工作量/成本、构建、测试与复现命令见[机器记录](source-bank-20261112.json)。
本报告不包含后继 `08a349ce` resident 资源资格或 `9ae62d0d` 增量规划，不能据此宣布 P1—P6 完成。

## 实现改变了什么

`SourceBlocks(values, owners)` 保存唯一实际变换结果及物理使用索引。
[来源绑定](../weighted-source-binding.md)仍分别保留物理边、参数分类、相同值的独立根和真实共享；
默认 Aggregate 直接读取来源 owner，显式观察时仍可取得独立投影。
[Aggregate](../weighted-aggregate-batch.md)的 `SourceIncidence` / C++ `AggregateSourceLayout` 保留 int64 重数、行与接收者映射。
令 C 为物理 incidence、m 为来源质量、z 为来源内容，其正总量分支为 `T=C@m`、`u=(C@(m*z))/T`；
零总量与原正支持守卫保持。反向通过 incidence 转置汇总，质量伴随使用中心化的 `z-u`。
一次仅保存一个接收者的 S×R×D 临时量，不再保存 G×R×A×D 接收者展开载荷。
这降低存储与根数量；中心化反向仍可能做 G×S×R×D 算术，不能从字节减少推断加速。
真实单 owner 的不变质量根保留数学上的连接零；不同独立根即使值相同也不合并。

[区域列](../weighted-region-batch.md)的 `RegionParameterViews` 在一次调用内复用真实参数的 device/dtype 转换，
按候选数分组时不重复转换同一参数；实际发送操作数按别名共享传输。
C++ allocation VJP 在版本检查之后，为完全未使用的组返回六个 undefined 输入伴随，与 Python 的 None 语义一致。
相关定向入口为 `test_aggregate_source_bank`、`test_region_column_transport` 及对应 standalone checks；
[总对应](../semantic-map.md)连接数学、实现、根与适用域。
源 `72e1d22c` 仍为每个候选预算前缀重建 SourceIncidence，本报告不把后继增量规划算作已实现。

## 独立资格与保留失败

两套 core 均从固定源码重新构建：Python-owned `ascend/dev-workspace-9.2.0` 和独立
`libtorch-npu/2.12.0-cann9.2.0`，GCC 10.3.1 / aarch64 / ABI1 / Release。
它们使用相同 C++ 源码 hash `d34a19cc196846500f8417b4b35847397840c802bc36a305706c5e53cb373b01`；
两份完整 manifest、二进制 hash、wheel hash 及 319 个安装文件的逐字节核对在机器记录中。
构建各用 160 核，两个非重叠 CPU 池、各 512 GiB / 2400 秒；本轮不是 CUDA 构建或 GPU 资格。

| 检查 | 结果与范围 |
| --- | --- |
| 固定 CPU 定向 | 334 项：325 passed、7 原 xfail、2 failed，0 errors；作业原状态为 failed/exit1 |
| standalone CPU | 7 个 CTest passed：columns、KV tensor/prepare、Read transport、Aggregate tensor、region transport、column checkpoint |
| standalone NPU | 1 个进程，五记忆 × mixed-a/b/c 的 15 个组合通过；独立 streaming、raw 状态、VJP、两次 SGD 窗口；物理 NPU11 |
| 安装 CPU/NPU | 各 6 个父调用，共 12 个：Python/native × positions、mixed-a、LH176；全部完成 whole3/prefix2/新进程 resumed1 |

安装小型 positions/mixed-a 每步核对独立标量日程，每次比较 483 个数值叶和 162 个离散叶；
CPU FP64、NPU FP32，三次 AdamW 与精确恢复通过。CPU 请求 mixed-a 最终全在 CPU，不是跨设备资格。
NPU positions 为 mixed-c；mixed-a 将 payload 放 NPU，Read/control/selection/events 放 CPU。
LH176 消费者检查实际最终输出、raw continuation、pending/history/逐物理端口 ledger、None/零梯度、
参数及全部 AdamW 槽的新进程精确恢复；它不是该规模的独立数学 oracle 或吞吐实验。
安装消费者与源码目录中的成本驱动分别记录，不把后者称为安装资格。

两个新失败都是 `test_general_ports_messages_states_and_gradients` 的 Python FP32：
`regional_total`、`regional_threshold`，根 `nodes.0.kernel.base[7,1]`。
原严格阈值 atol=1e-6 / rtol=1e-5 下，绝对差 8.851289749145508e-6、相对差 2.4057279006228782e-5。
七个原 xfail 是 native total、Python/native total_zero、Python proportional、native threshold、
Python/native hard_selection；没有改成 passed 或放宽阈值。

有界开发诊断显示：完整观察严格通过、梯度有限、None/连接零一致；
同一份已存 FP32 材料提升为 FP64 后，全部观察和梯度根通过。
失败坐标 candidate/scalar/FP64 分别为 -0.3679344952 / -0.3679256439 / -0.3679313453。
该根相对 FP64 的最大差为 5.932e-6 / 7.823e-6；所有根之间最大绝对差为 7.629e-5。
这些结果支持有限累加舍入差异的判断，不能替代原 FP32 严格通过。
诊断最初来自 dirty development；其中列出的 18 份源码 hash 已逐一匹配固定提交，仍不冒称固定提交重跑。
原开发中新增测试的失败/修正记录也保留，既不删去，也不叠加进 334 项资格计数。

## 成本设置与计时边界

32 个独立成本进程各一次 reset warmup、一次测量，均 exit0；这 32 点无 profiler 插桩。
后续三个独立 CPU profile/census 进程单列，不混入成本数据。
CPU Torch 1 线程、affinity 0–7；NPU 主机 2 线程、affinity 32–35、物理 NPU1。
每点 240 秒、服务 1800 秒；计划内存 128 GiB，但实际 launcher 上限为 256 GiB，两者分别记录。
计时前构建已结束；resident-resource 正确性作业使用其他 CPU/设备，外部争用未完整追踪。
这些是有限观察，不代表独占、峰值或多线程 CPU 上限。

native 表示 Python 客户端调用 C++/LibTorch；pure Python 使用 PyTorch；全部为主机日程。
NPU 公共/默认路径的数值 payload/Read/control/selection 在卡上，events 在 CPU（mixed-c），不是 resident。
standalone LibTorch 在本轮有正确性资格，没有单独性能数据；mixed-a/b 没有本轮吞吐数据。
LH 使用 IEEE 请求（NPU matmul/conv HF32 均关闭）；FC 使用 vendor-default，NPU matmul HF32 关闭、conv HF32 开启。
CPU 均为最高 FP32 matmul precision，原始请求与生效配置逐点保留。

完整训练计时包含输入生成/搬运、packing、前向、loss、有限性检查、反向、优化器、清理和同步；
FC 还包含 clip。构造单列，reset 排除在步计时外。host phase 时间是 enqueue/调用线程区间，不能作为可相加的设备耗时占比。
吞吐为真实输入位置数/完整秒数，不是生成 token 吞吐；含 grad 前向不是训练，也不能单靠模式时间差隔离全部 autograd 开销。

LH176：176 节点、22 区域、1058 条体边；区域层宽 1/2/4/8/4/2/1，每区域 8 节点。
B4/L16/d64、H1、FFN256、summary/adapter32，14,902,694 个训练参数。
56 个 qry / 120 个 sum 节点；扩散/收拢的 14 区域 att/content_stats，中间 8 区域 ind；
Read+、local/evidence=cat、context、regional_total/quad、k=1、source=region。
staggered-v1 的样本长度为 16/15/14/16，分子为 **61**，不是矩形容量 64。
任务为 `1e-6 Σ(cG)^2`，训练 SGD lr1e-5/momentum .3。

FC16：4×4 节点、48 条体边，B8/L128/d128、H2、FFN512、KV qry；
regional_total/quad、allocation temperature .25、k=2；局部/区域评分 cat/ind、context、evidence cat。
任务含 embedding/head，总参数 5,092,788（core 4,564,404）；使用固定下游任务及 AdamW、clip1。
推理只计算最后位置 head 并返回首 token；含 grad 前向/训练计算全序列 head/交叉熵，输出工作不同。
配置中旧下游 foundation lock 只是原请求记录，实际本轮基座身份始终为 `72e1d22c`。

## 32 点结果

表内每格为 **秒 / 输入位置每秒**，均为一次实测。
LH 24 点的实际工作一致：8368 候选、2572 selected、15340 边访问；公共/默认为 94 区域列调用、176 状态列调用。
直接公式计算 11264 个物理 Full 行再 mask，公共/默认为 2572；直接公式观察及独立根范围较窄，
不能把直接到公共的全部差额称作可消除调度开销，也不把直接公式视为绝对速度下界。

| 设备 | LH 入口 | 无 grad 推理 | 含 grad 前向 | 完整训练 |
| --- | --- | --- | --- | --- |
| CPU | 直接 Python | 0.202101 / 301.8 | 0.238903 / 255.3 | 0.760625 / 80.2 |
| CPU | 公共 native | 0.508552 / 119.9 | 0.554366 / 110.0 | 1.282720 / 47.6 |
| CPU | 默认 native | 0.605756 / 100.7 | 0.693567 / 88.0 | 1.405757 / 43.4 |
| CPU | 默认 Python | 1.485830 / 41.1 | 1.594181 / 38.3 | 2.677793 / 22.8 |
| NPU | 直接 Python | 0.175950 / 346.7 | 0.209123 / 291.7 | 0.614998 / 99.2 |
| NPU | 公共 native | 1.224504 / 49.8 | 1.306496 / 46.7 | 2.708257 / 22.5 |
| NPU | 默认 native | 1.422845 / 42.9 | 1.485219 / 41.1 | 2.902243 / 21.0 |
| NPU | 默认 Python | 2.341709 / 26.0 | 2.440282 / 25.0 | 4.141327 / 14.7 |

FC 8 点均有 16384 候选、4 区域列调用、16 状态列调用。
CPU selected=8344/边访问=25060；NPU selected=8345/边访问=25064，不能宣称两设备路由严格一致。
分子均为 1024；本推理列是该基准首 token 请求范围的完整时间，不是稳定 decode 的 token 间隔。

| 设备 | FC 入口 | 无 grad 推理 | 含 grad 前向 | 完整训练 |
| --- | --- | --- | --- | --- |
| CPU | 默认 native | 0.416408 / 2459.1 | 0.453829 / 2256.4 | 1.145259 / 894.1 |
| CPU | 公共 native | 未测 | 未测 | 1.156296 / 885.6 |
| NPU | 默认 native | 0.147039 / 6964.1 | 0.153032 / 6691.4 | 0.291365 / 3514.5 |
| NPU | 公共 native | 未测 | 未测 | 0.276464 / 3703.9 |

本轮没有重新测量 FC 直接公式/数学不等价 Transformer，保留[原对照](tensor-kernels-20261112.md)的源码与配置身份。
所有成本点的规模独立数学/恢复都未检查；不能由 finite/execution completed 推断 strict、学习、恢复资格。
各安装消费者的独立检查/恢复事实单独列于前节。

## 与上一固定提交比较

32 点的初始参数 SHA 均与 `1f0e6e43` 对应点一致，实际工作计数也逐点一致。
24 个 LH 点同时记录并匹配实际输入值/支持 SHA；FC **未记录 runtime 输入字节 SHA**。
FC 保留固定下游 `0439efd2af31cf10b07ce9e6423b0195f7449968`、archive SHA、生成器原文/源码 hash、
B8/L128、参数 seed11、warmup991/测量992；这些不能补称原输入字节已核对。
每点的旧/新观察、绝对/相对时间差在机器记录中；主要训练成本如下。

| 完整训练路径 | CPU 旧→新（秒） | NPU 旧→新（秒） |
| --- | --- | --- |
| LH 直接 | 0.779556→0.760625 | 0.612110→0.614998 |
| LH 公共 native | 1.285649→1.282720 | 2.630771→2.708257 |
| LH 默认 native | 1.390021→1.405757 | 2.840692→2.902243 |
| FC 默认 native | 1.185442→1.145259 | 0.285554→0.291365 |
| FC 公共 native | 1.152264→1.156296 | 0.269344→0.276464 |

上一轮 CPU affinity12–15、NPU 物理5/主机24–27；本轮 CPU0–7、NPU 物理1/主机32–35，线程数一致。
一次测量的这些差异不能证明因果加速/回退；当前结果没有显示 LH 公共层的主要差距缩小。
完整原始数据身份、warmup、construction、loss、实际工作及有效执行配置均保留。

## 静态审阅与下一边界

固定 30 份源文件 hash 的审阅已嵌入机器记录，并逐一匹配 `72e1d22c`。
这是源代码/数据流审阅，没有新的 profiler；下面的表达式数量不是设备 kernel 数或测得的耗时占比。

- KV 写入、Read、Full 的数学已经按 G/B/Q/K 合批，不是逐 token 的 PDG 数值 decode。
- 发布调用随节点/字段、候选数与预算分组增长，而非每个 B×L 事件一次；元数据扫描仍随行数增长。
- 当前写入按节点发布 4 个浮点字段；逐节点 `cat(old,new)+index_select` 在自定义 Function 外重建 raw bank。
- Read 重新打包 bank/query，再按节点发布 ind 的 4 个或 att 的 6 个字段；区域再打包字段并逐节点返回控制。
- selected Full 再打包 query/raw bank；默认接口立即导出逐节点/样本 Continuation，后续 import 又装配。

部分 copy 位于 no-grad 私有边界，仍有实际工作；不能把所有 copy 都说成外层 autograd 膨胀。
候选重写边界是 typed region-window KV owner，贯穿 DescriptorBlock、候选计划、ControlBlock、MessageBlock 与 typed continuation；
共享 initial/write bank，用 int64 快照/保留索引表达 Read/Full 可见域，不强制一个万能大张量。
保留现有数值核/求解器及独立日程，优先移除阶段字段的发布/重装配，再做同材料三层比较。
state→Read→区域→Full 的因果梯度必须保留，不能把互有依赖的中间输出变成一个黑盒 Function 的无依赖兄弟输出。
独立 raw 根、非空初态、三种 Read、重复边、版本守卫、梯度连接、原始逻辑时间与精确 checkpoint 是对应见证。
这是有源码依据的下一候选，不是本轮已经隔离计时并证明的唯一瓶颈。

## 独立的 CPU 操作与图诊断

同一固定 `72e1d22c`、LH176 材料由 `profile_lh_tensor.py --census` 分三进程运行；
CPU1、affinity40–43，每进程240秒、服务900秒，均完成。一次 reset warmup 后记录前向、真实反向与有限性检查；
没有 optimizer，插桩时间不是前述完整训练成本，也不能作为可消除开销的直接秒数。

| 入口 | Python 可见外图唯一节点 | native 已注册外层+私有图唯一节点 | native 节点执行数 |
| --- | ---: | ---: | ---: |
| 直接 Python | 11192 | 不适用 | 不适用 |
| 公共 native | 6069 | 15138 | 13157 |
| 默认 native | 8853 | 17922 | 13157 |

公共/默认两者的私有 KV write/read/Full 分别注册22次，区域私有图注册94次；对应唯一节点为
594/2282/1912/4281。各组和外图可能重叠，不能相加；保存字段仅为已注册范围，不是全部 ATen 保存量或内存峰值。
公共 native 前后向中的 `aten::cat` 有5210次（按全部shape汇总），直接公式985次。
这些是 ATen/图观察口径，非设备kernel数，也非每次copy都进入外层autograd。
直接公式的实际输出/根范围较窄且执行全部Full行，不能直接以节点差证明所有差额都可删除。
诊断支持继续检查字段跨阶段装配，而不支持“外层图更小，所以内部工作已更少”的结论。
完整汇总、实际命令和原始文件hash在机器记录的 `separate_cpu_profile`，可在新目录由跟踪入口重建。

## 复现入口

从固定提交按[开发入口](../development.md)建立 Python-owned 与 standalone 两套运行时；
机器记录嵌入原 build/test/consumer/measure 脚本和 argv，替换本机临时路径及输出目录即可重建。
不要把本报告的旧运行目录当成必要依赖；新长任务仍须冻结源码、有限服务/资源与合法设备分配。
LH plan 可由如下固定 recipe 重建；其 JSON-normalized 值已与原 plan 核对，完整计划身份在机器记录中。

```python
from tidegraph.planning.recipes import prepare
plan, _ = prepare(
    dict(topology="lh", width=64, region_width=8, depth=3,
         profile="textbook", family="kv"),
    samples=4, positions=16, seed=7, input_pattern="staggered-v1")
```

独立 CPU 定向完整 pytest argv 见 `cpu_directed.command`；standalone NPU 用
`tidegraph-column-check --device=npu:0 --dtype=float32`。
安装消费入口为 `examples/consumer_core5/tensor_window.py` 和 `planned_windows.py --plan PLAN`，
工作目录需在 source 外，PYTHONPATH 指向 install；机器记录保留每次子进程命令与身份。
成本入口为 `scripts/benchmark_lh_tensor.py` 与 `scripts/benchmark_dense_query.py`。
FC 需要一个含上述固定下游 commit 的只读仓库，驱动只 archive 指定 commit，不采纳 mutable HEAD。
[复现导航](../reproduction.md)、[性能经验](../performance-lessons.md)与[路线图](../ROADMAP.md)分别负责工具、经验和当前待办。
