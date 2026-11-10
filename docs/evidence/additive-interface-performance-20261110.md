# 加法接口接入：局部、层链和完整用例成本

计算源码`e6f7bd5f570b6227d7664e3be45c4f27eb5b611a`，探针源码
`0e6916dc8c74cdfe3fc6694a1423b85d191bccbd`，数学`0770593`。
[机器记录](additive-interface-performance-20261110.json)保存实际配置/工作量/身份/命令及失败；
[固定资格](additive-interface-qualification-20261110.md)单列小型公式、完整根、训练和恢复。

这轮采用了默认Full共享bank矩阵，但**没有消除公共图入口与紧凑公式的明显差距**。
前向减少历史展开，独立快照反向仍有成本；局部收益没有转化为FC16的NPU训练收益。
下列是有限单次观察，用于决定接口后继方向，不声明最优算法或统计稳定加速。

## 方法与执行身份

CPU/NPU各6个局部点、7个FC点、8个层链点；另有12个代表训练/精确下一窗、
1个NPU约1B推理/精确下一窗及独立图/阶段诊断。每配置独立进程，1次预热和1次正式测量。
CPU intra-op1，NPU host intra-op2，调度worker1；诊断census两侧固定host1。
线程数沿用上轮受控对照，**不是本机CPU最大吞吐扫描或构建线程限制**。
PyTorch/native基于Torch2.12/TorchNPU2.12.post3/CANN9.2，SDK环境独立。
空闲physical11→`npu:0`，未使用共享卡；外部CPU/内存争用未排除，不宣称全机独占。
无重型构建与计时重叠。没有新GPU、多卡、resident吞吐或9B测量。

完整训练包含准备/传输、调度、packing、前向/loss、真实反向、必要有限值检查、
optimizer和同步；构造、预热、图统计/阶段profile与checkpoint恢复检查单列。
FC及层链在预热后恢复初始模型；公共代表矩阵按自身契约从预热后的下一窗继续测量。
未在完整测量规模做独立数值/离散/学习资格，不能把恢复通过当成这些资格。

## 局部：相同独立快照契约

`benchmark_bank_attention.py`，B4/L128/d128/H4、FP32/IEEE，独立q/weights/K/V快照根，
相同因果前缀；只含投影后Attention和对应一阶VJP，不含Q/out投影、调度、区域和optimizer。
新bank执行Q×K/V矩阵；expanded显式展开每个query的历史。单位ms：

| 模式 | CPU bank / expanded | NPU bank / expanded |
| --- | ---: | ---: |
| 无梯度前向 | 8.147 / 27.908 | 6.639 / 40.317 |
| 记录梯度的前向 | 8.462 / 31.790 | 6.743 / 51.235 |
| 真实反向本身 | 58.862 / 42.924 | 35.174 / 19.362 |
| 同次前向+反向 | 68.244 / 76.663 | 41.852 / 58.458 |

新前向避免query×history×width的K/V物化，但反向仍返回独立snapshot adjoints，
本次两设备的反向本身均更慢。总计有所改善；不能把前向倍数用于完整训练。
这不是vendor fused attention或硬件kernel数量证据。

## FC16：默认入口与加法参照

固定下游`0439efd2af31cf10b07ce9e6423b0195f7449968`：16节点、4串行区域×4节点，
相邻区域全连接；B8/L128/d128/H2/FFN512，summary/adapter/score/region hidden32。
KV qry/Read+/cat/ind/context/cat/regional_total/quad，temperature=.25、k2。
完整任务5,092,788参数，core4,564,404。L128是每个样本的128个输入位置，合计1024位置。
Python→C++/LibTorch，FP32、auto component-window/packed-v2、Describe batch-v2、
minimal返回；NPU为mixed-c，窗口推进仍在host，不是resident。

| 模式 | CPU秒 | NPU秒 |
| --- | ---: | ---: |
| 无梯度完整序列及loss | 2.818699 | 3.915932 |
| 记录梯度的前向及loss | 4.578968 | 7.448917 |
| 完整训练更新 | 12.283111 | 15.510679 |
| 其中真实反向 | 7.538370 | 8.120481 |
| 推理首token返回 | 2.736881 | 4.054554 |
| 完整推理请求 | 2.738368 | 4.058731 |
| 直接FC公式完整训练（投影契约） | 1.425050 | 0.712580 |
| Transformer4完整训练（不同函数） | 0.246228 | 0.018551 |
| Transformer16完整训练（不同函数） | 0.755807 | 0.045640 |

训练输入吞吐=`1024/完整训练秒数`：CPU **83.37**、NPU **66.02位置/秒**。
推理输入吞吐按完整请求为373.95/252.30位置/秒；首token是整个已加载本地B8请求
完成prompt、末位置head、greedy选择、检查并返回host的时延，不除以B或L；
不含加载、排队、tokenizer或后续decode，也不是生成token吞吐。

相对[上轮](interface-heads-performance-20261110.md)的12.613559/15.497203秒训练，
CPU下降约2.6%，NPU基本不变；首token由3.083076/3.959563变成2.736881/4.054554秒，
NPU反升约2.4%。独立进程/一次测量不能据此宣布小幅变化显著。

两设备均16,384候选；实际Full CPU8,344/NPU8,345，消息25,060/25,064。
默认入口与同设备FC公式的这些计数相同，但这不证明全部轨迹/根/梯度等价。
FC公式承担输出/raw状态数值投影，默认通用实现保留公共接口及相关检查义务；
其8.6×/21.8×差距指明值得检查的边界，不是全部可消除开销的证明。
Transformer4/16没有Tide区域分配、消息与raw续接语义，参数/工作也不同，只作同维度计算参照。

## 层链：控制组合复杂度

四层链B4/L32/d128/H4/FFN512，KV qry或SSD state，区域单候选必活动；
同参数/输入、SGD完整训练，FP32/显式IEEE。原始时间短卷积保留；
直接公式无图调度、独立中间根/完整记录。它们不是标准Transformer/Mamba实现。

| 模块/模式 | CPU默认 / 公式，秒 | NPU默认 / 公式，秒 |
| --- | ---: | ---: |
| KV推理 | 0.065916 / 0.010623 | 0.152425 / 0.013383 |
| KV训练 | 0.231195 / 0.028691 | 0.573972 / 0.022429 |
| SSD推理 | 0.111125 / 0.015169 | 0.625357 / 0.014281 |
| SSD训练 | 0.689099 / 0.040144 | 2.174451 / 0.032392 |

每点512个候选/Full。KV默认已有矩阵/序列边界，仍比公式慢；SSD不使用本轮bank路径，
其差距继续指向循环状态/反向/校验与组装。不能把固定四层、四阶段视为内部工作已紧凑。

## 代表与图诊断

公共CLI生成的B4/P16、staggered-v1不齐输入、FP32/packed/outputs，SGD完整一步；
CPU/单NPU mixed-c。以下12项均完成，并通过实际选择路径的raw解码/新owner精确下一窗。
独立数值、离散和学习为not_checked。这里的native指Python调用C++/LibTorch。

| 拓扑/模块/客户端 | body节点/区域/d | 唯一参数 | CPU秒 | NPU秒 |
| --- | --- | ---: | ---: | ---: |
| 宽FC/KV sum/native | 4/2/512 | 16,223,730 | 0.307250 | 0.413203 |
| HB/KV sum/native | 176/22/32 | 5,220,726 | 5.660797 | 17.099047 |
| LH启发/KV qry+sum/native | 176/22/64 | 14,904,826 | 10.378583 | 31.164541 |
| LH启发/KV qry+sum/PyTorch | 176/22/64 | 14,904,826 | 19.845958 | 44.181674 |
| FC16/KV sum/native | 16/4/64 | 1,304,092 | 0.506446 | 1.331999 |
| FC16/KV sum/standalone LibTorch | 16/4/64 | 1,304,092 | 0.565280 | 1.195770 |

实际候选/Full/消息：宽FC两设备214/122/122；HB6340/3466/8840；sum-FC915/382/1228。
LH native的CPU为8416/2726/16020，NPU为8413/2724/16015；PyTorch CPU为8416/2727/16025，
NPU为8413/2724/16015。差异保留，不能称逐轨迹等价加速。HB/sum多数不经过本次Full
Attention新边界，其耗时有升有降，不能把所有波动归因于该优化。

额外LH约1B：176 body节点/22区域/d640、KV sum、k1、1,051,335,206可训练参数，
单NPU mixed-c/FP32/native，B1/P8推理完整窗**1.728373秒**；896候选、199 Full、1180消息，
Full/候选约22.2%。5,185,227,214 B checkpoint及新owner精确下一窗通过；
恢复检查单列37.94秒，不并入推理吞吐。含恢复的allocator peak allocated/reserved约
4.56/4.86 GB，RSS进程高水位21.54 GB；不含未被allocator追踪的驱动/vendor空间。
该短窗/稀疏度/语义不同于旧LH口述9B数据，不做直接吞吐对比；未认证本配置训练。

图统计另用FC16/B2/L8/d32、相同输入/参数、minimal、真正反向，禁用性能结论：

| 计算图 | CPU节点数 | NPU节点数 | 计数范围 |
| --- | ---: | ---: | --- |
| PyTorch默认图入口 | 9,920 | 9,920 | 外层与已注册私有图去重并集；其中外层1,141 |
| native默认图入口 | 10,298 | 10,298 | native注册并集；Python可见外层只有1,143 |
| FC紧凑公式 | 5,095 | 4,061 | 同输出投影的可见图 |
| Transformer4 | 183 | 143 | 不同函数，设备算子后端也不同 |

native实际节点执行CPU9,262/NPU9,246次，不能只看32个query外层边界。
PyTorch私有Read组5,120节点、Full组2,603节点；组间可能重叠，不能直接相加。
FC公式自身仍有数千节点，不把它称为零冗余最优下界。
Python/native观察宇宙重叠，不相加；saved-storage统计不是进程峰值。

另一个B2/L8/**d128** native阶段诊断，query私有梯度计算约CPU118.61/NPU258.59ms，
Read记忆51.11/114.11ms，Full记忆27.53/71.14ms，区域梯度23.14/57.15ms，
query结果发布1.74/1.67ms。它们是具名calling-thread排他墙钟，含偶然等待，
不是硬件kernel时间；不能与d32图统计或正式吞吐直接相减，也不外推到B8/L128。
这支持下一轮同时检查Read/Full私有反向与块接口，不能仅继续压缩Full前向或最外层调度。

## 复现与停止边界

基础构建和安装见[资格报告](additive-interface-qualification-20261110.md)。
局部入口如：

```sh
python scripts/benchmark_bank_attention.py --implementation bank --mode backward \
  --device cpu --threads 1 --output-dir NEW_LOCAL
python scripts/benchmark_layer_chain.py --family kv --implementation native \
  --native-library BUILD_NATIVE --device cpu --threads 1 --mode training --output-dir NEW_LAYER
python scripts/benchmark_dense_query.py --downstream-repo DOWNSTREAM_REPO \
  --native-library BUILD_NATIVE --trace-records minimal --device cpu --threads 1 \
  --mode training --output-dir NEW_FC
python scripts/diagnose_autograd_graph.py --baseline fc-tensor --client python \
  --width 32 --depth 4 --region-width 4 --batch 2 --length 8 --trace-records minimal \
  --mode backward --device cpu --output-dir NEW_CENSUS
```

公式census不能传`--native-library`；原驱动误传导致CPU/NPU各两项启动拒绝，
保留原失败，只补这四个未执行的诊断。既有计时不重跑以制造绿色总状态。
`gef-additive-measure`因此保持failed；`gef-additive-supplement`的四个诊断及CPU四项
默认调用观察全部完成。所有服务已终态、NPU锁已释放，没有把外层状态当内层资格。
公共代表由`scripts/prefill_regression_config.py`生成，实际完整配置保存在机器记录；
site路径须同时供matrix解析，不仅写进被覆盖的单项。NPU命令按当前空闲队列/环境执行。

本轮采用共享bank矩阵和准确首次借用计费；将有状态紧凑基线保留为独立投影参照。
拒绝“局部前向更快即完整训练加速”或“所有根可直接合并”的推论。
块内snapshot伴随、循环序列VJP、resident单bank投影及规模吞吐延期：
后继需先给出能保留所需根/精确状态的块接口、小组合对照及明确成本问题，
再做有界实施，不继续无限压缩同一局部前向。当前待办与授权状态只由ROADMAP/STATUS维护。
