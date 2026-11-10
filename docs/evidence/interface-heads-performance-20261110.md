# 通用Describe增量：有限性能与图统计

固定实现`554f2f5a00ea1e1dbff8f13c48c09dbb4366efac`，数学固定`0770593`。
[机器记录](interface-heads-performance-20261110.json)保存配置、实际工作量、源码/材料/结果hash、
计时与恢复边界；[独立资格](interface-heads-qualification-20261110.md)单列小型公式、完整根、
三客户端训练与新进程恢复。后继`ebfe075`只修正文档和NPU算子观察器，不改变所测计算。

本次完成CPU/NPU共14个代表训练与精确下一窗检查、10个FC16成本点、8个层链成本点和
8个独立诊断进程。它们回答不同问题，不能合计为“40项数学资格”。通用入口仍有明显开销，
没有达到接近直接张量公式或常规Transformer实现的目标。完整规模的独立数值、离散及学习
仍为not_checked；已有严格失败保持原阈值。

## FC16：区分推理、构图与真实训练

固定下游`0439efd2af31cf10b07ce9e6423b0195f7449968`，16节点/4串行区域×4节点，
相邻区域全连接；B8/L128、d128、heads2、FFN512、summary/adapter/score/region hidden32。
KV qry/Read+/cat/ind/context/cat/regional_total/quad，temperature=.25、发送强度k2。
完整任务5,092,788参数，其中core4,564,404；模型/head/loss/AdamW/clip沿用原固定材料。
Python调用native C++/LibTorch、FP32、component-window/auto packed-v2与Describe batch-v2、
minimal返回。CPU为1个intra-op线程，NPU为mixed-c/主机2线程；调度worker1，非resident。

每点独立进程，一次预热后恢复初始参数/optimizer，一次测量；已核对测量前参数hash相同。
计时包含必要准备/上传、调度/packing、前向、loss及对应模式的反向/检查/优化器/清理/同步。
构造、预热与独立诊断不在完整步内。以下backward行包含前向和反向，不是纯反向时间。

| 模式 | CPU秒 | NPU秒 |
| --- | ---: | ---: |
| 无梯度完整序列与loss | 3.215327 | 5.413898 |
| 开启梯度记录的前向与loss | 5.659086 | 8.826882 |
| 前向+真实反向 | 12.145675 | 16.583629 |
| 完整训练更新 | 12.613559 | 15.497203 |

完整训练输入吞吐分别为**81.18/66.08位置/秒**，
计算为1024/完整秒数。训练内反向分别7.507134/8.017826秒。
不同进程存在波动；不能用两种模式的秒数相减认定纯autograd引擎开销。

独立推理首token返回为**3.083076/3.959563秒**；
完整请求3.084443/3.963583秒，
对应331.99/258.35输入位置/秒。
这是已加载模型的本地B8 prompt→末位置head→greedy token→host返回，不含排队/tokenizer/加载/后续decode；
不除以B或L，也不是生成token吞吐。完整序列nograd计算所有位置的head/loss，不能替代此TTFT。

CPU/NPU均16,384候选，Full分别8,344/8,345，消息25,060/25,064；梯度范数
1362.2411/1359.8636。工作量接近不证明轨迹/梯度等价。
[上个47704c9增量](interface-batch-publication-20261110.md)训练12.4468/17.2122秒、
TTFT3.1589/5.0984秒；本次CPU训练略升、NPU训练和TTFT下降。每配置仅一次测量，
不宣称稳定加速或消除所有回归。当前NPU仍慢于该CPU配置。

## 代表回归：拓扑、模块与客户端

公共`prefill_regression_config.py`生成，B4/P16、staggered-v1不齐输入、FP32、seed7；
outputs返回、packed局部计算、scalar返回、一个调度worker、CPU1/NPU主机2线程。
Python两客户端finite-check为scalar；SDK保留自己的完整实际字段/梯度/optimizer守卫。
目标为weighted-output-square×1e-6，SGD(lr=.001,momentum=.3,weight_decay=.01)。
CPU放置cpu，单NPU放置mixed-c；ATen/LibTorch算子，无新专用融合或稀疏算子。
每项一预热、一完整训练更新，然后另计checkpoint/new-owner精确下一窗。
与FC16的每点reset不同，公共矩阵测的是预热后继续的下一训练窗；不混算两种计时。

| 用例/客户端 | 节点/区域/d | 唯一参数 | CPU完整步秒 | NPU完整步秒 |
| --- | --- | ---: | ---: | ---: |
| 宽FC/KV sum/native | 4/2/512 | 16,223,730 | 0.342263 | 0.447675 |
| HB/KV sum/native | 176/22/32 | 5,220,726 | 5.901692 | 16.729626 |
| LH启发/KV qry+sum/native | 176/22/64 | 14,904,826 | 11.033548 | 31.432143 |
| FC16/KV sum/native | 16/4/64 | 1,304,092 | 0.496732 | 1.242435 |
| FC16/SSD state/native | 16/4/64 | 1,419,852 | 2.345992 | 7.658685 |
| LH启发/KV qry+sum/PyTorch | 176/22/64 | 14,904,826 | 20.277586 | 43.985534 |
| FC16/KV sum/standalone LibTorch | 16/4/64 | 1,304,092 | 0.527108 | 1.162407 |

表中native为Python→C++/LibTorch；standalone的数值执行与调度在独立C++进程内，
其材料生成/导出和Python控制启动不属于表中的C++完整训练窗。三客户端来源分别记录。
所有14项执行与serial new-owner精确恢复通过；独立数值/离散检查均not_checked。
这个恢复范围不同于小型安装消费者的fresh-process恢复，两种证据分列。

这些recipe的k=1，区域采用Read+/cat/regional_total/context/cat/quad；
region_score在LH中混合ind/att，其余为ind。HB为扩散/收拢，LH启发拓扑还包含跳边与qry/sum
异构，局部定义都来自core-5。
KV heads/FFN依次为8/2048、1/128、1/256、1/256，summary/adapter32；
SSD为heads1、key/value64、FFN256、summary32，卷积4槽、原始逻辑时间stride6。
唯一参数包含frozen参数；实际Full与消息、局部配置变体、窗口和来源绑定由机器记录/生成器给出。
不能从k或静态全连接推出活动比例；不同客户端/设备的实际工作量存在少量差异。

本轮没有重测1B/9B、多卡、mixed-a/b或resident吞吐。它们的原资格和原失败仍见
[部署代表](prefill-deployment-20261110.md)和[回归导航](../prefill-regression.md)，
不能从本表称新提交已通过全部规模防退化。

## 同公式层链仍然显示内部开销

B4/L32/d128、4层/4头、FP32/IEEE，KV704,612参数、SSD669,332参数；CPU1/NPU主机2线程。
每层一个节点/区域、单位强度、零左状态，完整训练包含准备/传输、loss、反向、finite checks、SGD和同步。
图入口为Python→native mixed-c/CPU；张量公式为纯PyTorch SDPA或卷积/因果矩阵。
两者按相同种子/参数与reset流程执行；适用域和省略职责见[层链契约](../layer-chain-baseline.md)。

| 记忆 | CPU默认图秒 | CPU张量公式秒 | NPU默认图秒 | NPU张量公式秒 |
| --- | ---: | ---: | ---: | ---: |
| KV | 0.268754 | 0.029784 | 0.534760 | 0.024700 |
| SSD | 0.751136 | 0.038821 | 2.702812 | 0.028290 |

差距仍明显。直接公式省略辅助Describe/图记录及某些零连接，SSD展开限制L≤256，
不提供一般checkpoint或任意左状态；差距不能全部当作已经证明可消除的管理成本。
这组成本未做实测尺度的独立数值或恢复认证。原小型独立公式资格保持自己的提交和范围。

## 独立计算图诊断

另用d32/B2/L8的16节点FC（4区域×4节点）与独立FC张量公式，hidden-MSE真实反向。
Transformer为4个串行block、d32/head1/FFN128，是不同函数且只有65,536参数；
FC三种实现均533,940参数。所有诊断时间含观察开销，不作为吞吐。

| 已登记图的节点并集 | CPU | NPU |
| --- | ---: | ---: |
| PyTorch默认图（含已登记私有图） | 10,382 | 10,382 |
| native默认图（含已登记私有图） | 10,802 | 10,802 |
| 独立FC张量公式 | 4,083 | 3,339 |
| 4层Transformer | 183 | 143 |

native外层只有1,143节点，加入显式登记的私有图是10,802；实际反向执行9,766/9,750次。
各组以及Python/native观察集合重叠，不能相加。native显式保存的去重storage为2,634,044字节，
只覆盖插桩边界，不是所有ATen保存量或allocator峰值。后端可能选择不同autograd节点/工作区，
所以同函数跨设备的节点数和保存字节也不能直接换算速度。完整计量契约见[图统计](../autograd-census.md)。

## 作业、失败与复现

所有运行使用554f2f5只读快照、独立background.slice、7200秒上限、CPUQuota32000%、
MemoryMax1024GiB/Nice10。仅空闲NPU准入，实际均physical11→logical npu:0；
构建未与计时重叠，外部主机负载未控制。线程配置用于对照，并非构建或测试资源上限。

`interface-heads-measure`保持failed：首版driver把site放在case内，被matrix级site覆盖为空，
两平台各5个native和1个SDK在计算前拒绝，未产生训练成本。该流程其余FC16/层链/诊断及
两项PyTorch代表全部完成。后继`interface-heads-matrix`只补这12个未执行配置，将site放到
matrix级，并在运行前检查生效路径和构建身份；它已完成。原失败/日志hash保持，未重测成功项。

从干净固定源码按[development](../development.md)安装与分别构建native/SDK；在当前准入与
持久服务内运行，下列路径替换为新的安装/输出，不需要旧运行目录：

```sh
python scripts/benchmark_dense_query.py --downstream-repo READ_ONLY_DOWNSTREAM \
  --native-library MATCHING_BUILD --device cpu --threads 1 --trace-records minimal \
  --mode training --output-dir NEW_FC16
python scripts/prefill_regression_config.py --case many-hb --family kv \
  --samples 4 --positions 16 --device cpu --threads 1 --output NEW_CONFIG.json
tidegraph run NEW_CONFIG.json --site SITE.json --output-dir NEW_REPRESENTATIVE
python scripts/benchmark_layer_chain.py --family kv --implementation native \
  --native-library MATCHING_BUILD --device cpu --threads 1 --output-dir NEW_LAYER
python scripts/diagnose_autograd_graph.py --baseline graph --client native \
  --native-library MATCHING_BUILD --width 32 --depth 4 --region-width 4 \
  --batch 2 --length 8 --trace-records minimal --mode backward \
  --device cpu --threads 1 --output-dir NEW_CENSUS
```

FC16另选nograd/grad-forward/backward/inference；层链另选ssd/tensor；census另选python、
fc-tensor或transformer（张量基线不传native库）。NPU成本用npu:0/threads2，census保持threads1。
代表表逐行对应生成器case/family/client，完整请求和解析身份在JSON；matrix使用顶层site。
新增机制与限制分别进入[性能经验](../performance-lessons.md)与[ROADMAP](../ROADMAP.md)。
