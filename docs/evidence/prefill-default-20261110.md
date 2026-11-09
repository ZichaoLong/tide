# 默认prefill：固定资格与受控成本

本轮在TL-REQ-004上补齐默认能力选择、原始KV槽批量索引、区域跨时间批及代表回归入口。
这是有限基座交付；原语言任务的学习/大图逐张量验收与下游依赖升级仍由实验侧独立完成。
[机器证据](prefill-default-20261110.json)分列构建、实际设备、正确性、恢复和成本。

## 固定身份与范围

| 项目 | 身份 |
| --- | --- |
| 数学 | 0770593838a37c790a568226930567ed60f99b83，core-5 |
| 公共默认/核心实现 | 7fe97f2f485fa8b2a207957da3f8949091a16f80 |
| standalone checkpoint查询策略守卫 | 5f563f6a1b5e921bb644905015f70e870043ad07；Python及core源码与7fe97f2相同 |
| 下游原模型/head/loss | 0439efd2af31cf10b07ce9e6423b0195f7449968；只读取Git对象，未改依赖/文件/运行 |
| 受控成本旧查询 | Python26ca1b6、native75a5027，显式legacy；不是同源码切换查询开关 |
| 环境 | aarch64，CANN9.2、Torch2.12 CPU + TorchNPU2.12.0.post3；独立SDK环境和loader闭包另列 |

库/wheel/源码集合hash、实际物理卡映射和每个原失败状态见JSON。
通用模型/图v13/checkpoint v5不变；auto产生新执行身份，实际节点选择进入统计。
显式legacy保留历史查询，但复现整个旧调度还须固定旧源码。

## 实际资格

| 检查 | 结果与实际范围 |
| --- | --- |
| CPU安装局部/训练 | 58 passed、2 retained xfail，FP32/FP64；新布局独立字段/私有根/None/零/别名/版本/一阶及FC/HB/PDG训练/cut |
| CPU通用与配置回归 | 312 passed；一般端口/同秩/重复边、历史依赖、core-3程序/反馈/切面、planning与客户端配置 |
| CPU区域组合 | 27 passed，**仅FP64**；不冒充原FP32 case4修复 |
| NPU安装检查 | 26实际设备（13新packed/auto、13显式v1回归）+7CPU辅助检查通过；FP32单卡mixed-c |
| Python/native安装消费者 | CPU和NPU各两客户端，三步AdamW完整比较、新进程3 vs 2+1精确恢复通过 |
| standalone消费者 | CPU FP64/NPU FP32；FC4/d8、B2/P3、8190可训练参数、KV sum；独立完整记录/字段根/SGD、新owner及新进程精确恢复、跨query策略拒绝通过 |

standalone在每个设备检查16个root/cotangent用例；不能把CPU辅助项计作NPU设备测试。
基础Attention/归一化函数被局部标量与批量实现共同复用；它们的数学锚点沿用模块独立公式/方向差分。
本轮直接新增的反例针对忽略首值导致中心VJP相消；调度独立性由Python reference/greedy/native及端口测试负责。
新布局保留原始int64时钟、Read初态/快照、Full proposal快照和原始checkpoint解码。

原dense-v1 FP32 evidence梯度约1.044e-6的两个xfail保持。packed-v2保留evidence逐行投影并通过该原输入，
不宣称全部FP32误差消失。混合局部组合case4的input.1根最大差3.814697265625e-6、相对1.4105765e-5，
本轮在旧source-d/build-c的Python/native均复现同值；原断言与失败保持，不用新门禁计数将其覆盖。

## 原任务成本

范围：FC16/4区域/48边，B8/L128，内容128、summary/adapter32、2头×64、FFN512，KV qry、
Read+、cat/ind、regional_total、context/cat、quad、tau=.25、k2、gamma_max2。
recipe的history=512按Settle步长6展开：前三个区域的原始逻辑时间window=3072，末区域window=0；
position_origin为各区域rank 1–4，position_scale=6。不能将history直接称为逻辑时间窗口512。
总可训练参数5,092,788（包括原下游embedding/head），核心约4.564M；不是9B的LH模型。
CPU ATen1线程，NPU mixed-c/2 host线程，native单调度worker，FP32；无自定义设备融合核。

每配置一个独立进程，一预热一步测量。预热后恢复同初始模型和空optimizer状态，
测量前参数hash与原始hash核对；设备算子/缓存已预热，AdamW状态重新初始化的成本仍包含。
完整wall包括输入生成/上传、调度、packing、forward/loss、backward、finite/clip、optimizer、清理和同步；
构造、预热、profile单列。此口径不同于旧报告的连续预热更新，不能直接拼成同一加速比序列。

四项均执行完成；下表是每配置一次的受控成本观察，**不是原任务完整数值/离散/恢复资格**。

| 配置 | core前向 s | 反向 s | 完整一步 s | token/s | Full | 内部消息 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU旧源码legacy | 26.494 | 34.947 | 62.177 | 16.469 | 8344 | 25060 |
| CPU新默认auto | 15.868 | 22.492 | 38.953 | 26.288 | 8344 | 25060 |
| NPU旧源码legacy | 120.781 | 115.170 | 237.119 | 4.319 | 8345 | 25064 |
| NPU新默认auto | 53.823 | 46.978 | 101.756 | 10.063 | 8345 | 25064 |

CPU耗时下降37.35%，NPU下降57.09%。四项均16,384候选；每设备内Full/消息总数相等，
但总数相等不能证明具体作用支持、逐张量或梯度相等。CPU/NPU之间仍差一个Full。
CPU旧/新loss分别10.9906902313/10.9906883240，裁剪前梯度范数1362.03186035/1362.02502441；
NPU分别10.9906787872/10.9906778336、1360.36035156/1359.88085938。
四项初始和测量前参数SHA256均为`7f41057f0196bc2b259902f8dadafc9c3f46d197e02de5ebbda9517a632f9a1a`。

真实区域批调用512→4，最大区域批8→1024；state序列调用仍16、最大长128、标量序列后备0。
所以旧执行也并非全程逐token decode，原来的四阶段已经有部分时间批；本轮进一步处理了区域和查询内部碎片。
私有KV写投影仍12,288次、最大批4，不能从调度批1024推断所有矩阵都是千行GEMM。

内存为进程截至测量末端的高水位，**包含预热，未在测量步前重置峰值计数**。
CPU RSS旧/新9488.2/9556.0MiB；NPU主机RSS4939.5/3993.5MiB，
设备peak allocated6059.1/6536.0MiB，reserved6152/6652MiB。设备峰值上升，不能称峰值内存优化。
运行前/后主机load average分别56.45/59.55/62.44、56.86/57.62/59.15；affinity为320 CPU，
本项目构建/重型检查与计时无重叠，其他用户争用未控制。单NPU物理1映射为npu:0。

新NPU仍慢于CPU，两者均未达到下游36小时完成4,194,304 token所需约32.4 token/s。
该数是下游曝光预算，不是“设备必须快于CPU”的通用资格门槛。本轮未运行下游学习验收或改其依赖锁。

## 独立profile与可复用经验

另一个B2/L8进程在一次预热、reset、一次正常测量后插桩第三步；实际256候选、130 Full、388消息，
区域4次/最大批16，原始KV槽布局236次/772行，私有BMM192次/全部4行。
weighted MM324次/4396行，实际行数1–56，82次为16行。该步与全尺寸成本分开。

| profile观察 | 次数 | 主机self总时长 ms |
| --- | ---: | ---: |
| aten::index | 2759 | 325.485 |
| empty_tensor | 71062 | 230.849 |
| aten::copy_ | 10879 | 200.106 |
| aten::_local_scalar_dense | 6328 | 194.466 |
| aten::nonzero | 4 | 见JSON |

设备kernel_details有87,404条记录；task_time另含异步复制等任务，不能混作同一计数。
AiCPU只剩8次MaskedFill，未出现旧profile的Index/IndexPutV3；AiCPU不是主机CPU fallback。
厂商时间线Stage3.251秒、Computing.259秒、Free2.992秒。Free分类及嵌套host总时长
不能证明一个唯一瓶颈，也不能直接外推正常全尺寸利用率。
此前dense-v1的135,504条kernel记录及两类各1170次AiCPU索引来自不同更新前缀/实际工作量，
仅用作结构变化线索，不给逐算子的等价加速比。

这支持继续核对小投影、索引/根装配、重复元数据回读三个边界。
已有`kv_projection.projections`的common分支能在序列内批量计算私有投影，后继可检验让共享准备保留normal/base；
`weighted_kv.cpp`的时间/权重/惰性状态守卫也有重复`.item()`。这些是已定位的候选，尚未改动或测出收益；
必要校验、独立None/零根、所有使用者梯度及精确恢复不能省略。具体重启条件在ROADMAP。

## 可独立复现

1. 固定实现提交，按[开发入口](../development.md)建立匹配的私有wheel/native和独立SDK构建。
   构建可用全部affinity CPU；新SDK需保留构建工具PATH，不能将Python-wheel库混入SDK loader。
2. 激活单独安装wheel，在干净固定cwd执行：
   `bash scripts/qualify_dense_query.sh NEW_OUTPUT cpu MATCHING_NATIVE_BUILD auto`。
   单NPU队列内把cpu改为npu，物理卡由站点队列映射为npu:0。
3. SDK静态材料：`prepare_core5_study.py --topology fc --depth 2 --region-width 2 --width 8 --samples 2 --positions 3 --output-dir NEW_PLAN`，
   再`export_weighted_benchmark.py --binary --workload NEW_PLAN/workload.json --native-library MATCHING_BUILD --output NEW_FIXTURE`。
   在SDK环境用`check_query_sdk.py --binary SDK_BENCHMARK --fixture NEW_FIXTURE --device cpu --dtype float64 --output-dir NEW`；
   NPU改npu:0/float32。它不向候选提供参考轨迹。
4. 原任务：`python scripts/benchmark_dense_query.py --downstream-repo READ_ONLY_DOWNSTREAM --native-library MATCHING_BUILD --query-execution auto --warmup-state reset --device cpu --batch 8 --length 128 --output-dir NEW`。
   NPU队列内将device改npu:0；独立profile改B2/L8并加`--profile npu`。
   旧版本对照只从固定旧Python树/匹配native库加载legacy，记录代码身份；不修改当前默认来伪装旧版本。
5. profile CSV已生成后，`python scripts/summarize_query_profile.py --csv-dir CSV_DIRECTORY --output NEW_SUMMARY.json`
   只读汇总host、设备kernel、task类型、AiCPU及时间线，并保留原CSV hash；不重新运行模型。

SDK检查和profile汇总脚本在本报告的证据提交中；计算实现固定以上两份提交。
重建旧计算版本后，可将这两个无执行器逻辑的驱动从证据提交单独取出使用，hash见JSON。
原任务驱动显式从下游固定Git对象取静态模型/head/loss；其他资格与目录不需要旧运行目录。

成本/资格均由独立background.slice服务运行，时长、内存、线程和停止条件有限；构建与重型计时不重叠。
首次pip代理超时被主动停止，SDK首次缺Ninja失败；后继复用已验证nlohmann3.11.2头文件并修正构建PATH。
这些是保留的环境/构建终态，不是模型数值通过。全部生命周期审阅在JSON中，成本服务已终止且无存活后代。
本轮没有GPU/resident/FP16/多卡新资格，也没有运行目录中所有HB/LH/1B–16B尺度；目录是后续检查入口。

[默认契约](../weighted-dense-query.md)、[分层回归](../prefill-regression.md)、
[复用经验](../performance-lessons-prefill.md)连接实际接口/实现/测试。限制与重启条件只维护于[ROADMAP](../ROADMAP.md)。
