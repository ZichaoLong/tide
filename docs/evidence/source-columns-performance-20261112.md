# 来源列合批：固定00bb成本与LH176生命周期（2026-11-12）

固定实现 **`00bb5ac2dec61e215966f523063786682caa40c1`**，数学固定`0770593838a37c790a568226930567ed60f99b83`。
本报告整理已完成的22个成本点与四个独立安装消费者，没有重测。
[机器记录](source-columns-performance-20261112.json)保存逐点命令、原始字节hash、构建、参数/输入身份和完整数值。
对照仅使用[fb83原报告](rope-performance-20261112.md)的`fb83ee91cc34ba49bf00465916d6e7e7a7eacdec`实际对应点。

LH176默认native完整训练为CPU **2.3187s**、NPU **5.1350s**，相对前次观察下降6.88%/3.63%；
直接公式仍为0.7568/0.6238s。主要差距位于公共列计算及其状态/观察职责中，默认分派的增量较小。
本次来源变换合批没有消除LH差距；NPU推理几乎未变，1B NPU默认推理这次增加4.40%。
四格三AdamW与新进程精确恢复通过，**性能、同客户端恢复和独立数学资格分别成立**。

## 实际范围、配置和计时

| 用例 | 实际拓扑/尺寸 | 局部配置 | 规模与吞吐分子 |
| --- | --- | --- | --- |
| LH176 | 176节点、22区域、1058内部边；区域层宽1/2/4/8/4/2/1，每区域8节点；B4/L16，d64/H1/FFN256，summary/adapter32 | textbook KV：opening56 qry，closing56 sum，middle64 sum；相应Full/Read RoPE，窗口4608/0。Read+、cat、regional_total/context/cat/quad，T1/k1/gamma2；非middle att/content_stats，middle ind；source_mode=region、source_hidden32 | 14,902,694可训练参数；实际61个有输入的sample-position |
| LH约1B | 同176节点拓扑；B32/L16，d640/H10/FFN2560 | baseline KV sum，无RoPE；全部ind，其他分配同上；source_hidden32 | 1,051,335,206参数；实际481个输入位置；仅推理 |
| FC16 | 16节点/4区域，每区域4节点；B8/L128，d128/H2/FFN512，summary/adapter32 | KV qry，无RoPE，窗口3072/0；Read+、cat/ind/context/cat、regional_total/quad，T0.25/k2；source_mode=region | core4,564,404，含embedding/head共5,092,788；1024个输入位置；仅默认native训练 |

LH输入是`staggered-v1`物理端口日程，B×L是容量；不能用64/512代替61/481。
1B原始`config.batch=4`是未生效的解析默认，实际加载plan的`samples=32`，本表使用实际plan。
FC任务固定下游`0439efd2af31cf10b07ce9e6423b0195f7449968`，完整输入来自其`synthetic_batch(seed=992)`。
下游配置内旧foundation锁是请求元数据；本轮实际源码与native构建均为00bb。未补造FC运行时输入字节hash。

每点新进程、一次恢复初态的预热、一次恢复初态的测量。吞吐=实际输入位置/完整同步区间秒数，
不是生成token吞吐或TTFT。本轮没有prompt-to-first-token测量，也没有重测FC直接公式/Transformer。
LH完整训练含输入准备/上传、packing、前向、逐端口`1e-6*sum((cG)^2)`、有限检查、反向、
SGD(lr1e-5/momentum0.3)、清理及同步；构造、hash、重置在外。FC包含embedding/core/readout/head/loss、
有限检查、反向、clip、AdamW、清理及同步。含grad前向只记录图，不包含反向或更新。

CPU使用亲和池0–31；LH176/FC的ATen线程1、1B线程32，interop1。
NPU主机池32–39、ATen线程2、interop1；物理9→逻辑`npu:0`，不是32/39个线程。
PyTorch2.12.0+cpu、torch_npu2.12.0.post3、CANN9.2.0、Python3.12.13、aarch64。
LH为IEEE FP32（NPU matmul/conv HF32均false）；FC为vendor-default FP32（matmul false、conv true）。
public/default实际分派`tensor-window-v1 / settle-columns-v1 / bounded-common-origin-v1`，无fallback；
CPU全部在CPU，NPU为**mixed-c**：payload/Read/control/selection在NPU，事件推进在主机。
`native`是Python调用C++/LibTorch；`python`是纯PyTorch调度。这批没有standalone或resident计时。

两项性能服务在`background.slice`并行使用不相交CPU池，各256GiB/1800秒；
08:08:07–08:11:37 UTC范围内保留的P1P6作业记录仅这两项，没有本项目重构建重叠。
NPU是授权共享观察：两次准入采样为AICore0%、HBM5654/65536MiB，同租PID1711169/start2203892577/UID1010；
结束为5655MiB。没有连续利用率采样，不能声称独占峰值，也不能把一次前后差额全归于代码改变。

## LH176：22点中的16点

单元格为 **完整秒数 / 输入位置每秒**。

| 模式 | 路径 | CPU1 | NPU/host2 |
| --- | --- | ---: | ---: |
| 无梯度推理 | 直接公式/PyTorch | 0.2028 / 300.7 | 0.1828 / 333.6 |
| 无梯度推理 | 默认/native | 0.8495 / 71.8 | 2.2292 / 27.4 |
| 含grad前向 | 直接公式/PyTorch | 0.2367 / 257.7 | 0.2108 / 289.4 |
| 含grad前向 | 默认/native | 0.9650 / 63.2 | 2.4591 / 24.8 |
| 完整训练 | 直接公式/PyTorch | 0.7568 / 80.6 | 0.6238 / 97.8 |
| 完整训练 | 公共列/native | 2.1761 / 28.0 | 4.9468 / 12.3 |
| 完整训练 | 默认/native | 2.3187 / 26.3 | 5.1350 / 11.9 |
| 完整训练 | 默认/PyTorch | 3.7771 / 16.2 | 6.7390 / 9.1 |

本轮训练覆盖直接公式→公共列→默认执行器，并测默认Python/native两个客户端。
公共列的推理/含grad前向、默认Python的推理/含grad前向未在00bb重测；不以fb83旧值填入这些空格。
所有16点均记录candidate8368、selected2572、内部消息/访问边15340。
直接公式物理Full为11264行（dense Full后mask），public/default为2572行；后者区域列94次、状态列176次。
selected/candidate约30.7%，k/区域宽=1/8是发送强度预算，不等于活动比例。
相同计数不能证明逐条路由或全部独立根一致，直接公式的raw/观察职责也不同。

与fb83相同点的完整训练成本比较：

| 用例 | CPU1，秒 | NPU/host2，秒 |
| --- | ---: | ---: |
| LH176 public-native训练 | 2.3505 → 2.1761（-7.42%） | 5.2207 → 4.9468（-5.25%） |
| LH176 default-native训练 | 2.4899 → 2.3187（-6.88%） | 5.3286 → 5.1350（-3.63%） |
| LH176 default-python训练 | 4.0621 → 3.7771（-7.02%） | 7.1539 → 6.7390（-5.80%） |

默认native推理CPU0.9651→0.8495s（-11.98%）、NPU2.2282→2.2292s（+0.05%）；
含grad前向CPU1.0831→0.9650s（-10.91%）、NPU2.4427→2.4591s（+0.67%）。
直接公式三模式CPU变化-0.16%至+1.09%、NPU+0.73%至+4.13%，给出了本轮观察波动的背景；
不能由单次共享计时得出稳定微幅收益。逐点前后值和工作量比较在JSON中。

完整训练相减定位差距在哪个区间：

| 设备 | 相减路径 | 完整差额s | 主机forward差额s | 主机backward差额s | 其余完整差额s |
| --- | --- | ---: | ---: | ---: | ---: |
| cpu1 | public-native − direct-python | 1.4194 | 0.6489 | 0.7265 | 0.0439 |
| cpu1 | default-native − public-native | 0.1426 | 0.1095 | 0.0023 | 0.0308 |
| cpu1 | default-native − direct-python | 1.5620 | 0.7584 | 0.7288 | 0.0747 |
| cpu1 | default-python − default-native | 1.4583 | 1.0892 | 0.3698 | -0.0008 |
| npu2 | public-native − direct-python | 4.3230 | 2.1248 | 2.0399 | 0.1584 |
| npu2 | default-native − public-native | 0.1882 | 0.2892 | -0.0199 | -0.0811 |
| npu2 | default-native − direct-python | 4.5112 | 2.4140 | 2.0200 | 0.0773 |
| npu2 | default-python − default-native | 1.6040 | 1.0668 | 0.5410 | -0.0038 |

主机阶段没有逐段同步，NPU工作可以跨阶段完成；负差额不代表某段有“负开销”。
默认−公共列只有CPU0.1426/NPU0.1882s；公共列−直接公式仍为1.4194/4.3230s。
因此仅替换默认分派不能解释或消除主要差距。但相减本身不能把差额全部认定为可删除的调度、
区域求解或autograd成本；Full工作方式、raw状态和观察根职责仍须逐边界匹配。

## FC16及LH约1B：其余6点

| 用例/路径 | CPU 秒/位置每秒 | NPU 秒/位置每秒 |
| --- | ---: | ---: |
| FC16 默认native训练 | 1.1635 / 880.1 | 0.5933 / 1725.8 |
| LH1B 直接公式推理 | 1.2433 / 386.9 | 0.1865 / 2578.5 |
| LH1B 默认native推理 | 2.9107 / 165.3 | 3.6109 / 133.2 |

FC16对前次CPU1.2410→1.1635s（-6.24%）、NPU0.6050→0.5933s（-1.92%）。
候选均16384；CPU selected8344/边25060，NPU selected8345/边25064，区域列4次、状态列16次。
本轮没有跨设备同路由证明。此前默认FC已接近现有直接FC的结论仍按[fb83身份](rope-performance-20261112.md)引用；
不能把旧直接FC/Transformer秒数与本轮值混作一次配对实验。

LH1B使用CPU32，前次直接/默认1.2547/2.9835s，本轮1.2433/2.9107s（-0.91%/-2.44%）；
NPU前次0.1832/3.4587s，本轮0.1865/3.6109s（+1.83%/+4.40%）。
候选均54507，直接selected11970/边72161/物理Full90112，默认11971/72166/11971。
这一项活动与五条消息的差异保留，不能形成严格同路由、同工作量速度比。
NPU峰值allocated：LH176训练直接327,067,648字节、默认native271,434,752字节；
1B推理直接5,460,244,992字节、默认5,259,393,536字节；FC16训练556.42MiB。
较少Full行或较低峰值存储没有自然转化为更快执行。

## 四格安装消费者：三AdamW及新进程恢复

使用同LH176公开plan和默认张量列入口，CPU FP64、NPU FP32，Python/native各一格。
物理NPU5→逻辑npu:0，经空闲设备准入；这与性能物理9共享观察独立。
CPU亲和64–95、NPU主机56–63，各256GiB/1800秒；实际ATen/interop均1。
格ID中的`cpu64/npu32`表示dtype，不表示线程。NPU环境OMP/MKL/OpenBLAS为2，但消费者显式设置ATen1。
AdamW lr1e-5、betas(.9,.95)、weight_decay.01、foreach=false；每窗口后detach continuation。

| 设备/dtype | 客户端 | 实际运行 | 新进程零容差恢复 | 第三步selected/内部边 |
| --- | --- | --- | --- | ---: |
| cpu / float64 | python | 3步 + 2步保存/新进程1步 | 通过 | 2731 / 16166 |
| cpu / float64 | native | 3步 + 2步保存/新进程1步 | 通过 | 2731 / 16166 |
| npu:0 / float32 | python | 3步 + 2步保存/新进程1步 | 通过 | 2732 / 16171 |
| npu:0 / float32 | native | 3步 + 2步保存/新进程1步 | 通过 | 2732 / 16171 |

每格分别跑whole3步、prefix2步保存checkpoint、新进程加载后第3步；三个子进程均exit0。
最终比较实际输出、raw continuation/pending/history/ledger、输入与参数梯度（含None）、参数及全部AdamW槽，
`atol=rtol=0`。这是同一客户端/设备的精确续接，不是四格之间逐位相等，也不是完整中间根的独立数学验证。
CPU/NPU第三步活动数2731/2732已显示跨精度轨迹不能视为相同。
两生命周期服务分别于08:16:00/08:17:33 UTC终态passed，原结果与子结果hash在JSON。

调用记录使用独立安装目录`core-source-v2-install`，源码检查确认其304个Python文件与固定00bb逐字节相同；
子结果记录的native manifest hash为`07f46539892e985127b4551b99b8ddbd8b458a1cacf0365bef0ce19ac7d438f0`。
这是安装命令、固定脚本、构建身份及事后文件审计的证据链；该消费者未记录运行时`tidegraph.__file__`
或安装包树hash，不能补称拥有这两个运行时字段。性能点则显式记录了冻结源码包的导入路径。

## 机制、数值与运行身份的边界

00bb把[来源MLP](../../python/tidegraph/weighted/column_sources.py)按实际shape/dtype/device/alpha/epsilon及有序参数槽合批；
[原生组核/VJP](../../cpp/src/weighted_column_source_ops.cpp)在无梯度packing内使用共享GEMM或BMM，
保存原版本与实际输出支持，声明一阶转置。[区域Aggregate](../../cpp/src/weighted_column_aggregate.cpp)
只复用相同输入/参数身份/有序消息列表；同类不同原子仍各参与一次，重复边不被SrcClass合并。
这是固定源码静态机制，JSON保存对应文件hash；本次没有新的packing、图普查或设备profile，
不能把fb83的算子计数或静态预期写成00bb实测减少量。

[有限舍入见证](source-columns-rounding-20261112.md)单独保留：首Aggregate差约1.92e-9，
该固定见证将同FP32材料提升FP64后全根最大差约1.71e-13。原严格失败、阈值、None/零连接边界没有被22点exit0
或四格恢复通过改写。LH测量仍为`strict_numerical/restore=not_checked_at_measured_scale`；
FC仍为`strict_numerical=not_checked`、`recovery=see separate consumer`，原值分别保留。

原计划hash为`1fc477fe910acc6e074bc9268710d3b541a2fdcf676ab558cf8978c2488b4f71`。
其`status=planned_not_run`、42点总数及部分子计数、`build_repair=fb83`、顶层`core-rope`路径与5400秒预算
残留旧模板；**实际formal数组22条、服务预算1800秒，全部实际cwd/argv/package/source/native manifest均为00bb**。
驱动只读取逐项`preparation/formal_measurements`的cwd/argv/env，不采用上述旧顶层路径。
原始字节未改写，计划元数据瑕疵与真正运行身份分列记录。

## 从干净源码重复

按[开发入口](../development.md)固定00bb并建立匹配的native构建/安装，不复用旧shm路径。
[LH配方与三层runner](../lh-tensor-baseline.md)拥有生成器与计时契约：
`prepare(dict(CASES['many-lh'],family='kv'),samples=4,positions=16,seed=7,input_pattern='staggered-v1')`
得到workload `5539e0fd…`；`CASES['lh-1b']`与samples32得到`ebe54216…`，完整hash在JSON。
`CASES`来自`scripts/prefill_regression_config.py`。换用新输出目录并验证生成身份。

成本调用`benchmark_lh_tensor.py --plan PLAN --level direct|public|default --client python|native
--mode inference|grad-forward|training --device DEVICE --threads N --native-library BUILD --output-dir NEW`；
FC调用`benchmark_dense_query.py --baseline window --client native --fc-shape fc16 --batch 8 --length 128
--mode training --warmup-state reset --phase-fences step --query-execution auto --prefill-schedule auto`，
明确固定下游仓库/源码及匹配库。逐点完整实际argv保存在JSON，旧路径只是身份引用。
安装恢复使用[planned_windows消费者](../../examples/consumer_core5/planned_windows.py)，同plan、client/device/dtype与匹配库；
消费者默认即三步AdamW与2+1新进程比较。长任务仍需固定源码、有限服务和当前合法设备准入。

本报告不新增TTFT、1B训练/恢复、9B、mixed-a/b、standalone、resident、多卡或GPU真机资格；
也不把来源列局部收益当成P1—P6或LH主要差距已完成。当前任务与剩余范围只维护在STATUS/ROADMAP。
