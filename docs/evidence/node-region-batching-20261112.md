# 节点与区域合批：固定资格及有限成本（2026-11-12）

固定实现 `a95b8987af6f823366a25dcbfdeae225c986e184` 已完成相容 KV 节点准备、
按实际候选数量的区域合批和单位传输检查合批。LH176 默认 native 完整训练的本次观察为
CPU **2.451 秒／24.886 输入位置每秒**、共享 NPU **5.011 秒／12.174 输入位置每秒**；
直接公式分别为 0.849／0.661 秒。区域调用从 307 降至 94，但主要差距仍在公共张量执行内部。
本报告不把这轮局部改进、FP64 数值投影通过或组件资格扩为 P1—P6 已完成。

[机器可读结果](node-region-batching-20261112.json)保留26个成本点、材料/构建身份、
原失败和后继资格。上一比较边界见[LH三层成本](lh-three-layer-20261112.md)，
接口见[张量窗口契约](../weighted-tensor-windows.md)，持续范围由[ROADMAP](../ROADMAP.md)管理。

## 实现边界与数学对应

数学仍固定 `0770593838a37c790a568226930567ed60f99b83`，不改变发送、Read/Full 快照或参数绑定。

| 实现提交 | 采用的机制 | 保留的语义与验证 |
| --- | --- | --- |
| `78fc90b4` | 相容节点的 KV 准备使用整块 MM/BMM；相同参数与实际输入才复用计算，否则批量独立矩阵 | 私有状态、不同 epsilon、实际正写入支持与预算；`linear_blocks` 用 `dX=gW`、`dW=gᵀX` 手算检查，未使用块保持 None、连接零保持零，别名汇总全部贡献 |
| `96cb570a` | 区域由完整成员名单分组改为实际候选数 N 分组；每组保存 int64 `[R,N]` 真实成员，描述/控制按列交接 | 不填假候选，稳定节点次序与十种发送各自的反向；结构支持区分不共行节点、caps/普通分支、None/零；同成员或无梯度路径避免额外私有图 |
| `a95b8987` | 固定 unit transport 按 device/dtype 和显式字节预算合并值检查，no-grad；无跨调用缓存 | 每项仍检查不可训练、每个元素仍必须等于1；非连续/空 Tensor、最后一块非法值、变化后再次调用均检查 |

代码入口为[KV准备](../../python/tidegraph/weighted/kv_columns_state.py)、
[块线性伴随](../../python/tidegraph/weighted/row_linear.py)、
[区域布局](../../python/tidegraph/weighted/regional_packing.py)、
[区域组合/VJP](../../python/tidegraph/weighted/regional_groups.py)与
[单位传输检查](../../python/tidegraph/weighted/transport_validation.py)；
C++ 对应 `weighted_column_state.cpp`、`weighted_row_linear.cpp`、
`weighted_regional_packing.cpp`、`weighted_regional_groups.cpp`、`weighted_unit_transport.h`。
这些是列布局/求值粒度变更，未把数学区域合并，也未以同值代替参数或输入身份。

`test_unit_transport_validation` 的 CPU profiler 计数见证：2或2,132个相容标量均一次
`aten::equal`；预算迫使分块时逐块比较。该断言说明主机可见调用粒度，
不能据此给 NPU kernel 数或端到端节省时间定量归因。

## 来源、构建与执行资源

26个成本结果均记录 a95、干净源码、completed；两个13项矩阵的逐项 exit code 均为0。
原始构建先固定实现再启动；后继 checker 固定 `f681ef63`（CPU）／`3f99eb22`（NPU）。
`git diff a95b8987 3f99eb22 -- cpp python CMakeLists.txt cmake` 无差异；
后继只增加诊断/修正测试设备绑定，保留其不同资格源码身份。

两套新构建均为 aarch64、Release、GCC10.3.1、C++17/ABI1、CANN9.2.0，
各160 build jobs并行构建；这是本次资源安排，不是后续构建上限。
Python-owned 为 Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post3；
standalone 使用独立 LibTorch-NPU2.12.0 SDK。CPU native 点使用前者执行 CPU Tensor。

| 身份 | SHA-256 |
| --- | --- |
| C++/CMake 源码 | `f4596708e67ba24ba1d844452cb551593e8a46db25e641f3ce8c02304dc0130b` |
| Python-owned `_tide_native.so` | `b4ba06c78809ce84db098f0efa4f7fba9cde186125da02788e187d9d0d175086` |
| Python-owned `libtidegraph.a` | `4b040669ae6ac34823afe2a2e3e6d54292c121692f1ca14b0b25cb1d99a46e77` |
| standalone `libtidegraph.a` | `f0b2ab655e46ea10f085383bd72f09d0d561e044f4f8a592778a9d425cb42a0d` |
| standalone `tidegraph-column-check` | `b95817d72572a2b06a735e8035afc2297d6c6dc95f2e14648b58c9c6dede5b4a` |

源码 hash 和实际二进制 hash 已重算核对；完整 manifest 摘要在 JSON。

| 任务 | 申请资源 | 结果中实际 CPU affinity／线程 |
| --- | --- | --- |
| CPU成本13点 | CPU0–31 | 0–31；ATen1／interop1 |
| 共享NPU成本13点 | CPU0–31、64GiB、900秒；物理9→npu:0 | **0–319**；ATen2／interop1 |
| 原共享资格 | CPU64–127、128GiB、900秒；物理1→npu:0 | 不把申请 CPU 池当成实际隔离；安装案例未记录进程 affinity |
| 后继LH规模检查 | CPU192–223或64–127的申请池 | CPU和NPU checker实际均0–319 |

共享卡使用有本次对话授权、PID/start/UID绑定、资源预算、两次准入采样及组锁。
成本卡9两次准入均健康、0% AICore、5,653/65,536MiB HBM；max AICore20%、
max HBM49,152MiB、要求余8,192MiB、授权策略一小时有效。
这是准入时快照，不是整个计时区间无争用的证明。原资格卡1先有超预算样本，后续0%/3%准入。
本项目重型构建在成本运行前结束；轻量测试、共享主机/设备外部争用仍可能影响时间。
`AllowedCPUs` 申请未使这些共享任务形成所写的进程亲和性隔离，不能宣称 CPU 池互不争用。

## 独立检查、训练与恢复

| 固定源码／检查 | 终态与范围 |
| --- | --- |
| 96cb CPU FP32/FP64 | 178通过、9原数值xfail；区域布局/分组、节点准备/块伴随、一般Settle端口、窗口、共享、装配、放置与owner |
| a95 CPU FP32/FP64 | 150通过；含transport38、source binding22、native区域28、column training6、window18，也含resident调度11/selection6及floor诊断21，不能都计为批量化用例 |
| a95共享NPU定向 | 59通过、12失败；含CPU辅助项，不能称59个NPU案例。12项均为native区域对照设备不一致 `StructureMismatch: root[0]: tensor device mismatch` |
| 3f99后继区域检查 | **15通过**：12个实际NPU case及3个CPU辅助项；e3ac44fb仅使测试显式绑定payload control/selection，未修改生产默认；原12失败保持 |
| a95安装消费者 | CPU FP64、NPU FP32 mixed-c各 Python/native × KV/SSD 四项，共8项：三步AdamW、独立比较、新进程精确恢复通过；deterministic=true，小FC4/B2/L3 |
| a95独立SDK CPU | `tidegraph-column-check`实际执行：独立streaming、raw状态、VJP、两SGD窗口通过；并非Python调用native |

区域打包用字面关联映射/转置；分配组用逐帧标量区域调用检查组内不同成员及独立输出根，
它不替代独立求解算法对每个数学候选的证明。KV准备另有逐时间状态公式，
块线性使用显式 Jacobian。上述9项原xfail均为 `test_general_ports_messages_states_and_gradients`
的FP32严格数值见证；匹配FP64检查分开通过，不提高原阈值。

新增[LH尺度检查器](../../scripts/check_lh_tensor.py)对176节点本次材料比较独立StaticKV公式、
公共 native 与默认 native，候选只解码自己实际计算的列，不调用参考补全。检查控制、候选/活动
行、完整raw KV槽与int64元数据、逐物理端口cG，以及一个输出加raw字段损失的参数/输入VJP。

| 检查器／平台 | 严格数值投影 | 超差 numeric leaves（control／raw／output／gradient） |
| --- | --- | --- |
| f681 CPU FP64 | 通过；atol1e-10、rtol1e-8 | 0／0／0／0 |
| f681 CPU FP32 | 有限严格失败；atol1e-6、rtol1e-5 | 143／2256／8／2406 |
| 3f99 NPU FP32 | 有限严格失败；相同FP32阈值 | 126／2075／8／2308 |

每行public/direct与default/direct结果相同；候选/活动支持、raw int64和标量任务值均通过。
三种检查各有36个 None 与connected-zero连接差异，独立dense公式计算了额外未使用Full；
这是声明的None-as-zero**数值投影**，不是完整计算图连接等价。
CPU FP32示例：port0 maxabs差0.000672817（reference max13.30908）；输入梯度maxabs差
0.12937546（reference max554.8894）。FP64通过支持有界高精度诊断，不能将所有FP32超差自动认证为无害。
检查使用共同生成的空左态/单位输入；FP64参数从同一FP32初始化提升。
未覆盖该规模的全部独立中间根、逐边内容、多步optimizer或恢复；小安装恢复资格不能代替这些能力。

原共享资格外层failed还包含Scalar floor测量/profile的独立边界失败；Tensor两项exit0。
两resident组件构建/loader及有限真机检查单列通过，均不等于完整resident执行器/optimizer/恢复验收。
CPU f681与NPU3f99诊断外层因各自LH FP32严格失败保持failed；不以成功内项重写外层终态。

## LH176：同材料三层成本

沿用上一报告同一生成器：176节点／22区域，每区域8节点；rank区域数 `1,2,4,8,4,2,1`，
1,058条内部物理边，输入/输出各8端口，stride9。d64／H1／FFN256／summary32／adapter32；
扩散56节点KV qry、Full/Read RoPE64/32，收拢56节点KV sum、Full RoPE32，两者window4608；
中间64节点KV sum/noRoPE/window0。全部Read+、cat、regional_total/context/cat/quad，温度1、
k1、gamma_max2；14个非中间区域att/content_stats，8个中间区域ind。来源分类region，alpha_agg1。
这是core-5 LH启发图，不能与原LH Add/Attention视为数学相同。

B4/L16、staggered-v1、空左态、seed7，实际输入位置61；参数14,902,694、冻结2,132。
本轮20点及旧c83同材料hash：workload `5539e0fd…25d7`、模型 `939fbb10…a57f`、输入 `6fb02760…22fc`，
完整值见JSON。每个独立进程一次重置warmup、一次重置后测量；训练为逐端口
`1e-6 sum((cG)^2)`、SGD(lr1e-5,momentum.3)。区间含输入/上传、packing、前向、loss、
finite、反向、更新、清理及同步，构造/重置在区间外；不测TTFT。

每格为**秒（输入位置/秒=61÷完整秒）**。

| 设备／模式 | 直接公式Python | 公共native | 默认native |
| --- | ---: | ---: | ---: |
| CPU 推理 | 0.207656（293.756） | 0.903968（67.480） | 1.004557（60.723） |
| CPU 含grad前向 | 0.238812（255.431） | 1.072227（56.891） | 1.156607（52.740） |
| CPU 训练 | 0.849339（71.821） | 2.362251（25.823） | 2.451212（24.886） |
| 共享NPU 推理 | 0.193738（314.858） | 2.055867（29.671） | 2.699729（22.595） |
| 共享NPU 含grad前向 | 0.204282（298.607） | 2.303507（26.481） | 2.514147（24.263） |
| 共享NPU 训练 | 0.661101（92.270） | 4.976049（12.259） | 5.010665（12.174） |

默认纯PyTorch训练补点：CPU3.858339秒／15.810位置每秒，NPU7.193995秒／8.479位置每秒。
所有点FP32；LH请求IEEE精度，NPU matmul/conv HF32均false。native是Python调用LibTorch，
public/default NPU为mixed-c主机调度，不是standalone、resident或跨卡测量。

全部点candidate8,368、selected2,572、内部消息15,340一致；direct physical Full11,264，
public/default仅2,572。后两者state column仍176次，region column由旧307次降为94次。
相同计数不证明逐值/梯度等价；直接公式计算更多Full仍更快也不证明所有稀疏度都应dense Full。
当前默认/直接训练成本比CPU2.886、NPU7.579；公共减直接CPU1.513秒、NPU4.315秒，
默认减公共仅0.089／0.035秒。差额不是纯调度或纯求解器时间，包含各自观察/状态、布局与反向职责。

旧c83默认native训练CPU3.062758→2.451212秒（观察下降19.97%），
NPU7.495768→5.010665秒（观察下降33.15%）。旧CPU affinity0–63、新0–31；
旧NPU为空闲物理11、affinity144–159，新为共享物理9、affinity0–319。
因此不将差额全归因给三项优化，也不以一次测量给出统计稳定加速结论。
本轮没有LH kernel时间线或私有图普查；Read/Full仍有逐节点边界，剩余验证、查询布局和伴随的
各自成本尚未隔离，不能用本表推断“数学必然慢”或某一单项已解释全部差距。

## FC16：默认native三模式

同固定下游 `0439efd2af31cf10b07ce9e6423b0195f7449968` 的FC16/B8/L128/d128，
4区域×4节点、H2／FFN512／summary32／adapter32、KV qry/noRoPE/window3072，
Read+、cat/ind/context/cat、regional_total/quad、温度0.25、k2。
核心参数4,564,404、含embedding/head共5,092,788；初始及测量模型hash六点完全相同，见JSON。
主入口 `benchmark_dense_query.py --baseline window --client native --trace-records minimal --phase-fences step`，
生效tensor-window-v1/settle-columns-v1，包含raw续接导出。每点一warmup/一测量，输入seed991/992。
训练含输入生成上传、embedding/core/readout/head/loss、finite、实际反向、clip、optimizer、清理同步。
FC请求vendor-default；NPU matmul HF32=false、conv HF32=true，不能冒称与LH精度策略完全一样。

| 模式 | CPU完整秒 | CPU输入位置/s | 共享NPU完整秒 | NPU输入位置/s |
| --- | ---: | ---: | ---: | ---: |
| 推理 | 0.440323 | 2325.564 | 0.239234 | 4280.328 |
| 含grad前向 | 0.497009 | 2060.326 | 0.259675 | 3943.388 |
| 训练 | 1.278176 | 801.142 | 0.661738 | 1547.440 |

分子均为8×128=1,024输入位置，非输出生成速度。推理另实测warm-loaded批首token：
CPU0.439518秒、NPU0.238401秒；范围是输入创建/上传、prompt prefill、末位置head、
greedy选择、finite及host返回，不含排队、tokenizer或后续decode。
CPU/NPU候选均16,384，Full分别8,344／8,345，消息25,060／25,064；
没有据此认定两设备完整轨迹严格等价。阶段计时为主机提交区间，不能相加给NPU阶段归因。
前一aligned训练CPU1.272193／空闲NPU0.617622秒；本轮CPU相近，NPU略慢，
共享条件与单次样本不足以作严格防退化裁决。本轮未重测直接FC或Transformer，保持既有证据身份。

## 从固定源码复现

新检出a95，按[开发入口](../development.md)准备匹配环境与构建；长任务使用固定检出和有界
background.slice。下列BUILD/LAB/NEW为新路径，LAB须能读取上述固定下游Git对象；
不是要求找回旧运行目录或共享卡策略。设备必须按当前授权重新准入，不能继承本次共享许可。

```sh
TIDE_BUILD_DIR=BUILD python -m pytest -q tests/test_linear_blocks.py tests/test_kv_column_groups.py tests/test_regional_packing.py tests/test_regional_groups.py tests/test_regional_groups_native.py tests/test_unit_transport_validation.py --dtype both
python scripts/benchmark_lh_tensor.py --case many-lh --level direct --client python --mode training --device cpu --threads 1 --output-dir NEW_DIRECT
python scripts/benchmark_lh_tensor.py --case many-lh --level public --client native --native-library BUILD --mode training --device cpu --threads 1 --output-dir NEW_PUBLIC
python scripts/benchmark_lh_tensor.py --case many-lh --level default --client native --native-library BUILD --mode training --device cpu --threads 1 --output-dir NEW_DEFAULT
python scripts/benchmark_dense_query.py --downstream-repo LAB --baseline window --client native --native-library BUILD --trace-records minimal --phase-fences step --batch 8 --length 128 --mode training --device cpu --threads 1 --output-dir NEW_FC
```

三模式改 `--mode inference|grad-forward|training`；共享/空闲NPU合法准入后改device与threads2。
安装示例 `examples/consumer_core5/tensor_window.py` 显式family/client/device/dtype可复现小型训练恢复。
后继LH检查器须用f681或3f99源码，a95匹配的生产库可复用，命令为：

```sh
python scripts/check_lh_tensor.py --case many-lh --batch 4 --length 16 --client native --native-library BUILD --device cpu --dtype float64 --output-dir NEW_CHECK
```

原始结果在 ignored `artifacts/rewrite-20261112/` 的batched CPU/NPU矩阵、同名目录及四份JUnit，
独立LH检查目录、`shared-{cost,qual}-{policy.json,audit.jsonl,queue.json}`。
生命周期见 `artifacts/jobs/rewrite-{regions,batched-native,batched-sdk,batched-cost,shared-cost,shared-qual,diagnostic-f681-cpu,diagnostic-3f99-npu}`；
JSON保存其已核对终态和各结果hash。前三批构建/CPU成本及shared-cost passed；
shared-qual和两LH诊断外层failed保持。原始目录可按保留规范退役，本文、JSON和固定生成器保留复现入口。
