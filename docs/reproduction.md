# 从 Git 复现问题与维护失败经验

本页是诊断入口导航，不是第二份状态表。当前限制、停止/重启条件仅见[ROADMAP](ROADMAP.md)；
数学定义见[semantic-map](semantic-map.md)，已有正负经验见[performance-lessons](performance-lessons.md)。
历史源码/环境身份及观察值以各审阅证据为准。清理不重跑旧队列，不改变任何失败或容差。

## 接续时先做什么

1. 按[development](development.md)在固定提交重建私有环境及所需库；不假设旧venv、build或run目录仍存在。
2. 从下面找到问题、Git内输入或生成器、诊断代码和原证据。先读证据中的范围/资源及ROADMAP重启条件。
3. 只有出现具体新方案或新任务时，才在新目录运行有界检查。记录实际源码、环境、输入hash、完整命令和终态。
4. 诊断通过、原反例通过、完整功能资格和性能结论分别表述；小反例不能替代完整图、独立梯度根及恢复。

需要重演旧结论时使用报告所记提交及匹配环境；当前源码运行的是新检查，不继承旧失败或旧通过。
命令中的NEW路径必须不存在，CPU_CORE必须是对应环境和源码的构建。长任务遵守[verification](verification.md)。

## 复现入口

| 调查问题 | Git内材料和入口 | 原证据与范围 |
| --- | --- | --- |
| KV整组算术与resident组合录入任务过多 | [数值边界](kv-tensor-kernels.md)、[bank GEMM](resident-bank-gemm.md)、`test_tensor_kernel/test_kv_tensor_batches`、SDK `weighted-kv-tensor`和`block-kv-messages` | [6011d599资格/44点成本/四条profile](evidence/tensor-kernels-20261112.md)；9resident组件通过不代表第10组合通过；保留EE1019厂商摘录和依赖停止事实 |
| FC4/FC16默认、直接公式与Transformer的巨大差距 | [有限suite](../scripts/run_fc_comparison.py)、[任务投影检查](../scripts/check_fc_comparison.py)、[通用配对与附件](../scripts/cost_comparison.py)、`test_fc_comparison_diagnostics/test_cost_comparison` | [前置调查](evidence/fc-three-layer-comparison-20261111.md)、[查询块实施与复测](evidence/kv-query-blocks-20261111.md)；同材料FP64诊断保留原FP32失败，公开根/任务投影/成本分列。manifest不绑定旧运行目录，profile与完整计时各有身份 |
| 接口/文件/函数审阅、包身份与插件返回边界 | [当前生产边界与hash范围](evidence/p1p6-boundary-review-20261112.md)、[早期源码清单](evidence/implementation-audit-20261110.md)、[清单工具](../scripts/review_inventory.py)、`test_session_restore/test_task_observations/test_package_content` | 分开实际审阅、自审和固定资格，不相加重叠覆盖；按[维护影响表](core5-study.md#修改时同步维护什么)补当前改动层，不要求重审全部历史工具 |
| 默认张量路径、三客户端选择及SDK旧计数误判 | [窗口契约](weighted-tensor-windows.md)、`test_window_execution/test_experiment_columns/test_standalone_columns`、[SDK资格入口](../scripts/check_query_sdk.py) | 检查实际interface/schedule/roots/fallback；outputs列任务与complete事件根分别资格；[当前FC/LH/规模证据](core5-study.md#当前张量路径与结果)不继承旧default成本 |
| LH直接/公共列/默认执行的剩余差距 | [静态循环/形状/反向审阅](evidence/lh-block-static-20261112.md)、[配方与runner](lh-tensor-baseline.md)、`benchmark_lh_tensor.py`、`profile_lh_tensor.py`、`test_lh_source_slots` | [00bb的22点成本及生命周期](evidence/source-columns-performance-20261112.md)；精确Q/K分组和逐节点小操作、公共职责与默认分派增量分开。旧fb83来源逐项计数不描述新source-v2 |
| 新resident公共sum/qry、规模容量与内部复制 | [公共owner](resident-block-session.md)、[实验默认策略](experiment-resident-blocks.md)、`examples/consumer_core5/block_resident.py`、`test_standalone_resident_blocks` | [a3c资格/完整命令/配置重建](evidence/resident-block-public-20261112.md)；FC16多bank qry预留与LH176 optimizer/8 canonical layout拒绝不是OOM，旧EE1019未解除。角色分组与共享历史的替代边界见静态审阅 |
| 外层/私有autograd图、构图与真实反向成本 | [图统计方法和两种runner](autograd-census.md)、[唯一KV存储与转置](weighted-snapshot-bank.md)、[小反例](../tests/test_snapshot_bank.py) | Python/native统计宇宙不相加，保存字节非峰值；FP32旧scalar差异与同材料FP64见证分列 |
| FC16同尺寸Transformer/直接FC基线、VJP成本与复制隔离 | [归因契约](fc16-cost-attribution.md)、[独立公式](../scripts/fc_tensor_baseline.py)、[公式反例](../tests/test_fc_tensor_baseline.py)、[复制隔离](../tests/test_independent_copies.py)、[舍入见证](../scripts/diagnose_dense_rounding.py) | [固定结果与复现](evidence/fc16-cost-attribution-20261110.md)；训练吞吐与真实首token分列，历史退化/取消保留，直接公式不替代完整记录资格 |
| TL-REQ-004 默认prefill、区域跨时间批及剩余碎片化 | [原始槽/默认策略反例](../tests/test_weighted_query_layout.py)、[分层回归目录](prefill-regression.md)、[standalone新进程检查](../scripts/check_query_sdk.py)、[profile CSV汇总](../scripts/summarize_query_profile.py) | [默认增量交付与完整复建命令](evidence/prefill-default-20261110.md)；资格入口在旧命令末追加`auto`，原任务成本和小型严格资格分开，不依赖历史运行目录 |
| TL-REQ-004 Read/Full细碎查询与反向 | [独立局部根](../tests/test_weighted_dense_query.py)、[完整训练/cut](../tests/test_weighted_dense_training.py)、[安装消费者](../examples/consumer_core5/dense_query.py)、[舍入诊断](../scripts/diagnose_dense_rounding.py)、[原任务成本生成器](../scripts/benchmark_dense_query.py) | [具名dense-v1交付](evidence/dense-query-tl-req-004-20261110.md)：CPU FP32局部严格失败保留；小图完整检查/恢复与原任务有限成本分列，资格入口`scripts/qualify_dense_query.sh NEW cpu\|npu BUILD` |
| TL-REQ-003 Shannon正支持下溢 | [实际HB16行](../tests/data/tl_req_003_shannon_row.json)、[公共消费者](../examples/consumer_core5/positive_support.py)、[局部/整图检查](positive-support-policy.md#独立检查与复现) | 原FP32严格拒绝保留；新数值模式与下游完整训练分开，资格入口 `scripts/qualify_positive_support.sh NEW cpu\|npu` |
| 分配中的大数相减及残差拒绝 | [两份实际失败行](../tests/fixtures/allocation/provenance.json)、[Decimal独立求解](../scripts/diagnose_allocation_failure.py)、[局部回归](../tests/test_retired_allocation_witnesses.py) | [局部坐标修复](evidence/core5-local-allocation-factorized-kv-20261028.md)；仅局部分配，不能解除D2048全图keys失败 |
| D64 FP32完整训练、局部精度和反向累加 | [计划生成](../scripts/prepare_weighted_workload.py)、[完整诊断](../scripts/diagnose_weighted_workload.py)、[汇合点探针](../scripts/diagnose_weighted_joins.py) | [算术诊断契约](weighted-arithmetic-diagnostics.md)、[eager边界](evidence/core5-eager-boundary-20261029.md)；参数/梯度/槽及每调度自身恢复分别核对 |
| 原始逻辑时间反馈PDG | [四窗诊断](../scripts/diagnose_weighted_feedback.py) | [非GPU边界](evidence/core5-nongpu-boundaries-20261029.md)；W32/B4/seed7，FP32与显式FP64分开 |
| FP16 GDN/默认AdamW | [完整精度诊断](../scripts/diagnose_weighted_precision.py)、[干预检查](../tests/test_weighted_half_probe.py)、[原训练用例](../tests/test_weighted_precision.py) | [数值隔离](evidence/core5-numerical-isolation-20261028.md)、[后续半精度干预](evidence/core5-boundary-isolation-20261029.md)；原strict xfail/epsilon/loss scale保持 |
| 默认异步双NPU | [纯Torch小图](../scripts/diagnose_torch_peer.py)、[完整图诊断](../scripts/diagnose_weighted_async.py)、[一般图完整验收](#一般图异步策略的有限验收)、[桥接独立根检查](../tests/test_weighted_peer_bridge.py) | [数值/异步隔离](evidence/core5-numerical-isolation-20261028.md)、[完整CPU桥接失败](evidence/core5-boundary-isolation-20261029.md)；小图/队列探针不能认证完整图 |
| CPU autograd之后晚注册NPU | [完整小程序和退出码](evidence/core5-resident-boundary-20261029.json)的bootstrap.program/相关结果 | [初始化边界](evidence/core5-resident-boundary-20261029.md)；early/late必须两个新进程，原程序已在Git中，无需旧私有启动器 |
| native workers2跨进程恢复 | [安装消费者](../examples/consumer_core5/README.md)的check.py，显式FP32/native/workers2 | [原失败与命令](evidence/downstream-contract-20261030.md)；默认worker1不代表workers2修复，单次通过也不证明并发顺序确定 |
| 本轮NPU1B/B512精确下一窗 | [有界恢复诊断](../scripts/diagnose_experiment_resume.py)：原安装运行`CONFIG --checkpoint OLD --windows 3 --output-dir NEW`；无旧目录时省略checkpoint，由相同固定配置重建新见证 | [规模原失败](evidence/core5-platform-scale-20261031.md)；先分开raw-boundary与next-complete，再比较状态/参数stride、offset、对齐与NPU format及工作量。布局差异不自动证明因果，不运行CPU参考轨迹或改变算术/布局 |
| D2048首次写入的真实局部算术 | [有界读取诊断](../scripts/diagnose_weighted_first_write.py)、[参数匹配及导出](weighted-workloads.md) | [首次写入隔离](evidence/core5-boundary-isolation-20261029.md)；原输入binary及witness已退役，需重建相同种子/配置材料；约37GB静态材料的成本仍存在 |
| 常驻局部程序首次引用已持有参数库造成的预算拒绝 | [预算/服务诊断](resident-storage-accounting.md)、[独立小见证](../tools/device_online/weighted_budget_check.cpp)、`weighted-budget-construction` | [原审阅](evidence/resident-workspace-audit-20261110.md)及[修复资格](evidence/additive-interface-qualification-20261110.md)；构造、Describe实际执行及half保守拒绝分列，不代替原规模复验 |
| Resident单bank逐行矩阵复制、FC16任务队列失败 | [投影公式](../tools/device_online/weighted_projection_check.cpp)、[同局部尺寸KV见证](../tools/device_online/weighted_kv_query_check.cpp)、[公共阶段诊断](../scripts/trace_resident_step.py)、[服务递推见证](../tools/device_online/task_services_check.cpp) | [固定资格与原失败](evidence/resident-shared-bank-20261112.md)；机器记录的representatives保存完整configuration及展开局部绑定，可在新checkout生成用例；局部容量、完整训练、原规模队列和精确恢复分开验收 |
| 紧凑公式→公共Full共享bank的增量成本 | [bank对照](../scripts/benchmark_bank_attention.py)、[有状态基线](../tests/test_tensor_baseline_continuation.py)、[默认实际调用](../tests/test_bank_attention.py) | [资格](evidence/additive-interface-qualification-20261110.md)与[成本](evidence/additive-interface-performance-20261110.md)；数值投影/完整根分列，保存前向收益与反向负结果 |
| 批量Describe首evidence根的有限FP32差异 | [同材料FP64诊断](../scripts/diagnose_window_describe.py)、`test_component_window_weighted` | [47704c9原严格失败与限定后继](evidence/interface-batch-publication-20261110.md)；只诊断指定根，完整训练/恢复另验，其他字段和非有限仍严格失败 |
| 较大FC/HB数值、NPU流/任务资源及局部预算 | [有限目录](../benchmarks/core5-study-v1.json)、[单配置入口](../scripts/run_core5_study.py)、[配置/证据查询](core5-study.md) | [自然规模测量](evidence/core5-natural-study-20261029.md)、[流资源](evidence/core5-compact-fibers-20261029.md)、[文件预算](evidence/core5-checkpoint-budget-20261029.md)；停止条件保持，8B/16B仍无实测吞吐 |

SDK affine、normalization、源变换及query舍入的旧临时变体不作为第二套实现保留。
后续从上述完整诊断、[设备用例入口](../scripts/check_weighted_device.py)、相关独立测试及原证据进入；
这是承接调查问题，不宣称旧临时脚本与现行入口逐值等价。

张量窗口卷积的重复索引反向与精确续接可用
`python scripts/check_column_repeatability.py --device npu:0 --output-dir NEW`复验。
对于CPU Read/设备payload的共享列反向，另用
`python scripts/check_column_adjoint_order.py --device npu:0 --placement mixed-a --include-columns --output-dir NEW`。
它固定一张真实前向图重复VJP，`--client native --native-library BUILD`选择独立C++组合；
`--autograd-threads serial`只用于对照累加顺序，不是生产默认修复。再用安装窗口示例核验完整训练恢复。
完整配置先由`prefill_regression_config.py --case layer-chain --family ssd ...`生成，
配置中绑定实际native库，再传`--config CONFIG --output-dir NEW`；GDN和many-lh/mixed同理。
该检查保留普通gather的观察，不要求它必然失败；要求有序tap VJP逐位相等，且两个新进程
从同一checkpoint恢复的输出、loss、raw状态、全部参数/梯度/optimizer逐位相等。
全局deterministic开关不修改。计时与这些正确性检查分开。

## 一般图异步策略的有限验收

新队列或通信实现先保留原失败，再用`check_weighted_device.py`的现成小配置检验真实一般图。
下列三格沿用`scripts/weighted_device_cases.py`，无需专用新benchmark或预跑轨迹：

| case | 能区分的行为 |
| --- | --- |
| `delta-pdg` | 原反馈反例：3节点/2区域、重复边、不同正延迟及跨owner反馈；delta旧态参与跨cut反向 |
| `ssd-dag` | 非对齐TimedDAG：4条物理边，固定逻辑时间短卷积、SSD状态与跨owner消息 |
| `kv-pdg-initial` | 同一反馈图的KV `Read^0`绑定初态；Full读取实际更新快照，KV窗口/位置与pending完整保留 |

均为2样本、内容/summary宽2、FFN宽3，输入时间0/3，停止cut8，切面cut3；
输入包含真实零质量消息。节点按region映射两张卡。各客户端、各case使用独立新进程，
避免一个设备失败中断其余格。以native为例，Python客户端将implementation改为python并省略库参数：

```sh
TASK_QUEUE_ENABLE=1 PER_STREAM_QUEUE=1 python scripts/check_weighted_device.py \
  --device npu:0 --peer-devices npu:0 npu:1 --implementation native \
  --native-library MATCHING_CORE --placement mixed-c --scoring-dtype float32 \
  --dtype float32 --optimizer adamw --workers 1 --packing auto \
  --case delta-pdg --output-dir NEW_CASE
```

上式是显式候选策略，运行前应unset `ASCEND_LAUNCH_BLOCKING`并记录实际环境；
不自动设置公共默认，不把开关请求值当成驱动内部生效证明。按执行契约分配设备及持久作业。
检查覆盖完整记录、任务总根/首输出/逐字段Describe/控制独立根、raw状态根、streaming/prefill、
保留cut、三次AdamW更新加恢复后的第四次更新、同设备精确恢复、CPU handoff和owner重映射。
原严格数值失败、通信失败与精确恢复失败分列，不因为小图通过扩展成任意拓扑/设备或resident资格。

## 小型实际分配行

两份JSON共不足6KiB，保留原FP32输入、控制、响应、乘子、时间/节点和拒绝阈值，
[provenance](../tests/fixtures/allocation/provenance.json)记录生产源码的dirty事实及原文件hash。
它们是经过挑选的回归输入，不是完整模型快照。独立Decimal根据捕获的二进制浮点输入求解；
新测试分别检查旧返回确实越界及当前Python/native标量/批局部前向，不复写旧返回。

```sh
python scripts/diagnose_allocation_failure.py --failure tests/fixtures/allocation/wide2048.json --output NEW_WIDE.json
python scripts/diagnose_allocation_failure.py --failure tests/fixtures/allocation/matched2048.json --output NEW_MATCHED.json
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_retired_allocation_witnesses.py
```

其他诊断的完整命令、原始形状、精度及预算已有[算术诊断](weighted-arithmetic-diagnostics.md)和各证据，
本页不再复制另一套默认配置。原始大输入已退役时，hash只能核对重新生成的内容，不能恢复丢失的字节。

## 一次调查结束后维护什么

| 产生的内容 | 应维护的位置 |
| --- | --- |
| 当前结论、停止/重启条件 | ROADMAP已有行；STATUS只保留当前交接与活跃任务 |
| 已审阅结果、原失败、源码/环境/命令/输入身份、原始材料是否在线 | docs/evidence的小报告及索引；历史报告不随新结果改写 |
| 仍需执行的通用诊断 | scripts中的有界入口；相关测试及本页导航，输入参数显式传入 |
| 必要的小型实际输入 | tests/fixtures，保留来源/hash、适用范围，并以独立公式或原严格断言验证 |
| 性能与工程经验 | performance-lessons，记录正负收益、工作量、资源争用及适用边界 |
| build、环境、原始日志/张量、完整源码快照、checkpoint | 临时ignored目录；终态审阅并有清理授权后退役，不成为下次接续的隐含依赖 |

不要把所有失败脚本都升级为公共API，也不要为了清理把未采用补丁全部合并。
正式交付的代码必须已提交；实验差异被放弃时明确记录，不用“其HEAD已合并”代替这项判断。
已安装的下游应持有自己的包/库及依赖锁；不要依赖本仓库临时build目录长期存在。

[退役任务摘要](evidence/retired-jobs-20261030.json)保留520份旧任务的终态/来源/原记录hash，
外层passed不表示每个内部case通过。原始日志不再在线；判断具体语义/性能结论仍需相应审阅报告。
2026-10-31新增[41份任务终态及补充记录](evidence/retired-jobs-20261031.json.gz)，三平台204条查询结果仍在各原审阅归档；
[彻底清理记录](evidence/artifact-cleanup-tracked-only-20261031.md)声明本轮原始文件退役和未提交源码保全位置。


## CUDA 真机分层入口

`check_cuda_eager.py`按客户端/配置保留原严格断言与独立失败，包含四次更新、完整返回、独立根与同栈精确恢复；
CPU/remapped handoff另用原容差，不从首个KV或一次构建推断全部资格。
`check_cuda_peer.py`查询所有指定方向的P2P/原子能力与enable返回码，比较普通Python/native复制和显式Python CPU桥，
覆盖默认/非默认流及独立None/零/别名根。复制通过不证明直接P2P、NCCL、完整模型或resident。

```sh
python scripts/check_cuda_peer.py --devices 0 1 --native-library CUDA_CORE --output-dir NEW_PEER
python scripts/diagnose_weighted_async.py --device cuda:0 --peer-device cuda:1 --copy-policy cpu_bridge --scope complete --case delta-pdg --output-dir NEW_BRIDGE
```

CPU桥仍是显式一阶诊断路径，不改变公共通信协议。CUDA检查/诊断/测量入口显式关闭TF32并记录IEEE策略；
库消费者仍拥有自己的精度政策。真机资格与当前失败见STATUS及后续审阅证据，不继承CPU/NPU结论。

[RTX3090固定有限证据](evidence/core5-cuda-finite-20261031.md)给出从干净源码构建、实际配置/逐项查询及
Nsight SQLite分析命令。Git内`benchmarks/core5-cuda-finite-v1.json`与`run_core5_study.py`可重新生成
FC/HB/LH small/medium输入，不需要旧fixture或私有driver。HB small resident、medium keys/values和
LH queue64拒绝均保留配置/结果，原始材料须重新生成；显式`--resident-limits`只改变资源，不更改数学或原失败结论。
`inspect_cuda_timeline.py`只读单进程/单设备Nsight导出；实际GPU trace须重新采集，hash不能恢复退役的字节。

## GPU规模推理、批次与返回打包

[固定报告与完整命令](evidence/core5-cuda-inference-20261031.md)连接
`benchmarks/core5-cuda-inference-v1.json`、`run_core5_study.py`、`planning.inference_budget`、
`diagnose_core5_inference.py`及`test_weighted_return_packing.py`。原d256 FP32差异保留原输入/容差；
FP64是独立配置。先重建匹配core/standalone，按实际1B构造/窗口/恢复峰值准入近界，
不需要旧fixture/checkpoint。`--observe-work`只记实际桶/矩阵行，`--profile`另起有界进程，均不作吞吐。
实际证据从`core5-index.json`查询；六项后续停止/重启条件仍集中于ROADMAP和performance-lessons。

## 三平台大batch、同模型多卡与公开实验入口

本轮材料在[189有限配置](platform-matrix.md)及`configs/matrices/`，包括B128/B512、A/B/C、三图层级、
真实反馈/非对齐/一般端口、三客户端和同模型任意卡数放置。按[人类手册](human-workflow.md)从固定源码安装，
填写自己的site与构建来源；生成输入及调度不读取参考轨迹，不依赖旧运行目录或内部benchmark导入。

```sh
tidegraph plan configs/matrices/npu-batch.json --matrix --site site.json
tidegraph matrix configs/matrices/npu-batch.json --site site.json --output-dir jobs/new-npu-batch --detach
tidegraph job status jobs/new-npu-batch
tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --batch 512 --format table
tidegraph results docs/evidence/core5-platform-*-20261031.json.gz docs/evidence/core5-finite-packing-20261031.json.gz --format html --output matrix.html
```

选择单项可按[规模报告](evidence/core5-platform-scale-20261031.md)的JSON提取命令，随后公开`plan`/`job start`；
配置名称不代表通过，数值、离散、执行、恢复和性能资格分别查询。未准入/资源排队与实际执行不混为一类。
`configs/checkpoint-cost/`显式改变保存间隔或存储介质；磁盘/tmpfs与完整返回成本分列。
`configs/capacity-training/`的1e-9 loss是独立任务配方，不解除原0.001配方非有限。
主存准入6095c6a的实际欠估见证与固定45项检查在[训练容量](evidence/core5-platform-training-capacity-20261031.md)。

NPU1B/B512原下一窗逐值失败的有界入口为`scripts/diagnose_experiment_resume.py`：
使用待检查checkpoint所对应的固定安装与相同配置，先重放原三窗，再比较raw边界、实际所选
返回路径及下一窗完整raw状态/参数，记录stride/offset/格式。outputs/metrics不得为诊断偷偷
启用完整trace；`tests/test_resume_diagnostic.py`注入未返回的状态错误检验拒绝。
从干净仓库重新生成的checkpoint是新的见证，不能冒称复验了原文件字节；原结果/hash与失败仍保留。

```sh
# 旧checkpoint已按授权退役；从相同配置生成一个新的有界见证：
/path/to/fixed-install/bin/python scripts/diagnose_experiment_resume.py resolved.json --windows 3 --output-dir diagnostics/new-resume
# 仅在另有待检查文件时增加 --checkpoint /path/to/checkpoints。
```

此诊断不改布局、算术或精度，不让CPU轨迹指导候选；raw边界已不同须先区分新进程重放与序列化，
不能直接将后续差异归因于checkpoint。NPU blocking崩溃、resident服务资源和其他停止/重启条件仍只在ROADMAP维护。

## 可重用的三层对照收集

HB/LH沿[独立张量基线](lh-tensor-baseline.md)及`benchmark_lh_tensor.py`重建直接/公共/默认层，
保留真实候选、selected与dense Full工作量。下游REQ-006沿[固定材料诊断](req006-diagnostics.md)
进入四份压缩配置、固定Git源码及`diagnose_req006.py`，无需旧学习目录。
[四代表证据](evidence/req006-diagnostics-20261112.md)的clamp反向热点可单独用
`diagnose_scalar_floor.py`复验Scalar/Tensor阈值；measure/profile分进程，不修改下游生产函数。
`check_lh_tensor.py`独立检查实际列/端口/raw和所选根族VJP；`--left-entries/--start-position`
生成非空大时钟，`planned_windows.py`另验计划图的真实AdamW和新进程恢复。
[节点/区域证据](evidence/node-region-batching-20261112.md)与
[固定55a七根族/生命周期及显式队列证据](evidence/query-lifecycle-20261112.md)分列严格失败、
None/零差异、独立安装和源码消费者的实际范围。
REQ006跨进程配对用`--initial-state-out/in`复用实际模型、optimizer槽和RNG；小型固定配置位于
`tests/fixtures/req006-tiny-transformer.json`。原Scalar失败和显式dtype常量的成功分别见
[算子/初态证据](evidence/req006-floor-20261112.md)，不以同seed推定配对。

`run_fc_comparison.py --suite windows` 测默认窗口 Python/native、直接 FC 和 Transformer 的
FC4/FC16 三模式；`--suite window-profiles` 另测 B2/L32 的真实算子及 autograd census。
用 `collect_fc_suites.py --suite RUN_DIR [--suite MORE] --job STATUS_JSON
--consumer RESULT_JSON --junit CHECKS_XML --output NEW_JSON` 从显式目录收集，无需约定旧作业名。
该入口保留每个子命令退出码、缺失结果、源码/安装/配置/参数身份、计数和独立数值状态，
不以服务 passed 推导资格。Profile/census 的耗时不混入正式吞吐，CPU/NPU 图计数不可相加。


FP16张量窗口与双owner的最新有限验收见[固定记录](evidence/column-half-20261111.md)。
`test_half_columns`覆盖half/FP32 Attention独立公式、五记忆全字段和master恢复；
`test_experiment_columns -k half`覆盖默认实验配置/loss/恢复。双owner使用
`TIDE_COLUMN_PEERS=npu:0,npu:1`运行`test_column_payload_owners`；普通异步与
`ASCEND_LAUNCH_BLOCKING=1`必须分开记录，后者不是运行时修复。
SDK执行`tidegraph-column-check --device=npu:0 --dtype=float16`，peer显式设置
`TIDE_COLUMN_PEER_DEVICE=npu:1`；源码/构建/设备与安装消费者资格分别见报告。
