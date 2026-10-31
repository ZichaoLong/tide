# core-5 候选、配置与测试入口

从本页可以独立找到模型配置、公共实现、运行入口、检查和证据。数学固定
[0770593](upstream.json)；[当前进度](STATUS.md)和[待办](ROADMAP.md)分别维护。

| 要找的内容 | 直接入口 |
| --- | --- |
| 实验接入与需求反馈 | [协作规范](downstream-contract.md)、[可复制的core-5消费者](../examples/consumer_core5/README.md)；依赖锁/请求/交付/验收模板由规范导航 |
| 选模型/执行组合 | [有限目录](../benchmarks/core5-study-v1.json)、下方测试表；公共`resolve_model`可配置目录外合法组合 |
| 当前实测设置与结果 | [自然规模25配置](evidence/core5-natural-study-20261029.md)、[实际大文件后续窗口](evidence/core5-checkpoint-budget-20261029.md) |
| 已通过的功能门禁 | [实际26模型/52训练检查](evidence/core5-coverage-correction-20261030.md)、[三个消费者资格](evidence/core5-study-clients-20261029.md)、各模块独立公式/根测试见[semantic-map](semantic-map.md) |
| 在新clone中筛选结果 | [逐配置证据索引](evidence/core5-index.json)；`python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --kind correctness --format tsv` |
| 修改时同步维护什么 | 本页维护表；[实现分层](architecture.md)、[开发/资格](verification.md)、[历史与新经验](performance-lessons.md) |

配置摘要与覆盖可从同一入口查询（Python≥3.10，`-S`不加载Torch）：

```sh
python -S scripts/index_core5_study.py docs/evidence/core5-natural-study-20261029.json --view models --format markdown
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --topology fc --family ssd --kind correctness --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --kind performance --device npu:0 --format tsv
```

`models`展示全部节点/区域配置的分组、实际窗口及来源分类/参数共享；历史recipe须先匹配model_id。
`coverage`按实际model_id/workload_id/execution_id及源码展示，不按用例名称继承资格；可筛选
topology、profile、实际family、client、device、placement、dtype、kind和终态。正确性检查与性能
记录分别显示；没有测量窗的测试不产生吞吐资格。未出现的配置是未被本索引认证，不解释为不支持。
索引只引用明确的仓库报告，不扫描旧运行目录；历史仅有汇总计数的报告继续从证据文档进入，
不自动拆成未经记录的逐配置通过项。

静态事实由`planning/facts.py`生成；新run自动保存，JUnit中的实际recipe/执行/检查记录可由
`record_core5_checks.py --junit FILE --source EXACT_COMMIT --output NEW_JSON`提取。
失败、跳过和缺少梯度/恢复断言分别保留；测试耗时不能变成性能时间。

## 配置与测量身份

| 对象 | 公共入口 | 身份包含什么 |
| --- | --- | --- |
| 数学模型 | [resolve_model](../python/tidegraph/planning/model.py) | 物理图/端口、全部展开的局部配置、SrcClass、共享绑定、聚合尺度；独立角色清单 |
| 工作负载 | 同文件的 `workload` | 模型身份、输入策略、样本/原始时间窗、初值种子；不含参考轨迹 |
| 执行 | [ExecutionSpec](../python/tidegraph/planning/execution.py) | 三种client、设备/peer、放置、dtype、scoring dtype、streaming/prefill、算子/packing、线程；请求和生效设置分列 |
| 测量 | 同文件的 `MeasurementSpec` | 工作负载/执行身份、推理或完整训练、warmup/步数、恢复检查、超时 |

模型身份不因设备或样本窗改变，工作负载身份会因输入改变；实际测量还须固定源码、材料与环境。
身份是追溯键，**不是资格认证**。实现版本、构建、真机检查、数值失败与性能结果单独记录。
输入默认保留原密集材料和hash。显式`window.input_pattern="staggered-v1"`（公共
`workload/prepare`或`prepare_core5_study.py --input-pattern staggered-v1`）按样本提供
`P-(sample%min(3,P))`个原始逻辑位置；多端口时省略满足
`(sample+global_position+port)%4==0`的输入。未出现的物理输入保持缺席，不补数值零。
时间仍为`rank+stride*global_position`，每样本/端口的occurrence单独连续编号，跨窗按
该端口实际出现次数推进。内容随样本/原始位置改变，同一位置的真实广播共享张量；
三客户端消费同一静态输入，无参考轨迹预跑。B/P仍是逻辑上界，memory_floor保持原B/P容量口径，
仍只是所列存储分区的下界，不是完整运行峰值准入；
实际消息、活动和缓存差异须用完整运行记录检查，不能由输入策略名推断。
运行记录另存materialization ceiling、resident limits文件hash及native checkpoint文件预算；
这些资源准入设置不改变模型身份。standalone的`--checkpoint-max-bytes`默认4GiB，
较大训练记录可显式提高，详见[原生checkpoint](native-graph-checkpoint.md)；不是峰值内存承诺。
`resolve_model`直接复用公共`KVConfig/RecurrentConfig/RegionalConfig/SourceBinding`的适用性检查；
支持逐节点记忆和逐区域配置列表，不将有限recipe集合当成合法候选全集。
参数清单用独立shape公式，`materialize`再核对真实参数形状、trainability和所有共享别名。
模型参数、私有状态和计算共享仍是不同对象；构造没有真实轨迹预跑。

## 拓扑和自然规模

[recipes](../python/tidegraph/planning/recipes.py)提供如下起点。它们都属于core-5局部语义；
未指定的可训练矩阵使用公开构造器的确定性初始化，无任务头或填充参数。

| recipe | 连接 | 局部设置 |
| --- | --- | --- |
| `fc` | 相邻层全连接，深度/区域宽度可变 | baseline或FC实例§4.2 B的窗口/位置 |
| `hb` | 递归扩散/收拢，同秩独立区域，默认偏移{0,1} | baseline或HB实例§4.2的qry/sum、完整/部分RoPE、att/ind |
| `lh` | HB连接加局部邻域、跨区域线和跳层线 | 同一core-5模块配置；不引入LH局部函数 |

baseline采用KV sum、cat/ind、regional_total/context/cat、quad；可以切换五记忆、发送、
regularizer及显式覆盖。textbook配置借用固定实例的记忆/评分表，默认区域控制按实例§4.4；
选择`constrained`可用§4.2的固定增益发送。`mixed`在中间区域交替GDN/SSD，其余KV。
自然维度保持FFN=4d、summary/adapter/source/score宽32、KV头宽64；GDN/SSD按d扩展头数，
key宽64/value宽128；基础循环key/value宽64。小测试允许按内容宽度缩小这些尺寸。

`target_parameters`按128的内容宽度步长选择最接近的自然配置，报告实际参数数和偏差。
约1B/8B/16B不要求精确相等，不固定core-3节点/区域数，不压缩FFN去凑目标。
模型头参数为零，固定transport与去重可训练参数分列。旧core-3匹配入口仍可复现，见
[历史工作负载契约](weighted-workloads.md)。

稀疏预算记录请求的1/N、取整后的逐区域k；k限制发送强度，不能解释为活动节点比例。
`regional_total_zero`可以发送小数，不能替换成硬Top-k。实际candidate/Full/消息、区域批长和
solver/packing统计必须来自运行。位置窗口、衰减与短卷积都按原始逻辑时间绑定；
同秩区域不人为错开rank。

```sh
python -S scripts/prepare_core5_study.py --topology hb --profile textbook \
  --target-parameters 1000000000 --region-width 32 --sparse-denominator 8 \
  --samples 1 --positions 1 --output-dir NEW_PLAN
python scripts/benchmark_weighted.py --workload NEW_PLAN/workload.json \
  --device cpu --implementation python --mode training --checkpoint-check \
  --max-materialize-bytes EXPLICIT_BYTES --output-dir NEW_RUN
```

`--recipe FILE.json`接受`model_recipe/scaled_recipe`的完整关键字，不另造CLI默认语义。
任意合法组合可直接调用公共`resolve_model`和`workload`；`materialize`构造公共binding。
新计划v2展开默认配置且保留model_id；旧v1内容/hash不改写。两者使用同一个实际构造器。
完整消费者、二进制材料与计时边界见[性能契约](weighted-performance.md)。内存floor只是下界，
不是队列、tape、workspace、流资源或训练峰值准入。

## 修改时同步维护什么

| 改动 | 必须同步的入口和证据 |
| --- | --- |
| 数学配置/形状/共享 | Config、planning/shapes与inventory、实际绑定；[semantic-map](semantic-map.md)对应、模块契约；独立公式与实际角色检查 |
| 新拓扑/输入时钟 | planning/topology、recipes、端口/延迟/可达性检查；完整返回/续接；recipe身份 |
| 调度/packing/局部优化 | 公共LocalProgram能力及独立调度；[packed](weighted-packed.md)、[区域批](weighted-region-batch.md)；独立根VJP、工作量与完整计时 |
| 新设备/放置/dtype | ExecutionSpec、公共能力门禁、各client入口；构建、真机、训练/恢复证据分别记录 |
| 测试或测量 | 有限范围、固定源码、实际配置/材料身份、终态/失败和结果路径；[evidence](evidence/README.md)、[逐配置索引](evidence/core5-index.json)、[经验](performance-lessons.md)；JUnit提取实际检查，历史汇总不伪造拆分 |
| 移动文件/入口 | 本页、architecture、semantic-map、README相关链接；STATUS当前动作/ROADMAP待办，不新增第二份交接 |

新增代表用例同时断言生成后的层间边、实际记忆族、教材读/位置/评分设置及运行中事件；
不要仅凭`fc`、`mixed`或`textbook`名称计覆盖。修改配置摘要时运行`test_core5_coverage`，
其独立窗口/分类检查和生成表核对可防止把第一节点配置写成全图配置。

定向验证入口是`test_core5_planning`、`test_weighted_inventory`、`test_weighted_workload_plan`。
拓扑与教材手算1408节点/22区域/3584边和独立公共生成器对照；五族及混合小图检查真实
参数/梯度和int64元数据。全图独立调度、optimizer、磁盘换owner与设备资格由相应门禁负责，
这些静态/小图检查不替代性能规模或真机结论。

## 有限测试表与结果查询

[core5-study-v1.json](../benchmarks/core5-study-v1.json)是可选用例目录，不是自动全排列任务。
[CUDA小中规模目录](../benchmarks/core5-cuda-finite-v1.json)显式列出宽8/64的FC/HB/LH启发KV、三客户端及CPU/mixed-c/resident/scalar设置，逐项选择，未运行项不获资格。
最大配置约345万参数；先查生成的memory_floor，另保留默认resident的5GiB总预算、2GiB材料化上限和4GiB文件上限。静态下界不包含完整训练峰值，实际状态/journal/反向/workspace和checkpoint分别记录。
默认必须选一个case和一个execution；任何合法公共配置仍可经`resolve_model`接入，不限于此表。

| 组 | 目的 | 有限覆盖 |
| --- | --- | --- |
| `check-fc/hb/lh` | 入口、配置和训练恢复设施 | 宽8、五记忆/混合，小图B2/P2；不作规模性能结论 |
| `fc-medium-few/many`、`hb-medium` | 调度与矩阵成本、实际在线合批 | 少节点大模块/多节点小模块，显式scalar/packed、CPU/mixed/resident |
| `fc/hb/lh-1b` | 主架构自然规模 | 约1B、五记忆可选；逐个选择有价值的组合运行 |
| `hb-1b-rich/sparse32/zero/dense` | 关键局部组合和发送稀疏度 | qry/att/位置、异构、1/32、分数零温、稠密Attention |
| `fc-1b-prefill` | 在线时间批 | B8/P32，保持原输入并记录容量/时限拒绝 |
| `hb-8b/16b` | 较大参数规模 | KV/混合；先静态预算及同配置小检查，再决定有限实测 |

执行目录保留PyTorch、Python→native、standalone LibTorch，以及CPU、mixed-a/b/c、resident；
FP16/FP32、CPU或单卡CUDA eager FP64、线程、packing和peer各自显式选择；
CUDA FP64 payload/scoring同时为64位，设备资格独立验证。resident使用native客户端与公共设备库，
生效算子记为`aten+device-control`，不称为纯PyTorch设备调度。NPU FP64 scoring只能显式放在CPU。

```sh
python -S scripts/run_core5_study.py --list
python scripts/run_core5_study.py --case check-fc-kv --execution python-cpu --output-dir NEW_RUN
python scripts/run_core5_study.py --case hb-1b-kv --execution cpp-npu-mixed-c \
  --standalone-binary BUILD/client/weighted-benchmark --fixture MATCHED_FIXTURE \
  --max-materialize-bytes EXPLICIT_BYTES --output-dir NEW_RUN
python scripts/index_core5_study.py RUN_PARENT --format tsv
python scripts/index_core5_study.py RUN_PARENT --client libtorch --state failed --format json
python -S scripts/index_core5_study.py REVIEWED_EVIDENCE.json --state passed --format tsv
```

长运行仍由本机独立服务启动；runner只负责单次有限子进程、无重试和原子记录。
`run.json`连接模型/工作负载/执行/测量身份、精确源码、命令、阶段退出码、日志hash；
`workload.json`和`inventory.json`给出所有展开配置/尺寸/共享角色。
`measurement/result.json`保留实际配置、完整计时、candidate/Full/消息和原始批统计、内存、
严格核对与精确恢复。失败也能查询到已产生的时间，但`throughput_qualified=false`。
同一查询入口接受仓库内`tide-core5-finite-performance-evidence-v1`审阅记录，
按其中的模型表恢复参数/规模，按状态与完整核对标志判定资格，不要求旧日志或运行目录存在。
审阅记录的hash用于定位证据；查询不会重新认证历史构建/运行。失败计时不能覆盖失败终态。
standalone须提供本源码的构建记录与实际binary hash；CPU/Python-owned NPU/SDK分开验证。
可以先用另一环境导出纯静态fixture，再让SDK进程消费；导出不执行参考轨迹。

设施验证：`test_core5_study.py`覆盖全部目录的静态构造、显式命令、拒绝和证据门禁；
`check_weighted_benchmark.py`在真实standalone进程中核对KV/SSD训练和恢复。
CPU默认六个有区别的dtype/schedule/packing/worker组合；显式NPU是八个小组合。
这些检查与区域分配手算/独立根测试、性能测量的职责不同，不互相替代。
`test_core5_recipe_training.py`另验五族/混合及关键评分/分配小组合的完整CPU FP64多步训练、
独立调度与磁盘新owner；不是全排列，也不解除原较大FP32边界。
当前测试区分纯PyTorch/native，FC至少两层，FC mixed使用八层以实际包含rank4/8的GDN/SSD；
FC/HB各有textbook KV与异构组合；[f456160固定资格](evidence/core5-coverage-correction-20261030.md)73项通过，包含26模型×2客户端的52项训练。仅认证该小型CPU FP64范围。

[GPU规模推理目录](../benchmarks/core5-cuda-inference-v1.json)显式定义不齐输入、B1/8/32、FC约1B及有条件近界点，另含HB/LH和少量qry/循环代表。仅逐项选择；配置存在不授予资格，近界须依据实际资源再准入，结果从STATUS/证据索引进入。

目录的[实际GPU推理证据](evidence/core5-cuda-inference-20261031.md)分列三客户端、CPU/mixed/resident、FP32严格失败、显式FP64规模与资源边界；从审阅JSON直接查询，不依赖旧运行目录。
