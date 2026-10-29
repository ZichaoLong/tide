# core-5 候选、配置与测试入口

从本页可以独立找到模型配置、公共实现、运行入口、检查和证据。数学固定
[0770593](upstream.json)；[当前进度](STATUS.md)和[待办](ROADMAP.md)分别维护。

## 配置的三层身份

| 对象 | 公共入口 | 身份包含什么 |
| --- | --- | --- |
| 数学模型 | [resolve_model](../python/tidegraph/planning/model.py) | 物理图/端口、全部展开的局部配置、SrcClass、共享绑定、聚合尺度；独立角色清单 |
| 工作负载 | 同文件的 `workload` | 模型身份、输入策略、样本/原始时间窗、初值种子；不含参考轨迹 |
| 执行 | [ExecutionSpec](../python/tidegraph/planning/execution.py) | 三种client、设备/peer、放置、dtype、scoring dtype、streaming/prefill、算子/packing、线程；请求和生效设置分列 |
| 测量 | 同文件的 `MeasurementSpec` | 工作负载/执行身份、推理或完整训练、warmup/步数、恢复检查、超时 |

模型身份不因设备或样本窗改变，工作负载身份会因输入改变；实际测量还须固定源码、材料与环境。
身份是追溯键，**不是资格认证**。实现版本、构建、真机检查、数值失败与性能结果单独记录。
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
| 测试或测量 | 有限范围、固定源码、实际配置/材料身份、终态/失败和结果路径；[evidence](evidence/README.md)、[经验](performance-lessons.md) |
| 移动文件/入口 | 本页、architecture、semantic-map、README相关链接；STATUS当前动作/ROADMAP待办，不新增第二份交接 |

定向验证入口是`test_core5_planning`、`test_weighted_inventory`、`test_weighted_workload_plan`。
拓扑与教材手算1408节点/22区域/3584边和独立公共生成器对照；五族及混合小图检查真实
参数/梯度和int64元数据。全图独立调度、optimizer、磁盘换owner与设备资格由相应门禁负责，
这些静态/小图检查不替代性能规模或真机结论。

## 有限测试表与结果查询

[core5-study-v1.json](../benchmarks/core5-study-v1.json)是可选用例目录，不是自动全排列任务。
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
FP16/FP32、CPU FP64、线程、packing和peer各自显式选择。resident使用native客户端与公共设备库，
生效算子记为`aten+device-control`，不称为纯PyTorch设备调度。NPU FP64 scoring只能显式放在CPU。

```sh
python -S scripts/run_core5_study.py --list
python scripts/run_core5_study.py --case check-fc-kv --execution python-cpu --output-dir NEW_RUN
python scripts/run_core5_study.py --case hb-1b-kv --execution cpp-npu-mixed-c \
  --standalone-binary BUILD/client/weighted-benchmark --fixture MATCHED_FIXTURE \
  --max-materialize-bytes EXPLICIT_BYTES --output-dir NEW_RUN
python scripts/index_core5_study.py RUN_PARENT --format tsv
python scripts/index_core5_study.py RUN_PARENT --client libtorch --state failed --format json
```

长运行仍由本机独立服务启动；runner只负责单次有限子进程、无重试和原子记录。
`run.json`连接模型/工作负载/执行/测量身份、精确源码、命令、阶段退出码、日志hash；
`workload.json`和`inventory.json`给出所有展开配置/尺寸/共享角色。
`measurement/result.json`保留实际配置、完整计时、candidate/Full/消息和原始批统计、内存、
严格核对与精确恢复。失败也能查询到已产生的时间，但`throughput_qualified=false`。
standalone须提供本源码的构建记录与实际binary hash；CPU/Python-owned NPU/SDK分开验证。
可以先用另一环境导出纯静态fixture，再让SDK进程消费；导出不执行参考轨迹。

设施验证：`test_core5_study.py`覆盖全部目录的静态构造、显式命令、拒绝和证据门禁；
`check_weighted_benchmark.py`在真实standalone进程中核对KV/SSD训练和恢复。
CPU默认六个有区别的dtype/schedule/packing/worker组合；显式NPU是八个小组合。
这些检查与区域分配手算/独立根测试、性能测量的职责不同，不互相替代。
`test_core5_recipe_training.py`另验五族/混合及关键评分/分配小组合的完整CPU FP64多步训练、
独立调度与磁盘新owner；不是全排列，也不解除原较大FP32边界。
