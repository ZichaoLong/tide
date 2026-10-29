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
