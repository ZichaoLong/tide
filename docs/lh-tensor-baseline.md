# LH/HB 的独立张量对照

[固定三层结果](evidence/lh-three-layer-20261112.md)记录首轮176节点CPU/NPU三模式，
其数值/工作量/恢复与后继优化范围分别声明。

数学固定 `0770593838a37c790a568226930567ed60f99b83`。
[直接实现](../scripts/lh_tensor_baseline.py)按静态区域顺序执行，
[数值公式](../scripts/lh_tensor_math.py)与[原始状态](../scripts/lh_tensor_state.py)
不调用生产局部核、区域求解器、公共调度或参考轨迹。参数直接来自公共绑定，因此共享身份和
角色名称相同；共享参数的梯度汇总全部使用者。它是可测的独立候选，不是最优性能证明。

适用域是对齐 Settle 物理端口；支持同秩独立区域、跨区域和 skip 边、重复边、稀疏端口。
每区域一个 KV 配置，可在区域之间混合 qry/sum、窗口和位置；Read+、cat局部分数、ind/att、
content_stats key、fixed/context cat区域控制、regional_total/quad。对应固定教材的
[KV](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)及
[区域定义](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
`StaticKV` 构造会拒绝其他模块/配置，L≤256，FP32/FP64，单设备同dtype。
这是对照的有限定义域，不限制公共基座的支持范围。

区域间保持 `[node,B,L,…]` 张量。静态连接编译为每前驱区域一次 gather、源变换与 incidence
矩阵归约；重复边形成重复计数，不由 SrcClass 合并原子。同一实际输入且变换参数身份相同才
复用计算。身份包含完整有序参数角色与重复别名，不能使用去重后的参数集合代替；
`test_lh_source_slots`用相同前向值但不同独立梯度的低秩槽反例核验。
不同节点矩阵使用 bmm，相同矩阵使用共享 GEMM；参数stack的真实成本纳入前向。
更换参数owner、移动设备或dtype后须重新构造 StaticKV。

直接候选显式采用 **dense Full 后 mask**。candidate、selected Full、消息、物理 Full行数
分别报告；不能将实际支持比例当成物理计算缩减。分配独立枚举有限仿射区间，复杂度 O(N²)，
保存实际活动分支并实现一阶精确映射VJP。它不对迭代求导，也没有把求解换成硬Top-k。

返回域是逐端口 `(c,G)`、`cG` 数值投影、区域控制（显式records选项）和raw终态；
不认证完整逐事件独立根或None/连接零等价。生产公共计划仍承担不同的状态/观察职责，
其成本差不能全部命名为包装开销。Transformer与原LH属于不同函数，另列比较。
数值梯度对照将None与连接零分别标明后取零投影；它不认证带weight decay或已有动量时
不同梯度连接策略的optimizer轨迹等价。计时runner的单步SGD从相同空槽开始。

raw bank包含共同int64 `times[T]`，`valid[node,B,T]`，浮点 keys/values/summaries/weights，
以及int64 last_times/observations。补位不形成消息或记忆写入。过期仅按实际最后正写入发生；
静默窗口保留原始过期状态。`decode`仅在边界筛选候选实际结果，不递归重放参考。
连续窗口可保留反向图，`detached`显式声明训练截断。独立bank格式不是公共checkpoint v5，
其磁盘恢复资格与公共continuation资格分别记录。

## 重复三层成本比较

[runner](../scripts/benchmark_lh_tensor.py)一次进程选择一个层级和模式：

- `direct`：上述独立张量公式。
- `public`：公共 `SettleColumnPlan.run_ports`。
- `default`：默认 `eager_runner.run_ports`，记录实际分派。

public/default选择 `--client python|native`；native须明确匹配的 `--native-library`。
三模式为 `inference|grad-forward|training`。`--case many-lh`直接复用防退化目录的176节点
textbook配置，不把qry/sum与att/ind混配简称sum；`many-hb`、`lh-1b`、`lh-9b`沿用各自配方。
`--plan`接收固定展开workload，形状/初态材料和参数角色hash进入结果。

```sh
python scripts/benchmark_lh_tensor.py --level direct --case many-lh --mode training --device cpu --threads 1 --output-dir NEW_DIRECT
python scripts/benchmark_lh_tensor.py --level public --case many-lh --mode training --device cpu --threads 1 --output-dir NEW_PUBLIC
python scripts/benchmark_lh_tensor.py --level default --case many-lh --mode training --device cpu --threads 1 --output-dir NEW_DEFAULT
```

每个点一次恢复初态的预热和一次测量。训练完整计时包括输入准备/上传、packing、前向、逐物理
输出端口loss `1e-6*sum((cG)^2)`、有限检查、反向、SGD、结果观察/清理和同步；构造/参数hash/
预热前重置在外。SGD统一lr=1e-5/momentum=.3，是这个配对harness的更新配置。
吞吐分子是有实际物理输入的sample-position数量，另报候选/Full/消息及资源峰值。
主机提交阶段时间不是设备kernel时间；显式`--phase-timing`增加逐阶段同步，属于另一诊断身份，
不得把它混作无插桩完整步成本。长作业遵循仓库background.slice、固定源码与资源规范。

## 正确性与维护

[定向检查](../tests/test_lh_tensor_baseline.py)用独立标量调度检查实际输出、控制、活动、raw
字段/时钟/计数和共享梯度；手算分配VJP检查空/全上限/内部响应；跨窗测试覆盖超2^53时钟、
保留图和静默窗口；三步AdamW及磁盘新owner检查本投影的精确恢复。小型检查不代替176节点、
1B/9B的完整尺度数值资格，也不把运行成功等同于严格验证通过。

```sh
python -m pytest -q tests/test_lh_tensor_baseline.py --dtype both
```

尺度检查使用独立的[检查入口](../scripts/check_lh_tensor.py)，不会从参考轨迹生成候选输入。
[解码与比较](../scripts/lh_tensor_check.py)读取候选实际返回的控制、候选/Full支持、全部raw
终态字段及int64时钟、逐端口cG，再比较所选输出族对输入、共享参数和独立初态字段的数值VJP。
默认是输出+raw任务；`--root-family aggregate|descriptors|control|full|messages|raw`分别施加
不同符号、含零坐标的字段伴随，保留每个物理边/端口身份。`--left-entries 3`提供独立构造的
非空raw输入，含稀疏槽与int64时钟；它不是CPU预跑真实轨迹。`--start-position`可检验巨大时钟。
[材料及观察](../scripts/lh_tensor_observations.py)分别从直接公式和候选实际列提取所选字段，
不从参考补齐候选。完整逐事件快照根、optimizer和恢复不属于此命令的资格范围；
None/连接零差异继续单列，不把dense Full的数值投影认证为连接等价。
形状、离散/非有限失败与有限数值超差分开，原严格阈值不改。小型入口检查见
[test_lh_tensor_check](../tests/test_lh_tensor_check.py)。
[非空初态及输出族根](../tests/test_lh_tensor_observations.py)另核对巨大int64时钟、
共同初态编码的线性转置与七种独立任务边界。

```sh
python scripts/check_lh_tensor.py --case many-lh --device cpu --dtype float64 --client native --native-library BUILD --output-dir NEW_CHECK
python scripts/check_lh_tensor.py --case many-lh --device cpu --dtype float64 --client native --native-library BUILD --left-entries 3 --root-family messages --output-dir NEW_LEFT_ROOTS
```

默认B4/L16/d64、176节点，`--level both`检查public/default；NPU改为显式设备和FP32。
材料、workspace、节点行/Attention元素及比较捕获分别有预算；退出0为严格数值投影通过，
1为保留有限数值失败，2为结构/离散/非有限/执行/容量失败。该检查不测吞吐。

规模恢复与上述数学对照分开：将同一个公开展开plan交给
[planned_windows消费者](../examples/consumer_core5/planned_windows.py)，以默认Python/native
列路径执行三次AdamW、跨窗口raw续接及独立新进程恢复；完整比较None梯度、参数和槽，
不把直接公式的None-as-zero投影代入optimizer。其定向检查在
[test_planned_window_consumer](../tests/test_planned_window_consumer.py)，实际规模与设备须另记录。

修改公式同时维护适用域、独立检查和本契约；改变runner则维护计时/职责/参数身份。
固定运行证据从[证据索引](evidence/README.md)进入；当前完成范围与待办只维护在STATUS/ROADMAP。

## 重复算子与 autograd 诊断

`profile_lh_tensor.py --plan PLAN.json --level direct --client python --device cpu
--profile cpu --census --output-dir NEW_DIR` 使用与 `benchmark_lh_tensor.py` 相同的
参数/端口材料、目标和重置warmup。公共接口选择 `--level public`，默认入口选择
`--level default`；native须给 `--client native --native-library MATCHING_BUILD`。

CPU profile按实际shape记录算子数量/主机成本，`--census`记录返回根及已插桩私有图，
Python/native重叠计数不能相加。NPU选 `--profile npu --device npu:0 --trace-dir NEW_SCRATCH`，
由合法资源准入启动；厂商trace与审阅后的摘要分开保留。诊断包含forward/反向/finite检查，
不包含optimizer，不产生吞吐结论，也不代替独立数学或精确恢复。
