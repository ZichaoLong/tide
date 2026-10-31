# 实验运行、独立验收与图层级

固定数学和浮点阈值仍由 [semantics](semantics.md) 定义。公共实验可以不在每一步运行
CPU 参考；执行完成、独立数值、离散语义、精确续接和 checkpoint 是不同结果字段。
未检查的字段必须写 `not_checked`，不能从任务退出码、同栈恢复或有限耗时推断通过。
训练额外检查各独立根、共享参数、梯度、optimizer 槽和恢复后更新。

## 严格比较与探索观察

[comparison.py](../python/tidegraph/comparison.py) 提供：

```python
from tidegraph.comparison import finite, collect_comparison

finite(candidate_records)  # 无参考的运行也须检查非有限值
report = collect_comparison(candidate_records, reference_records,
                            check_device=False, max_differences=32)
```

`collect_comparison` 不运行或捕获求解器、通信、配置、容量或执行异常。
它先检查所有数学叶子的有限性，再遍历可比记录；仅诊断条数受上限约束，
不在第一个数值差异后停止离散检查。dataclass 的 `stats` 沿用既有忽略约定。

| 比较事实 | 类型 / 报告 | 实验解释 |
| --- | --- | --- |
| 有限浮点值超出原阈值 | `NumericMismatch`；`numeric=failed` | 严格失败；显式探索策略可保留有限观察 |
| 路由、端口、字典键、数量、整数/bool、None/零差异 | `DiscreteMismatch`；`discrete=failed`、`numeric=not_comparable` | 单列工作量差异；不称等价或探索验收通过 |
| dtype、shape、设备或记录类型错误 | `StructureMismatch` | 直接失败，不能由探索策略接纳 |
| 任一数学叶子 NaN/Inf | `NonFiniteMismatch` | 直接失败，即使叶子位于缺失路由或双方 Inf 相同 |

四类错误均继承 `AssertionError`，旧严格测试无需降低断言。
`tidegraph.compare.equivalent` 继续默认严格比较；FP64/FP32/FP16 tensor 阈值分别为
`(atol, rtol)=(1e-10,1e-8)/(1e-6,1e-5)/(1e-3,2e-2)`。
整数/bool 零容差，Python 标量保留原精确相等规则。不得使用
`except AssertionError: accept` 或错误文本匹配开启探索。

报告包含 `finite/numeric/discrete`、`strict_passed`、`compared_leaves`、
`mismatch_counts` 及有界 `differences`。只有显式启用探索且数值超差、离散通过、
完整执行有限时，启动器才能标 `exploratory_accepted`；它仍不是严格通过。
性能报告分列严格合格吞吐、探索成本、不同工作量成本和执行失败。
旧 [study records](../scripts/core5_study_records.py) 的严格完整记录门禁不因此放宽。

## 独立 LibTorch 入口

现有 `weighted-benchmark` 同时供公共启动器调度，不新增第二套 C++ 执行实现。
源码为 [tools/weighted_benchmark](../tools/weighted_benchmark/main.cpp)，默认仍严格验收。
新控制参数为：

```text
--validation strict|explore|none
--return-mode complete|outputs|metrics
--resume CHECKPOINT_DIR
--warmup N --steps N
--node-devices DEVICE,DEVICE,...
--communication ordinary|blocking|cpu_bridge
--loss-scale X --lr X --momentum X --weight-decay X
--check-roots [true|false]
```

`--threads` 是原 `--intraop-threads` 的别名；原 repeatable `--peer-device` 保留，
也可用 `--peer-devices` 的逗号列表。默认按 region 对有序设备列表取模；显式节点映射
逐节点覆盖，仍由公共参数共享/放置校验拒绝非法拆分。standalone resident 使用实际链接
的公共设备库，不接受 Python 的动态 `--resident-library` 参数冒充另一构建。
请求/生效节点映射、copy policy、线程及 dtype 均入结果。

`strict` 与 `explore` 只用于 `complete` 返回。C++ 的
[collect_compare/collect_tensors](../cpp/bench/comparison.h) 先验证所有双方叶子有限，
再完整遍历数值和离散字段；结果和命名参数/梯度/SGD 槽都使用同一分类。
旧 `compare` 仍严格并保留原 dormant-extension anchor 的身份范围；公共 collector
还要求图 identity 相等。求解器、通信、结构和非有限异常均直接失败。
数值超差但全部离散通过的显式探索运行写 `state=exploratory`、
`exploratory_accepted=true`、`strict_passed=false`；不改严格阈值或旧失败。

`none` 仍执行有限值/设备事务/optimizer 守卫，但不运行独立 CPU 参考，写
`state=completed` 及 `not_checked`，不产生严格吞吐资格。
standalone 的 `throughput_qualified` 还要求显式完整精确续接检查通过；
仅独立前向通过而未检查磁盘恢复时，仍分别保留对应事实。
`outputs` 只搬运所需输出，`metrics` 推理不额外求诊断 loss，结果中 loss 为 null。
两者保留设备上的 continuation，要求 `validation=none` 和 `checkpoint-check=false`。
完整独立验收与精确下一窗检查由公共 `check` 的完整返回流程承担；最小返回本身不能
继承这些通过字段。resident 最小返回使用现有 `advance_device/outputs/metrics`，
不在窗口内新增主机推进；未解码的 per-sample cache 分布明确记为 `not_checked`。

两种计时口径分别是 `complete-diagnostic-v2` 与 `selected-return-v1`。
新的完整口径含实际结果有限检查，不能与旧完整返回样本无说明混算。
`warmup` 默认 1、可以为 0；`steps` 为正整数，分别保存 `warmup_metrics` 和 `metrics`，
记录绝对窗口索引并检查 int64 时间域。执行超时、资源隔离由持久启动器负责。
v2 静态 fixture 不再继承旧 v1 的 B≤512/P≤128 目录上限；实际资源预算及合法时间域仍须通过。

每个完成窗口在计时外保存 `CHECKPOINT_DIR/window-N/{manifest.json,state.pt}`，
完成 fsync 后用 `latest.json` 原子发布；随后只删除当前 writer 的上一完整 generation。
pointer schema 为 `tide-weighted-checkpoint-pointer-v1`，manifest 为
`tide-weighted-run-checkpoint-v1`，固定 fixture hash、dtype、模式、task、codec、span
及 `next_window`。原始 native/resident checkpoint 格式不变；读入后再次核对实际 cut。
强停可恢复最后发布的完整窗口，半写入文件不替代旧边界。
每窗 `checkpoint_seconds` 与累计时间分列；预算同时包含新旧两份 checkpoint、
主存和恢复事务峰值。外部恢复的独立参考从原 fixture 重放历史前缀，
不把候选保存的数值状态当作独立参考初态，前缀时间不混入测量。

copy 计数来自公共显式 tensor 复制 API，区分 logical bytes、端点 tensor bytes 和同步；
它们不是 PCIe 测量，不证明普通复制使用直接 P2P。resident 内部协议/解码传输另行观测。
计算与 checkpoint 峰值按阶段采样，主存 RSS 保留进程生命周期峰值语义。

`--check-roots` 另开两个全新 eager owner，在计时外比较所选设备/放置/packing
与独立 CPU 标量、相反 streaming/prefill 调度的 VJP；不执行 optimizer 更新。
入口在模型分配前检查真实去重 trainable 参数不超过 5,000,000，适用于有界小材料。
先比较完整首窗，再分别独立运行前缀、克隆其浮点 state/slots/history/物理 pending
作为独立续接叶子，比较后缀完整记录和 VJP。共享参数按 canonical owner，外部输入按
真实 TensorImpl alias 取根，pending 名含全部物理身份；不按 shape 复用旧 root。
每段分别使用 all/output/control/descriptor 的逐字段均方和，施加 `.7` 与 `0` cotangent，
保留 None/连通零差异；FP16 诊断目标的平方累加显式使用 FP32。
输出 `independent_roots`（`tide-weighted-independent-roots-v1`）及独立诊断耗时；
其比较并入严格/探索结果，非有限或异常不能由探索接纳。
该入口要求 native、完整返回与 strict/explore；resident 的设备反向资格仍走其独立门禁。

CMake 提供 `weighted-comparison-check`、`weighted-run-check`、`weighted-roots-check`：
分别检查分类全遍历/严格整数/非有限拒绝、CLI 和 checkpoint generation/身份/续接元数据，
以及有解析答案的共享贡献/独立根/连通零/未连接 VJP。持久测试使用最小 owner，根测试
使用字面张量；它们不能代替真实图恢复或 `--check-roots` 的实际图资格。构建后执行
`ctest --test-dir BUILD/client --output-on-failure`；实际客户端推理/训练/恢复仍须独立记录。

## 同模型的图层级视图

[planning.hierarchy](../python/tidegraph/planning/hierarchy.py) 的
`execution_view(plan, family="auto")` 接受 `settle/timed-dag/pdg`。
它只校验并记录执行视图：同一模型、输入时钟、参数所有者、物理边及 workload 身份
全部不变。不同视图具有不同 `view_id`；实际客户端和调度器须另行记录。
视图合法性或同一二进制不能替代真实执行证据。

`workload(..., input_origins=[...])` 显式定义各物理输入端口的时间原点：
`time = input_origins[port] + stride * global_position`。
省略字段时保留旧默认和身份：ranked graph 用所属区域 rank，其他 graph 用 0。
携带显式原点的计划使用 `tide-weighted-workload-v3`，避免旧 v2 读取器忽略该字段；
无此字段的现有 v1/v2 继续可读，且默认生成的 v2 身份不变。
原点必须是 `[0,stride)` 内的 int64。Settle 视图另要求原点等于端口 rank、
边 delay 等于 rank 差，且 stride 大于所有 rank；独立同秩区域仍合法。
运行器还须对续接状态/pending 执行现有 Settle 时间码校验。

`non_aligned(graph, edge_index=..., extra_delay=...)` 显式改变一条边延迟，
`with_feedback(graph, source=..., target=..., delay=...)` 增加闭环物理边。
两者返回新拓扑，调用 `resolve_model` 产生新 `model_id`；不得称作同实例层级嵌入。
反馈构造器要求已有 target 到 source 的路径，不能用一个仍无环的新边假称 PDG 反馈。

`general_ports_plan(width=8, samples=2, positions=3)` 是可分配的 core-5 材料：
三个独立区域 rank `(1,1,3)`；两条 `0→2` 物理重复边和一条 `1→2` 边；
输入 `(0,1,2,2)`、输出 `(0,2,2)`，包含内部及重复端口。
`staggered-v1` 保留真实样例活动与缓存长度差异，不用重复样例冒充不定长输入。
测试将公共 KV 模块绑定到此材料，比较独立 region-major、标量参考和在线 greedy，
包括非整位置 cut、pending、完整记录和独立根 VJP；并未修改 Full 的固定一阶数学。

## 有限选点材料

下表为无 Torch、无模型分配的精确去重 inventory；它是选点依据，不是设备资格。
均采用自然 KV 基线、FFN=4d、默认共享绑定；发送预算和实际活动须分别记录。
最大规模先按站点逐卡准入，不能把参数字节数视为运行峰值。

| 材料 | recipe 参数 | 节点/区域/边 | trainable 参数 | P4 的逻辑 span |
| --- | --- | --- | --- | --- |
| FC 小型 | d64、region_width2、depth2 | 4/2/4 | 339,430 | 16 |
| FC 中型 | d256、region_width4、depth4 | 16/4/48 | 16,197,044 | 24 |
| HB 小型 | d64、region_width2、depth1 | 8/4/16 | 678,956 | 20 |
| LH 小型 | d64、region_width2、depth1 | 8/4/18 | 679,020 | 20 |
| HB 中型 | d128、region_width4、depth2 | 40/10/96 | 10,920,018 | 28 |
| LH 中型 | d128、region_width4、depth2 | 40/10/222 | 10,920,978 | 28 |
| FC 少节点约 1B | d4096、region_width2、depth2 | 4/2/4 | 1,010,339,302 | 16 |
| FC 多节点约 1B | d1024、region_width8、depth8 | 64/8/448 | 969,962,344 | 40 |
| HB 约 1B | d640、region_width8、depth3 | 176/22/448 | 1,051,330,022 | 36 |
| LH 约 1B | d640、region_width8、depth3 | 176/22/1058 | 1,051,335,206 | 36 |

小型三拓扑用于三平台公共客户端与层级覆盖；中型固定模型的 B1/8/32 曲线
与固定 B 的参数曲线分开。一般 Settle 端口、真实反馈和非对齐延迟各有独立材料。
PDG 的 Python 仅作独立参考，性能主候选为 LibTorch；TimedDAG/Settle 两者均可比较。
训练先选有预算的小中材料，约 1B 和接近容量点不自动继承训练资格。
少节点宽模块与多节点窄模块对照须记录实际 Full、候选、桶、时间批和消息工作量，
不能由相近参数量推断相同工作或 GPU 利用率。

复现本页共用边界的定向检查（匹配的项目 Python 环境，无 native 库）：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_comparison_report.py tests/test_compare_discrete.py \
  tests/test_core5_hierarchy.py tests/test_core5_planning.py
```
