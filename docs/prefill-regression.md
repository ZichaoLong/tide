# prefill代表回归与历史规模导航

本页规定性能改动的检查入口；结果必须引用固定源码证据，目录条目本身不是通过记录。
默认策略见[KV查询](weighted-dense-query.md)，当前活动和限制分别只见STATUS/ROADMAP。
[7fe97f2/5f563f6首轮固定结果](evidence/prefill-default-20261110.md)包含小图语义/恢复和原FC16成本；
下列HB/LH及1B–16B目录并未全部运行，历史结果不自动成为新默认的资格。
[分量窗口固定资格](evidence/component-window-20261110.md)补五记忆、变长B×L及三图族训练/恢复；
其NPU严格超差与通过范围分列，不能由此推断以下全部规模已验。

| 层级/触发 | 必查代表 | 入口与判定 |
| --- | --- | --- |
| 每次查询/packing改动 | 独立字段、异形/过期/空缓存、int64、None/零、参数别名、版本、一阶 | test_weighted_query_layout、test_weighted_dense_query；严格公式/根，旧dense数值xfail保留 |
| 每次调度/默认策略改动 | FC/HB/反馈PDG三步训练/切窗/恢复；一般端口、同秩、重复边、历史依赖、core-3程序 | 上述文件及test_weighted_dense_training、test_settle_ports、test_region_contract、test_region_schedules、test_greedy；不能仅输出对照 |
| 影响目标prefill成本 | 原FC16/B8/L128 CPU和NPU mixed-c完整一步 | benchmark_dense_query.py；同初始参数/输入、候选/Full/消息、forward/backward/optimizer/完整wall；数值与耗时分列 |
| 改变矩阵分桶/阈值或规模里程碑 | 少节点大矩阵、多节点小矩阵、HB扩散收拢、LH跳边 | 下列有限study目录的medium行；先选受影响项，不全排列 |
| 扩大规模资格时 | core-5 FC/HB/LH约1B、HB8B/16B；历史core-3宽/窄与LH约9B | 先预算再单项运行；历史通过/失败保持原提交，不把目录或CPU代替设备/吞吐资格 |

## 可运行目录

同时增大独立batch与sequence的代表配置使用已有安装CLI；生成器只产生配置，不另设执行器：

```sh
python scripts/prefill_regression_config.py --case many-hb --family kv \
  --samples 4 --positions 16 --device npu:0 --output NEW_CONFIG.json
tidegraph plan NEW_CONFIG.json --site SITE.json
tidegraph run NEW_CONFIG.json --site SITE.json --output-dir NEW_RUN
```

`wide-fc`为4节点/d512，`many-hb`为HB扩散收拢/区域宽8/d32，
`many-lh`为同类跳边/d64/qry-sum异构，`sum-fc`为16节点/d64，
`layer-chain`为4层/区域宽1/d128/固定总发送1；后者仍是core-5层链，
不自动等同特定Transformer/Mamba架构。`lh-1b/lh-9b`按真实参数清单自然定宽。
每项可选择五记忆或mixed、B/L、客户端、设备/placement、线程、推理/训练及auto/greedy。
默认返回outputs、独立数值not_checked；`--complete`才启用完整返回及explore比较。
不同返回契约不直接计算加速比；先运行相关独立小门禁，真实大规模数值/恢复仍另列。
单配置一次预热/一次计时，完整训练包括准备、packing、backward、finite及SGD/同步；
checkpoint成本按既有CLI单列。配置身份、精确恢复和资源拒绝沿用公共设施。

复用[现有study驱动](core5-study.md)，不增加第二套执行器：

```sh
python -S scripts/run_core5_study.py --study benchmarks/prefill-regression-v1.json --list
python scripts/run_core5_study.py --study benchmarks/prefill-regression-v1.json \
  --case hb-medium-kv --execution native-cpu --native-library MATCHING_BUILD --output-dir NEW_RUN
```

执行名称以`--list`输出为准。目录保留PyTorch/native/standalone、CPU/mixed/resident、dtype/线程维度，
默认auto，生效路径及构建/真机资格分开。需要显式legacy对照时复制到新的配置身份，不能改旧运行记录。
NPU需站点队列；长检查/测量需冻结源码、background.slice及有限内存/时限。构建和计时不重叠。

原下游模型/head/loss使用独立固定探针，不能用目录中core-only loss冒充原任务：

```sh
python scripts/benchmark_dense_query.py --downstream-repo READ_ONLY_DOWNSTREAM \
  --native-library MATCHING_BUILD --query-execution auto --warmup-state reset \
  --device cpu --batch 8 --length 128 --output-dir NEW_COST
```

下游固定0439efd2af31cf10b07ce9e6423b0195f7449968。NPU将device改为npu:0，放在队列内运行。
每配置一个独立进程，一预热一步测量；一次小B2/L8 profile只解释算子形态，不作为长prefill耗时。
`benchmark_dense_query --trace-records complete`保留历史native成本的完整事件返回；
`minimal`通过公共工厂关闭内部trace，是另一种返回契约，可进入KV整序列VJP。
固定下游的旧native适配器忽略其`run_features(trace=False)`请求，原记录实际仍完整；
不能将新minimal耗时直接标成原完整返回的等价加速。当前公共CLI的任务扩展仍可读取trace，
只选择outputs投影不自动删去内部trace；须按实际`joint_state_vjp_events`确认所测路径。
原始参数hash与测量前hash相等才是reset流程；设备间/路径间实际Full数相等也不自动证明完整等价。
出现非有限、非法配置、容量/通信/精确恢复错误停止；有限数值差异保留严格失败，不能写成等价加速。
不设固定加速比门槛；回归比较检查工作量/计时范围/争用，超过预算则明确终态，不自动扩时重跑。

## 历史代表到底是什么

| 记录 | 配置与原范围 | 独立入口 |
| --- | --- | --- |
| 用户确认的原LH约9B口述 | 旧Intel56核、B512/d2048、名义激活1/32；nograd3.3ms/sample-token、grad列3.6ms/token，阶段含义未细分 | [LH规模背景](lh-scale-benchmark.md)；不是本仓库实测 |
| 本机原LH/PDG实测 | 9,468,020,899参数、aarch64 56核；LH/PDG nograd5.824/5.598ms/token，grad-forward7.128/108.233 | [Add规模对照](evidence/add-scale-comparison.md)；grad-forward不是完整训练 |
| core-3 LH衍生宽图 | 480节点/2208边、d2048、B512/T12，Add9.468B、Attention17.521B | [B512 bound](evidence/formal-b512-bound-add-prefill-20261004.md)、[CPU训练/常驻推理](evidence/formal-b512-cpu-training-resident-inference-20261004.md) |
| core-3其他宽/多节点图 | DAG/Settle Add9.397B/N320；Attention17.722B/N384；窄d128/N46912约8.497B，只有四个区域活跃 | [跨家族对照](evidence/cross-family-performance.md)；Settle初次超时及后续延长时限完成都保留 |
| core-5自然约1B | FC64/d1024约.970B；HB/LH176/d640约1.051B，B1/P1 | [自然规模](evidence/core5-natural-study-20261029.md)；LH FP64通过，FC/HB训练尾数值失败分列 |
| core-5宽批平台 | FC4/d4096约1.010B、B128/B512/P2；FC16/d2048、HB/LH20/d1792 | [三平台规模](evidence/core5-platform-scale-20261031.md)；有限成本/严格失败/恢复限制不能合并为通过 |

core-5 CUDA4.561B/9.123B及NPU12.802B是容量点，HB8.118B/16.528B有静态计划，
均不能替换成“LH9B新吞吐已通过”。原LH参考仓库只读；历史源码用resolve_history_commit.py定位。
实际新经验统一维护[performance-lessons](performance-lessons.md)，不在本页积累第二份失败清单。
