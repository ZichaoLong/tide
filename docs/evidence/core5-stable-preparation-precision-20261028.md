# 稳定共享准备、显式精度与完整训练观测 — 2026-10-28

实现固定`1f1ec5780e054e460e3471547519ace3f38fa3af`；数学固定
`0770593838a37c790a568226930567ed60f99b83`，图v13/checkpoint v5不变。
本报告区分通过范围和严格失败；[同名JSON](core5-stable-preparation-precision-20261028.json)
记录源码、输入、构建、终态及原始证据hash。CPU扩大门禁整体失败，不把其中572项通过写成全部通过。

## 实现与独立验证

[共享契约](../weighted-kv-sharing.md)对应FC §6.2：相同参数和相同输入许可共同计算，
各使用者保有私有状态和完整VJP。`prepare_shared_state`原先按含TensorImpl地址的键排序，
换owner可能改变autograd创建及共享梯度累加顺序；现在按首个实际事件求值，键只判等。
独立C++反例故意令键的字典序逆于事件顺序，避免依赖allocator偶然排列。
真正的checkpoint测试保存实际状态、参数和optimizer，换owner精确解码，并比较下一窗口
完整记录、参数梯度、更新参数和optimizer，均要求零容差。测试不借参考递归补写状态。

[性能入口](../weighted-performance.md)的CPU Python、Python/native及standalone支持显式
`--dtype float64`。原FP32种子/二进制字节不变；执行前提升并保留参数/输入别名，
请求精度、生效精度、原材料精度和实际存储预算分别记录。FP64仍用原atol1e-10/rtol1e-8；
FP32默认、数学函数及容差不变。此入口不默许FP64设备/resident替代。

旧benchmark的`optimizer_values_checked=true`实际只核对最终参数。本次eager入口分别导出并比较
最后参数梯度与SGD动量，保留None/连接零；Python还比较完整state_dict。
故障注入验证：即使参数不变，缺梯度或损坏动量仍必须失败。resident本入口未导出这些字段，
两个标志现在为false；其独立完整训练门禁仍有原证据，旧报告不追改成新验证范围。

## 固定资格与新增失败

| 范围 | 结果与边界 |
| --- | --- |
| CPU相关门禁 | 572 passed、2 failed；涵盖材料/预算/别名、五族训练、共享/packed/字段根、时钟、评分、来源与恢复。完整命令见JSON；不是重跑所有历史矩阵 |
| 新恢复反例 | 开发210项通过，补充C++确定性反例及双dtype/Python/native恢复6项通过；固定门禁再次包含这些检查 |
| D64 KV显式FP64 | Python与native各两步；每步完整记录、参数梯度、参数与optimizer通过；实际checkpoint解码及各自下一窗口精确恢复通过 |
| standalone NPU | SDK2.10/CANN9.0，物理13→npu:0；FC/KV、HB/KV各4次AdamW更新、完整VJP/恢复及设备→CPU handoff通过；core/消费者构建与CPU两dtype反例通过 |
| CUDA | Torch2.10/cu128的core/消费者构建、loader/help、CPU两dtype反例通过；GPU真机未验。本次未改resident核，未重跑其全矩阵 |

CPU失败为`test_recurrent_multistep_optimizer_and_restore[float32-{python,native}-gated_delta]`。
改动前1850434在相同栈和原断言下复现同一失败；不是本次共享排序改动造成。
HB/GDN默认AdamW的第三次更新，`nodes.5.kernel.conv.weight[0,2]`两侧为
0.027985455468297005与0.027986867353320122，差1.4118850231170654e-6。
三个窗口的完整前向和梯度均通过原容差，但不足以保证optimizer后的参数也通过。

第一步对应梯度为−2.002002474910114e-8与−2.003758936552913e-8，差约1.76e-11；
默认AdamW的epsilon为1e-8。独立80位Decimal从实际梯度和旧槽值计算更新，
第三步预测0.027985456848606385与0.02798686885488567，复现差异放大。
这说明参数比较不能由梯度比较替代；不宣称所有上游梯度微差的来源已定位。
测试未改容差/输入、未标xfail，原CPU资格、旧baseline和见证任务分别保留failed。
原FP16 GDN、SDK affine、默认异步双NPU、反馈PDG、D64/D2048 FP32见证继续保留。

## 有限测量

测量全部终态passed/exit0。每配置一次独立进程、一次warmup及一次连续测量窗，
项目重型构建与计时不重叠；外部服务器负载未隔离。完整训练含输入准备、调度、packing、
完整返回、loss、反向、finite checks、SGD、detach及同步；构造、warmup、最终梯度/槽导出、
事后独立调度核对分列。候选不消费参考轨迹。

| 配置 | 正式窗口 | 完整核对及实际工作量 |
| --- | --- | --- |
| D64，Python prefill，训练 | 5.702109 s | 通过，包括最后梯度/SGD动量；479候选/344 Full/1604消息 |
| D64，Python调用native prefill，训练 | 2.896188 s | 同上 |
| D64，standalone prefill，训练 | 2.957849 s | 同上 |
| D64，standalone streaming，训练 | 2.903450 s | 同上 |
| D2048，standalone prefill，推理 | 151.415824 s | 两个窗口完整记录通过；424候选/199 Full/946消息 |

两份材料均为480主体节点/2208边、B1、每窗1位置。D64为9076975主体参数，
对照core-3 Add主体9076928，差+47；D2048为9262006511，对照9262008512，差−2001。
使用真实私有FFN宽4/5或319/320匹配，未用无效参数凑数；无任务头。
D2048构造213.597352 s、warmup154.864902 s，包含事后独立核对的整个进程1085.531 s；
测量结束时采样的进程生命周期RSS峰值97006063616字节（约90.35 GiB），不是训练峰值。

此有限D64输入下，native及standalone单次比纯Python更快，streaming/prefill接近；
State最大批1，只有34/479次State进入共享私有投影批，最大私有批32。
不能由此认定所有图/尺度都由prefill获益，也不能从少量BMM推断低垂果实耗尽。
D2048现在有与core-3主体参数量匹配的有效**CPU FP64推理**成本，
仍没有同精度/同完整训练工作量的core-3对照、设备训练或跨卡吞吐结论。
原FP32失败不由这些结果解除，不能同旧FP32计时算等价加速。
测量期间主机1分钟load从约45.78升至141.0；项目自身重型任务串行，外部争用未排除。

## 未采用的算术诊断

1850434固定源码上的临时探针试过row方向、product/reduction、BMM形状和局部Full/VJP边界。
某个变体使479条前向事件逐值一致，完整参数梯度与第二次更新仍失败；没有采用这些动态补丁，
也没有把显式FP64通过称作FP32修复。共享计算下逐使用者cotangent不能直接与独立计算相比，
须先解释真实别名与归约范围。所有探针终态、driver/日志身份见JSON。

首个`core5-d64-arithmetic-trace`的完整JSON和五个张量曾被row探针误覆盖。
原stdout、最初10条事件与计数的partial恢复、`artifact-issue.json`仍保留，不能当完整资格。
此前正式资格与大型材料没有受影响。随后算术诊断和正式测量各用新输出目录。

在上述测量终态后，另做一次1f1ec57上的FP32 GDN动态探针：将已用于half的normalized
L2/scaled-RMS一阶收缩应用于FP32，前向公式、输入和容差不变。
该Python三步探针通过，最后关注参数差约7.00355e-7；它尚不是本提交的公共实现，
也不认证native、独立Jacobian或checkpoint。后续采用须另有实现提交和固定资格，不能回写本阶段2项失败。

## 从干净源码复现

按[开发入口](../development.md)选择匹配Torch并构建CPU core；长任务遵循[固定资格](../verification.md)。
不依赖旧artifacts路径。原严格GDN失败可直接运行上述两个完整pytest node ID，
使用`--dtype float32`和显式`TIDE_BUILD_DIR=CPU_CORE`；应保留失败，不能放宽断言。

```sh
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_weighted_workload_training.py \
  tests/test_weighted_kv_sharing.py tests/test_cli.py::test_cpp_custom_state_kernel --dtype both
python scripts/build_weighted_benchmark.py --core-build CPU_CORE --output-dir NEW_BUILD
TIDE_BUILD_DIR=CPU_CORE TIDE_WEIGHTED_BENCHMARK=NEW_BUILD/client/weighted-benchmark \
  python -m pytest -q tests/test_weighted_benchmark_material.py tests/test_weighted_benchmark_precision.py
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float64 \
  --implementation native --native-library CPU_CORE --steps 2 --checkpoint-check --output-dir NEW_CHECK
```

生成器输出`workload.json`与`inventory.json`。测量使用`benchmark_weighted.py --workload PLAN --device cpu
--dtype float64 --implementation python|native --mode training --steps 1 --output-dir NEW`，native另给
`--native-library CPU_CORE`。standalone先用`export_weighted_benchmark.py --workload PLAN --binary
--native-library CPU_CORE --max-materialize-bytes BYTES --output FIXTURE`导出静态材料，
再运行`NEW_BUILD/client/weighted-benchmark --fixture FIXTURE --device cpu --dtype float64
--implementation native --schedule prefill|streaming --mode training|inference --steps 1 --output-dir NEW`。

D2048用相同生成参数但`--width 2048`，导出静态上限显式给137438953472字节。
本次复用旧FP32二进制，只在新manifest提高准入上限；原JSON/二进制、派生manifest及
实际生效dtype的身份分别记录。相同主体参数量不表示相同函数、任务头、激活量或core-3完整训练成本。

每个测量进程单线程；D64每例32GiB/360秒，D2048推理192GiB/1800秒。
background.slice/Nice10提供任务生命周期分离，不证明独占CPU/内存。
CUDA真机、x86_64、大型设备训练及跨卡吞吐待验；共享组外私有合批、resident组合矩阵和窗口反向复用
仍有优化空间，见[历史及本轮经验](../performance-lessons.md)。
