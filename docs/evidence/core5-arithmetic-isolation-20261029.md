# D64算术边界与跨步误差隔离

固定实现`d8f71327f2c1515d78c2ccf3e355fc427673708f`交付诊断设施，**没有改变生产算术或默认精度**。
探索基线为`e75e1f5bca347b45301990689d8d7d088dd6e16b`，数学仍固定
`0770593838a37c790a568226930567ed60f99b83`。可审查数据、hash、终态及完整关键见证见
[同名JSON](core5-arithmetic-isolation-20261029.json)。固定源码设施检查**19 passed**；
原D64最佳局部精度方案仍**strict_mismatch/exit 2**，两条调度各自精确恢复通过。

## 问题与材料

区分前向局部算术、反向/共享贡献汇总及optimizer的误差，不修改输入、容差、SGD或离散规则。
原计划SHA256为`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`：
480节点、2208边、D64、B1、每窗1个位置、9,076,975个主体参数；两个连续窗口，
SGD lr=.001、momentum=.3、weight_decay=.01。参考与候选独立构造、独立调度；无数值预跑供应候选。
FP32原容差atol=1e-6/rtol=1e-5不变。每步比较完整返回/离散结构/梯度/参数/动量槽。

CPU为本机aarch64、Torch2.12.0+cpu、单线程ATen/BLAS。探索采用临时Python dispatch探针，
固定资格采用入仓库的[公开诊断入口](../weighted-arithmetic-diagnostics.md)。这里“公开”指可复现工具，
不表示增加了生产执行配置。选中算子内部FP64、结果返回FP32；norm内部及其一阶收缩FP64。
optimizer始终在探针之外。参数/状态/消息仍FP32，时间/计数仍int64。

## 有限探索的结果

每项各一次独立进程。下表列**超原容差的梯度张量数**；并非全部精确不等张量数。

| 干预 | 首步完整返回 | 首步/第二步坏梯度张量数 | 首步/第二步坏参数张量数 | 两步完整训练 |
| --- | --- | --- | --- | --- |
| 原FP32 | 失败 | 2393 / 2566 | 3 / 6 | 失败 |
| 矩阵乘加内部FP64 | 通过 | 2 / 48 | 0 / 0 | 失败 |
| 归约内部FP64 | 失败 | 2371 / 2667 | 4 / 4 | 失败 |
| 矩阵乘加+归约 | 通过 | 2 / 22 | 0 / 0 | 失败 |
| RMS内部FP64 | 失败 | 2088 / 2387 | 0 / 1 | 失败 |
| 矩阵乘加+归约+RMS | 通过 | 0 / 2 | 0 / 0 | 失败 |
| 仅反向矩阵乘加+归约 | 失败 | 2392 / 2590 | 3 / 6 | 失败 |
| 三组加非线性内部FP64 | 通过 | 0 / 16 | 0 / 0 | 失败 |
| 三组加完整Read/区域控制FP64 | 通过 | 0 / 17 | 0 / 0 | 失败 |

九项离散结构均通过，第二步完整返回及动量槽均失败。仅最后三种前向完整干预
（三组、三组加非线性、三组加完整Read）首步梯度/参数/动量全部通过。
矩阵乘加族是主要误差来源的强线索；它覆盖多个模块和反向调用，**没有定位到某一个唯一算子**。
仅提升反向基本无效，不能恢复已经产生的前向差异。扩大精度范围也不保证最终差异单调下降。
这些探索均保留原failed/2，不作为已通过的生产候选或性能数据。

## 固定源码复现与精确续接

在d8f7132用仓库生成器重新生成相同hash的计划，复现“三组内部FP64”并新增实际checkpoint检查。

- 首步479条事件按原比较器逐值一致；6381个参与梯度的张量中，4186个精确不等但均在容差内。
  更新后12170个参数张量中1383个精确不等，同样均在容差内。完整记录、梯度、参数和动量槽通过原断言。
- 第二步最早精确事件差异在(time=18,batch=0,node=0)的content，最大1.4901161e-8。
  完整返回首个超容差见证是node294的keys，最大1.3411045e-6。
  两个坏梯度为node457和node171的score.vector，最大差分别2.7045608e-6和1.4267862e-6；
  参数仍通过容差，node457对应动量槽失败。这里的“首个”是指定遍历顺序下的观察，不能当作唯一根因。
- 参考与候选每步的离散支持/控制结构一致。首步实际479候选、331 Full、1541消息；
  第二步479候选、344 Full、1604消息。State最大实际batch仍1；不把prefill接口等同大批运算。
- 首步更新后，两条调度分别保存实际状态、参数、SGD槽；换owner精确解码。
  第二步各自的完整返回、梯度、参数和optimizer与未中断执行全部逐值一致。
  没有跨调度复制参考状态；精确恢复通过没有解除两种调度之间的浮点不等价。

首步事件相同而反向/更新有小差异，支持继续研究反向求值及汇总边界；
这还不能证明仅提高共享叶最终累加精度就足够。部分贡献在送达叶之前已经舍入，
共享计算与逐次计算也可能形成不同的反向收缩顺序。普通SGD更新的小差异会进入下一窗；
这与此前GDN近零梯度被默认AdamW显著放大的具体机制不同。

## 验证范围与作业终态

新工具的独立相消手算/literal Jacobian、int64、None/零、版本、适用域拒绝、异常清理、
小图两步checkpoint，以及既有Python/native两dtype的训练/恢复与精度设施共**19 passed、0 skipped**。
开发冻结dirty快照也19通过；实现先提交，再完成上述固定资格。
C++代码未变，复用前后核对当前C++源码hash`8c942fa686ed276204b3bd5a04a026513a3ca21d0a480506c2ae3ba32211d7b2`
及缓存23个binary hash。旧缓存manifest的1850434基提交保持原值，不冒充本次资格来源。

`core5-d64-precision-stage1-20261029`、`stage2-20261029`与
`core5-arithmetic-qualified-20261029`全部终态failed/2；最后一个设施子检查通过、D64子作业仍failed/2。
`core5-arithmetic-dev-20261029`为passed/0。不存在活动作业。
各任务独立background.slice/Nice10；stage1 32G/25min、stage2及固定资格32G/10min、开发16G/10min，
探索每进程180s，固定D64子任务240s。实时PID/Transient/slice已核对；固定资格另保存live-unit.json。
不以卸载后systemd的默认success字段推断通过。未隔离外部CPU/内存争用。

## 采用决策与复现

没有把失败的局部探针接入Python/C++公共执行。需要当前可用的更高精度配置时，
继续显式选择整条CPU FP64；它的独立完整训练/恢复和有限计时来自
[1f1ec57证据](core5-stable-preparation-precision-20261028.md)，不回填为FP32修复。
本轮没有新的生产候选，因此没有新增吞吐测量或重跑D2048/设备矩阵。
Python派发的诊断不认证native、NPU、CUDA或resident；所有硬件/异步待验项保持原状态。

从干净d8f7132按[开发入口](../development.md)准备匹配CPU core，运行：

```sh
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_weighted_arithmetic_probe.py \
  tests/test_weighted_workload_training.py tests/test_weighted_benchmark_precision.py --dtype both
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float32 \
  --implementation python --arithmetic linear_reduce_norm64 --checkpoint-check --output-dir NEW_CHECK
```

最后一条预计exit 2；分别检查首步通过、第二步严格失败、两条精确恢复通过。
各探索变体的flags在[诊断契约](../weighted-arithmetic-diagnostics.md)，无需旧运行目录；
原始探索与固定复现的源码身份及范围分别记录，不声称九项都重新进行了checkpoint资格。
