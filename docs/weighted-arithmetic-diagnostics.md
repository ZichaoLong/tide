# 带权执行的算术边界诊断

数学函数及指定一阶反向仍由[语义对应](semantic-map.md)和[精度契约](weighted-precision.md)定义。
`scripts/diagnose_weighted_workload.py`独立构造相同材料的scalar参考与greedy/packed候选，
检查完整返回、离散结构、参数梯度、SGD参数和动量槽。任何比较失败保留exit 2。
`--checkpoint-check`在首步更新后保存实际状态/参数/optimizer，再换owner解码并精确比较下一窗。
`--dtype float64`是已有的整条CPU FP64配置；从同一seeded FP32材料提升，不换输入或初始化。

## 干预的身份与边界

可选`--arithmetic`只供**Python CPU FP32诊断**，native和其他payload dtype提前拒绝。
默认`none`不干预算术；这些探针没有加入公共执行器、C++公式或benchmark候选。
实现为`scripts/weighted_arithmetic_probe.py`的显式上下文，两个独立调度使用相同干预。
调用方必须分别包住forward/loss和backward；optimizer、checkpoint编码/解码在上下文外。

| 名称 | 选定边界 |
| --- | --- |
| `linear64` | ATen mm/bmm/mv/dot/addmm的FP32输入临时提升FP64，结果转回FP32；同时覆盖实际出现的反向调用 |
| `reduce64` | sum/mean的FP32归约临时FP64，结果转回FP32 |
| `linear_reduce64` | 上述两组 |
| `norm64` | Architecture §1 RMS的分母、normalized向量和一阶收缩FP64；输出/输入梯度返回原dtype |
| `linear_reduce_norm64` | 上述三组 |
| `nonlinear64` | 三组加SiLU/sigmoid及其ATen backward、exp/log/sqrt边界 |
| `backward_linear_reduce64` | 只在backward干预矩阵乘加与归约，forward保持原样 |

算子每次返回就舍入，不等同于整个模块、完整计算图或共享叶累加保持FP64；
也不能直接等同于把Python表达式换成`.double()`后由autograd生成另一张图。
例如归约变宽不会恢复此前FP32乘法已舍掉的信息。增加干预不保证误差单调减少。
norm干预保存原tensor用于版本检查，保留None/连接零，拒绝二阶，作用域结束或异常时恢复原函数。
这是单进程单线程定位工具，不是供并发调用的全局精度切换器。
FP64/非浮点张量不被算子探针量化；逻辑时间/计数继续int64。

`--scoring-dtype float64`则调用**已有公共**Read/区域控制精度配置，参考与候选一致采用。
可与探针组合研究完整Read边界。raw状态、消息、参数和SGD仍用payload dtype；
这与整条`--dtype float64`不同。检查点不保存临时探针，诊断入口在恢复的两条调度上显式重用同一设置。
不能把这些CPU干预的效果外推到设备或native；不发生隐式CPU搬移。

## 记录与复现

结果保存原公共比较的首个失败，同时附加：

- 按(time,batch,node)排序的首个精确事件差异；这是观测位置，不证明因果来源。
- 唯一命名参数/梯度的精确不等张量数、超容差张量数和最多10个最差见证；None/零、
  非有限值和元数据错误单列。误差比使用该dtype的原容差，不改变完整比较。
- 前向/反向实际被干预的ATen调用计数。不是完整算子量、FLOPs或性能计时。
- 两步实际候选/Full/消息数、loss、optimizer槽、每条调度自身的精确恢复结果。

本机长任务遵守[固定源码资格](verification.md)，从干净checkout可生成同一D64静态计划：

```sh
python scripts/prepare_weighted_workload.py --topology ranked --layers 15 --region-width 32 \
  --width 64 --samples 1 --positions 1 --match-core3 add --output-dir NEW_PLAN
python scripts/diagnose_weighted_workload.py --workload NEW_PLAN/workload.json --dtype float32 \
  --implementation python --arithmetic linear_reduce_norm64 --checkpoint-check --output-dir NEW_CHECK
```

每个配置使用新目录/独立进程，不自动重试或扫描全部开关。当前原D64预期仍有严格不等价；
生成报告不等于获得通过资格。去掉`--arithmetic`是原FP32对照；增加`--scoring-dtype float64`
是完整Read边界探针；整条FP64用`--dtype float64`且不带算术干预。
native整条FP64另给`--native-library CPU_CORE --implementation native`。
原计划SHA256为`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`；
生成后的hash、参数清单与实际源码都要核对。

## 外层反向汇合点诊断

`diagnose_weighted_joins.py`复用上述CLI，仅允许Python CPU FP32，以FP64缓冲累加外层反向图
各汇合点的贡献，再转回原dtype交给局部VJP。它使用Torch2.12的私有node接口，只供有限定位；
自定义Function内部的`autograd.grad`仍由原引擎执行，所以不是整张反向图全精度。
广播输入梯度按原形状缩减，Python Function的非Tensor参数不当作梯度边；None/连接零分别保留。
只支持全标量loss、一阶、已清空leaf梯度、无hook/retain-grad的诊断用例，拒绝其他接口请求。
不能作为生产引擎、独立数学oracle、native/设备或性能资格。入口退出会还原临时补丁。

```sh
python scripts/diagnose_weighted_joins.py --self-check
python scripts/diagnose_weighted_joins.py --workload NEW_PLAN/workload.json --dtype float32 \
  --implementation python --arithmetic linear_reduce_norm64 --checkpoint-check --output-dir NEW_JOIN_CHECK
```

`join-probe.json`单列实际汇合/广播次数；`result.json`的完整记录/梯度/optimizer失败仍以exit2返回。
`test_weighted_join_probe`验证手算相消、广播VJP、非Tensor参数/None/零、两步小图真实恢复及禁止覆盖。
只提升外层汇总不能恢复已经在局部VJP里舍掉的信息，也不保证误差单调减少。

## 原反馈图与异步见证

`diagnose_weighted_feedback.py --native-library CPU_CORE --dtype float32 --output-dir NEW_FEEDBACK`
固定原W32/B4/每窗4位置/seed7的KV反馈PDG材料，比较独立Python标量与native streaming/prefill。
四个连续窗逐项核对完整记录、命名参数梯度、SGD参数/动量；第三次更新后各自保存实际checkpoint，
换owner后第四窗完整记录/梯度/参数/槽要求逐值恢复。`--dtype float64`是显式另一个配置。
任何严格差异返回2，元数据/数值有限性仍由公共比较检查；没有吞吐采样。

`diagnose_torch_peer.py --policy stock|single_thread|device_guard|cpu_bridge`保留同一个12轮
纯Torch小反例。每种政策使用独立进程并显式unset launch blocking；CPU桥是有意经主机传输的
定位方案。任一小反例通过都不代表完整图异步资格；不安装生产传输替换或修改公共环境。

`tests/test_weighted_arithmetic_probe.py`以手算相消、独立literal Jacobian验证干预本身，
并覆盖int64、None/零报告、非有限值、版本、异常恢复、适用域拒绝和小图两步实际checkpoint。
`test_weighted_workload_training.py`保持Python/native默认两dtype的完整训练/恢复检查。
局部测试通过只认证诊断机制，D64和生产算术是否等价由各自实际报告决定。


## 补充局部与运行时诊断

`diagnose_core5_inference.py --workload PLAN --device cuda:0 --implementation python|native
--placement mixed-c --dtype float32 --packing packed --output-dir NEW`保留原完整推理配置，
先完成候选，再由独立CPU时间递归比较；native另给匹配的`--native-library`。
默认仅第0窗，可显式两窗；不包含吞吐采样或checkpoint资格。
报告完整/离散首差、按原始`(time,batch,node)`排序的首个精确与严格事件差异，
以及最早严格新KV写的来源/归一化/低秩投影局部FP64锚点。
锚点仅消费双方已经返回的实际fiber/content和原参数；候选从未收到参考轨迹。
分别量化输入content差异的传播、各自局部写误差及连贯source→KV锚点；
CPU FP64只是明确的数值锚点，不称精确实数，也不由局部改善推断全图修复。
原FP32/FP64容差、int64及结构比较不变，严格不等返回exit2。
显式全CUDA eager FP64使用`--dtype float64 --scoring-dtype float64`，
它是另一个执行身份，须另验完整记录/同栈恢复；不解除原FP32失败。

`diagnose_weighted_first_write.py --fixture FILE --output-dir NEW`只读取已导出的v2材料中
首个无入边区域所需的实际张量，逐张量校验hash；累计操作数上限512MiB，单次读取上限128MiB。
必须各节点只有一个共同外部原子，A/B/embedding及base是实际共享身份。比较source变换、
private BMM与组合误差，以及连贯FP64局部公式一次返回FP32的误差。不会分配完整模型、
回放参考轨迹或认证下游状态/路由/反向；未读取的完整binary hash只记录声明值。

`diagnose_weighted_precision.py --implementation python --normalization-vjp float64 --output-dir NEW`
在原half GDN/default AdamW fixture上保持前向逐值不变，仅对三种归一化的解析VJP
使用CPU FP64中间值，公开梯度仍回原dtype，optimizer/loss scale不变。
`test_weighted_half_probe.py`用独立FP64 Jacobian验算干预、前向不变及hook恢复。
这是诊断，不是设备FP64支持或新的默认精度策略。

`diagnose_weighted_async.py --device npu:0 --peer-device npu:1 --copy-policy cpu_bridge --scope complete --output-dir NEW`
评估显式CPU staging：逐字段原版本、None/零根、完整图与保留图、三步AdamW、
实际checkpoint解码后第四步及CPU/换卡恢复。桥接提交数与字节另记；即使通过，
也不认证默认直接peer异步。`test_weighted_peer_bridge.py`只检查CPU上的结构契约；
真实事件和跨设备运行必须在对应设备另验，不修改公共环境或默认transport。
