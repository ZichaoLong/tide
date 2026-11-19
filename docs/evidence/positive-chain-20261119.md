# 005同类正量链：修复、资格与成本（2026-11-19）

当前FC/HB中确有“最终结果可表示，中间运算却提前溢出/下溢”的合法局部输入。
normalize不能保证避开：归一化后仍有无幅度约束的可训练readout、offset和Q/K投影，
也没有统一的正质量下界。旧真实训练故障说明有实际工程意义，但不能据此推断新绑定训练中的发生概率。
本轮修复当前pool、Shannon和普通弱prior Attention的具体路径，未重放退役Bern/direct训练。

005的三项原故障应分开理解：约1.27845e-40的Bern发送量可表示，属于提前归零；
约1.49804e-48和1.43837e-46的最终发送量则会舍入为FP32零。
当前数学不包含旧Bern/direct绑定，接受并落实的是适用的稳定求值方法；
旧positive-floor策略未迁入，不能称005全部问题消失或下游已验收。

## 实现身份与有界见证

| 身份 | 内容 |
|---|---|
| `fd70fefbc8dbb799828c59fa4a12d11760211fd9` | 本轮before，已有稳定Reception及质量缩放KV |
| `b4a8d93799eb24ed7f338ed969702b7ac0150e8a` | pool复合VJP、Shannon稳定重建/耦合反向、普通弱prior Attention |
| `3efcc6b1ae68e5c22f5895a7d68220836edfcef7` | NPU设备内FP32拆分，替换frexp/copysign主机回退 |

最终生产`src/tide` tree：`6863ff792e3fae1556a80e195c1358a3405e0e0e`。
固定math仍为`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件原字节通过。
公式、形状及范围见[positive-chain](../positive-chain.md)，没有改epsilon、支持规则、参数归属或私有状态。

| 当前独立见证 | 旧行为 | 修复后的目标 |
|---|---|---|
| pool零score、两个1e-40质量，根[1,0] | 质量梯度±Inf | [1,0] |
| pool score=[90,0]、质量=[exp(-90),1]、根[1e-30,0] | 首质量梯度Inf | 有限，约3.051e8；总质量约1 |
| 普通Attention零score、prior均1e-40、value=[1,1+2^-20] | 旧math/tiled梯度Inf；旧CPU auto此例原本有限 | prior梯度约±2.3842e33 |
| Shannon score=[2]×63+[-98]、单位质量/上界、预算32 | 先softmax丢失弱概率 | 最终弱发送量约1.88956e-44，保持正支持 |
| Shannon最小次正规质量及根[.25,0] | 耦合反向先乘小量，dw变0 | dw=[.125,.125] |

pool在恢复公开梯度前合并概率、幅度和伴随，并先归约至原始广播根；零prior保留右导数，
空行VJP为零。pool的M=1使用常量min分支，与KV clamp在M<=1的导数区分。
Shannon保持KKT活动面导数；还修复最小次正规行的midpoint落到全upper、总量翻倍问题，
只在残差严格更小时选择bracket端点，原阈值不改。

完整小FC在normalized features后使用有限offset[-110,1]，仍可触发真正的最终发送范围下溢。
该测试保留严格拒绝。这证明合法参数配置可达，不证明某优化器轨迹或高发生率。
正常FC记忆权重为1；当前完整9B quad图还有Hub及区域总质量下界，不能把整行弱pool例子
直接当作该9B的真实状态。具体可达性边界在公式文档中分别记录。

## 数值、训练与实际恢复

| 冻结源码 / 硬件 | 检查与结果 |
|---|---|
| b4a / CPU | 全套1182 passed、4原有strict xfailed、1 warning，670.75s |
| 3ef / CPU | 相关最终定向274 passed、3原有strict xfailed，19.72s |
| 3ef / 空闲单NPU物理11 | 未捕获输出的定向133 passed、3原有strict xfailed、2 warnings，4.85s |
| 后续harness / CPU | 4个设备日志门控检查通过：parent、restore、missing-parent、clean |

b4a全套没有转记为3ef全套。三个seed773近零dV超差和既有FFN超差保留；没有新阈值放宽。
公式回归含值/VJP、输入/参数及共享梯度、广播根、零质量、版本保护、FP64独立参考、BF16，
设备拆分另含标量/连续/转置/展开/切片、次正规、幂边界、±0、±Inf。
FP64只在CPU作有界诊断。定向及既有组合覆盖私有状态、原始int64、窗口淘汰和分块继续。

CPU在b4a、NPU在3ef各完成五个小图：B2/L8/chunk4，一次anchor更新、三步继续训练、
checkpoint及新进程step4→5。推理、含梯度前向、重复VJP、完整更新分别记录。

| 案例 | 核心形状 | 参数 | CPU / NPU |
|---|---|---:|---|
| FC KV128 FP32 | 单节点单层，d192/KV128/FFN128 | 338,809 | within_threshold |
| HB X pool FP32 | compact-5-dd4，n2/hub1，d64/KV32/FFN128，region_shared | 1,012,351 | within_threshold |
| HB X Shannon FP32 | 与pool同尺寸，改变分配数学 | 1,012,351 | within_threshold |
| HB Timed K FP32 | 小HB，d64/KV32/FFN64，checkpoint_reception | 490,619 | within_threshold |
| HB X pool BF16 | 与FP32 pool同尺寸；真实BF16投影 | 1,012,351 | within_threshold |

比较同一候选自己的恢复，不要求pool与Shannon相等。CPU五例saved_data、next_forward、actual_update
均exact；NPU saved_data和next_forward五例exact，actual_update仅FC和Timed逐位相同。
另外三例最大已列梯度差分别4.6566e-10、9.3132e-10、9.3132e-10，无支持、整数时间、
非有限或阈值失败。整图沿用atol1e-6/rtol0.002，不能与原语的rtol1e-5混同，也不声称逐位训练恢复。
本轮未新增隔离wheel资格、9B训练、CUDA、多NPU或下游学习验收；主环境隔离导入在退休后另核对。

## NPU归属更正与失败保留

旧[stable-mass报告](stable-mass-20261119.md)的原训练、恢复和六份after成本日志实际含
`aten::frexp.Tensor_out`、`aten::copysign.out`的CPU fallback。
旧pytest默认捕获通过测试的C++ stderr，不能据此声称设备内执行。旧数值与计时不改，
改归类为混合执行；旧+25.01%/+32.88%包括该回退成本。

本轮b4a训练再次显露回退，主动停止并保留terminated；b4a原语126 passed/3 xfailed
只证明相应数值检查，不建立无回退资格。3ef对NPU FP32用IEEE位解释，次正规先精确乘2^24；
signbit/where替代copysign，CPU/FP64继续原有函数。3ef的`pytest -s`、主训练日志及五个
恢复日志、六份after成本日志均未见主机回退。资格入口补日志门控和不捕获stderr，防止重犯。

物理7上的局部trace已导出且值exact断言通过，随后解析器误把JSON list当dict，服务为failed。
离线审阅保留失败：两次binary_parts/restore出现166 AIVEC、12 AICPU、4 DMA和4 event。
AICPU包括RightShift/Pow，4次实际HtoD、无DtoH事件；不是纯AICore或无搬运声明。
NPU格式转换/contiguous也可能有设备副本与工作区。原461,000字节trace一并归档。
两个旧CPU成本服务在依赖等待阶段因增加有限timeout/切换最终源码而停止，未完成测量；不得算成成本结果。

## 修复成本：固定小图普通计时

固定上表1,012,351参数的HB X pool，FP32，独立B2，prompt16/chunk4/decode8，4 Torch线程。
before=fd70，after=3ef；各三组成对重复，每组warmup1/samples3，第二组反转先后顺序。
表中为三组中位数的中位数；变化为成对比值的中位数，两列中位数不必严格相除一致。
训练吞吐按总位置/普通样本总时间计算，与样本中位耗时的倒数也不必相等。
训练含跨chunk前向、loss、backward、AdamW、同步和更新后detach；推理含输出就绪/释放及同步。
工作/存储观察另跑；首主干输出没有任务头或采样，不称完整语言模型TTFT。

### CPU：固定亲和80–83，共享主机

| 阶段 | before秒 | after秒 | 成对耗时变化 |
|---|---:|---:|---:|
| 首主干输出 | 0.113131 | 0.126174 | +11.24% |
| prefill，32输入位置 | 0.556961 | 0.621160 | +12.73% |
| 稳定decode，排除首步，14位置 | 0.950110 | 0.985013 | +4.03% |
| 完整推理 | 1.706360 | 1.769220 | +4.01% |
| 训练前向 | 0.715845 | 0.734442 | +1.85% |
| backward | 0.619765 | 0.749039 | +20.86% |
| AdamW | 0.007793 | 0.007756 | -0.47% |
| 完整训练更新 | 1.382364 | 1.504411 | +7.43% |

| 吞吐 | before | after | 成对变化 |
|---|---:|---:|---:|
| prefill位置/s | 57.4547 | 51.5165 | -11.29% |
| 稳定decode位置/s | 14.7351 | 14.2130 | -3.87% |
| 独立完整序列/s | 1.1721 | 1.1304 | -3.86% |
| 完整训练位置/s | 23.1843 | 21.2463 | -7.65% |
| 独立训练序列/s | 1.4490 | 1.3279 | -7.65% |

完整耗时成对比值：推理[1.04084,1.00502,1.04011]，训练[1.04239,1.11652,1.07428]。
有限三组有波动，不作显著性或整机CPU最优吞吐结论。

### NPU：空闲独占物理11，host亲和92–95

| 阶段 | before秒 | after秒 | 成对耗时变化 |
|---|---:|---:|---:|
| 首主干输出 | 0.181556 | 0.228947 | +25.65% |
| prefill，32输入位置 | 0.724856 | 0.912554 | +25.25% |
| 稳定decode，排除首步，14位置 | 1.243118 | 1.592042 | +26.81% |
| 完整推理 | 2.145278 | 2.731320 | +26.93% |
| 训练前向 | 0.802331 | 0.972745 | +22.70% |
| backward | 0.988837 | 1.484205 | +55.60% |
| AdamW | 0.009810 | 0.009944 | +4.82% |
| 完整训练更新 | 1.801802 | 2.467762 | +40.16% |

| 吞吐 | before | after | 成对变化 |
|---|---:|---:|---:|
| prefill位置/s | 44.1467 | 35.0664 | -20.16% |
| 稳定decode位置/s | 11.2620 | 8.7937 | -21.14% |
| 独立完整序列/s | 0.9323 | 0.7322 | -21.22% |
| 完整训练位置/s | 17.7577 | 12.9375 | -28.85% |
| 独立训练序列/s | 1.1099 | 0.8086 | -28.85% |

完整耗时成对比值：推理[1.25491,1.27318,1.26934]，训练[1.44134,1.32295,1.40159]。
before六份子日志均有frexp/copysign主机回退，after六份未见；此比较同时包含数值算法和设备求值改变，
没有消融两者的独立成本。本轮未重测9B/Transformer，不能把这组净增幅转移到旧大图数据。

### 工作与内存

所有成对初参、输入hash、配置、参数量及完整工作记录一致。推理Full816：KV选中/执行576行、SSD240行；
训练观察Full544：KV384、SSD160行，均没有Full padding。接收分别有3696/3312、2464/2208个
消息槽/正消息；这些padding及NodeLinear矩阵规模两侧一致。训练活动观察仅单独anchor含梯度前向，
没有逐更新在计时内追踪活动。耗时增量主要在前向/反向，不能归因于更多参数或Full padding。
额外尾数/指数操作、归约、检查与同步均可能贡献；未提供各自独立百分比或整网算子trace。

| NPU普通计时峰值，MiB | before | after |
|---|---:|---:|
| inference allocated | 36.7412 | 36.7412 |
| inference reserved | 58 | 58 |
| train allocated | 82.1260 | 82.1465 |
| train reserved | 174 | 174 |

独立存储诊断：权重4,049,404 B，AdamW状态8,099,508 B（含CPU step槽700 B），
结束私有记忆198,096 B，均未改变。前向额外保存激活的可见独占存储：
CPU 19,570,363→19,574,339 B；NPU 21,059,012→21,062,988 B。
分阶段allocator与可见类别的残差收入JSON，含未观测张量/后端开销，不能称纯临时工作区。
CPU没有等价设备allocator，RSS仅进程生命周期观察；持久状态和激活快照不替代峰值。

## 剩余边界与建议

建议采用包含3ef的FP32实现，BF16次选，接受本次测到的稳定性成本。
本轮默认API、状态格式及实例绑定不变，浮点求值改变不保证旧checkpoint的逐位后续轨迹。
训练consumer应在新私有安装验证后采用，不自动更新下游锁。

真正最终正发送量超出FP32表示范围仍严格拒绝；如需持续越过它，另行设计对数/更大范围质量表示，
同时贯穿生成、存储、聚合、KV和梯度。没有用floor、裁剪梯度或扩大epsilon掩盖问题。
Attention概率可能在后续缩放乘积前已下溢，前向分子也可能在除分母前溢出；
任意极端logits、被mask的中心化锚、相消及后续共享参数累加仍有范围限制。
Shannon仍以求解后存储的c为斜率；没有声明全次正规域未量化log-slope契约。
RMS平方、SSD极端参数/scan、9B训练、CUDA/多NPU均没有新增全域保证。

可选下一步是在有证明的普通数值范围内减少二进制拆分/恢复和同步，或融合复合VJP；
普通域快路径必须有范围准入和独立梯度/零边检查。移除主机回退后仍有小kernel及AICPU成本，
不能从局部替换推断整体加速。任务质量仍属于下游。

## 复现、资源与保留

原命令、环境、配置、输入及包hash、普通样本、工作/存储、恢复日志、失败、trace和终态在
[压缩审阅JSON](positive-chain-20261119.json.gz)，用[archive index](archive-index.json)核验：

```bash
python scripts/materialize_record.py --index docs/evidence/archive-index.json \
  --record docs/evidence/positive-chain-20261119.json --output /tmp/positive-chain-review.json
```

在[私有环境](../development.md)中用新任务名冻结资格，例如：

```bash
python scripts/launch.py --name positive-chain-cpu-recheck --revision 3efcc6b1 \
  --threads 4 --affinity 0-3 --seconds 900 --rss-gib 16 -- \
  python -m scripts.qualify_positive_chain --device cpu --threads 4 \
  --output-dir '{run}/qualification'
```

局部复现见`tests/test_allocation_range.py`、`test_ordinary_attention_range.py`、
`test_numerical_reachability.py`、`test_binary_parts.py`。设备原语用`TIDE_STABLE_DEVICE=npu:0`及`pytest -s`。
NPU必须重新检查空闲资源并经协作队列；旧物理卡号不是未来授权。本轮未共享占用卡。
成本入口为`scripts.measure_stable_mass_costs`，用`git archive fd70fefb`重建before源码；
配置原文在归档`files["artifacts/positive-chain-20261119/cost-config.json"].text`。
旧绝对路径只是身份，不是在线承诺。plan.json保持准备记录，不作为实验结果。

全部12个服务和148个记录后代PID/start身份已核验终态：MainPID=0、无匹配活后代或cgroup进程。
CPU类/NPU类服务累计墙钟耗时分别1467.042s/632.420s（包含依赖等待），不等于CPU计算或NPU忙碌时间；
峰值进程树RSS分别1.652/5.043 GiB。
各服务有限预算、冻结源码、background.slice及持久日志可审阅。
主机HiSilicon aarch64，320逻辑CPU、4 socket、8 NUMA；Ascend910、CANN9.2、
Torch2.12.0+cpu/TorchNPU2.12.post3。未改公共环境、参考仓库或下游。
原始张量/重复源码在证据提交和终态核验后退休，保留源码、独立见证与小记录，见最终STATUS及退休收据。

证据提交`f098d3b4`后，13个本轮临时目录退休，释放838,303,744 B；没有转存大载荷。
仓库含Git/主环境约151.2 MB。主editable安装用`python -I`重新导入当前源码，完成一个CPU FP32
前向/反向/AdamW smoke；不是新wheel资格。四份用户文档hash及math原字节未变，
公共与下游保持边界，详见[退休收据](positive-chain-retirement-20261119.json)。
