# 带权消息发送函数与节点增益

固定 [区域分配教材](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)
`0770593` §2、§4、§6.3、附录 S。共同 payload、Read 快照、状态保留、
Full 和 checkpoint 见 [带权绑定](weighted-message.md)；本页补发送函数的定义域与反向。
[评分扩展](weighted-scoring.md)独立绑定 cat/match、ind/att、cat/grouped 及两种 key；资格状态见 STATUS。

## 公共配置与描述

`RegionalConfig(selection_mode=...)` 选择下表中的完整函数。
Python `programs.RegionalTotal` 保留原公开类名，现承载全部发送函数；
C++ `weighted_region.cpp::Sending` 有独立公式。两者均向普通引擎返回
`Selection(active, controls, singleton_history)`；不替换调度、采用或 Full。

| 函数 / 定义 | 实际发送 | 正典容量 K |
| --- | --- | --- |
| regional_total，§2/§4.1.1 | Alloc(s,μmax;B,B) | 区域大小 |
| regional_total_zero，§4.4.2 | 按 score 降序/ID 升序，用 μmax 逐项填 B | 区域大小 |
| regional_threshold，§4.1.2 | Alloc(s−ηj,μmax;min(1,Σw),k) | 区域大小 |
| regional_threshold_hard，§4.4.3 | 正上限中调整分为正的前 k 个；全非正时最高一个 | k |
| constrained，§4.1.3 | Alloc(s−η,μ;min(1,Σμ),k) | 区域大小 |
| hard_selection，§4.4.3 | 与阈值硬规则相同，使用节点增益 μ | k |
| direct_attention，式 (11) | μp，base=μ/received | 区域大小 |
| unit_attention，式 (12) | p，base=uniform/μ | 区域大小 |
| pooled_attention，式 (13) | min(1,Σμ)p，base=μ | 区域大小 |
| proportional，式 (14) | μ min(k,Σμ)/Σμ；全零时为零 | 区域大小 |

regional_total/threshold 及各自配对函数使用区域控制描述
`{w:[], mu:[], score:[], evidence:[dreg]}`。mu=min(1,gamma_max*w)，
w=min(1,r)。fixed 控制返回零证据；context 依接收 w 汇集证据，
head 输入是 `[Σw*e/max(1,Σw), log1p(Σw), log1p(Σmu)]`。
total 对 head 使用 sigmoid 得比例，threshold 直接使用其有限实数输出作 ηj。

其余规则使用节点增益描述 `{w:[], mu:[], score:[]}`，无无意义 evidence 槽。
`send_gain=fixed` 可给 fixed_gain∈[1,gamma_max]，省略时为 gamma_max；
或给 gain_logit，绑定独立可训练标量参数 ζ，γ=1+(gamma_max−1)sigmoid(ζ)。
二者互斥。`send_gain=context` 绑定独立 cat 特征 ScalarHead，给 γ 的相同 sigmoid 形状。
节点增益 μ=min(1,γ*r)；原始非负接收 r 与截断 w 分开。score/gain 默认参数独立。
固定增益不依赖特征，全部节点的状态仍按各自记忆的正接收规则保留。

所有配置先解析默认值，再冻结进入图身份。旧 regional_total 默认图身份保持。
新函数使用 weighted-region-v1 哈希。新图绑定才能切换函数；不能保留旧缓存后改配置。
只为适用字段填默认值：正温约束默认 quad/temperature=1；Attention 默认 temperature=1，
direct/pooled base=mu，unit base=uniform；有 k 的函数默认 1；节点固定阈值默认 0。
regional 默认 context/cat/dreg=2；fixed total 必须给 total_fraction，fixed threshold 默认 eta=0。

不适用字段必须缺席（None）：硬/零温/比例不读温度或正则，Attention 不读 k/eta/regularizer；
regional 不读 send_gain，节点增益函数不读区域控制/证据/宽度，total 不叠加 eta。
用 dataclasses.replace 从正温函数切到硬函数时必须明确清除旧求解字段。
节点 ID 是不同的非负 int64；k 是正 int64，绑定时校验不超过区域大小。
C++ RegionalConfig(mode) 提供适用默认配置，赋值后 validate() 检查同一域。

## 可独立调用的数学原语

Python [sending.py](../python/tidegraph/weighted/sending.py) 与 C++
[weighted_sending.h](../cpp/include/tide/weighted_sending.h) 提供对应接口：
attention_shares、attention_send、proportional、total_zero、hard_selection。
输入是同 dtype/device 的一维 FP64/FP32 完整真实候选，允许空向量；
caps/received∈[0,1]，received≤caps；shares 的非负 base 不限于一。
total、temperature 是同 dtype/device 标量，后者严格为正。
返回控制向量按原输入顺序；节点 ID 仅用于稳定平票。
这些原语不计算状态或调用参考解码。每个控制和描述字段保留独立公开梯度根。

Attention 在正基权重上对 `s/temperature+log(base)` 作稳定归一化。
零基权重候选仍保留 s，其 base 右导数不为自动的零。
uniform 由真实候选存在性定义，不能把 padding 当候选；引擎只传完整候选集合。
pooled 的单项可以超过对应 mu；direct 的外乘 mu 与发送 payload 中的 c 不重复相乘。
数学正支持若在请求 dtype 下溢为零，明确失败，不改活动集合、不提升精度。
Full 仅对 c>0 执行，仍发 `(c,G)`，数值投影才得到 cG。

## 一阶反向与边界身份

Attention 声明 `weighted-attention-map-v1`。Z>0 时使用教材式 (20) 的精确映射 VJP：
ds=p*(g−Σpg)/temperature，da=exp(s/temperature)/Z*(g−Σpg)。
da 在 a=0 处取向正值移动的右导数；温度导数沿同一份额映射计算。
保存实际中心化 log 比率，避免不必要的指数中间量溢出；不可表示的 VJP 明确失败。
direct/unit/pooled 再分别按式 (21) 沿外乘、B、增益和评分依赖回传。
全零 base 声明连接输入的零 VJP；它通常不是普通 Jacobian，也不等于各方向导数。
空候选返回空，普通引擎不调用空 SelStep。

proportional 声明 `weighted-proportional-map-v1`，不接收 score。
Σmu<k 时 c=mu，包括原点非负域的一阶恒等；Σmu>k 沿归一化公式求导。
Σmu=k 取未缩放侧。gain、B、接收的 clamp 在精确上界取线性侧（ATen clamp 约定）。

零温填入声明 `weighted-sorted-fill-branch-v1`：固定稳定顺序内，满项对自身 cap 导数为 1；
若存在唯一部分项，其微分为 dB−Σ(之前已满 cap 的微分)，其他项零。
精确填满端点取 cap 侧，dB=0；零项取零侧。score 的排序不连接反向，返回 None。
全上限端点直接返回原 cap，避免浮点前缀减法制造一个伪部分项。
hard 声明 `weighted-hard-cap-branch-v1`：固定所选集合，仅选中 cap 导数为 1，
score/阈值无反向连接。平票、正评分边界、支持切换处不宣称通常可微；无代理反向。
以上均独立于 quad/bern/ent 最优解映射、有限迭代导数及旧 HST。只声明一阶。
自定义反向保存输入版本，前向后的非法原地修改会失败。
Python RegionalConfig.backward_identity 给出所选反向身份，固定 alpha/全部函数参数仍由图哈希记录。

## 定义 → 符号 → 独立检查

| 数学条目 | 实现符号 | 验证内容 |
| --- | --- | --- |
| §4.2、§6.3、S.3 | attention_shares / attention_send | 独立 exp 商公式及全部根、零 base 手算右导数、全零角方向导数、真实零候选、支持下溢 |
| §4.3 | proportional | 比例手算，原点恒等导数，没有 score 输入 |
| §4.4.1–3 | total_zero / hard_selection | 独立枚举 LP 顶点、稳定 ID、四个 0.3 cap 发送总量 1 的反例、分支方向差分、无 HST |
| §4.1.2–3、S.1 | Describe / RegionalTotal、C++ Describe / Sending | 原始阈值头、独立增益参数/三元描述、适用字段和 k/K 校验 |
| 架构/正典作用与继续 | bind、原有独立 Python/native 调度 | 十种发送的全记录/独立根、HB/非对齐 DAG、PDG 反馈、每切点、三步 AdamW/checkpoint/no_grad |

检查入口：[公式](../tests/test_weighted_sending.py)、[native 公式](../tests/test_weighted_sending_native.py)、
[整图](../tests/test_weighted_sending_execution.py)、[standalone 局部](../cpp/test/weighted_sending.cpp)。
配置身份兼容由原熵分配检查继续验证。完整记录等价与不同函数的性质比较分别进行。
开发/不可变资格状态见 STATUS；局部 C++ 手算不替代完整 standalone 训练矩阵。
本增量尚不认证 packed 联合算法、设备常驻、跨卡或性能。
