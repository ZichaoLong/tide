# 正量归零链路：实际失败、静默见证与待讨论方案

2026-11-11，TL-POSITIVE-v6。三项历史失败均已重现；另一次原native边界前向补齐direct发送的实际操作数。全部CPU1/FP32，四个服务已终态且无存活后代；分配约0.2395 CPU核时，运行材料约49MiB，不含短命局部公式探针和源码归档。数学0770593、基座74e7bbd、原native二进制与安装锁不变。

**结论：三个异常包含两类原因；无异常不代表没有数值内容或离散支持丢失。** constrained-Bernoulli的真实结果可表示，但sigmoid提前归零；regional-total-Bernoulli和direct的最终控制确实不可表示。小型公共图另证实旧strict聚合的静默内容丢失，以及Describe前置精度转换的静默支持变化。

本页为实验线程的讨论材料。未创建新上游需求、未修改基座、未接入当前性能优化，也未让原失败轨迹继续训练。[执行表](positive-path-audit-v6.md)、[实际行和100位Decimal参考](../evidence/positive-path-v6-local-analysis.json)、[运行/恢复/Lens审阅](../evidence/positive-path-v6-review.json)。

## 三个实际失败

| 原配置 | 重放及历史观察 | 捕获位置与数值 | 分类 |
| --- | --- | --- | --- |
| HB40 direct_attention / received | 第65步checkpoint；72、74步的loss/梯度范数/token/LR逐值一致；第75次训练前向失败 | 实际候选批row5/col0：mu=3.9022370605e-21，p=3.6860240702e-26，乘积=1.4383739733e-46；两因子本身均为FP32正规数 | 最终乘法真范围下溢；份额先通过正支持检查，最终控制守卫拒绝 |
| HB40 regional_total / bern / tau=.25 | 第135步checkpoint；136、141步观察逐值一致；第142次训练前向失败 | time17，nodes24–27，第0项mu=1.0963971953e-9，x/tau=-89.4886856；100位约束解c=1.4980425046e-48 | sigmoid先归零，稳定求值后最终乘容量仍不可表示；只修sigmoid不能解除该失败 |
| HB40 constrained / bern / tau=.25 | 第125步checkpoint；128、129步观察逐值一致；129步后的验证前向失败 | time73，nodes20–23，第2项mu=1，x/tau=-91.8577576；100位解c=1.2784471168e-40，可舍入为1.2784466280e-40 | 可表示结果被当前sigmoid提前算零；slack，无约束乘子误差，残差为0 |

FP32最小正次正规数为2^-149≈1.4012984643e-45，舍入到零的半单位界为2^-150；不是finfo.tiny≈1.17549435e-38。两个Bern失败的原残差均为0，容差仍为3.0517578125e-5。原严格数值阈值1e-6/1e-5未放宽。

三个实际局部行通过公共Python scalar API均重现FP32拒绝，FP64单行完成并与独立Decimal解相符；这不是完整FP64训练/反向/恢复资格。direct额外保存了实际max-centered评分、温度1、received、caps、shares和控制行；绝对未居中评分、全局节点及事件时间未被native通用RuntimeError携带，因此不把局部row5/col0当作全局节点ID。输入与失败边界可再次运行原native，尚未追溯该小received在整张HB图上的完整来源链。

原生ATen只读观察器重跑同一前向，四次均得到原异常；B1/L2独立见证还验证了观察器不改变native输出。Bern求解器的试探/残差复算可重复出现归零，观察器计数不是唯一事件数或最终失败坐标数。重放检查仅覆盖现有历史标量观察，未声称所有参数/梯度/状态的独立长轨迹位级等价。

## 已执行的静默见证

1. **聚合丢失内容，质量和写入记录仍为正。** 4节点公共小图，每个输入mass=2^-149、content=[.25,.25]。旧strict Python和native均完成，但Aggregate的`sum(m*z)/sum(m)`返回内容[0,0]，KV保留正质量和一次写入，写入keys为零。已有Python positive-floor-v1的缩放均值返回[.25,.25]，KV keys非零。这里不是新增修复，而是证明已接受的稳定聚合范围尚未覆盖strict/native；补一个支持bool无法恢复内容。
2. **Describe前置cast能静默改变Full支持。** 同一个公共CPU小图、FP64输入mass=1e-46、proportional发送：scoring_dtype=float64时四个节点均active，control=1e-46；显式float32评分时四个节点均inactive、control=0。两次都执行完成，KV仍各有一次写入。后置control→payload支持检查没有报错，因为支持在Describe临时视图转换时已丢失。该混合精度见证不解释本轮三个CPU纯FP32失败，也不宣称两个precision配置原本获得了严格等价资格。
3. **前向通过而Bern反向斜率静默丢失。** 公共scalar Bern slack，scores=[90,0]、caps=[1,1]、tau=1、loss根=[约1e30,0]：controls=[1,.5]，返回score梯度[0,0]；独立100位公式的第一项约8.1940127473e-10。当前对向sigmoid(-90)提前归零。这是合成局部VJP见证，不证明实际三条训练的梯度在此前已发生该错误。
4. **复合表达式的最终值可表示，不等于中间概率可表示。** KV AttentionMean所用表达式的独立例子：logits=[0,-120]，values=[0,约1e30]；FP32 softmax尾部为零，输出0，100位复合值约7.6676481891e-23。只验证公式风险，未捕获训练中的该例。已有positive-floor-v1的forward也没有承诺任意分差/值域的尾部比例。
5. **配置正性与允许衰减须单列。** KVConfig接受host epsilon=1e-50，但其FP32表示为0；本例只验证配置/转换，不新增完整图结论。exp(-200)在FP32归零而FP64约1.38e-87；现有正支持契约明确允许短通道的这种衰减，要求raw KV项、时间和未衰减通道保持。

## 链路审计表

“无专门检查”是本次所读代码中的局部事实，不声称所有上层/后端都没有验证。这里只审计固定Python eager/native主要weighted路径；resident、CUDA/NPU执行、通信后端冲零与所有未来组合未作动态覆盖。

| 环节/量 | 正性条件与零的含义 | 现有检查或消费 | 本次结论 |
| --- | --- | --- | --- |
| 外部/消息Payload `(m,z)` | 原始payload允许非正m；numerical使用m+。z可正可负可零 | encode/decode主要校验metadata；m*z无正支持守卫 | 不能把所有payload数值零改成最小正数；输出存在性由记录表达 |
| Aggregate质量与内容 | 至少一个正m时总质量应正；内容可能合法抵消为零 | strict先乘后除；检查total>0后进入正常分支，不检查内容是否中间丢失 | 公共图已证静默丢失；Python floor模式已有scaled mean |
| Describe临时content/state cast | 原始正质量可能在低精度临时视图变零 | mapper/placement_tensors无质量支持前后对照 | 公共FP64→FP32图已证静默Full支持改变；后置守卫无法倒查 |
| Describe received/caps | r=min(m,1)，gamma>=1，mu=min(gamma*r或gamma*m,1) | 对已经为零的输入按零处理 | 正常可表示输入下gamma不会令其变小；主要审查上游丢失/后端冲零，不能见乘法就认定有风险 |
| 区域总量rho/gamma | sigmoid数学在(0,1)；gamma有1的基底，B有low的基底 | 有限/区间检查；无“sigmoid不得到端点”守卫 | rho=0可使B取low但不自动使B=0；端点/梯度问题须与正支持分开 |
| Bern/Shannon分配控制 | 正cap普通分支要求正；zero/full单点分支独立 | scalar/batch均检查；native抛结构化AllocationFailure | 实际两类失败；strict拒绝保留，Shannon floor仅已有明确范围 |
| quad/Tsallis/硬发送 | 零可由公式、面条件、Top-k合法产生 | quad/Tsallis仅其正支持面检查；硬发送允许零 | 禁止全局floor；有限浮点评分在到达守卫前的边界舍入仍需另界定 |
| attention份额与发送 | 正base⇒p>0；direct再乘mu，pooled再乘总量因子 | shares和最终control各有守卫；proportional也有守卫 | direct实际例证明两处检查不可互相替代 |
| control cast / Full payload | active按control>0；Full将c与内容拼接，物理输出边保留 | 控制转回payload后比较active集合 | 已覆盖后置cast；不证明更早cast没有丢失，也不替代稳定内容计算 |
| KV raw写入 | received>0才写入，weight=min(mass,1) | zero received跳过；raw state validate允许[0,1]，未验证丢失历史 | 保留时间/计数不足以保证写入内容没丢失；聚合见证已显示 |
| KV短衰减与内部attention | 正raw weight经exp衰减/乘积可变零；有限logits的softmax尾部数学为正 | Read筛选weights>0；内部softmax无逐项正支持守卫 | 衰减归零有既有许可，raw项不应删除；归一化/放大后的误差须另衡量 |
| Affine/Delta/SSD/GDN记忆 | exp(log_rate)、softplus(delta)、sigmoid(write)、retain等有正性；H/Z内容可为零 | 以received决定结构写入；更新公式没有逐门值非零守卫 | 静态风险清单，未捕获真实训练反例；不据此否定已学习候选 |
| 正参数/温度/epsilon | host有限正数不保证目标dtype转换后仍正 | 分配temperature的tensor验证较早；normalization epsilon仅host配置约束 | epsilon转换局部见证；应在使用dtype确定后验证必要分母/尺度 |
| VJP与loss/readout | 梯度一般有正负/合法零；局部正斜率可能先丢失再乘大伴随 | 若干VJP只查非有限，零通过；Shannon floor有专用log收缩 | Bern静默梯度见证；不能给零梯度统一补正数 |

主要固定来源：[分配与响应](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/cpp/src/weighted_regularizers.h)、[native发送](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/cpp/src/weighted_sending_batch.cpp)、[Describe placement](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/placement.py)、[控制cast](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/placement_tensors.py)、[均值](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/mean.py)、[KV](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/kv.py)、[循环记忆](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/recurrent.py)、[SSD](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/ssd.py)、[GDN](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/python/tidegraph/weighted/gated_delta.py)、[已接受策略的边界](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/docs/positive-support-policy.md)。

## 待本线程讨论的修复层次

1. **先修可避免的求值丢失。** 稳定sigmoid及正/负尾部斜率、先归一化再混合、复合VJP缩放/log收缩；先保证最终结果可表示时不因中间量而丢失。这不需要把所有量补成最小正值，也不是本页已实现的修复。
2. **对结构性正支持定义生成—转换—消费的一致契约。** 发送c、正received/cap、raw KV写入质量和决定Full的cast应覆盖上游来源；守卫应知道该量来自正支持分支，不能只看到已变零的值。默认strict继续拒绝；若扩展显式有限正支持模式，应在规定边界储存最小正次正规数，检查cap/总量/补入相对质量，并配套各族声明VJP、聚合/记忆和checkpoint身份。Bern不能借用Shannon的VJP公式。检查可以批量执行，不要求正式训练逐算子同步到CPU。
3. **连续衰减、内部概率、门值与梯度单列误差政策。** 对短衰减已有允许舍入的范围继续保留；其余先定义最终复合函数误差及观测，不要求每个指数或梯度永不为零。合法稀疏、窗口淘汰、掩码与真实零不补值。调温度/评分归一化是另一个训练配方，不冒充数学相同的数值修复。
4. **若需要保留任意小质量的比例，则讨论log/缩放质量表示。** 单一bool只能保留存在/激活信息，无法恢复质量、加权内容和导数；最小值存储也会把不同尾部压成相同质量。全链log表示影响组合、持久状态和成本，比有界floor扩展大，尚无本次实现/性能证据。

建议先对齐第1层和第2层的具体边界，同时明确第3层的许可；第4层作为“尾部比例确实影响目标”时的备选。不承诺有限dtype可处理任意长链或任意尺度。三条路径解除本次守卫后是否还会遇到下一处问题、是否改善学习、NPU/native新方案的梯度/恢复，均未检查。上游正式需求待讨论后再形成。

## 复现与查看

Lens项目`tide-learning-lab`，组`TL-POSITIVE-v6`，四个名称：`tl-v6-positive-direct-received`、`tl-v6-positive-regional-total-bern`、`tl-v6-positive-constrained-bern`、`tl-v6-positive-direct-boundary-context`。四次最终投影均verified/readback_matches；direct首次投影因本地下游控制台日志文件名冲突而在0事件时失败，原报告保留，使用新报告名重试。不是训练/数值失败，私有实例入口仍仅在`.local/LENS.md`。

重放源码固定a87bc30；边界前向固定7e1142d。运行命令、原输入hash、预算与service在`artifacts/jobs/v6-positive-*/job.json`，项目材料在对应`runs/`目录。原训练目录均只读。检查：`python scripts/job.py status artifacts/jobs/v6-positive-direct-received`；日志：同命令将status改为logs。任务已停止，无待排队项；如重现，必须选择新job/output，复用原配置与checkpoint hash，禁止覆盖本次证据。

有限局部分析入口：激活`.venv-positive-floor-v1`后运行`python scripts/probe_positive_formulas_v6.py --root "$PWD" --output docs/evidence/positive-path-v6-local-analysis.json`。它只读取原运行和公共安装、更新独立review证据；不执行训练或修补。源文件hash随结果记录。

记录与页面验证：四个诊断脚本语法检查、研究图14项检查、review写入隔离2项检查、确定性渲染核对及Firefox离线交互检查通过。a87bc30/7e1142d两个冻结源码目录内容再次核对未变。上述检查认证诊断材料/展示，不升级为数学、修复、恢复或学习通过。
