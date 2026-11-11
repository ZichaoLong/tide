# TL-REQ-005：正量链路的稳定前向、声明反向与有限正支持

提出方：`ZichaoLong/tide` / `tide-learning-lab`，协议 `tide-downstream-contract-v1`。
类型：数值求值错误、正支持能力、转换检查及反向接口边界。
状态：**用户已对齐方案并授权向基座提交；待基座回复，未实施、未交付、未下游验收。**
2026-11-11授权仅在本轮提交需求及基座目录交接；不在实验分支复制执行核心，不接入正在另线程推进的性能优化。
权威请求为本文件；基座本地 `docs/tide-learning-lab-TL-REQ-005-handoff.md` 仅作接续入口。

## 目的与固定证据

服务于自然文本next-token质量与规模化、实际稀疏、局部结构的联合研究。
小型HB40覆盖中三项发送候选真实失败；另有无异常但丢失内容、支持或梯度的独立见证。
请求修复可避免的中间求值损失，并扩展显式正支持策略；不把FP64、消除所有浮点差异或全平台矩阵作为学习门槛。
只有依赖这些路径的工作受阻；本请求不自动恢复旧训练或扩大新实验预算。

- 数学：`0770593838a37c790a568226930567ed60f99b83`。
- 被测基座及采用的协议提交：`74e7bbd62341d2ccd382194a2c52ad4fea033665`，无下游核心补丁。
- 证据提交：`d6582a172c3d76f26e67d9c4824b11fc619d4b36`。[报告](../studies/positive-path-audit-v6-results.md)、
  [100位Decimal/实际完整行/公共接口见证](../evidence/positive-path-v6-local-analysis.json)、
  [来源、作业、逐文件hash和恢复检查](../evidence/positive-path-v6-review.json)。
- 实际训练源码：`c4585715b3ffe3aa7e7bd88043ad26897ed76df5` / `4087310162338e4f4e90102989cd2c9b98db62b9`。
  诊断源码：`a87bc3019f407fbc3a88ac96fcc60ba35a0d2f95` / `7e1142df5225efc3fd5f1e9b3648e80544079d60`。
- 安装：私有 `.venv-positive-floor-v1`，Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post3、CANN9.2、aarch64。
  三次真实失败均为CPU1/FP32/native；局部公共Python scalar FP32亦拒绝、FP64行级结果与独立Decimal相符。
- `dependencies/positive-floor-v1/foundation.lock.json` SHA256：
  `ba5464d7e23fe6b8c7453a52bfc7b961f040c92d0b80786c18132e40d90308b7`。
  安装锁：`aaa03d8367a04080f9852ba883dc3c03c41f8946a95d99fd017c52e96ca86b45`；
  环境锁：`3662d5f7befd9f1a7aa5da6e0f6d65337ad3daa05d58a1a6854f10962e8ec30b`。
- native来自同一74e7bbd的独立构建；binary SHA256
  `27ecaa1315d250d9ab2de2a42f2a2487dc4d8225d739c61ffb333b7d66b08289`，
  manifest SHA256 `34d92b66394bad569838606230e3412938a074c06f99871a50a44a07c2b4bbcd`。
  来源锁中native字段为null，不能由该字段单独推断native身份；实际运行与构建证据另载上述hash。

不是对基座当前HEAD的回归判定。提交时基座另线程已有后续工作；请先对照当前实现与本固定反例，复用已解决部分。
TL-REQ-003已有策略限于Python eager/KV/Shannon，不能外推native、Bern或direct；本请求不撤销其限定验收。
TL-REQ-004的批量/性能工作独立，不以逐算子host同步或全量日志实现本需求。

## 真实失败及独立见证

三项均为HB40/width32/region4，语言覆盖配置，全局B8/L32，具体配置、数据及执行身份在各运行manifest。
运行入口：`configs/studies/language/v6/cover-hb-direct-received.json`、
`cover-hb-send-regional_total-bern.json`、`cover-hb-send-constrained-bern.json`（同目录）。

| 原运行 | 实际失败边界 | 独立结果与分类 |
| --- | --- | --- |
| `v6-cover-hb-send-constrained-bern` | 更新129后validation；time73/nodes20–23；slack；tau=.25 | 第2项cap1、x/tau=-91.8577576；c≈1.2784471e-40可表示，当前sigmoid提前归零 |
| `v6-cover-hb-send-regional_total-bern` | 训练前向142；time17/nodes24–27；equality；tau=.25 | 第0项cap≈1.0963972e-9、x/tau=-89.4886856；约束解c≈1.4980425e-48；稳定sigmoid后最终乘积仍不可表示 |
| `v6-cover-hb-direct-received` | 训练前向75；native局部候选batch row5/col0 | cap≈3.9022371e-21、share≈3.6860241e-26均为正规正数；乘积≈1.4383740e-46，最终发送归零 |

FP32最小正次正规数为2^-149≈1.40129846e-45，舍入到零半单位为2^-150；不是finfo.tiny。
Bern两行原残差均0、容差3.0517578125e-5，说明单看残差不能发现支持损失。
direct保存的是实际max-centered scores；无全局节点/事件时间和绝对未居中score，不能把局部row5当节点5。
小received更早的全图来源链尚未完整追溯。

额外见证的范围必须保留：

1. 公共4节点图，mass=2^-149、content=[.25,.25]：strict Python/native执行完成，Aggregate内容和KV key变零，正weight/写入次数仍存在；已有Python floor模式的scaled mean保持内容。
2. 公共图FP64 mass=1e-46，Describe scoring_dtype=float64时4节点active，转float32时全部inactive，两者执行完成。前置cast无支持对照，末端control cast守卫无法倒查。不是上述纯FP32失败的原因。
3. 公共Bern slack scores=[90,0]、caps=[1,1]、tau1、loss根[约1e30,0]：前向[1,.5]，score梯度0；独立第一项约8.1940127e-10。
4. 内部AttentionMean表达式logits=[0,-120]、values=[0,约1e30]：FP32输出0，独立复合值约7.6676482e-23。这是表达式见证，未捕获为训练事件。
5. KVConfig(epsilon=1e-50)通过host正性验证，但FP32转换为0。这仅证明配置转换缺口。
6. exp(-200)在FP32归零；既有契约允许短通道衰减归零，须保留raw KV和未衰减通道，不能归入统一禁止零策略。

本次未证明静默现象在原训练中的频率或质量影响；静态SSD/GDN/门控风险不升级为真实失败。
四项诊断外层完成并捕获原异常，不代表原模型执行成功。六个历史loss/梯度范数/token/LR观察逐值一致，
不宣称完整状态长轨迹独立等价；未新增学习、全模型FP64或NPU修复资格。

## 请求A：稳定公式与完整一阶VJP

以下为用户已对齐的数学义务与实现建议；基座可选择满足相同边界的实现，不要求特定专用核。

### Bernoulli响应、求解和反向一致

普通正cap分支，x=s-eta-lambda、z=x/tau、p=sigmoid(z)、c=mu*p。
稳定响应使用两侧sigmoid，或log(p)=-softplus(-z)、log(c)=log(mu)+log(p)。
求解器响应、残差复算和VJP采用一致原语；不能仅修最终返回而求解仍使用提前归零的响应。

局部斜率a=mu*p*(1-p)/tau，cap斜率e=p，温度斜率b=-a*z。
保存log(a)=log(mu)-log(tau)-softplus(-z)-softplus(z)，避免先形成不可表示a再乘大伴随。
对h非零，用sign(h)*exp(log(a)+log(abs(h)))融合；h=0保留零。归约也需适当缩放/中心化。

激活总量约束时，G=sum(a*g)/sum(a)、h=g-G；ds=a*h、dmu=p*h、dtau=-sum(a*z*h)、dB=G。
用log斜率归一化与中心化根求G，不因sum(a)提前归零丢失KKT耦合。
slack取G=0；适用的shift梯度为-sum(ds)，约束激活时共同shift为连接零。
zero/full-cap及不可微边界沿原声明；零cap规则、None/连接零、共享参数汇总和版本检查保持。
Bern在有限z/正cap下的数学内点不能由舍入后c==cap判成饱和；它不能复用Shannon的c/tau斜率或cap面导数。

### Aggregate与复合读取

M=sum(m)、u=sum(m*z)/M。先按max(m)缩放、归一化再混合内容，输出总质量仍为M。
正质量普通分支：dz_i=(m_i/M)*bar_u；dm_i=bar_M+dot(bar_u,z_i-u)/M；单消息内容均值对质量导数为0。
推广已有稳定均值到适用strict/native路径，按新源码固定身份；不保证任意比例/内容值域都仅靠max缩放解决。

两个m=delta、内容0/1时，质量导数1/(4delta)超FP32，但沿m=exp(s)回到score可为有限值。
现有公开mass伴随不可表示应明确失败，不能裁剪；更大范围融合或log-mass才可能消除该中间边界。

KV读出R=f(T)*sum(p_i*v_i)，f=min(T,1)、T=sum(w)、p=softmax(logits+log(w))。
令h_i=dot(bar_R,v_i-u)，则dl_i=f*p_i*h_i、dv_i=f*p_i*bar_R、dw_i|T=f*p_i*h_i/w_i、dT=f'(T)*dot(bar_R,u)。
T对w的依赖另链回；边界取原声明。融合小概率/小质量与大value/伴随，必要时使用带符号的log乘积与缩放归约。
不能给每个内部概率统一floor；已有floor模式的部分融合VJP不意味着其softmax前向覆盖任意尾部。

## 请求B：显式有限正支持覆盖生成—转换—消费

默认strict保留真范围下溢拒绝；稳定公式可修复可表示结果的提前丢失，原历史失败仍保留。
扩展具名数值策略到本次Bern和attention最终发送，并覆盖实际使用的Python/native路径及消费端。
Shannon已交付范围和待扩展范围分别列，不静默回退Python、不用关闭检查替代能力。

- 数学支持S由分支/原输入确定；稳定求值得到c0后，仅S且c0==0时存该dtype最小正次正规数delta。
  已可表示正数、真实零、zero/full-cap、mask/硬排除等保持。quad/Tsallis合法零不得全局floor。
- 保留原残差和适用的cap/总量边界；增加相对补入质量预算，拒绝全行/弱总量下实质畸变。
  参考已有added<=64*N*eps*max(sum(c0),lower)，但各发送family应采用自身不变量。
  原cap小于目标dtype最小正数时不能同时保正和不越cap，须拒绝或显式改变表示。
- 前向存补值，分配/发送VJP按未补值响应声明，后续函数按已存输入求导；不是普通clamp导数，
  不是用floored c重新构造斜率，也不声称整链等于高精度原实值程序。
- 检查后端实际cast及消费是否冲零；不能因delta常数创建成功就认证所有算子支持次正规数，
  也不能自动替换为更大的finfo.tiny。新增策略/实现/图身份与恢复边界明确。

Attention：p_i=b_i*exp(s_i/tau)/Z，direct c_i=mu_i*p_i。
保存log(p)、log(c)，优先在完整发送边界存储；避免未说明的内部份额和最终控制两次独立补值。
令v_i=g_i*mu_i、G=sum(p*v)，则ds_i=p_i*(v_i-G)/tau、db_i=(p_i/b_i)*(v_i-G)、dmu_i|b=g_i*p_i。
小量乘除与归约稳定求值；b=mu时合并base与外乘根，b=received时保留它对cap等上游的真实依赖。
零base右导数、全零base声明零VJP及温度导数沿原契约，不用支持mask统一切断所有梯度。
unit/pooled/proportional及其他发送按各自范围回复，不把direct cap约束误套到pooled/unit。

Describe前置质量/state视图cast也要对照正支持；末端control cast已有检查但不足以覆盖前置损失。
strict尽早指出转换字段和dtype；显式cast修复需检查扰动/约束。保留质量高精度是另一公开配置/表示选择。
raw KV写入质量、内容、int64时间/次数与Full活动必须一致；仅保存bool或次数不能认证内容保持。

## 请求C：连续量许可、配置检查与可选后继

- rho=sigmoid(a)、B=L+rho*(U-L)和gamma=1+(gamma_max-1)*sigmoid(h)的端点舍入，
  不等于消息消失；修稳定尾部导数，不统一floor门值。
- SSD/GDN的d=softplus(a)、q=w*exp(theta)*d、r=exp(-q)、beta=w*sigmoid(b)按复合更新检查。
  dr/dtheta=-q*r、dbeta/db=w*sigmoid(b)*sigmoid(-b)等需避免提前丢失后再乘大状态/伴随。
  本项为静态风险及定向检查要求，不要求本轮扩成所有局部语义动态矩阵。
- 短时间衰减归零的原许可保留，raw KV不删除。若请求更强复合读出保证，显式说明衰减/读取融合范围。
- epsilon、tau等必要分母/尺度在实际计算dtype确定后验证有限且>0；不能只依赖host配置正性。
  不偷偷将epsilon替换为delta或改温度/score归一化来冒充同数学修复。
- log-mass/缩放质量是有条件的后继：ell=log(m)、log(M)=LSE(ell)、alpha=exp(ell-log(M))；
  对内容均值dell_i=alpha_i*dot(bar_u,z_i-u)。若M也输出，其梯度依赖另计。
  真零用独立支持表达；涉及载荷、记忆、图/检查点和反向边界，需要独立设计与交付。
  当前不要求先完成它才能交付A/B，但不能声称bool/floor保存任意小质量比例或解决所有中间伴随。

发送行总补入质量小不保证下游误差小：某接收点若只汇聚两个被floor的尾部，原比例可能变成1:1。
记录修复数量/位置、补入相对量和必要机制观察，不能用全局added很小认证原模型质量不变。

## 可复现入口和材料保留

实际完整局部行已入Git中的 `docs/evidence/positive-path-v6-local-analysis.json`，SHA256
`6dc7989a6e15a18e4591670ba6d0140e95840431c63d6532e2c26d0164406821`。
独立行级复现不依赖大型训练checkpoint；Bern取`actual_bern.*.captured.tensors`，direct取`actual_direct`。
公共接口为`tidegraph.weighted.allocate`与`tidegraph.weighted.sending.attention_send`，保持真实dtype/字段，不导入benchmark。
普通CPU FP64行级检查可以作为参考，不是正式训练的强制dtype。

本机根目录 `/mi/data2T/zlong/home/llm/tide-learning-lab`；新的复现输出不能覆盖现有证据：

```sh
# 在上述固定私有安装与原依赖激活后；只运行小型局部分析，不续训。
python scripts/probe_positive_formulas_v6.py --root "$PWD" \
  --output docs/evidence/tl-req-005-local-replay-new.json
# 只读查看原诊断作业身份、终态与精确命令。
python scripts/job.py status artifacts/jobs/v6-positive-direct-received
python scripts/job.py status artifacts/jobs/v6-positive-regional-total-bern
python scripts/job.py status artifacts/jobs/v6-positive-constrained-bern
```

原完整重放配置在`configs/studies/language/positive-path-v6/`；精确argv/env/source/输入hash在
`artifacts/jobs/v6-positive-*/job.json`，诊断入口为`diagnose_positive_paths_v6.py`、`capture_sending_boundary_v6.py`。
原训练65/135/125步checkpoint起，最多16次更新；额外一次捕获边界native前向。
原预算3个CPU1作业各30min/8GiB，局部补证合计15min，合计<=1.75 CPU核时、输出<=1GiB；
已用约0.2395 CPU核时、约49MiB运行材料。新复现使用新作业/输出并单独扣账，本次未提交新计算。
保护三项原训练的恢复输入、四个`runs/v6-positive-*`、相应固定源码/安装/native和小型原始失败记录，
直至上游取得独立复现或明确退役；另行磁盘审计不删除这些输入。

## 有界验收与交付

1. 三个真实局部行：constrained-Bern可表示尾部经稳定求值保持正；两个真范围下溢在strict仍拒绝，
   新显式策略按支持/cap/总量/扰动契约完成，原失败保留。支持mask/消息/Full检查不由loss下降替代。
2. 独立前向/一阶反向：Bern slack/equality/lower/upper及端点，score/cap/tau/target/shift；
   direct的base/cap/received依赖、零base右导数、全零分支；大伴随和双侧尾部、合法零、边界选择。
   使用Decimal/独立闭式公式；普通尺度gradcheck。floor处不能用clamp的有限差分认证原函数声明VJP。
3. 公共小图：极小mass的Aggregate内容/KV key保持，cast不静默改支持，raw/int64/窗口规则、
   共享根汇总、None/连接零、输入/状态梯度、optimizer槽及新进程精确恢复。不可表示公开伴随明确拒绝。
4. Python/native CPU FP32先覆盖真实失败；CPU FP64局部独立参考；NPU FP32按实际拟交付路径做有限真机验证，
   记录subnormal、cast、反向/恢复能力。CUDA/resident/standalone等未检查组合不冒称通过，也不要求全矩阵前置。
5. 执行、严格数值、离散语义、恢复、学习分别报告。保留原atol1e-6/rtol1e-5；有限超差经有界定位与机制检查后
   可继续明确标记学习探索；非有限、错误支持/共享/状态/恢复仍阻塞相应路径。无需逐正式训练步跑CPU参考。
6. 提供实现/证据提交、实际构建/安装锁、公共参数与能力拒绝、数值/图/检查点身份、同模式恢复及跨模式迁移说明。
   新模式不改旧checkpoint身份，不把旧权重引入新运行称作原轨迹精确续接；必要观测可批量收集，不逐算子同步。

基座回复：**待回复**。下游验收：**待交付后在新私有安装中实施，当前锁不变**。
交付可分A稳定求值/B正支持/C许可与边界，不以可选log-mass设计阻塞已具备前提的修复。
本请求由用户明确授权提交至本机基座目录；不自动发送外部issue、推送基座分支或启动上游实现作业。
