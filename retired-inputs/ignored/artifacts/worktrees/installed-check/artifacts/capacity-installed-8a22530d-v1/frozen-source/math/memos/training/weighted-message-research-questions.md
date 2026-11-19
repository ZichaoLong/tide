---
type: training-research-memo
status: optional-research-questions
tags:
  - tide
  - training
  - experiments
---

# 带权消息架构：可选研究问题与对照

本文保存按需取用的问题、观测和训练目标。整体设计的阅读入口是[设计总览](../design/weighted-message-design.md)；函数以[局部架构](../../local-semantics/weighted-message/architecture-learning-note.md)、[区域分配](../../local-semantics/weighted-message/region-allocation-learning-note.md)及共同架构列出的记忆教材为准。实现与定义的对应见[实现核对](../../reference/weighted-message-implementation-checks.md)。

当前主干对照从[K／S／X 三组记忆绑定](../../instances/hb-multiscale-weighted-message.md#2-两类记忆与三个主候选)出发，分别是全 KV／sum、全 SSD／state、普通节点 KV 与 hub SSD；共同编码与区域 self_attention 形成评分，分配主对照为 quad、约束 Shannon 与 pool。参数对照收在[independent／region_shared／layer_shared 三档](../../instances/parameter-profiles.md)，尺寸和预算见[主干比较绑定](../../instances/backbone-comparison.md)。本页的更广目录供研究问题需要时选取。

以下条目没有预设执行顺序，也不要求全部实施。研究者掌握设计空间后，根据当前问题、已有结果和资源选择优先级。一个配置失败、难以训练或过于昂贵，本身可成为缩小探索范围的依据；复杂耦合可能使某些机制判断长期只有间接支持。

## 1. 问题与可选比较

沿用教材中的接收强度 $r_v^{\mathrm{in}}$ 与截断接收量 $w_v=\min(1,(r_v^{\mathrm{in}})_+)$。指定总量与区域阈值使用固定最大倍率上限 $\mu_v^{\max}=\min(1,\gamma_{\max}w_v)$；节点增益规则使用 $\mu_v=\min(1,\gamma_v(r_v^{\mathrm{in}})_+)$，其中 $\gamma_v$ 为发送增益。区域评分产生 $s_v$，发送分配产生控制 $c_v$，即本次实际发送强度。下表按计算位置列出可替换部分；每项比较都保留其余函数的明确绑定。名称与配置键的对应见[设计总览 §1](../design/weighted-message-design.md#1-从整体看各个选择)及[区域教材附录](../../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)。

| 当前关心的问题 | 可选比较 | 有用观测与解释边界 |
|---|---|---|
| 候选之间的关系能否改善选择 | 固定 encoded 描述，比较 ind 与 self_attention；也可按问题比较旧 att 的完整绑定 | 重复与互补证据的利用、区域准备成本和端口质量；原 att 的表示也不同，不能只归因于头数 |
| 传播保障是否过度强制放大 | 角色分配 received／mu 下界，固定其余绑定；按需比较 interval／fraction | 弱接收分支的可行余量、真实总量、路径持续性；旧 mu 下界可强制全上限，连续性并不要求它 |
| 共同编码是否形成瓶颈 | 按当前问题调整描述宽度，或与原独立编码组合比较 | 局部判断、关系判断与可选总量证据的相互影响；增宽不等于增加可见信息 |
| 区域适合直接决定总量还是调节阈值 | regional_total／regional_threshold，在相同节点上限、接收下界、评分和正则下比较；各自可选 fixed／context | 控制头与评分的梯度、总量范围、实际总和、饱和分支；两者的零温目标分别声明 |
| 增益是否承担过多作用 | 区域 query 按 $\mu_v$ 或 $w_v$ 汇集；直接 Attention 的份额基权重分别取这两者 | 增益、query、评分和发送的梯度；移除依赖也移除了学习通路 |
| 弱线索唤醒与噪声放大如何取舍 | 直接 Attention、固定总强度 Attention、总量缩放 Attention、约束 Shannon | 接收、写入、发送、旧记忆利用与错误输出；各函数的零边界不同 |
| 稀疏分配能否改善质量与成本 | 保持评分，比较二次正则、Shannon 熵、固定形状广义熵、二元熵的约束分配 | 正控制量的数量、准备成本、Full、下游候选和任务质量 |
| 评分是否需要按当前内容查记忆 | 时间摘要与 KV 内容检索，或时间摘要与循环矩阵查询 | Mem 提供的证据、读取成本及历史使用；各模块的两个描述通道含义按其定义 |
| 本地证据怎样形成评分 | 拼接评分与分组匹配评分，保持记忆读取和区域评分 | 内容、统计量与匹配项的实际使用及尺度；评分、区域证据与增益的函数分别绑定 |
| 区域控制需要怎样的节点证据 | region_evidence=cat／grouped，保持局部评分、区域头形式与发送规则 | 总量／阈值的任务作用、全部候选的描述成本、信息压缩及统计依赖；独立编码与显式共享的反馈分别核对 |
| 是否需要跨候选的内容比较 | 独立评分与区域 Attention 评分，保持记忆和发送函数 | 节点互补性、共同 query 的信息压缩及额外准备成本 |
| 写入统计是否直接参与区域匹配 | region_key=content_stats／content，保持记忆读取、局部评分、query 与发送函数；明确 alpha_loc | 匹配尺度、统计使用与描述成本；同时改变了输入、非线性和归一化，整体差异不能仅归因于移除统计。alpha_loc=0 时仅关闭局部评分通路 |
| 本次写入是否干扰评分 | Read$^+$／Read$^-$，Full 保持读取新状态 | 当前自匹配、历史检索、写入到未来的反馈 |
| 状态压缩是否值得 | KV、基础仿射、基础 Delta、Gated DeltaNet、SSD，或按区域组合 | 任务质量、长程检索、状态更新成本、存储和训练反馈；原模型对应与 Tide 改造分别记录 |
| 卷积、多头和具体门是否值得 | 基础 Delta 与 Gated DeltaNet；SSD 与 Gated DeltaNet；所选模块的受控部件消融 | 同时变化的特征、门、状态和读出须列明；匹配参数量或工作量并记录剩余差别；消融改变函数时单独声明定义 |
| 图的不同邻域是否有用 | 全连接、不同丰富程度 HB、不同区域划分 | 可达信息、父邻域重叠、候选覆盖、广播工作和分工 |
| 是否需要退火或零温部署 | 固定正温、有限降温、从初态求值的配对零温或硬函数 | 评分间隔相对温度、激活集合变化、梯度尖峰、质量与严格容量需求；指定总量按区域分配 §7.2，区域阈值分配、节点增益与固定阈值分配按 §7.4、§7.5 分别确定极限目标与精确相等条件 |
| 位置和记忆范围如何配置 | 相同 KV 窗口下 NoPE／部分／完整 RoPE，再考察异构区域 | 长程任务、外推、读取带宽；分别记录 Read 与 Full 的配置 |

比较以明确的问题为单位。局部分配实验可以固定同一评分、接收、节点上限及总量范围；完整训练会改变消息与历史，结果应按整体方案解释。数据暴露、参数量、总算术工作和设备时间通常无法同时匹配，可给互补比较并记录剩余差别。

准备比较前，先查[全连接实例的单位接收与等值结论](../../instances/fully-connected-layered-weighted-message.md#3-每层的接收写入与输出)以及 [HB 的弱接收算例](../../instances/hb-lattice-weighted-message.md#41-稠密发送的强度传播)，确定所选形状能否区分这些函数。

两个区域控制候选的完整绑定见两份实例 §4.4。全连接单位输入可在 $k>1$ 时比较，$k=1$ 则总量固定；弱接收还需核对 $b^{\mathrm{recv}}$ 与 $U$ 是否重合。若同时比较 constrained，记录其下界按 $\sum\mu_v$ 计算的差异。局部导数与完整训练效果分别判断，不从区域头梯度非零直接推出更好的分工。

## 2. 可选的任务与训练目标

### 2.1 语言模型接入

两个架构实例均给出因果表示 $y_t\in\mathbb R^d$。独立语言模型可用 token 嵌入作为入口内容 $h_t$，声明词表大小 $n_{\mathrm{vocab}}$ 和矩阵 $W_{\mathrm{vocab}}\in\mathbb R^{n_{\mathrm{vocab}}\times d}$。令 $\mathcal T_{\mathrm{sup}}$ 为有限非空监督位置集，$N_{\mathrm{tok}}=|\mathcal T_{\mathrm{sup}}|$，$x_{t+1}^{\mathrm{tok}}\in\{1,\ldots,n_{\mathrm{vocab}}\}$ 为位置 $t$ 的下一 token 标签。沿用局部教材的归一化函数，定义

$$
\begin{aligned}
p_t&=\operatorname{softmax}(W_{\mathrm{vocab}}\mathcal N_d(y_t)),\\
\mathcal L_{\mathrm{task}}
&=-\frac1{N_{\mathrm{tok}}}\sum_{t\in\mathcal T_{\mathrm{sup}}}
  \log p_t(x_{t+1}^{\mathrm{tok}}).
\end{aligned}
\tag{1}
$$

其他任务独立声明读出、监督位置与损失。

### 2.2 稠密暴露与教师

持续稠密的各发送函数可独立训练、推理，也可以作为教师。[按比例分配](../../local-semantics/weighted-message/region-allocation-learning-note.md#43-按比例分配)是另一个已定义参照；它没有分配到评分的直接梯度。是否使用教师取决于当前研究问题。

稠密教师在同一外部序列上运行自己的轨迹与缓存，产生 $p_t^{\mathrm{teach}}$；学生运行自己的轨迹。可先训练并固定教师，蒸馏时停止教师目标的梯度。教师、间歇稠密校准或 shadow Full 的额外工作均计入训练预算，注明参数共享、状态写入和梯度归属。

### 2.3 成本代理与联合损失

记 $\mathcal E_C$ 为学生序列中全部候选的 $(v,\theta)$ 坐标。给节点固定参考成本 $C_v^{\mathrm{ref}}>0$，取 $\epsilon_{\mathrm{cost}}>0$ 和非负系数 $\lambda_{\mathrm{cost}},\lambda_{\mathrm{dist}}$，定义

$$
\begin{aligned}
\phi(c)&=\frac{c}{c+\epsilon_{\mathrm{cost}}},\\
\mathcal L_{\mathrm{cost}}
&=\frac1{N_{\mathrm{tok}}}\sum_{(v,\theta)\in\mathcal E_C}
 C_v^{\mathrm{ref}}\phi(c_{v,\theta}),\\
\mathcal L&=\mathcal L_{\mathrm{task}}+
 \lambda_{\mathrm{cost}}\mathcal L_{\mathrm{cost}}
 +\frac{\lambda_{\mathrm{dist}}}{N_{\mathrm{tok}}}
  \sum_{t\in\mathcal T_{\mathrm{sup}}}
  \operatorname{KL}\bigl(\operatorname{stopgrad}(p_t^{\mathrm{teach}})\,\|\,p_t\bigr).
\end{aligned}
\tag{2}
$$

stopgrad 的前向为恒等，反向对其输入贡献为零。无教师时取 $\lambda_{\mathrm{dist}}=0$ 并省去教师计算。发送强度 $c$ 远大于 $\epsilon_{\mathrm{cost}}$ 时，该成本代理近似计一次；$c$ 趋零时连续归零。它的导数为

$$
\phi'(c)=\frac{\epsilon_{\mathrm{cost}}}{(c+\epsilon_{\mathrm{cost}})^2},\qquad
\phi''(c)=-\frac{2\epsilon_{\mathrm{cost}}}{(c+\epsilon_{\mathrm{cost}})^3}.
\tag{3}
$$

固定发送强度之和与相同参考成本时，凹代理偏好集中分配；较小的正发送强度有较大的成本导数，经共同预算反向后可进一步减小。完全相等的发送强度给出相同成本导数，中心化后评分梯度为零；任务反馈与初始差异仍影响分化。精确零发送强度还可能处于零评分梯度分支。

$\phi'(0)=1/\epsilon_{\mathrm{cost}}$。过小的 $\epsilon_{\mathrm{cost}}$ 会增大近零处斜率，与低温响应叠加；应同时观察温度、成本系数、未触及上下限的候选及实际梯度。该目标表达候选上的激活成本，下游候选存在、邻域重叠和端到端时间按全图测量。引入可微下游成本预测时，另行定义目标、反向与校准依据。

仅使用任务目标、加入成本项、加入教师分别对应不同训练设置。持续稠密时，减小若干正发送强度可能降低上述成本代理，但实际 Full 与消息仍存在，需按真实运行计量。

### 2.4 中间辅助监督

选定前面的一项训练目标，记为 $\mathcal L_{\mathrm{base}}$。令 $\mathcal I_{\mathrm{aux}}$ 为有限辅助观测位置集合；$h_r$ 为学生的中间表示，$h_r^{\mathrm{teacher}}$ 为教师目标，$P_r$ 为到共同表示空间的线性或可学习投影。给定非负权重 $\alpha_r,\beta_r$ 和辅助任务损失 $\mathcal L_{\mathrm{aux},r}$，可定义

$$
\mathcal L=\mathcal L_{\mathrm{base}}
 +\sum_{r\in\mathcal I_{\mathrm{aux}}}\alpha_r
  \|P_rh_r-\operatorname{stopgrad}(h_r^{\mathrm{teacher}})\|_2^2
 +\sum_{r\in\mathcal I_{\mathrm{aux}}}\beta_r\mathcal L_{\mathrm{aux},r}.
\tag{4}
$$

教师生成方式、观测位置及额外工作由所选比较明确给出；没有教师时省去相应项。辅助头可以只在训练时存在。它可能缩短监督距离，也可能限制层间分工；未选硬分支的具体反事实输出仍须额外计算。跨架构的反馈覆盖和干预代价见[学习风险](sparse-routing-risks-and-diagnostics.md#6-可选训练干预及其代价)。

## 3. 训练配置的独立选择

| 选择 | 需要明确的内容 |
|---|---|
| 初始化 | 参数共享、私有差异、残差尺度；适配器两因子同时零初始化会阻断乘积路径的初始梯度 |
| 温度 | 固定或变化的尺度，评分间隔与 Q/K 范数；“中等温度”要相对评分尺度解释 |
| 推理函数 | 持续正温度，或使用指定总量的 regional_total_zero、区域阈值的 regional_threshold_hard、节点增益的 hard_selection；描述与控制头按[配置表](../../local-semantics/weighted-message/region-allocation-learning-note.md#s1-配置名称)配对 |
| 激活数量要求 | 两种硬选择保证至多 $k_j$ 个激活，严格最多 $n$ 个时取 $k_j=K_j=n$；regional_total_zero 保留 $K_j=\lvert\mathcal R_j\rvert$，其总量上限 $k_j$ 一般不限制激活数 |
| 状态管理 | 每序列时钟、重置、打包、反向截断；矩阵与卷积一起继续；参数更新后的缓存保留或重算策略 |
| 附加训练信号 | 教师、成本、辅助目标、探索或指定替代反向及额外计算 |

退火是可选训练操作。正温度精确零、温度极限中的数值收敛、激活数上限与训练效果分别评价。比较正温度函数及其配对函数时，各自从初态运行完整轨迹；中途换函数而保留旧缓存是另一项部署选择。

## 4. 按问题选取观测

| 观测 | 能帮助判断什么 |
|---|---|
| 任务质量，按类别和长度分组 | 某组合是否解决当前任务；极差结果是否值得继续定位 |
| 接收 $r^{\mathrm{in}}$、截断量 $w$、节点上限 $\mu^{\max}$ 或增益相关量 $\gamma,\mu$、控制 $c$、总发送 $S=\sum_vc_v$、候选数 $\lvert C\rvert$、激活数 $\lvert\mathcal A\rvert$ | 幅度放大、传播衰减、约束饱和及实际激活 |
| 模块梯度与更新量 | 参数是否生效、控制是否获得反馈；非零导数与有效更新分别解释 |
| 区域头输出、$b^{\mathrm{recv}},U,B,\eta_j,\lambda$ 与未饱和节点数（$0<c_v<\mu_v^{\max}$） | 区分模型控制量与求解乘子；记录总量范围退化、总和被上下界固定、头输出饱和以及实际反馈覆盖 |
| query/key 范数与评分差相对温度 | 固定温度下的集中、低温尖峰或长期饱和 |
| 写入到读取的时间与反向截断 | 私有记忆是否被使用、是否保留所需信用链 |
| 条件发送份额、记忆使用及内容互补 | 稠密或稀疏模型中的分工；次数均衡无法单独说明分工 |
| Agg、Upd、Read、SelStep、Full、消息、缓存成本 | 当前控制成本与后续节省是否值得，预填充和解码分别记录 |

清零、打乱、冻结或替换某个模块会改变分布；退化说明模型依赖了它，不独立证明该设计优于重新训练的其他模型。一次训练中的相关变化也可能有共同原因。数值异常和定义偏离先按实现核对定位，机制解释保留替代假设。

## 5. 研究记录

一份结果记录问题、完整组合、上游与下游提交、数据与预算、训练和状态策略、随机性、正确性证据、主要观测与解释。原始日志和设备测量由下游维护。

结论可以是保留、缩小范围、停止尝试或继续观察，也可以只更新当前偏好。标出数学保证、外部依据、当前实测和尚未区分的解释即可；无需把每个直觉都转成预先承诺的独立实验。具体实验安排由研究者另定。
