"""Screen text, speaker notes, sources and stable stop IDs for the Tide deck.

Narrative: Transformer -> explicit residual MoE -> local network -> history.
The new numeric binding is independent of the retained HB topology example.
"""
SOURCES = {
    'design': ('研究意图与探索路径', '../../memos/design/design-requirements-and-architecture.md'),
    'moe': ('显式残差 MoE 教学绑定', '../../instances/explicit-residual-moe.md'),
    'history': ('MoE §4：本地激活计数', '../../instances/explicit-residual-moe.md#4-单独增加本地历史'),
    'hb': ('HB 实例 §2', '../../instances/hb-lattice-weighted-message.md#2-三层图上的分叉汇合与记忆'),
    'topology': ('HB 拓扑', '../../topologies/hb-lattice.md'),
    'framework': ('TimedDAG §4—5', '../../timed-dag-region-selector-learning-note.md#4-节点的局部函数'),
    'chunk': ('分块教材 §7.1—7.3', '../../timed-dag-chunk-prefill-learning-note.md#71-串行-transformer-block'),
    'kv': ('KV §4.2：节点内残差', '../../local-semantics/weighted-message/kv-memory-learning-note.md#42-内容函数'),
    'agg': ('带权聚合 §2.1', '../../local-semantics/weighted-message/architecture-learning-note.md#21-来源分类与聚合'),
    'memory': ('带权架构 §3：Mem 与描述', '../../local-semantics/weighted-message/architecture-learning-note.md#3-状态与记忆描述'),
    'cost': ('带权架构 S.2：工作量', '../../local-semantics/weighted-message/architecture-learning-note.md#s2-工作量'),
    'choices': ('局部设计总览', '../../memos/design/weighted-message-design.md'),
    'lm': ('研究问题 §2.1：语言模型接入', '../../memos/training/weighted-message-research-questions.md#21-语言模型接入'),
    'review': ('既有实现审视 §3.3', '../../records/reviews/2026-10-28-ascend-implementation-review.md#33-常驻设备路径保留数学稀疏但仍执行填充行上的算术'),
    'layered': ('全连接实例 §2', '../../instances/fully-connected-layered-weighted-message.md#2-两层图上的两次输入'),
}


def step(id, title, lede, view='html', body='', figure=None, notes='', **kw):
    return dict(id=id, title=title, lede=lede, view=view, body=body,
                figure=figure, notes=notes, **kw)


def callout(text):
    return f'<div class="callout">{text}</div>'


def table(headers, rows):
    return '<table class="comparison"><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


SLIDES = [
    dict(chapter='01 · 从 Transformer 出发', status='结构示意 · 粒度可选', sources=['chunk','kv','moe'], steps=[
        step('transformer', '先看 Transformer：节点边界可以放在哪里？',
             'Attention、前馈计算与 residual add 都要有明确归属。', 'transformer', preview=True,
             notes='以 pre-norm Transformer block 为熟悉的起点；原论文的 post-norm 把 Norm 放在 Add 后，残差职责仍需保留。一个节点可以包住整个 block，也可只表达 Attention 或一个 MoE expert。完整 block 包为节点最直接，但是否与现成 Transformer 数值等价，还要匹配 Norm、Attention/KV、参数、选择与输出边界。图中的两个 Add 都真实保留；继续细化时要重新标明它们的位置。专家粒度将在下一页展开。')]),
    dict(chapter='02 · 展开 MoE 的职责', status='新增教学绑定 · 使用现有接口', sources=['moe','framework','chunk'], steps=[
        step('moe-roles', '专家贡献，与残差基线分开表达', '把前馈支路展开：选择专家 → 加权贡献 → 合并 → 加回 x。', 'illustration', figure='moe-roles',
             body='<div class="eyebrow">保留原计算的含义</div><p><b>专家：</b>Eᵢ(x)<br><b>控制：</b>选谁、给多少权重<br><b>合并：</b>r = Σ aᵢ Eᵢ(x)</p><div class="key">y = x + r</div>'+callout('基线 x 沿旁路只加入一次。')+'<p class="small">Eᵢ(x) 可包含本支路的 Norm。</p>',
             notes='这里展开的是 FFN/MoE 子层，x 是该子层的残差基线。专家函数可以包含本支路的 Norm；数值示例稍后以简单仿射函数替代它。专家计算、router/selection、加权合并、残差 add 是不同功能。蓝线为数值载荷，金色虚线为区域控制，深色旁路携带 x。局部节点边界并不规定整个网络必须围绕唯一残差主干。'),
        step('moe-mapping', '每个专家一个节点，选择属于区域', '五个节点：分发 S、三个专家、合并 M；三个区域各自完成选择。', 'illustration', figure='moe-mapping', preview=True,
             body='<div class="eyebrow">映射到 Tide</div><p><b>专家区域：</b>SelStep 返回 A、aᵢ<br><b>专家节点：</b>Full 发送 aᵢEᵢ(x)</p><p><b>M 的 Agg：</b>分别取得 x、r<br><b>M 的 Full：</b>执行 x + r</p>'+callout('旁路时延 2 = 专家路径 1 + 1。'),
             notes='三个区域分别是 {S}、{E1,E2,E3}、{M}；容量 1、2、1。S 和 M 的单节点区域遇到输入必选，图中省略其平凡选择器。router 在这套绑定中是专家区域的 SelStep，不另计一个节点。M 的 Agg 根据来源边分开基线与贡献，并对贡献求和；M 的 Full 做 Add。源节点向三个候选及 M 各发一条消息；未选专家不发送。先取单点状态，即无有效持久状态。')]),
    dict(chapter='03 · 完成一次计算', status='数值事实 · 标量教学例', sources=['moe'], steps=[
        step('moe-select', '输入 x = 2，先选两个专家', '评分为 (x, 1, 0)；Top-2 后，在所选集合上归一化 2 的评分次幂。', 'illustration', figure='moe-select',
             body='<div class="eyebrow">θ = 2 · 专家区域</div><p>E₁(x) = 2x<br>E₂(x) = −x<br>E₃(x) = x + 5</p><div class="key">评分 (2, 1, 0)</div><p>选择 E₁、E₂<br>权重 (⅔, ⅓, 0)</p>'+callout('先看候选描述，尚未计算专家贡献。'),
             notes='时间 1 的 S 已分发 x=2；专家收到值后产生评分。按评分降序取两位，同分时编号小的在前。所选权重为 2²/(2²+2¹)=2/3 和 1/3。E3 收到候选输入、计算描述，却不执行 Full。评分是教学给定函数，未用训练得到。S 到 M 的旁路在时间 3 到达。'),
        step('moe-contributions', '两个专家，给出不同的非零贡献', '先乘控制权重，再发送；未选专家没有贡献消息。', 'illustration', figure='moe-contributions',
             body='<div class="eyebrow">θ = 2 · Full</div><p>E₁(2) = 4<br>⅔ × 4 = <b>8/3</b></p><p>E₂(2) = −2<br>⅓ × (−2) = <b>−2/3</b></p>'+callout('E₃ 本次没有执行。<br>专家消息中不含残差基线 x。'),
             notes='E1、E2 的贡献分别为 4 和 −2，加权后产生两条内部消息。E3 的函数虽然有定义，本次不求值也不发零占位消息。图中灰色虚线只表示静态上存在的未用边。当前蓝色输入边仍包括三个候选，因此减少专家 Full 不等于省去全部候选输入传播。'),
        step('moe-result', '贡献合并，再加回基线：y = 4', 'θ = 3，旁路与两份贡献在同一逻辑时间到达 M。', 'illustration', figure='moe-result', preview=True,
             body='<div class="eyebrow">一次计算已完成</div><p>r = 8/3 − 2/3 = <b>2</b><br>基线 x = <b>2</b></p><div class="big-result">y = 4</div>'+callout('M 直接给出端口值。<br>图外不再重复加 x。'),
             notes='M 的完整时间纤维包含 S->M、E1->M、E2->M 三条消息；按来源取得基线 2、贡献和 2。未选 E3 没有消息，不是必须等待的一个零值。节点 Full 输出 4。这是通用 TimedDAG 下新定义的求和绑定，不能直接套用带权消息的归一化均值。')]),
    dict(chapter='04 · 从一个小组走向网络', status='已有拓扑＋设计要求', sources=['topology','hb','design','moe'], steps=[
        step('local-network', '多个局部选择组，怎样连成网络？', '这里切换到 HB 拓扑示意：节点的具体功能另行绑定。', 'illustration', figure='hb-neighbors', preview=True,
             body='<div class="eyebrow">两项约束一起检查</div><p>每区只协调本地候选；<br>每节点只连有限邻居。</p><p>避免所有内部计算<br>共同依赖固定主干<br>或单一决策中心。</p>'+callout('允许经过全部区域；<br>区内仍可稀疏参与。'),
             notes='这张图改用已有 HB 的 4 区域、12 节点、24 边，蓝色仅突出 I0 的四个邻居，没有表示一次新运行。前页的 S/E/M 五节点模型不是 HB 的数值绑定。单个 MoE 小组有局部 router 不能证明整网去中心化；直接扩大同一组会增加输入分发、候选协调和合并扇入。扩展时须声明多个有界组及组间连接，逐一检查实际依赖。局部残差仍可使用，不要求所有路径经过唯一内部残差主干。')]),
    dict(chapter='04 · 边界与规模扩展', status='有条件的结构推论', sources=['design','topology'], steps=[
        step('bounded-propagation', '固定端口下，逐步扩散与收拢', '端口数量固定、直接通信对象数有界，而可达内部容量继续增加。', 'expansion',
             notes='同时固定输入和输出端口数、每个节点的入出通信对象数上界，要求更多节点结构上能受输入影响且影响输出，则输入侧和输出侧的最大路径长度均须增加。有限跳数只能覆盖有限节点。图是多跳结构示意，不要求每次输入访问所有区域或所有节点。不同模型可有不同有限端口数，每个固定配置分别讨论。外部端口本身可以汇总，不因此免除内部固定主干的检查。无界端口仅作范围外答疑，未在本方案中引入其适配机制。')]),
    dict(chapter='05 · 再加入历史', status='新增扩展 · 节点持有激活计数', sources=['history','framework'], steps=[
        step('history-first', '先走无状态计算，再明确增加什么状态', '新增 qᵢ：该专家此前被激活的次数；选择改看 ℓᵢ(x) − 3qᵢ。', 'illustration', figure='moe-history1',
             body='<div class="eyebrow">第一轮 · x₀ = 2</div><p>旧计数 q = (0, 0, 0)<br>评分仍为 (2, 1, 0)</p><p>选择 E₁、E₂ → y₀ = 4</p><div class="history">新计数 (1, 1, 0)</div>'+callout('Next 根据激活指示加 1；<br>不需要读取 Full 输出。'),
             notes='明确切换到有状态扩展。各专家节点持有自己的非负整数计数，初态零。Upd 保持旧计数，Read− 用旧计数修正评分，Next=q+激活指示；区域历史依旧单点。专家内容函数继续是之前的无状态仿射函数。系数 3 只是让手算变化可见的教学参数，不是已验证的负载均衡方案。第一轮输出与无状态版相同。'),
        step('history-second', '再次输入 2，历史让选择改变', '固定当前输入，只观察本地历史的作用。', 'illustration', figure='moe-history2', preview=True,
             body='<div class="eyebrow">第二轮 · x₁ = 2</div><p>评分 (−1, −2, 0)<br>选择 E₃、E₁；权重 ⅔、⅓</p><p>r = ⅔ × 7 + ⅓ × 4 = <b>6</b></p><div class="big-result">y₁ = 8</div><p class="small">重置计数重算本轮：y = 4。</p>',
             notes='第二轮在时间 4 注入，5 做专家选择，6 输出。旧计数 (1,1,0) 将原评分 (2,1,0) 改为 (−1,−2,0)。E3 的贡献为 7、E1 为 4，加权和 6，输出 8；新计数 (2,1,1)。重置计数后同输入仍选 E1、E2，输出 4，直接验证历史在此模型中的作用。输入表示 x=2 不等于任何持久计数。输出变化没有优劣含义，任务收益仍需验证。')]),
    dict(chapter='06 · 回看共同接口', status='定义事实 · 阶段与功能两种划分', sources=['framework','history'], steps=[
        step('phases', 'P / S / U / F 是作用阶段', '专家、router、合并、Add 是功能；它们与阶段没有一一对应关系。', 'illustration', figure='moe-phases',
             body='<div class="phase-stack"><div><b>P</b><span>聚合输入，形成候选状态<br>Read 取得选择描述</span></div><div><b>S</b><span>区域 SelStep<br>给出激活集合和控制</span></div><div><b>U</b><span>确定计算快照<br>Next 确定下一持久状态</span></div><div><b>F</b><span>激活者执行 Full<br>沿边或端口输出</span></div></div><p class="small">M 也经过这些阶段：<br>其 P 求贡献和，F 做 Add。</p>',
             notes='三个区域各自执行 P/S/U/F，而不是分别充当某一个阶段。S 节点的功能名表示分发节点，不能与 S 选择阶段混为一谈。专家 P 读取输入和旧计数，区域 S 选择，U 使本次计算快照保持旧计数但下一持久计数加激活指示，F 才求专家贡献。M 的 P 从输入原子中分类求和，单节点 S 必选，U 保持单点状态，F 执行 Add。Full 不回写状态，Next 不读取或重算 Full。')]),
    dict(chapter='06 · 历史怎样变成选择依据', status='一般接口／带权支线', sources=['framework','memory','choices'], steps=[
        step('memory-interface', 'Read 负责描述；Mem 是一条具体实现路径', '当状态从一个计数变成 KV 或循环矩阵，就需要说明如何读取有用信息。',
             body='<div class="memory-comparison"><div><div class="eyebrow">刚才的计数例子 · 一般框架</div><h2>本地 qᵢ → Read⁻ → 分数</h2><p>读取旧激活计数，结合当前 x。<br>无需额外的 Mem 模块。</p></div><div><div class="eyebrow">已有带权消息支线</div><h2>记忆 → Mem → Describe</h2><p>Mem 取得记忆描述；<br>Describe 结合当前接收，交给区域选择。</p></div></div>'+callout('接口要说明读的是旧状态还是候选新状态，也要计入查询与描述成本。')+'<p class="small">计数、KV 与循环记忆承载的信息不同；替换时重新检查状态、签名和组合条件。</p>',
             notes='计数例子使用原生 Read−，未调用带权 Mem。既有带权架构将 Read 实现为 Mem 与 Describe 的组合；Mem 提供两个向量和两个统计量，具体含义依模块而定。KV 与循环矩阵有不同的存储与查询成本。Read0、Read−、Read+ 选择是否以及何时读取状态；当前计数例子的 Upd 恒等，因此 Read− 与 Read+ 恰好相同，这是例子性质。原 HB 使用接收后写入的 KV/Read+，不将其历史悄悄混入本例。')]),
    dict(chapter='06 · 从绑定回到架构', status='定义事实＋设计解释', sources=['framework','moe','choices'], steps=[
        step('architecture', '共同框架留出位置，完整绑定填入规则', '局部函数可以不同；边界、状态与依赖必须能接起来。',
             body=table(['层次','本次例子如何填入','更改时要检查'],[
                 ['共同框架','节点、区域、状态；P/S/U/F','谁读取谁，谁持有下一状态'],
                 ['拓扑与时间','五节点；专家路径 1 + 1，旁路 2','直接邻居、选择范围、到达对齐'],
                 ['局部语义','求和；Top-2；加权贡献与 Add','载荷含义、归一化、未选行为'],
                 ['完整模型与任务','输入端口、初态、参数、输出读出','组合条件与任务质量']])+callout('增加规模、引入历史或更换聚合，都要检查整条依赖；可表达性还不等于目标已达成。'),
             notes='新例子是通用 TimedDAG 的一个独立教学绑定，当前并非带权消息分支。接口层允许逐边输出与来源敏感的 Agg，所以可以表达显式残差。更换为真实 MoE、加入 Attention/KV、嵌入 HB，均需重新给定完整函数及组合条件。共同架构提供表达能力，具体配置是否实现局部性、稀疏性仍须检查。')]),
    dict(chapter='07 · 把工作分别计量', status='由新例子的路径计数', sources=['moe','cost','review'], steps=[
        step('cost', '少做一个专家，还剩哪些工作？', '本例每次 5 个候选节点、4 次 Full、6 条内部消息；不同角色分开统计。',
             body=table(['工作','本次次数','含义'],[
                 ['候选准备','5','S + 三个专家 + M；专家输入并未省去'],
                 ['区域选择','3','一次三选二；两次单节点选择'],
                 ['Full：专家计算','2','E₁/E₂，或第二轮的 E₃/E₁'],
                 ['Full：分发／合并输出','1 + 1','复制输入；对基线执行一次 Add'],
                 ['内部消息','3 + 1 + 2 = 6','候选输入 + 残差旁路 + 专家贡献'],
                 ['其他算术与边界','分别记录','两次加权、一次贡献求和；外部输入输出各 1']])+callout('真实专家的计算、描述查询与通信开销分别测量；Full 的调用数不是统一成本单位。'),
             notes='静态图 5 节点、7 边；本次所有节点都是候选，4 个执行 Full，6 条不同内部边产生消息，3/3 区域参与。专家 Full 2 次与分发、Add 的 Full 各 1 次成本异质。标量仿射专家本身很便宜，只有换成真实 FFN 才能测量专家开销。本次统计不沿用旧 HB 的 10/4/8。已有设备实现审视提示填充行仍可能算术求值，因此数学稀疏、字节量、设备成本应分别记录。')]),
    dict(chapter='07 · 上下层相互牵引', status='设计解释 · 收益待验证', sources=['design','moe','memory'], steps=[
        step('feedback', '从目标选机制，再用具体代价修正设计', '每一项局部选择，都会反馈到网络结构和成本约束。', preview=True,
             body=table(['上层需要','局部机制','行为与代价如何反推设计'],[
                 ['减少参与节点与连接','每个有界区域只选部分专家','仍给全部候选传 x → 重审邻域与候选生成'],
                 ['避免统一内部中心','多个局部选择与局部合并','是否绕回唯一主干 → 检查全网路径'],
                 ['保留基线和贡献含义','显式旁路、求和与 Add','多一条旁路与到达约束 → 调整粒度和时间'],
                 ['利用本地历史作选择','计数或 Mem/Describe','信息够不够、查询多贵 → 调整状态与描述']])+callout('在任务效果可比时，再判断完整计算与传播是否减少；局部结构只提供待检验的机制。'),
             notes='这里表达当前设计解释，不声称架构在历史上按此顺序形成。已确认优先级是模型信息/决策局部性和减少每次参与的节点/连接，进而减少完整计算和消息；设备部署、学习规则另外讨论。旁路外显让职责清晰，也带来实际消息和对齐约束。上下层需相互校准，不预设越细粒度一定越好。')]),
    dict(chapter='08 · 任务验证', status='实验优先项 · 尚无新性能证据', sources=['lm','design','moe'], steps=[
        step('experiment', '在 Next Token 预测上检验这些选择', '实验优先采用单输入单输出；有限端口是模型配置的一部分。',
             body='<div class="flow"><div class="flow-box"><strong>token 嵌入</strong><span>当前输入表示</span></div><div class="flow-arrow">→</div><div class="flow-box accent"><strong>Tide 网络</strong><span>输出表示 yₜ</span></div><div class="flow-arrow">→</div><div class="flow-box"><strong>词表分布</strong><span>预测下一 token</span></div></div><div class="evidence"><div><h3>教学例已经核对</h3><p>基线、贡献与 Add 的位置<br>两轮路径、状态和数值</p></div><div><h3>任务仍须验证</h3><p>局部选择和历史是否有用<br>可比质量下是否减少总工作</p></div></div>'+callout('固定数据、上下文与效果容差，再比较参与节点、连接、专家计算、消息和设备成本。'),
             notes='单输入单输出是近期任务实验的实际选择。不同规模可配置不同数量的有限端口；固定端口下的扩散/收拢推论仍分别成立。标量 4/8 没有词表或损失，不能解释为预测效果；真正的语言模型须补齐嵌入、Attention/记忆、网络函数和词表读出。此轮只改上游教学与展示，不包含训练实验。')]),
    dict(chapter='附页 A · 残差位置答疑', status='已有定义／条件等式', sources=['kv','layered','hb','moe','agg'], steps=[
        step('existing-residuals', '当前实例的 residual add 在哪里？', '节点内、图中显式合并、图外读出，是三个不同位置。',
             body=table(['材料','残差实际归属','解读'],[
                 ['现有 KV 内容函数','G 内有 Attention 与前馈两次 Add','Full 调用 G；取完整 block 是较粗的边界'],
                 ['全连接数值例','内部保留 Attention residual；另有图外 residual','前馈残差尺度为零'],
                 ['HB 数值例','内部两条尺度均为零，G = u；图外 y = h + O','旧展示的 y = 2h 来自图外 Add'],
                 ['本次 MoE 教学例','专家只给贡献，M 求和并做 Add','图外不再加一次 x']])+callout('把 Attention 残差尺度设为零，只关闭该内容项；KV 写入不会因此自动消失。'),
             notes='KV §4.2 的 x_v=u+αatt W^O[R] 与 G=x_v+αff SwiGLU(N(x_v)) 都在节点内容里。全连接 §2 取 αatt=1、αff=0；HB §2 两者都为零，两套实例另有外部残差。完整 Transformer 粗封装是直接起点，但确切等价还需核对完整绑定。本次新增例子不修改这些既有定义。'),
        step('residual-equivalence', '把 x 包进每个专家，何时数值等价？', '所有专家收到同一个 x，并且所选权重和为 1。',
             body='<div class="equivalence"><div>Σ aᵢ [x + Eᵢ(x)]</div><div class="muted">= (Σ aᵢ) x + Σ aᵢ Eᵢ(x)</div><div>= x + Σ aᵢ Eᵢ(x)</div></div>'+callout('主线保持专家单独占节点，同时把残差旁路显式画出，以保留贡献与基线的职责。')+'<p class="small">不同输入、权重和不为 1，或跨过 Norm／非线性移动加法，都要重新核对。带权均值也不能直接当作残差相加。</p>',
             notes='这是一条有条件的等式，不要求将基线实际复制进每个 expert 的内容函数。比如来源残差聚合 source_residual 在关闭来源修正时，将 (1,x) 与 (1,F(x)) 聚合得到的是 (x+F(x))/2。若希望复用该分支，需要显式定义缩放或其他映射并核对语义；本次选的是另一套求和绑定。局部 Add 并不要求唯一全局残差流。')]),
    dict(chapter='附页 B · HB 独立算例', status='已有绑定 · 与新 MoE 分开', sources=['hb','topology'], steps=[
        step('hb-detail', 'HB 留作拓扑与既有计算的备查', 'b = 2，h = 1，n = 3；这是另一套带权 KV 绑定。', 'illustration', figure='hb-output0',
             body='<div class="eyebrow">原实例的第一轮</div><p>h₀ = +1 → 图读出 O₀ = +1<br>图外 residual → y₀ = +2</p><p>候选 <b>10</b> · Full <b>4</b><br>内部消息 <b>8</b></p>'+callout('这里 G = u；<br>不承担专家贡献的演示。')+'<p class="small">第二轮、KV 参数与完整证明<br>通过备注中的原实例查阅。</p>',
             notes='b=2,h=1,n=3,N=2，偏移{0,1}，D=5。KV/sum/Read+，全部候选保留写入；独立评分与正温度二次分配，k=1、η=0、τ=1/8、容量K=3。给定评分间隔使每区恰好一位激活，k=1本身不普遍保证单激活。第一轮路径 I0,L0,R1,E1；第二轮 I1,L1,R2,E1，输出 −2。每轮 10/4/8，外部注入3、输出1另计。完整函数参数见原实例。')]),
]
