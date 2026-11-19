"""Memory modules shown through the shared Upd / Mem / G interfaces."""
from drawing import Figure, BLUE, GREEN, GOLD, PURPLE, MUTED, INK, EDGE
from catalog import REGISTRY, register, KV, REC


@register('kv-memory', 'KV：保存条目，再按当前内容读取',
          [KV+'#1-状态与接收写入', KV+'#3-记忆描述', KV+'#4-激活后的内容读取'],
          '把条目保存、时间窗口、区域描述和激活后的 Attention 读取分开。',
          ['图中 Mem 采用 Read+；sum 与 qry 是组成模型时选择的两个候选。',
           '条目使用示意符号；完整状态、投影和带权 Attention 以教材定义为准。'])
def kv_memory():
    f=Figure('KV：保存条目，再按当前内容读取',
             '同一个记忆状态，分别服务于区域描述 Mem 与激活后的完整内容 G',
             REGISTRY['kv-memory']['sources'])
    f.card(70,260,480,165,'当前接收',tone='blue')
    f.math(310,369,'h = (r^{in}, u)',35,BLUE)
    f.arrow([(310,425),(310,495)],GREEN)
    f.card(70,500,480,240,'接收写入 Upd',tone='green')
    f.math(310,597,'u → (k, ν, p)',34,GREEN)
    f.math(310,656,'w = min(1, r^{in})',30,GREEN)
    f.text(310,710,'正接收时追加一条',28,GREEN)
    f.arrow([(550,619),(665,619)],GREEN,5)

    f.rect(670,260,980,480,'green',GREEN)
    f.text(1160,305,'本次正接收后保留的条目',35,GREEN,True)
    for x,label in [(780,'写入时间'),(995,'key'),(1170,'value'),(1345,'摘要'),(1530,'权重')]:
        f.text(x,370,label,27,MUTED)
    for row,(xi,idx) in enumerate([('ξ_1','1'),('ξ_2','2'),('θ','new')]):
        y=403+row*98
        f.rect(705,y,910,80,'white','#bfd8ce',radius=12)
        for x,term in [(780,xi),(995,'k_{'+idx+'}'),(1170,'ν_{'+idx+'}'),
                       (1345,'p_{'+idx+'}'),(1530,'w_{'+idx+'}')]:
            f.math(x,y+39,term,32,GREEN)
    f.text(1160,712,'k、ν 保存原值，权重 w 单独保存',28,GREEN)

    f.rect(1760,260,570,480,'white','#cbd7e3')
    f.text(2045,310,'每次读取先看时间窗口',32,BLUE,True)
    f.math(2045,379,'0 ≤ θ − ξ_i < T_v',32,BLUE)
    f.arrow([(1810,510),(2270,510)],EDGE,3,13)
    for x,xi,active in [(1845,'ξ_0',False),(2000,'ξ_1',True),(2200,'θ',True)]:
        f.circle(x,510,26,color=GREEN if active else MUTED,
                 fill='green' if active else 'gray')
        f.math(x,568,xi,30)
    f.line([(1920,445),(1920,607)],GOLD,3,True)
    f.text(1845,650,'已清理',27,MUTED)
    f.text(2110,650,'窗口内条目参与读取',28,GREEN)
    f.text(2045,706,'窗口按时间跨度计量',29,MUTED)

    f.arrow([(1160,740),(1160,795),(610,795),(610,850)],GREEN)
    f.arrow([(1160,795),(1775,795),(1775,850)],GREEN)
    f.rect(70,855,1080,310,'gold',GOLD)
    f.text(610,900,'Mem：为当前区域选择提供记忆描述',32,GOLD,True)
    f.text(610,962,'sum：时间摘要    或    qry：当前内容检索',30)
    f.math(610,1031,'(M_{short}, M_{long}, W_{short}, W_{long})',32,GOLD)
    f.text(610,1114,'与当前内容一起进入 Describe → 区域选择',29,GOLD)
    f.rect(1220,855,1110,310,'blue',BLUE)
    f.text(1775,900,'G：区域返回正强度后读取新状态',32,BLUE,True)
    f.text(1775,951,'当前 query × 历史 key → 带权汇集 value',29)
    f.text(1775,1006,'归一化加权平均，再乘 min(1, W)',30,BLUE)
    f.text(1775,1056,'W 为窗口内总写入权重',27,MUTED)
    f.text(1775,1114,'读出 → 投影 + 当前内容残差 → 前馈残差',29,BLUE)
    f.footer('图中 Mem 采用 Read+；零权重保持旧状态，未激活候选仍保留正接收写入。',
             '定义：带权 KV §§1、3、4 · 读取的投影、位置与残差参数按接口分别给定')
    return f


@register('affine-delta-memory', '循环矩阵：叠加与误差修正',
          [REC+'affine-delta-learning-note.md#1-状态与逻辑时间',
           REC+'affine-delta-learning-note.md#2-接收更新',
           REC+'affine-delta-learning-note.md#3-描述与完整内容'],
          '对照仿射写入与 Delta 写入，保留两个时间尺度及共同读取接口。',
          ['中心省略时间尺度下标；同一种更新分别作用于 short 与 long 矩阵。',
           'Mem 绘出 state 候选并采用 Read+；sum 候选改为读取独立摘要。'])
def affine_delta_memory():
    f=Figure('循环矩阵：叠加与误差修正',
             '选定其中一种更新，分别作用于 short、long · 中央省略时间尺度下标',
             REGISTRY['affine-delta-memory']['sources'])
    f.card(75,240,600,200,'旧矩阵与逻辑时间',tone='green')
    f.math(375,335,'H_a^{-} = exp(−λ_a Δ) H_a',31,GREEN)
    f.math(375,397,'Δ = (θ − s)_{+}',31,GREEN)
    f.card(855,240,1470,200,'当前接收形成写入与保留参数',tone='blue')
    f.math(1590,335,'u → k, ν, p       w = min(1, r^{in})',35,BLUE)
    f.text(1590,398,'g：保留比例    β：Delta 误差写入比例    w：接收权重',29)
    f.arrow([(375,440),(375,490),(630,490),(630,532)],GREEN)
    f.arrow([(1140,440),(1140,490),(800,490),(800,532)],BLUE)
    f.arrow([(1930,440),(1930,532)],BLUE)
    f.arrow([(675,365),(735,365),(735,475),(1680,475),(1680,532)],GREEN,3)

    f.rect(75,537,1090,360,'green',GREEN)
    f.text(620,586,'affine：叠加当前关联',37,GREEN,True)
    f.math(620,666,'H̃ = g H^{-} + w ν k^{T}',43,GREEN)
    f.card(130,733,390,105,'保留旧矩阵',tone='green')
    f.text(621,785,'+',47,GREEN,True)
    f.card(725,733,390,105,'写入当前外积',tone='green')

    f.rect(1235,537,1090,360,'purple',PURPLE)
    f.text(1780,586,'delta：按当前预测误差修正',37,PURPLE,True)
    f.math(1780,666,'H̃ = g H^{-} + β (ν − g H^{-} k) k^{T}',37,PURPLE)
    for x,title in [(1280,'旧预测'),(1620,'与目标作差'),(1960,'按误差写入')]:
        f.rect(x,741,290,96,'white','#cfbfdc',radius=15)
        f.text(x+145,788,title,28,PURPLE,True)
    f.arrow([(1570,789),(1615,789)],PURPLE,3,12)
    f.arrow([(1910,789),(1955,789)],PURPLE,3,12)

    f.arrow([(620,897),(620,955)],GREEN)
    f.arrow([(1780,897),(1780,955)],PURPLE)
    f.rect(75,960,2250,208,'white','#cbd7e3')
    f.text(1200,1001,'两种更新共用：保存矩阵、时间、摘要与写入量',32,GREEN,True)
    f.text(650,1073,'Mem（state，Read+）：查询两份新矩阵',28,GOLD)
    f.math(650,1126,'(M_{short}, M_{long}, W_{short}, W_{long})',30,GOLD)
    f.text(1780,1073,'G：激活后查询新矩阵',30,BLUE)
    f.text(1780,1126,'两路读出 → 内容残差 → 前馈残差',29,BLUE)
    f.footer('接收门 g、β 由本次输入决定；区域控制 c 决定后续发送。未激活候选保留更新。',
             '定义：基础循环记忆 §§1—3 · Mem 还可选择 sum，从独立摘要取得描述')
    return f


def gated_state_reads(f, *, ssd=False):
    """Shared interface row; preserve each module's different output ordering."""
    f.rect(70,878,2260,300,'white','#cbd7e3')
    f.text(1200,915,'矩阵、卷积历史、摘要与时间共同保存在新状态中',31,GREEN,True)
    f.line([(965,963),(965,1144)],'#d5dee7',2)
    f.text(510,979,'Mem（state，Read+）：供区域形成描述',27,GOLD,True)
    f.text(510,1037,'short：时间摘要    long：矩阵查询',28,GOLD)
    f.math(510,1104,'(M_{short}, M_{long}, W_{short}, W_{long})',27,GOLD)
    f.text(1660,979,'G：区域返回正强度后，在新状态上计算',29,BLUE,True)
    if ssd:
        stages=[(1010,280,'查询 + 直通项'),(1340,240,'输出门'),(1630,270,'全宽 RMS'),(1950,325,'投影与残差')]
    else:
        stages=[(1010,280,'每头矩阵查询'),(1340,240,'每头 RMS'),(1630,270,'输出门'),(1950,325,'拼接投影与残差')]
    for i,(x,w,title) in enumerate(stages):
        f.rect(x,1035,w,100,'blue',BLUE,radius=15)
        f.text(x+w/2,1084,title,26,BLUE,True)
        if i:
            px,pw,_=stages[i-1]
            f.arrow([(px+pw,1084),(x-5,1084)],BLUE,3,12)


@register('gated-delta-memory', 'Gated DeltaNet：让写入修正预测误差',
          [REC+'gated-delta-learning-note.md#2-投影与接收更新',
           REC+'gated-delta-learning-note.md#3-矩阵读取记忆描述与完整内容'],
          '展示投影历史、接收门、矩阵误差修正和每头先归一化再乘输出门的完整内容路径。',
          ['图中省略头下标，逐头使用同一形式；被动衰减包含在 H− 中。',
           'Mem 绘出 state 候选并采用 Read+；sum 候选使用两个时间摘要。'])
def gated_delta_memory():
    f=Figure('Gated DeltaNet：让写入修正预测误差',
             '正接收先更新记忆；区域激活后，再完成内容变换 · 图中省略头下标',
             REGISTRY['gated-delta-memory']['sources'])
    f.card(70,245,450,200,'当前接收',tone='blue')
    f.math(295,346,'x = N_d(u)',34,BLUE)
    f.math(295,407,'w = min(1, r^{in})',28,BLUE)
    f.card(675,245,760,200,'投影 → 短卷积 → SiLU',tone='blue')
    f.text(1055,347,'Q、k 归一化，Q 按宽度缩放',30,BLUE)
    f.text(1055,407,'卷积历史保存在固定逻辑时间槽',28,MUTED)
    f.arrow([(520,345),(670,345)],BLUE)
    f.card(1590,245,740,200,'输入决定接收门',tone='gold')
    f.math(1960,347,'g = exp(−w A^{rate} Δ)',31,GOLD)
    f.math(1960,407,'β = w σ((a^{β})^{T} x + b^{β})',30,GOLD)
    f.arrow([(295,245),(295,209),(1960,209),(1960,240)],GOLD,3)

    f.rect(70,535,2260,260,'green',GREEN)
    f.text(1200,579,'Upd：保留旧关联，再修正当前 key 的预测误差',35,GREEN,True)
    f.arrow([(1055,445),(1055,530)],BLUE)
    f.arrow([(1960,445),(1960,530)],GOLD)
    f.math(520,668,'H̄ = g H^{-}',42,GREEN)
    f.arrow([(850,668),(990,668)],GREEN,5)
    f.math(1625,668,'H̃ = H̄ + β (ν − H̄ k) k^{T}',43,GREEN)
    f.text(520,745,'当前时间的旧矩阵 × 保留门',29,GREEN)
    f.text(1625,745,'目标 value − 旧预测 → 关联修正',30,GREEN)
    f.arrow([(1200,795),(1200,873)],GREEN)
    gated_state_reads(f)
    f.footer('Δ 由输入决定；逻辑时间衰减计入旧矩阵有效值。输出门调制内容，c 控制发送。',
             '定义：Gated DeltaNet §§2—3 · G 的顺序为每头 RMS → 输出门 → 拼接投影 → 残差')
    return f


@register('ssd-memory', 'SSD：用输入步长控制保留与写入',
          [REC+'ssd-learning-note.md#2-投影门与写入',
           REC+'ssd-learning-note.md#3-矩阵读取记忆描述与完整内容'],
          '展示输入步长、共享 B/C、多头状态和先乘输出门再全宽归一化的内容路径。',
          ['图中省略头与组下标；每头有独立矩阵，同组共享 B、C。',
           'Mem 绘出 state 候选并采用 Read+；输出按教材的 Mamba2Simple 形状排列。'])
def ssd_memory():
    f=Figure('SSD：用输入步长控制保留与写入',
             '每头保存独立矩阵；同组共享写入向量 B 与读取向量 C · 图中省略头与组下标',
             REGISTRY['ssd-memory']['sources'])
    f.card(70,245,440,200,'当前接收与投影',tone='blue')
    f.math(290,347,'x_0 = N_d(u)',34,BLUE)
    f.math(290,407,'w = min(1, r^{in})',28,BLUE)
    f.card(665,245,770,200,'短卷积 → SiLU',tone='blue')
    f.math(1050,347,'ν, B, C',42,BLUE)
    f.text(1050,407,'这三个分支使用固定时间槽的历史',28,MUTED)
    f.arrow([(510,345),(660,345)],BLUE)
    f.card(1590,245,740,200,'输入步长与保留比例',tone='gold')
    f.math(1960,347,'d = w Δ',37,GOLD)
    f.math(1960,407,'g = exp(−A^{rate} d)',34,GOLD)
    f.arrow([(290,245),(290,209),(1960,209),(1960,240)],GOLD,3)

    f.rect(70,535,2260,260,'green',GREEN)
    f.text(1200,579,'Upd：标量保留 + 按有效步长写入外积',35,GREEN,True)
    f.arrow([(1050,445),(1050,530)],BLUE)
    f.arrow([(1960,445),(1960,530)],GOLD)
    f.math(1200,669,'H̃ = g H^{-} + d ν B^{T}',49,GREEN)
    f.text(1200,748,'弱接收使 d 趋零、g 趋一：写入与遗忘一起减弱',31,GREEN)
    f.arrow([(1200,795),(1200,873)],GREEN)
    gated_state_reads(f,ssd=True)
    f.footer('矩阵查询使用 C，直通项 Dν 进入 G；输出门与步长由当前投影直接形成。',
             '定义：SSD §§2—3 · G 的顺序为查询 + 直通项 → 输出门 → 全宽 RMS → 投影与残差')
    return f
