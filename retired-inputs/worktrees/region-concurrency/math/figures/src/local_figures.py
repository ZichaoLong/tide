"""Teaching scenes; their mathematical sources are tracked with each figure."""
from drawing import Figure, BLUE, GREEN, GOLD, PURPLE, MUTED, INK, EDGE, BG
from catalog import REGISTRY, register, ARCH, ALLOC, KV, REC


@register('weighted-message-flow', '带权消息的一次处理',
          [ARCH+'#2-接收聚合', ARCH+'#3-状态与记忆描述', ARCH+'#4-分配完整计算与发送'],
          '把消息强度、内容、接收更新、描述、区域控制和同值广播接到同一次计算。',
          ['采用 Read+，Mem 与 Full 使用新状态；所有候选保留接收后的状态。',
           'Agg 可选来源残差或同刻局部自注意力，两者均交付接收强度与内容。',
           '同值广播属于带权消息支线的 Full 绑定，正典的一般 Full 可逐边返回不同载荷。'])
def weighted_message_flow():
    f = Figure('带权消息：接收、记忆、分工与发送',
               '同一逻辑时间 θ · 示例节点 v · 本图采用 Read+，描述读取本次新状态',
               REGISTRY['weighted-message-flow']['sources'])
    for i,y in enumerate([330,480,630],1):
        f.rect(75,y,265,110,'blue',BLUE)
        f.text(207,y+30,'输入消息 '+str(i),26,BLUE)
        f.math(207,y+77,f'(m_{i}, z_{i})',31)
        f.arrow([(340,y+55),(405,y+55),(465,400+35*(i-1))],BLUE)
    f.card(470,325,510,200,'接收聚合 Agg', ['来源残差／同刻自注意力'], 'blue',30)
    f.math(725,487,'h = (r^{in}, u)',34,BLUE)
    f.card(470,700,510,190,'记忆更新 Upd',['正接收驱动，候选保留更新'],'green',29)
    f.math(725,855,'q → q̃',37,GREEN)
    f.arrow([(725,525),(725,695)],GREEN)
    f.tag(725,611,'接收强度与内容',GREEN,27)
    f.card(1190,325,470,200,'Mem → Describe',['当前内容与记忆形成描述'],'gold',28)
    f.math(1425,487,'d_v',37,GOLD)
    f.arrow([(980,410),(1185,410)],BLUE)
    f.tag(1082,375,'当前接收',BLUE,25)
    f.arrow([(980,755),(1100,755),(1100,480),(1185,480)],GREEN)
    f.tag(1100,606,'新状态',GREEN,25)
    f.card(1850,325,470,200,'区域共同选择',['汇集同一逻辑时间的候选描述'],'gold',25)
    f.math(2085,487,'c_v ∈ [0,1]',34,GOLD)
    f.arrow([(1660,425),(1845,425)],GOLD)
    for x in [1970,2110,2250]:
        f.circle(x,237,18,color=GOLD,fill='gold')
        f.arrow([(x,260),(x,320)],GOLD,3,12)
    f.text(2110,196,'其他候选的描述',25,GOLD)
    f.rect(1165,655,1180,255,'white',GOLD,radius=29,sw=2)
    f.text(1755,683,'Full：激活后计算内容并组成载荷',27,GOLD,True)
    f.card(1190,715,470,175,'完整内容 G',['使用新状态与当前内容'],'blue',29)
    f.math(1425,855,'z_v = G_v(q̃, θ, u)',31,BLUE)
    f.arrow([(980,815),(1185,815)],GREEN)
    # Current content passes around the state update; it remains an input to G.
    f.arrow([(980,500),(1037,500),(1037,950),(1145,950),(1145,860),(1185,860)],BLUE,3)
    f.tag(1040,925,'保留当前内容 u',BLUE,24)
    f.card(1850,715,470,175,'发送载荷',['以区域控制量作为强度'],'blue',29)
    f.math(2085,855,'(c_v, z_v)',37,BLUE)
    f.arrow([(1660,795),(1845,795)],BLUE)
    f.arrow([(2085,525),(2085,650)],GOLD)
    f.rect(1995,567,180,56,BG,radius=8)
    f.math(2085,596,'c_v > 0',31,GOLD)
    f.arrow([(725,890),(725,1015)],GREEN)
    f.rect(470,1020,510,107,'green',GREEN)
    f.text(725,1073,'新状态留给下一次接收',31,GREEN,True)
    for x in [1920,2250]:
        f.arrow([(2085,890),(2085,946),(x,1010)],BLUE)
        f.circle(x,1055,38,label='后继',color=BLUE,size=24)
        f.math(x,1130,'(c_v, z_v)',29,BLUE)
    f.text(2085,1190,'每条出边复制同一载荷',29,BLUE,True)
    f.footer('接收决定记忆更新；区域决定发送强度。未激活候选仍保留本次更新。',
             '定义：共同局部架构 §§2—4 · 广播是本支线的 Full 绑定')
    return f


@register('region-allocation', '区域怎样分配发送强度',
          [ALLOC+'#2-指定总量的一次分配', ARCH+'#45-一组贯穿例子'],
          '使用教材的指定总量算例，区分节点上限、评分、总发送强度和激活数量。',
          ['w=μmax=(1,1,0)，s=(2,1,100)，区域证据=(-1/2,-1/2,0)。',
           'γmax=2，k=2，区域头给出 ρ=0.4，指定 B=1.4；quad，τ_sel=2。'])
def region_allocation():
    f=Figure('区域分配：发送多少，由谁发送',
             '同一逻辑时间 θ · 指定总量 + 二次分配（quad）· 教材的三个候选算例',
             REGISTRY['region-allocation']['sources'])
    for x,name,w,score,z in [(90,'a',1,2,'−1/2'),(475,'b',1,1,'−1/2'),(860,'c',0,100,'0')]:
        f.card(x,245,330,300,'候选节点 '+name,tone='blue')
        f.math(x+165,344,f'w_{name} = {w}',31)
        f.math(x+165,401,f'μ_{name}^{{max}} = {w}',31)
        f.math(x+165,458,f's_{name} = {score}',31)
        f.text(x+165,514,'区域证据 '+z,26,MUTED)
        f.arrow([(x+165,545),(x+165,597)],GOLD)
    f.line([(255,597),(1025,597)],GOLD)
    f.arrow([(640,597),(640,661)],GOLD)
    f.rect(90,665,1100,310,'gold',GOLD)
    f.text(640,711,'先确定总量，再按评分分配',36,GOLD,True)
    f.text(360,782,'区域证据 → 区域头',29,GOLD)
    f.math(360,844,'B = 1 + 0.4(2−1)',31)
    f.math(360,907,'B = 1.4',38,GOLD)
    f.arrow([(610,837),(674,837)],GOLD)
    f.lines(915,782,['在节点上限内分配','偏向高分，并抑制集中'],28,step=53)
    f.math(915,907,'τ_{sel} = 2',32,GOLD)
    f.arrow([(1190,820),(1355,820)],GOLD,5)
    f.rect(1360,245,950,850,'white','#cbd7e3')
    f.text(1835,303,'最终发送强度',38,BLUE,True)
    bottom,scale=850,390
    for x,name,value,cap in [(1505,'a',.95,1),(1775,'b',.45,1),(2045,'c',0,0)]:
        if cap:
            f.rect(x,bottom-scale*cap,135,scale*cap,'gray','#c8d3df',radius=12)
        if value:
            f.rect(x,bottom-scale*value,135,scale*value,'blue',BLUE,radius=12)
            f.math(x+67,bottom-scale*value-38,str(value),36,BLUE)
        else:
            f.line([(x,bottom),(x+135,bottom)],MUTED,4)
            f.math(x+67,bottom-43,'0',36,MUTED)
        f.text(x+67,893,'节点 '+name,30)
        f.text(x+67,941,'激活' if value else '未激活',28,BLUE if value else MUTED)
    f.text(1605,399,'浅色轮廓表示节点上限',25,MUTED)
    f.rect(1425,993,390,68,'gold',radius=15)
    f.text(1620,1027,'总强度  1.4',31,GOLD,True)
    f.rect(1855,993,390,68,'blue',radius=15)
    f.text(2050,1027,'激活节点数  2',31,BLUE,True)
    f.text(640,1035,'w：接收权重    μ：发送上限    s：评分',28,MUTED)
    f.text(640,1100,'节点 c 收到零强度消息：仍为候选，发送上限为 0。',26,MUTED)
    f.footer('发送总强度与激活节点数分别计量；全部候选都保留接收更新。',
             '定义与数值：区域评分与发送分配 §2.5；共同局部架构 §4.5')
    return f


@register('remember-before-send', '收到时记住，激活时发送',
          [KV+'#52-未激活时保留的记忆'],
          '沿用两个接收时刻的 KV 算例，展示未激活时的写入如何影响之后的选择和内容。',
          ['沿用 KV §5.1 的标量投影与 Full；Read+、sum 描述、单位权重，T≥4。',
           '节点增益与固定阈值硬选择：γ=1，k=1，η=0；s_a=0.5，s_b=0.75−M_long(new)。',
           '两节点在时间 1、4 收到 (1,+1)、(1,−1)；Attention 匹配相等，内容残差尺度 1，前馈尺度 0。'])
def remember_before_send():
    f=Figure('收到时记住，激活时发送',
             '同一区域的节点 a、b · 观察节点 b 的两次接收 · KV 教材算例',
             REGISTRY['remember-before-send']['sources'])
    f.math(1200,193,'s_a = 0.5     s_b = 0.75 − M_{long}',29,GOLD)
    for x,theta,inp in [(85,1,'+1'),(1300,4,'−1')]:
        f.rect(x,235,1015,920,'white','#cbd7e3',radius=30)
        f.math(x+507,294,f'θ = {theta}',40,BLUE)
        f.text(x+507,358,'两个节点都收到内容 '+inp+'，强度为 1',31)
        f.arrow([(x+507,400),(x+507,447)],GREEN)
        f.rect(x+75,450,865,240,'green',GREEN)
        f.text(x+507,495,'b 的 KV 记忆',33,GREEN,True)
        positions=[(x+285,'+1','时间 1')]
        if theta==4: positions.append((x+700,'−1','时间 4'))
        for cx,value,label in positions:
            f.rect(cx-130,542,260,112,'white','#a8cbbb',radius=15)
            f.math(cx,576,value,38,GREEN)
            f.text(cx,627,label,25,MUTED)
        f.arrow([(x+507,690),(x+507,737)],GOLD)
        f.rect(x+75,740,865,150,'gold',GOLD)
        if theta==1:
            f.text(x+507,783,'评分  a: 0.5，b: −0.25',31)
            f.text(x+507,844,'区域激活 a；b 保留记忆',33,GOLD,True)
            f.rect(x+155,965,705,118,'gray','#cbd7e3')
            f.text(x+507,1024,'b 本次不发送',38,MUTED,True)
        else:
            f.text(x+507,783,'评分  a: 0.5，b: 0.75',31)
            f.text(x+507,844,'区域激活 b',33,GOLD,True)
            f.arrow([(x+507,890),(x+507,939)],BLUE)
            f.text(x+507,981,'记忆读取',29,BLUE)
            f.math(x+507,1037,'R = (1−1)/2 = 0',34,BLUE)
            f.math(x+507,1100,'G_b = u_b + R = −1',35,BLUE)
    f.arrow([(1025,584),(1370,584)],GREEN,5)
    f.tag(1200,535,'保留到下一次接收',GREEN,25)
    f.footer('本例：Read+、sum 描述，硬选择取最高正分节点；单位权重、相等匹配分数、窗口 T ≥ 4。',
             '定义与数值：带权 KV §5.2 · Full 的内容残差尺度为 1，前馈尺度为 0')
    return f


# Import scenes after the shared registry has been loaded.
import memory_figures  # noqa: E402,F401
import control_time_figures  # noqa: E402,F401
