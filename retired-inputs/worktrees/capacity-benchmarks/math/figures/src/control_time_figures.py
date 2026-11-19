"""Regional controls and logical-time examples from the local textbooks."""
from drawing import Figure, BLUE, GREEN, GOLD, PURPLE, MUTED, EDGE
from catalog import REGISTRY, register, ALLOC, REC


@register('logical-time-slots', '历史按逻辑时间对齐',
          [REC+'causal-convolution-learning-note.md#1-缓存卷积与写入',
           REC+'causal-convolution-learning-note.md#22-与通常短卷积的对应',
           REC+'causal-convolution-learning-note.md#3-弱接收与固定时间'],
          '用教材的三抽头算例展示固定时间槽、缺失补零与弱历史的连续消失。',
          ['标量卷积，L_conv=3，δ_conv=2，系数 (1,2,3)，偏置为零。',
           '时间 0 写入 (w,z)=(1,4)，时间 2 写入 (ε,5)，时间 4 当前投影为 6；0<ε≤1。',
           '第二行删去时间 2 的接收事件，其余输入与查询时间保持。'])
def logical_time_slots():
    f=Figure('历史按逻辑时间对齐',
             '固定时间槽的短卷积 · 当前逻辑时间 θ = 4 · 三个抽头的间距为 2',
             REGISTRY['logical-time-slots']['sources'])
    positions=[535,795,1055,1315,1575]
    f.arrow([(400,284),(1705,284)],EDGE,4,15)
    for xi,x in enumerate(positions):
        f.line([(x,273),(x,296)],EDGE,3)
        f.math(x,241,str(xi),32)
    f.text(1955,283,'原始逻辑时间 θ',30,BLUE)

    for y,weak in [(365,True),(795,False)]:
        f.text(195,y+112,'保留弱接收' if weak else '删去该接收',31,GREEN if weak else MUTED,True)
        f.math(195,y+172,'w_2 = ε' if weak else 'θ = 2',30,GREEN if weak else MUTED)
        f.rect(355,y,1975,350,'white','#cbd7e3')
        for xi,x in enumerate(positions):
            current=xi==4
            present=xi==0 or (xi==2 and weak)
            color=BLUE if current else GREEN if present else MUTED
            tone='blue' if current else 'green' if present else 'gray'
            f.rect(x-95,y+80,190,147,tone,color,radius=17)
            label='当前投影' if current else '已存历史' if present else '缺失槽'
            f.text(x,y+111,label,25,color)
            value='6' if current else '4' if xi==0 else '5ε' if xi==2 and weak else '0'
            f.math(x,y+179,value,42,color)
            if xi in (0,2,4):
                weight={0:3,2:2,4:1}[xi]
                f.math(x,y+33,'× '+str(weight),31,GOLD)
                f.arrow([(x,y+54),(x,y+75)],GOLD,3,10)
        f.text(1055,y+291,'历史读取 w z；当前项直接使用 z',29,BLUE)
        f.arrow([(1695,y+152),(1770,y+152)],BLUE,4,13)
        f.text(2035,y+87,'卷积结果',31,BLUE,True)
        f.math(2035,y+160,'18 + 10ε' if weak else '18',42,BLUE)
        f.math(2035,y+228,'12 + 10ε + 6' if weak else '12 + 0 + 6',30,MUTED)
    f.arrow([(2035,715),(2035,789)],GREEN,3)
    f.tag(2035,754,'ε → 0',GREEN,29)
    f.footer('缺失槽按零读取；其他历史保留原坐标。无接收的时间不产生节点事件。',
             '定义与数值：固定逻辑时间上的短卷积 §§1—3 · 当前项只计入一次')
    return f


@register('sending-rules', '发送强度的几种形成方式',
          [ALLOC+'#24-按评分分配并返回', ALLOC+'#41-约束分配', ALLOC+'#42-通过-attention-份额分配'],
          '区分约束分配的总量与上限，以及 Attention 份额到发送强度的三种绑定。',
          ['图中仅比较标出的六种发送规则；比例分配与零温配对函数见教材。',
           'Attention 的 a 是基权重；Z>0 时 p 的总和为 1，Z=0 时 p=0。',
           '右侧 μ 用于增益与份额，只有直接 Attention 保证各 c_v≤μ_v。'])
def sending_rules():
    f=Figure('发送强度的几种形成方式',
             '共同输出为 c；各函数分别规定总量、节点上限与评分的作用',
             REGISTRY['sending-rules']['sources'])
    f.text(625,247,'约束分配：在允许范围内优化',37,GOLD,True)
    f.text(1785,247,'Attention：先求份额，再定幅度',37,BLUE,True)
    left=[
        ('指定总量', '区域头 → B', 'Σ c_v = B', '评分在节点上限 μ^{max} 内分配'),
        ('区域阈值', '区域头 → η_j', 't_v = s_v − η_j', '总量在 [b^{recv}, k] 内，节点上限 μ^{max}'),
        ('节点增益与固定阈值', '节点增益 → μ_v', 't_v = s_v − η', '总量在 [min(1,Σμ), k] 内，节点上限 μ'),
    ]
    for y,(title,_,formula,_) in zip([320,600,880],left):
        f.rect(70,y,1110,235,'gold',GOLD)
        f.text(625,y+45,title,33,GOLD,True)
        if y==320:
            f.text(338,y+115,'区域头给定 B',30)
            f.math(900,y+115,formula,32,GOLD)
            f.text(625,y+184,'在节点上限内分配，μ 取最大允许上限',28,MUTED)
        elif y==600:
            f.math(338,y+115,'s_v − η_j',34)
            f.math(900,y+115,'Σ c_v ∈ [b^{recv}, k]',29,GOLD)
            f.text(625,y+184,'区域阈值调节发送意愿，总量由优化得到',28,MUTED)
        else:
            f.math(338,y+115,'s_v − η, μ_v',32)
            f.math(900,y+115,'Σ c_v ∈ [b, k]',30,GOLD)
            f.math(625,y+184,'b = min(1, Σ μ_v)',29,MUTED)
        f.arrow([(575,y+116),(655,y+116)],GOLD,3,12)

    for y,title,formula,note,base in [
        (320,'直接 Attention','c_v = μ_v p_v^{sel}','每节点受 μ 上限约束；总量随份额改变','a = μ 或 w'),
        (600,'固定总强度 Attention','c_v = p_v^{sel}','有正基权重时，总发送强度为 1','a = 1 或 μ'),
        (880,'总量缩放 Attention','c_v = B p_v^{sel}','B = min(1,Σμ)；个别分量可以超过 μ','a = μ'),
    ]:
        f.rect(1220,y,1110,235,'blue',BLUE)
        f.text(1775,y+45,title,33,BLUE,True)
        f.text(1415,y+116,base,27)
        f.arrow([(1585,y+116),(1655,y+116)],BLUE,3,12)
        f.math(1975,y+116,formula,35,BLUE)
        f.text(1775,y+184,note,28,MUTED)
    f.math(1200,1180,'p_v^{sel} = a_v exp(s_v/τ_{sel}) / Z',31,BLUE)
    f.footer('p 是归一化份额：Z 为分子之和；Z = 0 时取 p = 0。最终控制量统一进入 Full。',
             '定义：区域评分与发送分配 §§2.4、4.1、4.2 · 发送强度之和与激活节点数分别计量')
    return f


@register('temperature-support', '温度改变分配与激活数量',
          [ALLOC+'#55-分配与饱和', ALLOC+'#414-发送总量与激活数量'],
          '沿用四候选算例，以共同纵轴展示分配、饱和与精确零。',
          ['采用节点增益与固定阈值分配，quad，μ=(1,1,1,1)，调整后评分 t=(4,3,2,1)，k=2。',
           '三个温度分别为 5、2、0.4；本例各自达到总量上界 2。'])
def temperature_support():
    f=Figure('温度改变分配与激活数量',
             '同一组评分与上限，只改变选择温度 · 二次分配 quad 的教材算例',
             REGISTRY['temperature-support']['sources'])
    f.rect(75,225,2250,110,'gold',GOLD)
    f.math(1200,280,'t = (4, 3, 2, 1)        μ = (1, 1, 1, 1)        k = 2',38,GOLD)
    for x,tau,values in [(75,5,(.8,.6,.4,.2)),(840,2,(1,.75,.25,0)),(1605,.4,(1,1,0,0))]:
        f.rect(x,390,720,772,'white','#cbd7e3')
        f.math(x+360,451,'τ_{sel} = '+str(tau),42,BLUE)
        bottom,scale=929,350
        for i,value in enumerate(values):
            bx=x+73+i*152
            f.rect(bx,bottom-scale,110,scale,'gray','#d1dce5',radius=9)
            if value:
                f.rect(bx,bottom-scale*value,110,scale*value,'blue',BLUE,radius=9)
            else:
                f.line([(bx,bottom),(bx+110,bottom)],MUTED,4)
            f.math(bx+55,bottom-scale*value-35,f'{value:g}',34,BLUE if value else MUTED)
            f.text(bx+55,975,chr(97+i),32)
        active=sum(v>0 for v in values)
        f.text(x+360,1051,f'激活节点数  {active}',35,BLUE,True)
        f.text(x+360,1112,'发送总强度  2',29,GOLD)
    f.arrow([(750,451),(820,451)],GOLD,3,12)
    f.arrow([(1515,451),(1585,451)],GOLD,3,12)
    f.footer('本例在正温度就出现精确零。k 约束总发送强度；激活数量由正分量的个数给出。',
             '定义与数值：区域评分与发送分配 §§4.1.4、5.5 · 浅色轮廓为每节点上限 1')
    return f
