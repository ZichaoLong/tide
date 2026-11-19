# Restored from the October 2026 drawing calls; see ../../PROVENANCE.md.
from pathlib import Path
from PIL import Image, ImageFont
from svg_export import VectorDrawing
import math

import argparse

_parser = argparse.ArgumentParser(description='Render the restored Tide figure.')
_parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2])
_output_dir = _parser.parse_args().output_dir
_output_dir.mkdir(parents=True, exist_ok=True)
OUT = _output_dir / 'PSUF-complete.png'
W,H,SCALE = 2000,2170,2
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG = '#F8FAFD'
INK,MUTED,BLUE,GREEN,GOLD,EDGE = '#20364B','#60758A','#286DA7','#27816C','#A67937','#7B91A5'
im = Image.new('RGB',(W*SCALE,H*SCALE),BG)
d = VectorDrawing(im, scale=SCALE, background=BG)

def font(n,bold=False):
    return ImageFont.truetype(BOLD if bold else FONT,round(n*SCALE),index=2)

def txt(x,y,s,n=30,color=INK,bold=False,anchor='mm'):
    d.text((round(x*SCALE),round(y*SCALE)),s,font=font(n,bold),fill=color,anchor=anchor)

def box(bounds,fill='#FFFFFF',outline='#D2DDE8',radius=24,width=2):
    d.rounded_rectangle(tuple(round(a*SCALE) for a in bounds),round(radius*SCALE),
                        fill=fill,outline=outline,width=round(width*SCALE))

def arrow(points,color=EDGE,width=4,head=16):
    d.line([(round(x*SCALE),round(y*SCALE)) for x,y in points],fill=color,
           width=round(width*SCALE),joint='curve')
    (px,py),(x,y)=points[-2:]
    a=math.atan2(y-py,x-px)
    triangle=[(x,y),(x-head*math.cos(a-.45),y-head*math.sin(a-.45)),
              (x-head*math.cos(a+.45),y-head*math.sin(a+.45))]
    d.polygon([(round(xx*SCALE),round(yy*SCALE)) for xx,yy in triangle],fill=color)

def label(x,y,s,color=INK,n=27):
    width=d.textlength(s,font=font(n))/SCALE
    box((x-width/2-10,y-n/2-7,x+width/2+10,y+n/2+9),BG,BG,radius=7,width=1)
    txt(x,y,s,n,color)

def circle(x,y,r,s):
    d.ellipse(tuple(round(a*SCALE) for a in (x-r,y-r,x+r,y+r)),
              fill='#FFFFFF',outline=BLUE,width=3*SCALE)
    txt(x,y-1,s,28,BLUE,True)

txt(1000,65,'区域 E：同一逻辑时间 θ 内的 P / S / U / F',50,INK,True)
txt(1000,133,'示例候选 {u, v}，仅激活 v；κ = 1，两者的计算快照均采用候选状态。',30,MUTED)
txt(1000,186,'向下表示函数依赖顺序；四步使用相同的逻辑时间 θ。',31,BLUE)

xu,xv = 570,1430
for x,name in [(xu,'u'),(xv,'v')]:
    txt(x,249,'示例节点 '+name,38,INK,True)
    box((x-360,300,x+360,435),'#EDF3F9')
    txt(x,341,'同一逻辑时间 θ 的完整消息集合 B',30)
    txt(x,393,'旧状态 q；θ 的输入与缺席均已确定',27,MUTED)
    arrow([(x,435),(x,484)])
    box((x-360,485,x+360,670),'#E8F1FB','#A9C5DF')
    txt(x,528,'P 准备：Agg → Upd → Read',35,BLUE,True)
    txt(x,578,'聚合为 h，准备候选状态与描述量',29)
    txt(x,629,'返回 (h, q_tilde, d)',32)
    arrow([(x,670),(x,764)],BLUE)
    label(x,715,'描述量 d 交区域；h 与状态量留在本节点',BLUE,26)

box((210,765,1790,1005),'#FFF6E9','#DFC7A2')
txt(1000,809,'S 区域选择：SelStep',39,GOLD,True)
txt(1000,863,'区域 E 的候选节点共同确定激活与调制控制',32,GOLD)
txt(1000,917,'读取：旧历史 y、逻辑时间 θ、描述量 (d_u, d_v)',30)
txt(1000,969,'返回：激活集合 {v}、局部控制 (c_u, c_v)、下一历史 y_next',30)

for x,name,bit in [(xu,'u','0'),(xv,'v','1')]:
    arrow([(x,1005),(x,1109)],GOLD)
    label(x,1054,('未激活，仍取得 c_u' if bit=='0' else '激活，取得 c_v'),GOLD,30)
    box((x-360,1110,x+360,1370),'#ECF7F4','#AACFC3')
    txt(x,1152,'U 状态作用',36,GREEN,True)
    txt(x,1196,'确定计算快照与下一持久状态',29,GREEN)
    txt(x,1240,'本例 q_cmp = q_tilde',32)
    txt(x,1290,'q_next = Next(q, q_cmp, θ, h, '+bit+', c_'+name+')',27)
    txt(x,1338,'返回 (q_cmp, q_next)',30,MUTED)

arrow([(xu,1370),(xu,1464)],EDGE)
label(xu,1414,'激活指示为 0',MUTED,28)
box((210,1465,930,1710),'#EFF2F5','#D3DAE2')
txt(xu,1527,'u 本次不调用 Full',37,MUTED,True)
txt(xu,1590,'仍由 Next 确定下一持久状态',30,MUTED)
txt(xu,1649,'后续节点事件读取 q_next',29,MUTED)

arrow([(xv,1370),(xv,1464)],BLUE)
label(xv,1414,'v 已激活；使用 q_cmp、h、c_v',BLUE,28)
box((1070,1465,1790,1710),'#E8F1FB','#A9C5DF')
txt(xv,1508,'F 输出：Full(q_cmp, θ, h, c_v)',33,BLUE,True)
txt(xv,1560,'沿不同出边分别确定载荷信号',30)
txt(xv,1610,'返回 (f^A, f^O)：逐出边、逐端口值族',27)
txt(xv,1663,'各坐标可给出相同或不同载荷，也可取 ⊥',26,MUTED)

# Two outgoing messages correspond to the two successors in the overview.
arrow([(1275,1710),(1275,1750),(1210,1794)],BLUE)
arrow([(1585,1710),(1585,1750),(1650,1794)],BLUE)
label(1130,1750,'载荷 p1',BLUE,27)
label(1730,1750,'载荷 p2',BLUE,27)
circle(1190,1830,38,'w1')
circle(1670,1830,38,'w2')
txt(1190,1890,'逻辑时间 θ + δ(a1) 到达',25,BLUE)
txt(1670,1890,'逻辑时间 θ + δ(a2) 到达',25,BLUE)
txt(xv,1942,'输出端口有值时，记录在逻辑时间 θ',26,MUTED)
txt(xv,1982,'本例画出向两个直接后继发送的消息。',26,MUTED)

box((210,1760,930,1995),'#FFFFFF','#D2DDE8')
txt(xu,1802,'q_cmp：本次 Full 的计算快照',28)
txt(xu,1850,'q_next：后续节点事件读取的状态',28)
txt(xu,1898,'y_next：下次区域选择读取的历史',28)
txt(xu,1946,'旧 q 保留至 Next 使用',28,MUTED)

txt(1000,2051,'c 是局部控制量，可用于调制本地计算或输出；每个候选都取得自己的 c。',29,INK)
txt(1000,2102,'区域 E 的另一个节点在 θ 无输入，状态保持；各列的 q、h、d 等均属本节点。',28,MUTED)
txt(1000,2143,'q_tilde 表示候选状态；κ = 1 是本例的状态采用规则。',26,MUTED)

im.resize((W,H),Image.Resampling.LANCZOS).save(OUT)
print(OUT)
print(d.save_svg(OUT.with_suffix('.svg'), source=Path(__file__).name))
