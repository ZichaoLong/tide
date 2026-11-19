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
OUT = _output_dir / 'tide-structure.png'
W, H, SCALE = 2800, 1880, 2
REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG, INK, MUTED = '#F8FAFD', '#20364B', '#60758A'
EDGE, BLUE = '#7B91A5', '#286DA7'
COLORS = {
    'A': ('#EDF4FC', '#ADC6E0', '#457CB4'),
    'B': ('#ECF7F4', '#AACFC3', '#2C806D'),
    'C': ('#ECF7F4', '#AACFC3', '#2C806D'),
    'D': ('#ECF7F4', '#AACFC3', '#2C806D'),
    'E': ('#FFF6E9', '#DFC7A2', '#A67937'),
    'F': ('#FFF6E9', '#DFC7A2', '#A67937'),
    'G': ('#EDF4FC', '#ADC6E0', '#457CB4'),
    'R': ('#F6F1FA', '#CBBBDC', '#9073AB'),
}
im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
d = VectorDrawing(im, scale=SCALE, background=BG)

def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, round(n*SCALE), index=2)

def txt(x, y, s, n=30, color=INK, bold=False, anchor='mm'):
    d.text((round(x*SCALE), round(y*SCALE)), s, font=font(n, bold),
           fill=color, anchor=anchor)

def rect(bounds, fill, outline=None, width=2, radius=28):
    d.rounded_rectangle(tuple(round(t*SCALE) for t in bounds),
                       radius=round(radius*SCALE), fill=fill,
                       outline=outline, width=round(width*SCALE))

def ellipse(bounds, fill, outline, width=3):
    d.ellipse(tuple(round(t*SCALE) for t in bounds), fill=fill,
              outline=outline, width=round(width*SCALE))

def line(points, color=EDGE, width=4):
    d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points],
           fill=color, width=round(width*SCALE), joint='curve')

def arrow(points, color=EDGE, width=4, head=15):
    line(points, color, width)
    (px, py), (x, y) = points[-2:]
    angle = math.atan2(y-py, x-px)
    pts = [(x, y), (x-head*math.cos(angle-.44), y-head*math.sin(angle-.44)),
           (x-head*math.cos(angle+.44), y-head*math.sin(angle+.44))]
    d.polygon([(round(xx*SCALE), round(yy*SCALE)) for xx, yy in pts], fill=color)

def bezier_path(points):
    p0, p1, p2, p3 = points
    path = []
    for i in range(101):
        t = i/100
        path.append(tuple((1-t)**3*p0[k]+3*(1-t)**2*t*p1[k]
                          +3*(1-t)*t*t*p2[k]+t**3*p3[k] for k in (0,1)))
    return path

def bezier(points, color=EDGE, width=4):
    arrow(bezier_path(points), color, width)

nodes = {
    'a1': (420, 580, 'A'), 'a2': (420, 690, 'A'), 'a3': (420, 800, 'A'),
    'b1': (980, 320, 'B'), 'b2': (980, 420, 'B'), 'b3': (980, 520, 'B'),
    'c1': (1230, 700, 'C'), 'c2': (1230, 780, 'C'), 'c3': (1230, 860, 'C'),
    'd1': (980, 1080, 'D'), 'd2': (980, 1180, 'D'),
    'e1': (1700, 470, 'E'), 'v': (1700, 590, 'E'), 'e3': (1700, 710, 'E'),
    'f1': (1700, 1040, 'F'), 'f2': (1700, 1140, 'F'),
    'g1': (2455, 675, 'G'), 'g2': (2455, 795, 'G'), 'g3': (2455, 915, 'G'),
    'r1': (1270, 1280, 'R'), 'r2': (2110, 1280, 'R'),
}

def radius(name):
    return 48 if name == 'v' else (29 if name.startswith('r') else 31)

def edge(a, b, color=EDGE, width=4):
    x1, y1, _ = nodes[a]
    x2, y2, _ = nodes[b]
    x1 += radius(a)+4
    x2 -= radius(b)+7
    dx = x2-x1
    bezier([(x1, y1), (x1+.44*dx, y1), (x2-.44*dx, y2), (x2, y2)], color, width)

txt(1400, 66, 'Tide / TimedDAG：发散、并行处理与逐步汇合', 53, bold=True)
txt(1400, 132, '节点构成 DAG；同一逻辑时间有输入的同区域节点共同选择，确定激活节点与调制控制。', 32, MUTED)

regions = {
    'A': (265, 465, 575, 880),
    'B': (825, 205, 1135, 570),
    'C': (1075, 595, 1385, 915),
    'D': (825, 970, 1135, 1240),
    'E': (1540, 355, 1860, 775),
    'F': (1540, 925, 1860, 1220),
    'G': (2290, 560, 2615, 980),
}
for name, bounds in regions.items():
    fill, border, dark = COLORS[name]
    rect(bounds, fill, border, 2, 36)
    txt((bounds[0]+bounds[2])/2, bounds[1]+48, '区域 '+name, 34, dark, True)

main_edges = [
    ('a1','b1'), ('a1','b2'), ('a1','b3'), ('a1','c1'),
    ('a2','c1'), ('a2','c2'), ('a2','c3'), ('a2','d1'),
    ('a3','d1'), ('a3','d2'),
    ('b1','e1'), ('b2','v'), ('b3','v'), ('c1','v'), ('c2','e3'),
    ('c2','f1'), ('c3','f1'), ('d1','f2'), ('d2','f2'),
    ('e1','g1'), ('v','g1'), ('v','g2'), ('e3','g2'),
    ('f1','g2'), ('f1','g3'), ('f2','g3'),
    ('b3','c1'), ('b1','g1'),
]
extra_edges = [('d2','r1'), ('r1','f2'), ('f2','r2'), ('r1','r2'), ('r2','g3')]

# All ordinary regions are internally edgeless. R is a quiet boundary example.
assert all(nodes[a][2] != nodes[b][2] for a,b in main_edges)
assert all(nodes[a][0] < nodes[b][0] for a,b in main_edges+extra_edges)
assert sum(b == 'v' for a,b in main_edges) == 3
assert sum(a == 'v' for a,b in main_edges) == 2

for a,b in main_edges:
    if a != 'v' and b != 'v' and (a,b) != ('b1','g1'):
        edge(a,b)
skip_path = bezier_path([(1015,320),(1260,190),(1450,215),(1700,215)])
skip_path += bezier_path([(1700,215),(2170,215),(2200,675),(2417,675)])[1:]
arrow(skip_path, EDGE, 4)
for a,b in extra_edges:
    edge(a,b, '#A2ADBA', 3)
for a,b in main_edges:
    if a == 'v' or b == 'v':
        edge(a,b,BLUE,5)

# Input and output records enter and leave through separate ports.
for y in [580,690,800]:
    arrow([(120,y),(381,y)],EDGE,4)
for y in [675,795,915]:
    arrow([(2492,y),(2740,y)],EDGE,4)
txt(175, 520, '输入', 29, MUTED)
txt(2710, 610, '输出', 29, MUTED)

for name,(x,y,region) in nodes.items():
    fill,border,dark = COLORS[region]
    r = radius(name)
    if name == 'v':
        ellipse((x-r-8,y-r-8,x+r+8,y+r+8),'#D5EAF8','#D5EAF8',1)
        ellipse((x-r,y-r,x+r,y+r),'#FFFFFF',BLUE,4)
        txt(x,y-1,'v',42,BLUE,True)
    elif region == 'R':
        ellipse((x-r,y-r,x+r,y+r),fill,dark,3)
        txt(x,y-1,name,23,dark,True)
    else:
        ellipse((x-r,y-r,x+r,y+r),'#FFFFFF',dark,3)
        display_name = {'e1':'u', 'g1':'w1', 'g2':'w2'}.get(name)
        if display_name:
            txt(x,y-1,display_name,27,dark,True)
        else:
            ellipse((x-5,y-5,x+5,y+5),dark,dark,1)

# Only a few reading cues are needed; the branches do the main explanation.
txt(420, 940, '分发到多条分支', 30, INK, True)
txt(980, 181, '多组候选并行展开', 28, MUTED)
txt(1700, 309, '来自不同分支的消息汇合', 28, MUTED)
txt(2455, 1040, '继续收拢', 30, INK, True)
txt(1840, 180, 'B → G：跨过中间区域直达', 27, MUTED)
txt(1190, 553, 'B → C', 25, MUTED)

txt(210, 1080, '浅色轮廓：共同选择的区域', 26, MUTED, anchor='lm')
txt(210, 1132, '圆点：有自身状态的节点', 26, MUTED, anchor='lm')
txt(210, 1182, '箭头：沿 DAG 传递消息', 26, MUTED, anchor='lm')
txt(1690, 1337, '补充区域 R：跨位置划分，允许内部连边', 27, COLORS['R'][2])
txt(1690, 1376, '仅展示语义允许的范围；设计价值需另行验证。', 25, MUTED)

# The local flow corresponds to the highlighted three-input/two-output node v.
rect((110,1450,2690,1805),'#FFFFFF','#D2DDE8',2,25)
txt(155,1494,'示例节点 v 的一次处理',33,INK,True,'lm')
txt(2640,1494,'聚合到达逻辑时间相同的完整消息集合；本例中 v 被激活',28,MUTED,anchor='rm')

blocks = [
    (445,945,'P  准备',('聚合 Agg → 本地内容 h','准备候选状态与描述量'),'#EDF4FB','#B7CDDF',BLUE),
    (1005,1370,'S  区域选择',('与区域 E 其他候选节点','共同确定激活与调制控制'),'#FFF6E9','#DFC7A2','#A67937'),
    (1430,1830,'U  状态作用',('得到本次计算快照','与下一持久状态'),'#F4F6F9','#C3CFDA','#536E86'),
    (1890,2270,'F  输出',('若 v 被激活，调用 Full','各出边分别产生载荷信号'),'#EDF4FB','#B7CDDF',BLUE),
]
for left,right,title,detail,fill,border,dark in blocks:
    rect((left,1570,right,1730),fill,border,2,18)
    txt((left+right)/2,1608,title,33,dark,True)
    txt((left+right)/2,1660,detail[0],25,INK)
    txt((left+right)/2,1700,detail[1],25,INK)
for left,right in [(945,1005),(1370,1430),(1830,1890)]:
    arrow([(left+8,1650),(right-9,1650)],EDGE,4,13)
for sy,ey in [(1605,1618),(1650,1650),(1695,1682)]:
    arrow([(205,sy),(434,ey)],BLUE,4,13)
    ellipse((180,sy-8,196,sy+8),BLUE,BLUE,1)
for sy,ey,name in [(1627,1612,'w1'),(1673,1688,'w2')]:
    arrow([(2282,sy),(2530,ey)],BLUE,4,13)
    ellipse((2538,ey-25,2588,ey+25),'#FFFFFF',BLUE,2)
    txt(2563,ey,name,20,BLUE)
txt(2420,1583,'载荷 p1',25,BLUE)
txt(2420,1720,'载荷 p2',25,BLUE)
txt(309,1753,'同一逻辑时间的',24,MUTED)
txt(309,1787,'多路输入消息',24,MUTED)
txt(2424,1767,'发给直接后继',24,MUTED)
txt(1400,1847,'各出边载荷可相同或不同；⊥ 表示该边本次无消息。同值广播是常用特例。',27,MUTED)

im.resize((W,H),Image.Resampling.LANCZOS).save(OUT)
print(OUT)
print(d.save_svg(OUT.with_suffix('.svg'), source=Path(__file__).name))
