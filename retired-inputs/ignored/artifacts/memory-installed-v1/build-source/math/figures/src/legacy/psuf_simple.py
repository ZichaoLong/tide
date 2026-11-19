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
OUT = _output_dir / 'PSUF-simple.png'
W, H, SCALE = 2400, 1320, 2
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG = '#F8FAFD'
INK, MUTED = '#20364B', '#60758A'
BLUE, GREEN, GOLD, EDGE = '#286DA7', '#27816C', '#A67937', '#7B91A5'
im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
d = VectorDrawing(im, scale=SCALE, background=BG)

def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, round(n*SCALE), index=2)

def text(x, y, s, n=32, color=INK, bold=False, anchor='mm'):
    d.text((round(x*SCALE), round(y*SCALE)), s, font=font(n, bold),
           fill=color, anchor=anchor)

def box(bounds, fill, outline=None, radius=24, width=2):
    d.rounded_rectangle(tuple(round(v*SCALE) for v in bounds),
                       radius=round(radius*SCALE), fill=fill,
                       outline=outline, width=round(width*SCALE))

def circle(x, y, r, fill, outline=None, width=2):
    d.ellipse(tuple(round(v*SCALE) for v in (x-r, y-r, x+r, y+r)),
              fill=fill, outline=outline, width=round(width*SCALE))

def arrow(points, color=EDGE, width=4, head=17):
    d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points],
           fill=color, width=round(width*SCALE), joint='curve')
    (px, py), (x, y) = points[-2:]
    a = math.atan2(y-py, x-px)
    pts = [(x, y), (x-head*math.cos(a-.45), y-head*math.sin(a-.45)),
           (x-head*math.cos(a+.45), y-head*math.sin(a+.45))]
    d.polygon([(round(xx*SCALE), round(yy*SCALE)) for xx, yy in pts], fill=color)

def label(x, y, s, n=28, color=MUTED):
    tw = d.textlength(s, font=font(n))/SCALE
    box((x-tw/2-14, y-n/2-9, x+tw/2+14, y+n/2+9), BG, radius=8)
    text(x, y, s, n, color)

def stage(cx, top, bottom, title, detail, fill, border, color):
    box((cx-300, top, cx+300, bottom), fill, border, width=2)
    text(cx, top+41, title, 39, color, True)
    text(cx, bottom-34, detail, 32)

xu, xv = 685, 1715
text(1200, 66, '区域 E：同一逻辑时间 θ 内的处理顺序', 53, INK, True)
for x, name in [(xu, 'u'), (xv, 'v')]:
    text(x, 151, '候选节点 '+name, 38, INK, True)

# Incoming messages match the example nodes in the structure overview.
circle(xu, 204, 8, BLUE)
arrow([(xu, 218), (xu, 259)], BLUE, 4, 13)
for x0, x1 in [(xv-135, xv-95), (xv, xv), (xv+135, xv+95)]:
    circle(x0, 204, 8, BLUE)
    arrow([(x0, 218), (x1, 259)], BLUE, 4, 13)

for x in (xu, xv):
    stage(x, 265, 393, 'P  准备', '汇聚消息，准备本地量',
          '#E8F1FB', '#A9C5DF', BLUE)
    arrow([(x, 393), (x, 473)], BLUE)
    label(x, 433, '描述量', 28, BLUE)

# Shared regional selection is the only box spanning the two node columns.
box((385, 475, 2015, 612), '#FFF6E9', '#DFC7A2')
text(1200, 519, 'S  区域选择', 42, GOLD, True)
text(1200, 575, '共同决定激活与调制', 35, GOLD)

for x, active in [(xu, False), (xv, True)]:
    color = BLUE if active else MUTED
    arrow([(x, 612), (x, 732)], color)
    label(x, 670, '激活' if active else '未激活', 35, color)
    stage(x, 735, 870, 'U  状态作用', '确定本次快照与后续状态',
          '#ECF7F4', '#AACFC3', GREEN)

# Future state is retained independently of whether the node was activated.
for bounds in [(70, 750, 325, 855), (2075, 750, 2330, 855)]:
    left, top, right, bottom = bounds
    box(bounds, '#F0F5F3', '#BDD5CD', radius=18)
    text((left+right)/2, 786, '后续状态', 30, GREEN, True)
    text((left+right)/2, 829, '留给下次处理', 25, MUTED)
arrow([(385, 802), (330, 802)], GREEN, 4, 13)
arrow([(2015, 802), (2070, 802)], GREEN, 4, 13)

# The unactivated branch ends after U; the activated branch uses its snapshot.
arrow([(xu, 870), (xu, 990)], '#96A4B2', 4, 14)
circle(xu, 1003, 10, '#96A4B2')
text(xu, 1051, '本次结束', 36, MUTED, True)
arrow([(xv, 870), (xv, 982)], BLUE, 5)
label(xv, 924, '本次快照', 29, BLUE)
stage(xv, 985, 1113, 'F  输出', 'Full 沿出边发送载荷',
      '#E8F1FB', '#A9C5DF', BLUE)

# Two direct successors match w1 and w2 in the structure overview.
arrow([(1570, 1113), (1570, 1140), (1545, 1172)], BLUE, 4, 14)
arrow([(1860, 1113), (1860, 1140), (1885, 1172)], BLUE, 4, 14)
text(1484, 1140, 'p1', 26, BLUE)
text(1946, 1140, 'p2', 26, BLUE)
for x, name in [(1520, 'w1'), (1910, 'w2')]:
    circle(x, 1207, 35, '#FFFFFF', BLUE, 3)
    text(x, 1205, name, 28, BLUE, True)
text(xv, 1272, '直接后继节点', 28, MUTED)

im.resize((W, H), Image.Resampling.LANCZOS).save(OUT)
print(OUT)
print(d.save_svg(OUT.with_suffix('.svg'), source=Path(__file__).name))
