"""Small SVG vocabulary for editable, reproducible teaching diagrams."""
from html import escape
import json
import math
from pathlib import Path
import re

import cairosvg
from PIL import ImageFont

BG = '#f8fafc'
INK = '#20364b'
MUTED = '#60758a'
BLUE = '#286da7'
GREEN = '#27816c'
GOLD = '#a67937'
PURPLE = '#8566a1'
EDGE = '#7b91a5'
PALE = {'blue': '#eaf3fc', 'green': '#eaf6f1', 'gold': '#fff4e4',
        'purple': '#f2edf8', 'gray': '#eef2f6', 'white': '#ffffff'}
COLORS = {'blue': BLUE, 'green': GREEN, 'gold': GOLD, 'purple': PURPLE, 'gray': MUTED}
FONT_DIR = Path('/usr/share/fonts/opentype/noto')
MATH_FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def font(size, bold=False, math_font=False):
    if math_font:
        return ImageFont.truetype(str(MATH_FONT), round(size))
    return ImageFont.truetype(str(FONT_DIR / ('NotoSansCJK-Bold.ttc' if bold else
                                            'NotoSansCJK-Regular.ttc')), round(size), index=2)


class Figure:
    def __init__(self, title, subtitle, sources, width=2400, height=1350):
        self.width, self.height = width, height
        self.sources = sources
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      f'<title>{escape(title)}</title>',
                      f'<desc>{escape(subtitle)}</desc>',
                      f'<metadata>{escape(json.dumps(sources, ensure_ascii=False))}</metadata>',
                      f'<rect width="{width}" height="{height}" fill="{BG}"/>']
        self.bounds = []
        self.text(width/2, 77, title, 53, bold=True)
        self.text(width/2, 145, subtitle, 29, MUTED)

    def rect(self, x, y, w, h, fill='white', stroke=None, radius=23, sw=2):
        color = PALE.get(fill, fill)
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def text(self, x, y, text, size=30, color=INK, bold=False, anchor='middle', math_font=False):
        width = font(size, bold, math_font).getlength(text)
        left = x - (width/2 if anchor == 'middle' else width if anchor == 'end' else 0)
        self.bounds.append((left, y-size*.55, left+width, y+size*.6, text))
        family = 'DejaVu Sans' if math_font else 'Noto Sans CJK SC'
        self.parts.append(f'<text x="{x}" y="{y+size*.35}" text-anchor="{anchor}" font-family="{family}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" xml:space="preserve">{escape(text)}</text>')

    def lines(self, x, y, lines, size=30, color=INK, step=45, bold=False, anchor='middle'):
        for i, line in enumerate(lines):
            self.text(x, y+i*step, line, size, color, bold, anchor)

    def math(self, x, y, expression, size=34, color=INK, anchor='middle'):
        """Place basic subscripts/superscripts explicitly, with editable SVG text."""
        if re.search(r'[\u3400-\u9fff]', expression):
            raise ValueError('Use text() for Chinese labels; math() uses a Latin/Greek font.')
        runs, pos = [], 0
        for match in re.finditer(r'([_^])(?:\{([^{}]*)\}|([A-Za-z0-9]+))', expression):
            if match.start() > pos:
                runs.append((expression[pos:match.start()], size, 0))
            runs.append((match[2] if match[2] is not None else match[3], size*.65,
                         size*.27 if match[1] == '_' else -size*.45))
            pos = match.end()
        if pos < len(expression):
            runs.append((expression[pos:], size, 0))
        width = sum(font(s, math_font=True).getlength(t) for t, s, _ in runs)
        cursor = x - (width/2 if anchor == 'middle' else width if anchor == 'end' else 0)
        for t, s, shift in runs:
            self.text(cursor, y+shift, t, s, color, anchor='start', math_font=True)
            cursor += font(s, math_font=True).getlength(t)

    def line(self, points, color=EDGE, width=4, dashed=False):
        pts = ' '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="10 8"' if dashed else ''
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash}/>')

    def arrow(self, points, color=EDGE, width=4, head=16, dashed=False):
        self.line(points, color, width, dashed)
        (px, py), (x, y) = points[-2:]
        a = math.atan2(y-py, x-px)
        triangle = [(x,y), (x-head*math.cos(a-.45),y-head*math.sin(a-.45)),
                    (x-head*math.cos(a+.45),y-head*math.sin(a+.45))]
        self.parts.append('<polygon points="'+' '.join(f'{xx},{yy}' for xx,yy in triangle)+f'" fill="{color}"/>')

    def circle(self, x, y, radius=35, label=None, color=BLUE, fill='white', size=28):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" stroke="{color}" stroke-width="3" fill="{PALE.get(fill, fill)}"/>')
        if label:
            self.text(x,y,label,size,color,bold=True)

    def card(self, x, y, w, h, title, lines=(), tone='blue', size=30):
        color = COLORS[tone]
        self.rect(x,y,w,h,tone,color,sw=1.5)
        self.text(x+w/2,y+43,title,35,color,True)
        if lines:
            self.lines(x+w/2,y+98,lines,size,step=43)

    def tag(self, x, y, text, color=MUTED, size=26):
        w = font(size).getlength(text)+24
        self.rect(x-w/2,y-size*.7,w,size*1.5,BG,radius=7)
        self.text(x,y,text,size,color)

    def footer(self, text, source):
        self.text(self.width/2,self.height-88,text,28,MUTED)
        self.text(self.width/2,self.height-37,source,23,MUTED)

    def save(self, name, output_dir):
        outside = [entry for entry in self.bounds if entry[0]<15 or entry[1]<10 or
                   entry[2]>self.width-15 or entry[3]>self.height-10]
        if outside:
            raise ValueError(f'{name}: text outside canvas: {outside}')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        svg = '\n'.join(self.parts+['</svg>'])+'\n'
        path = output_dir/(name+'.svg')
        path.write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(output_dir/(name+'.png')))
        return path
