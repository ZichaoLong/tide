"""Native SVG figures for the design introduction.

HB states use instance §2: b=2, h=1, n=3, offsets={0,1}. The explicit
residual diagram uses instances/explicit-residual-moe.md and checked traces
supplied by build.py. Transformer is a pre-norm structural illustration.
"""
from html import escape

INK = '#193d46'
MUTED = '#677e82'
BLUE = '#327bb1'
GREEN = '#267b5c'
GOLD = '#bd7c25'
BOX = '#eef2ef'

REGIONS = {
    'I': (58, 152, 164, 264, '入口 I', [228, 298, 368]),
    'L': (354, 36, 164, 208, '左区 L', [104, 164, 224]),
    'R': (354, 340, 164, 208, '右区 R', [408, 468, 528]),
    'E': (650, 152, 164, 264, '出口 E', [228, 298, 368]),
}
POSITIONS = {f'{r}{s}': (x + w / 2, ys[s]) for r, (x, y, w, h, label, ys) in REGIONS.items() for s in range(3)}
EDGES = [(f'{a}{s}', f'{b}{(s+d)%3}') for a, b in [('I','L'),('I','R'),('L','E'),('R','E')] for s in range(3) for d in (0,1)]
ROUTES = [dict(I=0,L=0,R=1,E=1), dict(I=1,L=1,R=2,E=1)]


def text(x, y, value, size=22, fill=INK, anchor='middle', weight=500):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'


def start(label, width=900, height=580):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label, quote=True)}" font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif"><defs><marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L6 3L0 6" fill="none" stroke="{BLUE}" stroke-width="1.6"/></marker><marker id="arrow-ink" markerWidth="10" markerHeight="10" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L7 4L0 8" fill="none" stroke="{INK}" stroke-width="1.6"/></marker></defs>'


def snapshot(round_index, level):
    route = ROUTES[round_index]
    candidate = {f'I{s}' for s in range(3)} if level >= 1 else set()
    active = {f'I{route["I"]}'} if level >= 2 else set()
    used = []
    if level >= 3:
        used += [(a,b) for a,b in EDGES if a in active and a.startswith('I')]
        candidate |= {b for a,b in used}
        active |= {f'L{route["L"]}', f'R{route["R"]}'}
    if level >= 4:
        used += [(a,b) for a,b in EDGES if a in active and a[0] in 'LR']
        candidate |= {b for a,b in used}
        active.add(f'E{route["E"]}')
    return candidate, active, used


def graph(stage='neighbors'):
    states = {'regions':(0,0), 'neighbors':(0,0), 'input0':(0,1), 'select0':(0,2), 'middle0':(0,3), 'exit0':(0,4), 'output0':(0,5), 'memory0':(0,0), 'input1':(1,1), 'select1':(1,2), 'middle1':(1,3), 'output1':(1,5)}
    r, level = states[stage]
    candidate, active, used = snapshot(r, level)
    remembered = set(candidate)
    if r == 1 or stage == 'memory0':
        remembered |= snapshot(0,4)[0]
    highlighted = [(a,b) for a,b in EDGES if a == 'I0'] if stage == 'neighbors' else used
    parts = [start(f'HB 三层图，{stage}，本轮累计候选 {len(candidate)}，Full {len(active)}，内部消息 {len(used)}')]
    parts.append(f'<g data-candidates="{len(candidate)}" data-full="{len(active)}" data-messages="{len(used)}">')
    for region, (x,y,w,h,label,ys) in REGIONS.items():
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{BOX}" stroke="#d4dfd9"/>')
        parts.append(text(x+w/2,y+29,label,21,INK,weight=650))
    if stage != 'regions':
        for a,b in EDGES:
            x1,y1=POSITIONS[a]; x2,y2=POSITIONS[b]
            parts.append(f'<path d="M{x1+21} {y1} C{x1+135} {y1} {x2-135} {y2} {x2-23} {y2}" fill="none" stroke="#bbc9c7" stroke-width="1.5" opacity=".58"/>')
        for a,b in highlighted:
            x1,y1=POSITIONS[a]; x2,y2=POSITIONS[b]
            parts.append(f'<path class="message-edge" data-edge="{a}-{b}" d="M{x1+21} {y1} C{x1+135} {y1} {x2-135} {y2} {x2-25} {y2}" fill="none" stroke="{BLUE}" stroke-width="3.5" marker-end="url(#arrow-blue)"/>')
    for node,(x,y) in POSITIONS.items():
        if node in candidate:
            parts.append(f'<circle cx="{x}" cy="{y}" r="27" fill="none" stroke="{BLUE}" stroke-width="2"/>')
        fill = GOLD if node in active else '#fafbf8'
        stroke = GOLD if node in active else '#8caaa5'
        parts.append(f'<circle data-node="{node}" data-active="{str(node in active).lower()}" cx="{x}" cy="{y}" r="20" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        parts.append(text(x,y+7,node[1],22,'#ffffff' if node in active else INK,weight=650))
        if node in remembered:
            parts.append(f'<rect x="{x+23}" y="{y+17}" width="10" height="10" rx="2" fill="{GREEN}"/>')
        if stage=='memory0' and node=='I1':
            parts.append(f'<circle cx="{x}" cy="{y}" r="29" fill="none" stroke="{GREEN}" stroke-width="3"/>')
    if level >= 1:
        for s in range(3):
            x,y=POSITIONS[f'I{s}']
            parts.append(f'<path d="M10 {y} H{x-32}" fill="none" stroke="{BLUE}" stroke-width="2.6" marker-end="url(#arrow-blue)"/>')
        parts.append(text(138,448,'h₀ = +1' if r==0 else 'h₁ = −1',24,BLUE,weight=650))
    if level >= 5:
        x,y=POSITIONS['E1']
        parts.append(f'<path d="M{x+32} {y} H888" fill="none" stroke="{BLUE}" stroke-width="3" marker-end="url(#arrow-blue)"/>')
        parts.append(text(735,448,'O₀ = +1' if r==0 else 'O₁ = −1',24,BLUE,weight=650))
    parts.append('</g></svg>')
    return ''.join(parts)


def chain():
    p=[start('熟悉的顺序计算：输入、两次计算、输出',1050,220)]
    for i,(label,small) in enumerate([('输入','h'),('计算','f₁'),('计算','f₂'),('输出','y')]):
        x=28+i*270
        p.append(f'<rect x="{x}" y="38" width="185" height="132" rx="18" fill="{BOX}" stroke="#c9d8d2"/>')
        p.append(text(x+92,91,label,27));p.append(text(x+92,136,small,29,BLUE))
        if i<3:p.append(f'<path d="M{x+195} 104H{x+252}" stroke="{INK}" stroke-width="2" marker-end="url(#arrow-ink)"/>')
    return ''.join(p)+'</svg>'


def expansion():
    p=[start('有限入口和出口，经有界邻域逐步扩散与收拢',1000,390)]
    layers=[[195],[110,280],[60,150,240,330],[110,280],[195]]
    for l in range(4):
        for i,y in enumerate(layers[l]):
            targets=[i*2,i*2+1] if l<2 else [i//2]
            for j in targets:
                if j>=len(layers[l+1]):continue
                y2=layers[l+1][j];x=80+l*210
                p.append(f'<path d="M{x+19} {y}C{x+105} {y} {x+105} {y2} {x+191} {y2}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for l,ys in enumerate(layers):
        for y in ys:
            p.append(f'<circle cx="{80+l*210}" cy="{y}" r="18" fill="{INK if l in (0,4) else "#f7f8f3"}" stroke="{INK}" stroke-width="2"/>')
    p += [text(80,378,'固定入口',21),text(920,378,'固定出口',21),text(500,378,'多跳传播 · 示意图',21,MUTED)]
    return ''.join(p)+'</svg>'


def box(x, y, w, h, title, detail='', color=INK, fill=BOX):
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="1.7"/>']
    p.append(text(x+w/2, y+(h/2+8 if not detail else h/2-3), title, 23, color, weight=650))
    if detail:
        p.append(text(x+w/2, y+h/2+24, detail, 18, MUTED))
    return ''.join(p)


def wire(path, color=BLUE, dashed=False):
    dash = 'stroke-dasharray="6 5"' if dashed else ''
    marker = 'arrow-ink' if color == INK else 'arrow-blue'
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" {dash} marker-end="url(#{marker})"/>'


def transformer():
    p = [start('Pre-norm Transformer block：Norm、Attention、Add、Norm、FFN、Add；两条显式残差旁路', 1320, 420)]
    p.append('<rect x="105" y="40" width="1040" height="300" rx="20" fill="none" stroke="#9bb2ac" stroke-dasharray="8 6"/>')
    p.append(text(625, 73, '一个完整 Transformer block（pre-norm 示意）', 23, MUTED))
    for a,b in [(60,130),(260,310),(490,536),(584,650),(780,830),(1010,1056),(1104,1260)]:
        p.append(wire(f'M{a} 240H{b}'))
    p += [box(130,200,130,80,'Norm'),box(310,200,180,80,'Attention'),
          box(650,200,130,80,'Norm'),box(830,200,180,80,'FFN / MoE')]
    for x in (560,1080):
        p.append(f'<circle cx="{x}" cy="240" r="24" fill="#f7f6f0" stroke="{INK}" stroke-width="2"/>')
        p.append(text(x,248,'+',30))
        p.append(text(x,307,'Add',21,MUTED))
    p.append(wire('M90 240V110H560V214',INK))
    p.append(wire('M615 240V145H1080V214',INK))
    p += [text(40,247,'h',27,BLUE),text(1230,218,'输出',22,BLUE),
          text(615,277,'x',24,BLUE),text(330,132,'保留 h',20,INK),text(910,128,'保留 x',20,INK),
          text(625,395,'节点边界可以覆盖整个 block，也可以继续拆开内部计算。',26,INK)]
    return ''.join(p)+'</svg>'


def number(value):
    return str(value).replace('-', '−')


def moe(stage='roles', trace=None):
    """One stable geometry; route/values are rendered from the checked model."""
    numeric = trace is not None
    mapped = stage != 'roles'
    complete = stage in ('result','history1','history2','phases')
    computed = stage in ('contributions','result','history1','history2','phases')
    selected = set(trace['active']) if numeric else set()
    p = [start('显式 residual MoE：分发、区域选择、专家贡献、加权和、残差相加',900,560)]
    if numeric:
        p.append(f'<g data-example="explicit-residual" data-expert-full="{len(selected) if computed else 0}" data-output="{number(trace["output"]) if complete else "pending"}">')
    else:
        p.append('<g>')
    p.append('<rect x="286" y="50" width="295" height="424" rx="18" fill="#eef2ef" stroke="#b6c8c1"/>')
    p.append(text(434,78,'专家区域 · 最多激活 2 个' if mapped else '三个候选专家',20,INK,weight=650))
    p.append(box(326,96,214,58,'区域选择' if mapped else '选择专家',
                 'SelStep → A、aᵢ' if mapped else '集合 A · 权重 aᵢ',GOLD,'#f4ead8'))
    # Gold dashed control path, separate from blue payload paths.
    p.append(f'<path d="M326 125H303V409" fill="none" stroke="{GOLD}" stroke-width="2" stroke-dasharray="5 5"/>')
    for y in (217,307,397):
        p.append(f'<path d="M303 {y}H326" stroke="{GOLD}" stroke-width="2" stroke-dasharray="5 5"/>')
    p.append(box(40,240,154,92,'分发 S' if mapped else '分发输入','x = '+number(trace['input']) if numeric else '当前表示 x'))
    for i,y in enumerate((192,282,372),1):
        node=f'E{i}'
        active = not numeric or node in selected
        color = GOLD if numeric and active else INK if not numeric else MUTED
        detail = f'E{str(i).translate(str.maketrans("123","₁₂₃"))}(x) · 返回贡献'
        if numeric:
            detail = ('贡献 '+number(trace['values'][node])) if computed and active else ('已选 · a = '+number(trace['weights'][node])) if active else '未选 · 不计算'
        p.append(box(326,y,214,68,f'专家 E{str(i).translate(str.maketrans("123","₁₂₃"))}',detail,color,'#f4ead8' if numeric and active else '#fafbf8'))
        p.append(wire(f'M194 286C250 286 255 {y+50} 324 {y+50}'))
        shade = BLUE if active and (computed or not numeric) else '#bcc9c6'
        dash = 'stroke-dasharray="5 5"' if not active else ''
        p.append(f'<path d="M540 {y+36}H622C647 {y+36} 647 {218+i*15} 659 {218+i*15}" fill="none" stroke="{shade}" stroke-width="2.5" {dash}/>')
        if computed and active:
            p.append(text(610,y+22,number(trace['weighted'][node]),21,BLUE,weight=650))
    p.append('<rect x="648" y="170" width="220" height="296" rx="18" fill="none" stroke="#b6c8c1"/>')
    p.append(text(758,197,'合并节点 M' if mapped else '合并贡献与基线',21,INK,weight=650))
    p.append(box(661,220,194,77,'贡献求和 Σ', 'r = '+number(trace['contribution']) if complete else 'r = Σ aᵢ Eᵢ(x)'))
    p.append(wire('M758 298V368'))
    p.append(f'<circle cx="758" cy="395" r="26" fill="#f7f6f0" stroke="{INK}" stroke-width="2"/>')
    p += [text(758,404,'+',32),text(812,377,'Add',20,MUTED)]
    p.append(wire('M116 333V500H758V423',INK))
    p.append(text(400,525,'残差旁路：x'+(' = '+number(trace['input']) if numeric else '')+(' · 时延 2' if mapped else ''),22,INK))
    p.append(wire('M784 395H898'))
    p.append(text(809,447,'y = '+number(trace['output']) if complete else 'y = x + r',24,BLUE,weight=650))
    if mapped:
        p.append(text(124,214,'θ = 1 + 3t',19,MUTED))
        p.append(text(434,465,'θ = 2 + 3t',19,MUTED))
        p.append(text(758,153,'θ = 3 + 3t',19,MUTED))
    p.append('</g></svg>')
    return ''.join(p)


if __name__=='__main__':
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/'design-introduction'
    folder.mkdir(exist_ok=True)
    for name in ['regions','neighbors','input0','select0','middle0','exit0','output0','memory0','input1','select1','middle1','output1']:
        (folder/f'hb-{name}.svg').write_text(graph(name))
    (folder/'sequential.svg').write_text(chain())
    (folder/'bounded-propagation.svg').write_text(expansion())
