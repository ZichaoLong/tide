"""Build the offline backbone atlas from shared reference constructions/counts.

Only potential paths are evaluated, not regional selection or neural functions.
"""
from collections import Counter, defaultdict
from pathlib import Path
import base64
import html
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'reference'))
sys.path.insert(0, str(ROOT.parents[2] / 'tools'))
from backbones import multiscale as g
from backbones import settle_graphs as sg
import capacity_reports
from native_svg import render_png
N, H, T = 16, 4, 4
PROFILES = g.PROFILES
KINDS = ['DDL','DDX','DD1','DD2','DD4','DH1','HH1','HH2','HH4','HD1','HD2','HD4']
COLORS = {'DDL':'#2563eb','DDX':'#168476','DD1':'#2563eb','DD2':'#7353bf','DD4':'#a21caf',
          'DH1':'#0f8b82','HH1':'#b57a13','HH2':'#b57a13','HH4':'#b57a13',
          'HD1':'#d65c36','HD2':'#d65c36','HD4':'#d65c36'}
PHASE_COLORS = ['#2374ab','#b45516','#7651b9','#168476']
INK, MUTED = '#23344b', '#5c6d81'
OUT = ROOT / 'generated'


def phase_counts(delay, origins, source_width, period=T):
    answer = []
    for phase in range(period):
        mask = 0
        for d in g.support(delay):
            if d % period == phase:
                mask |= (origins >> (d * source_width)) & ((1 << source_width) - 1)
        answer.append(bin(mask).count('1'))
    return answer


def make_case(key, widths, add_dd4=False, family='multiscale', hubs=H, n=N,
              profile=None, expansion=None, rounds=None):
    if family == 'multiscale':
        graph = g.construct(widths, n=n, h=H, placement='all' if hubs else 'none',
                            probes=g.probes_for(n), add_dd4=add_dd4)
    else:
        graph = sg.construct(widths, n=n, h=hubs, exchange=family == 'exchange')
    nodes, ids, outs, ins, kinds, regional = graph
    assert all(len(es) == len({v for v,k in es}) for es in outs)
    assert all(nodes[u][0] < nodes[v][0] for u,es in enumerate(outs) for v,k in es)
    potential, origins = g.reachability(nodes, outs, widths)
    detail, dorigins = g.reachability(nodes, outs, widths, only_detail=True)
    mid = len(widths) // 2
    middle, morigins = g.reachability(nodes, outs, widths, only_detail=True, origin_layer=mid)
    terminal = [u for u,x in enumerate(nodes) if x[0] == len(widths)-1 and x[2] == 'd']
    spans = ([1,2,4] if add_dd4 else [1,2]) if family == 'multiscale' else [1]
    expected = [set() for _ in widths]; expected[0] = {0}
    for l in range(1, len(widths)):
        expected[l] = {d+1 for s in spans if l>=s for d in expected[l-s]}
    assert all(g.support(detail[u]) == sorted(expected[-1]) for u in terminal)
    cycle = T if family == 'multiscale' else len(widths)+(2 if family == 'lattice' else 1)
    phase_d = [phase_counts(detail[u], dorigins[u], widths[0], cycle) for u in terminal]
    phase_all = [phase_counts(potential[u], origins[u], widths[0], cycle) for u in terminal]
    mid_counts = []
    for u in terminal:
        mask = 0
        for d in g.support(middle[u]):
            mask |= (morigins[u] >> (d * widths[mid])) & ((1 << widths[mid])-1)
        mid_counts.append(bin(mask).count('1'))
    bundles = Counter()
    edges = []
    for u,es in enumerate(outs):
        for v,kind in es:
            edges.append([u,v,KINDS.index(kind)])
            a,b = nodes[u],nodes[v]
            bundles[a[0],a[1],a[2],b[0],b[1],b[2],kind] += 1
    node_delays = [g.support(d) for d in potential]
    node_ddelays = [g.support(d) for d in detail]
    hub_counts = [sum(x[0] == l and x[1] == 0 and x[2] == 'h' for x in nodes)
                  for l in range(len(widths))]
    middle_workers = [u for u,x in enumerate(nodes) if x[0]==mid and x[1]==min(3, widths[mid]-1) and x[2]=='d']
    selected_d = max(middle_workers, key=lambda u:(
        len({k for v,k in ins[u]}), sum(k.startswith('HD') for v,k in ins[u]), len(ins[u])))
    stats = {
        'path_timing':'unit', 'phase_period':cycle,
        'positions': sum(widths), 'regions': len({x[:3] for x in nodes}), 'nodes':len(nodes),
        'ordinary_nodes':n*sum(widths),'hubs':sum(x[2]=='h' for x in nodes), 'edges':len(edges),
        'degrees':{r:[max(len(ins[u]) for u,x in enumerate(nodes) if x[2]==r),
                      max(len(outs[u]) for u,x in enumerate(nodes) if x[2]==r)] for r in (('d','h') if hubs else ('d',))},
        'ordinary_delays':sorted(expected[-1]),
        'ordinary_phase_sources':[[min(x[p] for x in phase_d), max(x[p] for x in phase_d)] for p in range(cycle)],
        'all_phase_sources':[[min(x[p] for x in phase_all), max(x[p] for x in phase_all)] for p in range(cycle)],
        'middle_sources':[min(mid_counts),max(mid_counts),widths[mid]],
        'earliest':dict(sorted(Counter(node_delays[u][0] for u in terminal).items())),
        'edges_by_kind':dict(sorted(kinds.items())),
    }
    if family == 'multiscale':
        stats['source_D_nodes'], stats['ordinary_phase_node_sources'] = g.source_node_coverage(nodes,outs,widths,True)
        _, stats['all_phase_node_sources'] = g.source_node_coverage(nodes,outs,widths)
    return {'key':key,'depth':len(widths),'widths':widths,'ordinary':n,'hubs':hubs,
            'profile':profile or ('multiscale-'+key if family=='multiscale' else key.removesuffix('-h')),
            'expansion':expansion, 'rounds':rounds,
            'hub_counts':hub_counts,'family':family,
            'label': key + ' · ' + ('多跨度' if family=='multiscale' else '原 HB' if family=='lattice' else '位交换'),
            'default_timing':'timed' if family=='multiscale' else 'settle',
            'settle_period':len(widths)+(2 if family=='lattice' else 1),
            'period':T,'probes':dict(zip((1,2,4),g.probes_for(n) if family=='multiscale' else (1,1,1))),
            'add_dd4':add_dd4,'nodes':nodes,'edges':edges,
            'bundles':[list(k)+[v] for k,v in sorted(bundles.items())],
            'delays':node_delays,'ordinary_delays':node_ddelays,
            'selected_d':selected_d,'selected_h':ids.get((mid,min(3,widths[mid]-1),'h',1)), 'stats':stats}


def esc(s): return html.escape(str(s))


def text(x,y,s,size=17,color=INK,weight=400,anchor='start', extra=''):
    return '<text x="{}" y="{}" font-size="{}" fill="{}" font-weight="{}" text-anchor="{}" {}>{}</text>'.format(x,y,size,color,weight,anchor,extra,esc(s))


def start(width,height,title):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(width,height,width,height,esc(title)),
           '<style>text{font-family:"Noto Sans CJK SC","Droid Sans","Microsoft YaHei",sans-serif}path{fill:none;stroke-linecap:round;stroke-linejoin:round}.position{cursor:pointer}.bundle{cursor:pointer}</style>',
           '<rect width="100%" height="100%" fill="#ffffff"/><defs>']
    for kind,color in COLORS.items():
        out.append('<marker id="arr{}" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="6" markerHeight="5" orient="auto"><path d="M0 0 L10 4 L0 8 Z" style="fill:{};stroke:none"/></marker>'.format(kind,color))
    out.append('</defs>')
    return out


def region_svg(case, static=False):
    depth,widths = case['depth'],case['widths']
    dx,dy = 164,70
    width,height = 170+(depth-1)*dx,max(widths)*dy+260
    out = start(width,height,'HB 区域连接')
    out += [text(24,35,'{} 层候选 · 区域连接{}'.format(depth,'（普通网络）' if static else ''),25,weight=700),
            text(24,66, '每位置 D 普通区含 {} 节点；'.format(case['ordinary'])+('非末层另有 4 节点 H 区，末层只有 D。' if case['hubs'] else '无 hub。')+'同层无消息边。',16),
            text(24,92,'一条线汇总一个边族的实际连线；δ=1。线的交叉不是连接点。',15,MUTED, extra='id="region-timing-label"')]
    def xy(l,p,role=None):
        y=180+(max(widths)-widths[l])*dy/2+p*dy
        return 85+l*dx,y+(16 if role=='h' else -5 if role=='d' else 0)
    for l,w in enumerate(widths):
        x,_=xy(l,0)
        out.append(text(x,119,'ℓ={} · {}位置'.format(l,w),13,MUTED,anchor='middle'))
    for l,p,r,m,q,t,kind,count in case['bundles']:
        if static and not kind.startswith('DD'):continue
        x,y=xy(l,p,r); xx,yy=xy(m,q,t)
        sx,tx=x+35,xx-35
        bend=min(62,(tx-sx)/2)
        # Curves joining longer spans use a small off-center route; no new vertices.
        lift = -20*(m-l-1) if m-l>1 else 0
        path='M {} {} C {} {} {} {} {} {}'.format(sx,y,sx+bend,y+lift,tx-bend,yy+lift,tx,yy)
        attrs='data-src="{},{}" data-dst="{},{}" data-kind="{}" data-count="{}" data-span="{}"'.format(l,p,m,q,kind,count,m-l)
        opacity='.52' if static else '.32'
        out.append('<path class="bundle" {} d="{}" stroke="{}" stroke-width="1.5" opacity="{}" marker-end="url(#arr{})"><title>{}({},{}) → {}({},{}) · {} · {} 条消息边 · δ=1</title></path>'.format(attrs,path,COLORS[kind],opacity,kind,r.upper(),l,p,t.upper(),m,q,kind,count))
    for l,w in enumerate(widths):
        for p in range(w):
            x,y=xy(l,p)
            out.append('<g class="position" data-pos="{},{}" tabindex="0" role="button" aria-label="选择位置 {},{}">'.format(l,p,l,p))
            out.append('<rect class="position-outline" x="{}" y="{}" width="82" height="61" rx="10" fill="#f7f9fc" fill-opacity=".95" stroke="#dce4ed" stroke-dasharray="3 3"/>'.format(x-41,y-29))
            out.append(text(x,y-16,'{},{}'.format(l,p),10,MUTED,anchor='middle'))
            for role,fill,stroke,label in [('d','#e8f0ff','#7599d2','D'+str(case['ordinary']))]+([('h','#fff1d8','#c5a05c','H4')] if case['hub_counts'][l] else []):
                _,yy=xy(l,p,role)
                out.append('<rect x="{}" y="{}" width="68" height="18" rx="5" fill="{}" stroke="{}"/>'.format(x-34,yy-9,fill,stroke))
                out.append(text(x,yy+4,label,11,anchor='middle'))
            out.append('</g>')
    footer=height-45
    out.append(text(24,footer,'{} 个位置 / {} 个区域 / {} 个节点；横向为层号，纵向为位置编号。'.format(case['stats']['positions'],case['stats']['regions'],case['stats']['nodes']),16))
    out.append(text(24,footer+26,('DD：普通节点传播；DH／HH／HD：可选摘要通路。')+('此静态图省略 hub 边；交互页可逐族查看全部边。' if static else '点击位置看邻接；点击连线看该区域对的完整节点矩阵。'),14,MUTED))
    out.append('</svg>')
    return ''.join(out)


def neighbors(case,u):
    ins,outs=defaultdict(list),defaultdict(list)
    for a,b,k in case['edges']:
        if b==u:
            l,p,r,i=case['nodes'][a];ins[l,p,r,KINDS[k]].append(i)
        if a==u:
            l,p,r,i=case['nodes'][b];outs[l,p,r,KINDS[k]].append(i)
    return ins,outs


def ego_svg(case,u):
    ins,outs=neighbors(case,u)
    height=max(650,220+max(len(ins),len(outs))*85)
    width=1540;cy=(height+100)/2
    l,p,role,i=case['nodes'][u]
    out=start(width,height,'单节点的完整上下游')
    out += [text(26,38,'{}({},{}; {}) · 完整一跳上下游'.format(role.upper(),l,p,i),26,weight=700),
            text(26,70,'每张侧卡按区域、角色与边族合并列出全部端点；括号内数字是节点编号。未画邻居之间的边。',17,MUTED),
            text(260,110,'全部入边 · {} 条'.format(sum(map(len,ins.values()))),19,weight=700,anchor='middle'),
            text(1280,110,'全部出边 · {} 条'.format(sum(map(len,outs.values()))),19,weight=700,anchor='middle')]
    for side,groups in [('in',ins),('out',outs)]:
        gx=25 if side=='in' else 1040
        top=145+(height-185-len(groups)*85)/2
        for row,((nl,np,nr,kind),idxs) in enumerate(sorted(groups.items())):
            y=top+row*85;color=COLORS[kind]
            sx,sy,tx,ty=(gx+475,y+33,630,cy) if side=='in' else (910,cy,gx,y+33)
            path='M {} {} C {} {} {} {} {} {}'.format(sx,sy,sx+55,sy,tx-55,ty,tx,ty)
            out.append('<path d="{}" stroke="{}" stroke-width="2" opacity=".72" marker-end="url(#arr{})"/>'.format(path,color,kind))
            out.append('<rect x="{}" y="{}" width="475" height="67" rx="10" fill="#fff" stroke="{}"/>'.format(gx,y,color))
            out.append(text(gx+16,y+25,'{}({},{})  ·  {}  ·  {} 条'.format(nr.upper(),nl,np,kind,len(idxs)),17,color,600))
            out.append(text(gx+16,y+51,'节点 '+', '.join(map(str,sorted(idxs))),16))
    color='#2563eb' if role=='d' else '#b57a13'
    out.append('<rect x="630" y="{}" width="280" height="190" rx="16" fill="#f5f8fd" stroke="{}" stroke-width="2"/>'.format(cy-95,color))
    out.append(text(770,cy-51,'{}({},{}; {})'.format(role.upper(),l,p,i),25,color,700,anchor='middle'))
    out.append(text(770,cy-16,'所有相邻消息边 δ=1',16,anchor='middle'))
    out.append(text(770,cy+18,'首层输入的潜在总时延',15,MUTED,anchor='middle'))
    out.append(text(770,cy+46,', '.join(map(str,case['delays'][u])),19,weight=600,anchor='middle'))
    out.append(text(770,cy+75,'Full 沿固定出边广播',14,MUTED,anchor='middle'))
    out.append(text(25,height-22,'同刻多来源在该节点聚合；异刻到达通过其持久状态衔接。这里不模拟选择与实际内容。',16,MUTED))
    out.append('</svg>');return ''.join(out)


def matrix_svg(case,src=(3,2),dst=(4,2)):
    cell=29;x0,y0=180,168
    source=[u for u,x in enumerate(case['nodes']) if tuple(x[:2])==src]
    target=[u for u,x in enumerate(case['nodes']) if tuple(x[:2])==dst]
    rows,columns=len(source),len(target)
    out=start(1240,850,'区域对的完整节点连接矩阵')
    out += [text(25,38,'位置 {} → {}：全部 {}×{} 个端点组合'.format(src,dst,rows,columns),25,weight=700),
            text(25,70,'行是上游节点，列是下游节点；一个彩点是一条真实消息边。空格没有边。',17,MUTED)]
    si={u:i for i,u in enumerate(source)};ti={u:i for i,u in enumerate(target)}
    counts=Counter()
    out.append(text(x0+N*cell/2,y0-49,'下游普通区 D',17,'#2563eb',600,anchor='middle'))
    if columns>N:
        out.append(text(x0+(N+(columns-N)/2)*cell,y0-49,'下游 hub 区 H',16,'#b57a13',600,anchor='middle'))
    for j in range(columns):
        label='D'+str(j) if j<N else 'H'+str(j-N)
        out.append(text(x0+j*cell+cell/2,y0-14,label,12,anchor='middle'))
    for i in range(rows):
        label='D'+str(i) if i<N else 'H'+str(i-N)
        out.append(text(x0-18,y0+i*cell+20,label,13,anchor='end'))
        for j in range(columns):
            fill='#f3f7fe' if i<N and j<N else '#fdf6e9' if i>=N and j>=N else '#fafbfd'
            out.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#dce4ed" stroke-width=".7"/>'.format(x0+j*cell,y0+i*cell,cell,cell,fill))
    for a,b,k in case['edges']:
        if a in si and b in ti:
            i,j=si[a],ti[b];kind=KINDS[k];counts[kind]+=1
            out.append('<circle cx="{}" cy="{}" r="5.8" fill="{}"><title>{} → {} : {}</title></circle>'.format(x0+(j+.5)*cell,y0+(i+.5)*cell,COLORS[kind],esc(case['nodes'][a]),esc(case['nodes'][b]),kind))
    y=205
    for kind,count in sorted(counts.items()):
        out.append('<circle cx="850" cy="{}" r="7" fill="{}"/>'.format(y-5,COLORS[kind]))
        out.append(text(870,y,'{}：{} 条'.format(kind,count),19,COLORS[kind],600));y+=39
    for j,line in enumerate(['DD：细节传播','DH：普通结果进入 hub','HH：hub 背景交换','HD：hub 钩子进入普通节点','', '矩阵只显示这一对位置。','其他上游、下游在单节点图中查看。']):
        out.append(text(840,y+35+j*32,line,16,MUTED))
    out.append(text(25,820,'完整计数：{} 条消息边。矩阵中的四个角色块不是四种额外节点角色。'.format(sum(counts.values())),17))
    out.append('</svg>');return ''.join(out)


def ladder_svg(cases):
    out=start(1640,805,'层数阶梯与相位覆盖')
    out += [text(26,38,'层数阶梯：相位覆盖与来源混合分别看',28,weight=700),
            text(26,72,'每位置 16 个普通节点，非末层另加 4 个 hub；T=4，所有边 δ=1。彩格为每个末端的普通路径总时延。',17),
            text(26,112,'候选 / 总节点',17,weight=600),text(215,112,'区域宽度轮廓',17,weight=600),
            text(460,112,'普通路径总时延 d',17,weight=600),text(1155,112,'每相位可达入口数的最小值',17,weight=600),
            text(1155,138,'φ=d mod T；入口共 4 组',14,MUTED)]
    x0,dx=460,38
    for d in range(1,17):out.append(text(x0+(d-1)*dx+16,148,d,13,MUTED,anchor='middle'))
    for p in range(4):out.append(text(1182+p*74,164,'φ='+str(p),14,PHASE_COLORS[p],600,anchor='middle'))
    for row,case in enumerate(cases):
        y=185+row*72;s=case['stats'];label=str(case['depth'])+' 层'+(' + DD4' if case['add_dd4'] else '')
        out.append('<rect x="18" y="{}" width="1598" height="65" rx="10" fill="{}"/>'.format(y-5,'#faf6fe' if case['add_dd4'] else '#f6f8fc'))
        out.append(text(30,y+23,label,19,weight=700));out.append(text(30,y+48,'{:,} 节点 / {} 位置'.format(s['nodes'],s['positions']),13,MUTED))
        for l,w in enumerate(case['widths']):
            bh=w*2.3
            out.append('<rect x="{}" y="{}" width="9" height="{}" rx="2" fill="#7697bd"/>'.format(215+l*12,y+48-bh,bh))
        for d in range(1,17):
            on=d in s['ordinary_delays'];color=PHASE_COLORS[d%4] if on else '#e5eaf1'
            out.append('<rect x="{}" y="{}" width="32" height="36" rx="6" fill="{}"/>'.format(x0+(d-1)*dx,y+9,color))
            if on:out.append(text(x0+(d-1)*dx+16,y+33,str(d),15,'white',600,anchor='middle'))
        for p,mm in enumerate(s['ordinary_phase_sources']):
            out.append('<rect x="{}" y="{}" width="60" height="36" rx="6" fill="{}" stroke="#d2dce8"/>'.format(1152+p*74,y+9,'#e5f4ed' if mm[0]==4 else '#fff0dc' if mm[0] else '#fbe6e3'))
            out.append(text(1182+p*74,y+33,str(mm[0])+'/4',15,weight=600,anchor='middle'))
    out += [text(26,730,'基础规则：M 层的普通路径时延为 ceil((M−1)/2)…M−1；T=4 的完整相位覆盖从 M=7 开始。',17),
            text(26,759,'5 层 + DD4 新增普通节点跨 4 层连线，时延变为 1…4；15 层起，每相位至少有两种普通路径时延。',17),
            text(26,788,'来源数表示传播可能性，未声称这些来源在实际选择后都到达或被内容函数保留。',14,MUTED)]
    out.append('</svg>');return ''.join(out)


def main():
    OUT.mkdir(exist_ok=True)
    cases=[make_case(*p) for p in PROFILES]
    for depth in (5,7,9):
        widths, _ = sg.lattice(depth)
        cases.append(make_case('lattice-'+str(depth), widths, family='lattice', hubs=0))
    for m in (2,3,4):
        widths = sg.exchange_widths(m)
        for hubs in (0,4):
            key = 'exchange-'+str(len(widths))+('-h' if hubs else '')
            cases.append(make_case(key, widths, family='exchange', hubs=hubs, expansion=m, rounds=m))
    for name,widths,extra in g.COMPACT_PROFILES:
        for n in (8,16,32):
            for hubs in (0,4):
                cases.append(make_case(f'{name}-n{n}-h{hubs}',widths,extra,n=n,hubs=hubs,
                                       profile='multiscale-'+name))
    for m,rounds,n in ((1,3,8),(1,4,16),(1,5,32),(1,10,32),(2,5,32),(2,9,32)):
        name=f'exchange-m{m}-r{rounds}'
        for hubs in (0,4):
            cases.append(make_case(f'{name}-n{n}-h{hubs}',sg.exchange_widths(m,rounds),
                                   family='exchange',n=n,hubs=hubs,profile=name,
                                   expansion=m,rounds=rounds))
    default=next(c for c in cases if c['key']=='9')
    figures={
        'regions-9':region_svg(default,static=True),
        'ordinary-neighborhood':ego_svg(default,default['selected_d']),
        'hub-neighborhood':ego_svg(default,default['selected_h']),
        'node-pair-matrix':matrix_svg(default),
        'depth-ladder':ladder_svg([c for c in cases if c['key'] in {p[0] for p in PROFILES}]),
    }
    for name,svg in figures.items():
        (OUT/(name+'.svg')).write_text(svg)
        render_png(OUT/(name+'.svg'),OUT/(name+'.png'))
    report={c['key']:{'depth':c['depth'],'widths':c['widths'],'family':c['family'],'ordinary_per_position':c['ordinary'],'hubs_per_position_by_layer':c['hub_counts'],'settle_period':c['settle_period'],'add_dd4':c['add_dd4'],**c['stats']} for c in cases}
    (OUT/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for c in cases:c['svg']=region_svg(c)
    data={'kinds':KINDS,'colors':COLORS,'cases':{c['key']:c for c in cases},'capacity':capacity_reports.build(OUT)}
    (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    template=(ROOT/'page.html').read_text()
    page=template.replace('__STYLE__',(ROOT/'style.css').read_text()).replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')).replace('__SCRIPT__',(ROOT/'explore.js').read_text())
    page=page.replace('src="depth-ladder.svg"','src="data:image/svg+xml;base64,'+base64.b64encode(figures['depth-ladder'].encode()).decode()+'"')
    page=page.replace('__FORMULAS__',(ROOT/'formulas.js').read_text())
    page=page.replace('__CAPACITY__',(ROOT/'capacity.js').read_text())
    (OUT/'index.html').write_text(page)
    print(f"Built backbone atlas: {len(cases)} spatial cases, {len(data['capacity']['rows'])} parameter configurations; potential paths and shape counts only.")


if __name__=='__main__':main()
