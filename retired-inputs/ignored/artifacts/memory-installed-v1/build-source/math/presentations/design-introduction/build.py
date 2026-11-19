#!/usr/bin/env python3
"""Build the self-contained HTML, speaker notes and editable SVG figures."""
from pathlib import Path
from html import escape
import importlib.util
import json
import sys

sys.dont_write_bytecode=True

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


content=module('intro_content',HERE/'content.py')
fig=module('intro_figures',BASE/'figures/src/design_introduction.py')


model=module('intro_example',BASE/'examples/explicit_residual_moe_checks.py')
FIRST=model.run_one()
HISTORY_FIRST=model.run_one(counts=dict.fromkeys(model.EXPERTS,0))
HISTORY_SECOND=model.run_one(counts=HISTORY_FIRST['new'],start=4)
FIGURES={f'hb-{name}': fig.graph(name) for name in (
    'regions','neighbors','input0','select0','middle0','exit0','output0',
    'memory0','input1','select1','middle1','output1')}
FIGURES.update({'sequential':fig.chain(), 'bounded-propagation':fig.expansion(),
                'transformer':fig.transformer()})
for name in ('roles','mapping','select','contributions','result','history1','history2','phases'):
    trace = None if name in ('roles','mapping') else HISTORY_SECOND if name=='history2' else HISTORY_FIRST if name in ('history1','phases') else FIRST
    FIGURES['moe-'+name]=fig.moe(name,trace)


def body(s):
    if s['view']=='html':
        return s['body']
    if s['view']=='transformer':
        return '<div class="transformer-figure">'+FIGURES['transformer']+'</div><div class="granularities"><div><b>整个 block → 一个节点</b><span>两个 Add 可随计算一起留在内部。</span></div><div><b>Attention → 一个节点</b><span>明确 Norm 与对应 Add 的归属。</span></div><div><b>一个 expert → 一个节点</b><span>下一步展开选择、合并与残差旁路。</span></div></div>'
    if s['view']=='illustration':
        legend = '<span class="blue">蓝线：数值载荷</span><span class="gold">金色虚线：控制</span><span>深色旁路：基线</span>' if s['figure'].startswith('moe-') else '<span>区域与邻接取自 HB 拓扑</span>'
        if s['figure']=='hb-output0':
            legend='<span>蓝圈：候选</span><span class="gold">金色：Full</span><span class="green">绿块：记忆</span><span class="blue">蓝线：消息</span>'
        return '<div class="graph-layout"><div class="graph-area">'+FIGURES[s['figure']]+'<div class="legend">'+legend+'</div></div><div class="side">'+s['body']+'</div></div>'
    if s['view']=='expansion':
        return '<div class="spread-layout"><div>'+FIGURES['bounded-propagation']+'<div class="callout">增长的是可达容量；每次参与范围可随输入变化。</div></div><div><div class="eyebrow">固定一套模型配置</div><div class="condition"><b>1</b> 输入、输出端口数固定</div><div class="condition"><b>2</b> 单节点通信对象数有界</div><div class="condition"><b>3</b> 扩大输入到输出的可达容量</div><p class="small" style="margin-top:28px">不同规模可选不同有限端口数；<br>分别考察固定配置下的扩展。</p></div></div>'
    raise ValueError(s['view'])


def main():
    generated=BASE/'figures/design-introduction'
    generated.mkdir(exist_ok=True)
    print(model.check())
    for name,svg in FIGURES.items():
        (generated/f'{name}.svg').write_text(svg)
    frames=[];metadata=[];notes=['# Tide 展示讲者备注\n','由 [content.py](content.py) 生成；讲述主线见[脚本](script.md)。\n']
    for i,slide in enumerate(content.SLIDES):
        notes.append(f'## {i+1}. {slide["steps"][0]["title"]}\n')
        refs=[dict(label=content.SOURCES[k][0],href=content.SOURCES[k][1]) for k in slide['sources']]
        for j,s in enumerate(slide['steps']):
            idx=len(frames); html=body(s)
            html=html.replace('arrow-blue',f'arrow-blue-{idx}').replace('arrow-ink',f'arrow-ink-{idx}')
            source_label=' · '.join(r['label'] for r in refs)
            footer=f'<footer class="footer"><span class="ref">依据：{escape(source_label)}</span><span class="folio">{i+1:02d} / {len(content.SLIDES):02d} &nbsp; · &nbsp; {j+1}/{len(slide["steps"])}</span></footer>'
            frames.append(f'<section class="frame {"active" if idx==0 else ""}" data-slide="{i}" data-step="{j}" data-stop-id="{s["id"]}" aria-label="{escape(s["title"],quote=True)}"><div class="kicker"><span><strong>TIDE</strong> {escape(slide["chapter"])}</span><span class="status">{escape(slide["status"])}</span></div><h1>{s["title"]}</h1><p class="lede">{s["lede"]}</p><div class="content">{html}</div>{footer}</section>')
            metadata.append(dict(slide=i,step=j,title=s['title'],notes=s['notes'],refs=refs,figure=s['figure'],view=s['view'],id=s['id'],preview=s.get('preview',False)))
            notes += [f'### 停点 {i+1}.{j+1}：{s["title"]}\n',s['notes']+'\n']
        notes.append('出处：'+'；'.join(f'[{r["label"]}]({r["href"]})' for r in refs)+'。\n')
    css=(HERE/'deck.css').read_text(); js=(HERE/'deck.js').read_text()
    data=json.dumps(metadata,ensure_ascii=False).replace('</','<\\/')
    output=f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><link rel="icon" href="data:,"><title>Tide · 去中心化与稀疏化</title><style>{css}</style></head>
<body><main id="viewport"><div id="stage">{''.join(frames)}</div></main>
<div class="controls"><div class="control-buttons"><button id="prev" aria-label="上一步">← 上一步</button><button id="next" aria-label="下一步">下一步 →</button><span id="position"></span></div><span class="shortcut">← → 逐步推进 · N 备注 · G 目录</span><div class="control-buttons"><button id="replay">重看本页</button><button id="toc">目录</button><button id="notes">备注与出处</button><button id="fullscreen">全屏</button></div></div>
<dialog id="notes-dialog"><button class="close" data-close>关闭</button><h2 id="notes-title"></h2><p id="notes-body"></p><h3>定义出处</h3><ul id="notes-sources"></ul></dialog>
<dialog id="toc-dialog"><button class="close" data-close>关闭</button><h2>展示目录</h2><div id="toc-list" class="toc-list"></div></dialog><div id="live" class="sr-only" aria-live="polite"></div>
<script id="frame-data" type="application/json">{data}</script><script>{js}</script></body></html>'''
    (HERE/'index.html').write_text(output)
    (HERE/'speaker-notes.md').write_text('\n'.join(notes))
    print(f'Built {len(content.SLIDES)} scenes / {len(frames)} stops; HTML {len(output.encode()):,} bytes')


if __name__=='__main__':
    main()
