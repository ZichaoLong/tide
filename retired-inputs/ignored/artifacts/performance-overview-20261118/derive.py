import collections,csv,datetime,hashlib,json,math,subprocess
from pathlib import Path
root=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
out=root/'artifacts/performance-overview-20261118'
out.mkdir(exist_ok=False)
rows=[];subjects=[];identities={}
def load(group):
 p=root/f'docs/evidence/roadmap-{group}-20261118.json'
 identities[group]=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 return json.loads(p.read_text())

def make(group,device,case,precision,mode,batch,length,config,observations,audit,matches,source,run,evidence):
 obs={x['subject']:x for x in observations if x['mode']==mode}
 assert all(x in obs for x in ['tide']+['transformer_'+c for c in 'ABCD'])
 tide=obs['tide'];work=audit['work'];params=audit['parameters']['unique_parameters'];packing=audit.get('query_packing',{}).get('K',{})
 row=dict(group=group,device=device,case=case,precision=precision,mode=mode,batch=batch,length=length,
  d=config['d'],memory=config['memory'],aggregation=config['aggregation'],allocation=config['allocation'],sharing=config['sharing'],
  total_parameters=params,full_percent=100*work['full_actions']/work['candidate_updates'],
  touched_percent=100*audit['effective_touched']['mean']/params,touched_mean=audit['effective_touched']['mean'],
  touched_axis=audit['logical_axis'],cumulative_touched_per_input=audit['effective_touched']['cumulative_per_input_position'],
  candidate_updates=work['candidate_updates'],full_actions=work['full_actions'],
  packing_ratio=packing.get('executed_rows',0)/packing['selected_rows'] if packing.get('selected_rows') else None,
  tide_ms=1000*tide['median_seconds'],tide_peak_mib=tide.get('peak_device_bytes')/2**20 if tide.get('peak_device_bytes') else None,
  source=source,run=run,evidence=evidence,
  samples=len(tide['seconds']),tide_sample_range_over_median=(max(tide['seconds'])-min(tide['seconds']))/tide['median_seconds'])
 for c in 'ABCD':
  baseline=obs['transformer_'+c];match=matches[c]
  assert baseline['parameters']==match['actual']['parameters']
  assert match['relative_deviation']['blocks']==0
  row[c+'_ms']=1000*baseline['median_seconds']
  row[c+'_time_ratio']=tide['median_seconds']/baseline['median_seconds']
  row[c+'_peak_mib']=baseline.get('peak_device_bytes')/2**20 if baseline.get('peak_device_bytes') else None
  row[c+'_memory_ratio']=row['tide_peak_mib']/row[c+'_peak_mib'] if row['tide_peak_mib'] is not None and row[c+'_peak_mib'] else None
  row[c+'_blocks']=match['actual']['blocks'];row[c+'_parameters']=match['actual']['parameters']
  row[c+'_width']=match['constructor']['d'];row[c+'_ffn']=match['constructor']['ffn_width']
  row[c+'_parameter_deviation_percent']=100*match['relative_deviation']['parameters']
 for subject,ob in obs.items():
  subjects.append(dict(group=group,device=device,case=case,precision=precision,mode=mode,batch=batch,length=length,
    subject=subject,parameters=ob['parameters'],median_ms=1000*ob['median_seconds'],
    peak_device_mib=ob.get('peak_device_bytes')/2**20 if ob.get('peak_device_bytes') else None,
    sample_ms=[1000*x for x in ob['seconds']],source=source,run=run))
 return row

for group in ['representative','supplementary','scale-small']:
 d=load(group)
 for r in d['records']:
  main=group=='representative';device=r['device_kind'] if main else 'npu';case=r['config_name'] if main else r['case'];precision=r['arguments']['precision']
  scope=group if group!='supplementary' else r['suite']
  config=r['config'] if main else d['configs'][r['config_ref']]
  for mode in ['inference','grad_forward','train']:
   audit=r['initial_audits'][mode] if main else d['semantic_audits'][r['initial_audit_refs'][mode]]
   matches=r['joint_matches_by_mode'][mode] if main else d['joint_matches'][r['joint_match_refs'][mode]]
   row=make(scope,device,case,precision,mode,r['arguments']['batch'],r['arguments']['length'],config,r['observations'],audit,matches,d['source'],r['run'],identities[group]['path'])
   for c in 'ABCD':
    comparable=[x for x in d['comparisons'] if x['precision']==precision and x['mode']==mode and x['joint_class']==c and ((x['run']==r['run'] and x['config']==case) if main else (x['suite']==r['suite'] and x['case']==case))]
    assert len(comparable)==1
    assert math.isclose(row[c+'_time_ratio'],comparable[0]['tide_over_transformer_time'],rel_tol=1e-12)
   rows.append(row)
large=load('scale-large');buckets=collections.defaultdict(list)
for r in large['completed_comparison_rows']:buckets[r['job'],r['precision'],r['mode']].append(r)
for (job,precision,mode),items in buckets.items():
 item=next(r for r in items if r['subject']=='tide');ob=item['observation'];device='cpu' if '-cpu-' in job else 'npu';case='hb-17b' if '17b' in job else 'hb-9b'
 rows.append(make('scale-large',device,case,precision,mode,1,8,ob['spec']['config'],[r['observation'] for r in items],item['initial_audit'],item['matching'],large['source'],job,identities['scale-large']['path']))
assert len(rows)==184 and len(subjects)==932,(len(rows),len(subjects))
assert len({tuple(r[k] for k in ['group','device','case','precision','mode']) for r in rows})==184
order={'fc-k':0,'fc-s':1,'hb-lattice-k':2,'hb-exchange-x':3,'hb-multiscale-x-settle':4,'hb-multiscale-x-timed':5,'fc-k-149m':6}
rows.sort(key=lambda r:(r['group'],r['device'],r['precision'],r['mode'],order.get(r['case'],99),r['case']))
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 scope='Read-only derived overview of existing zero-state measurements, no new experiment or qualification; source identities retained per row.',
 ratio='Tide / same-precision same-mode Transformer; below 1 means Tide uses less time/memory.',
 exclusions='Continuous segments use different timing boundaries; read roadmap-continuous separately. Multi-device capacity rows have no jointly measured A-D and are not synthesized here. CPU has no independent per-model allocator peak.',
 full='Full actions / actual candidate updates, initial audit. Not an average fraction over all graph nodes.',
 active='Initial semantically effective parameter scalars, deduplicated per sample/input step for Settle or per logical tick for Timed. Not FLOPs, memory savings or nonzero-gradient count.',
 sources=identities,comparison_rows=len(rows),subject_rows=len(subjects))
(out/'overview.json').write_text(json.dumps(dict(metadata=manifest,rows=rows,subjects=subjects),ensure_ascii=False,indent=2)+'\n')
for name,data in [('comparisons.csv',rows),('subjects.csv',subjects)]:
 with (out/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

lines=['# 当前性能对比汇总','',f"来源：HEAD `{manifest['head']}` 的已提交证据；不新增测量。",'',
 'A=近节点/block数×近总参数；B=近节点/block数×近实际触及参数；C=近层深×近总参数；D=近层深×近实际触及参数。',
 '时间比为 Tide/Transformer，小于1表示Tide用时较短；各行均同精度/模式配对。Full%为初始Full/候选；触及%按初始实际有效触及/总参数，Timed按tick另计。',
 '完整原始样本、实际基线宽度/FFN/参数及内存比见 overview.json、comparisons.csv、subjects.csv；interactive.html 可筛选查看。',
 '短样本存在噪声。人工活跃率改变评分参数和支持分布，不构成同函数稀疏消融；组合同时改变多个因素。CPU未有可比较的逐模型设备allocator峰值。',
 '不含不同计时边界的连续段或未配A–D的多卡容量记录；它们保持原报告身份。','']
for key,group_rows in __import__('itertools').groupby(rows,key=lambda r:(r['group'],r['device'],r['precision'],r['mode'])):
 lines+=['## '+' / '.join(key),'','| 配置 | B/L | Full% / 触及% | Tide ms | A ms | B ms | C ms | D ms | Tide/A | Tide/B | Tide/C | Tide/D | Tide peak MiB |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in group_rows:
  v=[r['case'],f"{r['batch']}/{r['length']}",f"{r['full_percent']:.2f} / {r['touched_percent']:.2f}"]+[f"{r[k]:.2f}" for k in ['tide_ms','A_ms','B_ms','C_ms','D_ms']]+[f"{r[c+'_time_ratio']:.3f}" for c in 'ABCD']+[f"{r['tide_peak_mib']:.1f}" if r['tide_peak_mib'] else '未测']
  lines.append('| '+' | '.join(v)+' |')
 lines+=['']
(out/'overview.md').write_text('\n'.join(lines)+'\n')
html=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tide · 已测性能对比</title>
<style>body{font:15px/1.6 system-ui,sans-serif;color:#162238;background:#f5f7fb;margin:0;padding:28px}main{max-width:1420px;margin:auto}h1{font-size:28px;margin:0}p{max-width:1050px}label{display:inline-flex;flex-direction:column;gap:4px;margin:0 14px 14px 0;font-weight:600}select{padding:9px 12px;border:1px solid #c2cad8;border-radius:6px;background:white;font-size:14px}section{background:white;border:1px solid #dde3ee;border-radius:10px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{border-bottom:1px solid #e5eaf2;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f1f4f9}small,.note{color:#526175}strong{font-weight:650}svg{display:block;min-width:700px;width:100%}.fast{color:#087a66}.slow{color:#aa3b23}button{padding:9px 14px;border:1px solid #c2cad8;border-radius:6px;background:white;cursor:pointer}</style>
<main><h1>Tide 与四类 Transformer：已测结果</h1><p>A：近 block 数 × 总参数；B：近 block 数 × 实际触及参数；C：近层深 × 总参数；D：近层深 × 实际触及参数。时间/内存比均为 <strong>Tide ÷ Transformer</strong>，小于1表示Tide用量较少。不同数学函数的计算参照，不说明模型质量。</p>
<label>测量组<select id="group"></select></label><label>设备<select id="device"><option value="npu">NPU</option><option value="cpu">CPU</option></select></label><label>精度<select id="precision"><option>fp32</option><option>bf16</option></select></label><label>模式<select id="mode"><option value="train">完整训练</option><option value="inference">推理</option><option value="grad_forward">含梯度前向</option></select></label><label>指标<select id="metric"><option value="time">耗时（ms）</option><option value="memory">peak allocated（MiB）</option></select></label><button id="download">下载当前表 CSV</button>
<p id="count" class="note"></p><section><div id="chart" class="scroll"></div></section><section class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></section>
<p class="note">Full% = 初始 Full/实际候选更新数；触及% = 初始有效触及参数/总参数。Timed触及按逻辑tick去重，并包含完整flush，不能与每输入位置混为一个分母。训练参数会在采样间更新。人工稀疏输入改变评分参数/支持分布；候选组合改变多个因素。短时测量有噪声，细小次序不表示稳定优势。</p>
<p class="note">CPU只有进程累计RSS，不能作为逐模型内存比。BF16包含FP32参数/master、梯度、AdamW和声明的稳定子计算。连续decode/恢复和多卡容量使用不同边界，未并入本表。每行保留独立源码身份。数据来自本轮已提交报告，未新跑实验。</p><details><summary>来源与完整记录</summary><pre id="sources"></pre><a href="overview.md">完整静态表</a> · <a href="comparisons.csv">全部对比 CSV</a> · <a href="subjects.csv">各主体与样本 CSV</a> · <a href="overview.json">完整 JSON</a></details></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),rows=data.rows;
const labels={representative:'七配置主矩阵',sparse:'活跃比例',shape:'B/L/节点形状',combinations:'接收/分配/共享组合','scale-small':'0.5B / 1B','scale-large':'9B / 17B推理'};
const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
for(const [value,label] of Object.entries(labels))$('group').add(new Option(label,value));
$('sources').textContent=JSON.stringify(data.metadata,null,2);let selected=[];
const fmt=(x,n=2)=>x==null?'未测':x.toFixed(n);
function render(){selected=rows.filter(r=>['group','device','precision','mode'].every(k=>r[k]===$(k).value));const memory=$('metric').value==='memory',key=memory?'peak_mib':'ms',ratio=memory?'memory_ratio':'time_ratio',unit=memory?'MiB':'ms';
$('count').textContent=`${selected.length}个测点；比率小于1：Tide用量较少。图使用对数横轴；每行样本数和波动见完整JSON。`;
$('head').innerHTML='<tr>'+['配置（B/L）','Full%','触及%','Tide '+unit,'A '+unit,'B '+unit,'C '+unit,'D '+unit,'Tide/A','Tide/B','Tide/C','Tide/D','源码'].map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>';
$('body').innerHTML=selected.map(r=>'<tr><td>'+esc(r.case)+` (${r.batch}/${r.length})</td><td>`+fmt(r.full_percent)+'</td><td>'+fmt(r.touched_percent)+'</td>'+['tide','A','B','C','D'].map(c=>'<td>'+fmt(r[c+'_'+key])+'</td>').join('')+['A','B','C','D'].map(c=>'<td class="'+(r[c+'_'+ratio]<1?'fast':'slow')+'">'+fmt(r[c+'_'+ratio],3)+'</td>').join('')+'<td>'+esc(r.source.slice(0,8))+'</td></tr>').join('');
if(!selected.length){$('chart').textContent='此选择没有已测数据。9B/17B仅有单设备/CPU推理A–D；补充矩阵仅有NPU。';return}
if(selected.every(r=>r.A_memory_ratio==null)&&memory){$('chart').textContent='CPU没有独立逐模型设备allocator峰值，未生成内存比。';return}
const vals=selected.flatMap(r=>['A','B','C','D'].map(c=>r[c+'_'+ratio])).filter(x=>x!=null),lo=Math.min(-1,Math.floor(Math.log10(Math.min(...vals)))),hi=Math.max(1,Math.ceil(Math.log10(Math.max(...vals)))),x=v=>275+(Math.log10(v)-lo)/(hi-lo)*650,h=75+selected.length*66,colors=['#235ca9','#008875','#b96516','#9849a7'];
let s=`<svg role="img" aria-label="Tide除以四类Transformer的对数比率" viewBox="0 0 1020 ${h}"><rect width="1020" height="${h}" fill="white"/>`;
for(let p=lo;p<=hi;p++){const xx=x(10**p);s+=`<line x1="${xx}" y1="30" x2="${xx}" y2="${h-30}" stroke="${p===0?'#53647c':'#e0e5ee'}" stroke-dasharray="4 4"/><text x="${xx}" y="18" text-anchor="middle" font-size="12">${10**p}×</text>`}
selected.forEach((r,i)=>{const y=52+i*66;s+=`<text x="5" y="${y+12}" font-size="12">${esc(r.case)}</text>`;['A','B','C','D'].forEach((c,j)=>{let v=r[c+'_'+ratio];if(v==null)return;const yy=y+j*12;s+=`<circle cx="${x(v)}" cy="${yy}" r="4" fill="${colors[j]}"/><text x="${x(v)+7}" y="${yy+4}" font-size="10" fill="${colors[j]}">${c} ${v.toFixed(2)}×</text>`})});s+='</svg>';$('chart').innerHTML=s}
for(const k of ['group','device','precision','mode','metric'])$(k).addEventListener('change',render);
$('download').onclick=()=>{if(!selected.length)return;const keys=Object.keys(selected[0]),quote=x=>'"'+String(x??'').replace(/"/g,'""')+'"',csv=[keys.map(quote).join(','),...selected.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\r\n'),a=document.createElement('a'),url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='tide-current-comparison.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></html>'''
html=html.replace('__DATA__',json.dumps(dict(metadata=manifest,rows=rows),ensure_ascii=False).replace('<','\\u003c'))
(out/'interactive.html').write_text(html)
(out/'derive.py').write_text(Path(__file__).read_text())
print(json.dumps(dict(output=str(out),comparisons=len(rows),subjects=len(subjects),published_ratio_checks=720,source_hashes=identities),ensure_ascii=False))
