'use strict';
// All values come from the same Python inventory as the downloadable CSV.
const cap=ATLAS.capacity;
const capRows=cap.rows.map(values=>Object.fromEntries(cap.fields.map((key,i)=>[key,values[i]])));
const capVariants=cap.variants;
const capAggProfiles=cap.agg_profiles;
const capAllocations=cap.allocation_profiles;
const capSharing={independent:'独立参数',region_shared:'区域共享',layer_shared:'层级共享'};
const capNumber=x=>(x/1e6).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:3});
const capFraction=(numerator,denominator)=>`<mfrac>${numerator}${denominator}</mfrac>`;
const capGraph=new Map(Object.values(ATLAS.cases).map(c=>[`${c.profile}/${c.ordinary}/${c.hubs}`,c.key]));
function capacityProfiles(preferred='all'){
 const variant=$('#cap-family').value,family=capVariants[variant]?.family;
 $('#cap-profile').innerHTML=option('all','全部轮廓')+Object.values(cap.profiles).filter(p=>variant==='all'||p.variant===variant).map(p=>option(p.key,p.label)).join('');
 $('#cap-profile').value=[...$('#cap-profile').options].some(o=>o.value===preferred)?preferred:'all';
 const canHub=!['fc','lattice'].includes(family);
 $('#cap-h').disabled=!canHub;if(!canHub)$('#cap-h').value='0';
 const canTimed=variant==='all'||variant==='multiscale-timed';
 $('#cap-schedule').disabled=!canTimed;if(!canTimed)$('#cap-schedule').value='native';
 capacityMemory();
}
function capacityMemory(){
 const old=$('#cap-memory').value,h=+$('#cap-h').value;
 $('#cap-memory').innerHTML=option('K','K · 全 KV')+option('S','S · 全 SSD')+(h?option('X','X · 普通 KV / hub SSD'):'');
 $('#cap-memory').value=h&&old==='X'?'X':old==='S'?'S':'K';
}
function capacityBudgets(){
 const n=+$('#cap-n').value,previous=$('#cap-k').value;
 const ks=cap.budgets.filter(k=>k<=n);
 $('#cap-k').innerHTML=ks.map(k=>option(k,`k=${k} · 预算 ${100*k/n}%`)).join('');
 $('#cap-k').value=ks.includes(+previous)?previous:'1';
}
function capacityAgg(aggProfile,d){
 const p=capAggProfiles[aggProfile],width=p.attention_width==='d'?d:p.attention_width;
 const symbol=d==='all'?mi('d'):mn(d);
 const hidden=p.hidden_divisor===1?symbol:(d==='all'?capFraction(symbol,mn(p.hidden_divisor)):mn(+d/p.hidden_divisor));
 const attention=width==='all'?symbol:mn(width);
 $('#capacity-agg-summary').textContent=`${p.label}：${p.mode==='source_residual'?'逐条来源非线性后汇聚。':'同刻消息先相互读取，再逐条非线性与汇聚；一个块，两项残差尺度均为 1。'} 接收在区域选择前执行，实际工作按到达数计量。`;
 $('#capacity-agg-formula').innerHTML=mathBlock(sub(mi('p'),mi('a'))+mo('=')+hidden)+
  (width?mathBlock(sub(mi('d'),mi('A'))+mo('=')+attention+mo(',')+sub(mi('H'),mi('A'))+mo('=')+(width==='all'?capFraction(symbol,mn(32)):mn(+width/32))+mo(',')+sub(mi('d'),mi('h'))+mo('=')+mn(32)):'');
}
function renderCapacity(){
 const n=+$('#cap-n').value,h=+$('#cap-h').value,d=$('#cap-d').value;
 const share=$('#cap-sharing').value,shared=share!=='independent',rank=shared?+$('#cap-rank').value:0;
 $('#cap-rank').disabled=!shared;
 const allocation=$('#cap-allocation').value,k=$('#cap-k').value;
 const ratio=allocation==='quad'?+k/n:1;
 $('#cap-k').disabled=allocation==='pool';
 const variant=$('#cap-family').value,profile=$('#cap-profile').value,memory=$('#cap-memory').value;
 const aggProfile=$('#cap-agg').value;
 if(aggProfile==='all'){
  $('#capacity-agg-summary').textContent='同时比较全部接收配置；各行列出注意力总宽度 d_A 与逐条 MLP 隐藏宽度 p_a。来源残差没有注意力支路。';
  $('#capacity-agg-formula').innerHTML='';
 }else capacityAgg(aggProfile,d);
 const common=$('#cap-schedule').value==='common';
 const limit=$('#cap-limit').value==='256'?256e6:Infinity;
 const rows=capRows.filter(r=>r.n===n&&r.h===h&&r.memory===memory&&r.sharing===share&&r.rank===rank&&
  (d==='all'||r.d===+d)&&(variant==='all'||cap.profiles[r.profile].variant===variant)&&
  (profile==='all'||r.profile===profile)&&(aggProfile==='all'||r.agg_profile===aggProfile)&&r.total<=limit).sort((a,b)=>a.total-b.total||a.profile.localeCompare(b.profile)||a.agg_profile.localeCompare(b.agg_profile));
 $('#capacity-summary').textContent=`${rows.length} 个配置 · ${aggProfile==='all'?'全部接收配置':capAggProfiles[aggProfile].label} · 每普通区 ${n} 节点 · ${capAllocations[allocation].label}${allocation==='pool'?' · 不使用 k':`，k=${k}，预算比例 ${100*(+k/n)}%`} · ${capSharing[share]}${rank?`，适配秩 ${rank}`:''}。以下单位为百万参数（M）。`+(common?' Timed 行采用与 Settle 相同的输入周期；CSV 列出主周期。':'');
 $('#capacity-allocation-summary').textContent=allocation==='pool'?
  'D 区 pool：总量 min(1, Σμ)，按 μ 与评分分配；正接收候选稠密，允许单节点超过 μ。H 区仍为角色 quad。':
  `D 区 ${capAllocations[allocation].label}：普通边界 [0, μ]，接收量下界、区间总量，固定增益 2、阈值 0、温度 1。H 区仍为角色 quad，边界 [μ/4, μ]，k_H=2。${allocation==='shannon'?'Shannon 对正接收普通候选稠密；k 约束强度，不缩减活跃代理。':'quad 可稀疏，k 不保证正控制数量。'}`;
 const dsym=d==='all'?mi('d'):mn(d),dim=(den)=>d==='all'?capFraction(dsym,mn(den)):mn(+d/den);
 $('#capacity-region-formula').innerHTML=
  mathBlock(sub(mi('d'),mi('z'))+mo('=')+dim(2)+mo(',')+sub(mi('H'),mi('R'))+mo('=')+dim(64)+mo(',')+sub(mi('d'),mi('rf'))+mo('=')+dsym)+
  mathBlock(sub(mi('P'),mi('R'))+mo('=')+mn(2)+sup(mi('d'),mn(2))+mo('+')+mn(2)+mi('d'));
 $('#capacity-sharing-summary').textContent={independent:'节点矩阵与区域头分别独立；不受适配秩限制。',region_shared:'g(v) 按实际区域划分相容节点：区内共享基座，节点适配器私有，区域头独立。',layer_shared:'g(v) 按层与角色等划分相容节点：基座跨区域共享，节点适配器私有，区域头按层与区域类型共享。'}[share];
 const weightFormula=shared?mathBlock(sub(mi('W'),mi('v'))+mo('=')+sub(sup(mi('W'),mn(0)),mi('g(v)'))+mo('+')+sub(mi('L'),mi('v'))+sub(mi('R'),mi('v'))):'';
 $('#capacity-formula').innerHTML=weightFormula+mathBlock(sub(mi('P'),textMath('总'))+mo('=')+mi('B')+mo('+')+sub(mi('P'),mi('D'))+mo('+')+sub(mi('P'),mi('H')))+
  mathBlock(sub(mi('P'),textMath('活跃代理'))+mo('=')+mi('B')+mo('+')+mn(ratio)+sub(mi('P'),mi('D'))+mo('+')+sub(mi('P'),mi('H')));
 let table='<thead><tr><th>拓扑与时间</th><th>接收配置</th><th>层数</th><th>输入周期</th><th>d</th><th>d_A / p_a</th><th>D / H 节点</th><th>总参数 M</th><th>活跃代理 M</th><th>代理 / 总参数</th><th>连接</th></tr></thead><tbody>';
 for(const r of rows){
  const p=cap.profiles[r.profile],a=r.base+ratio*r.D+r.H,graph=capGraph.get(`${p.spatial_profile}/${n}/${h}`);
  const mode=common&&p.timing==='timed'?'timed-common':p.timing,period=mode==='timed-common'?p.settle_period:p.input_period;
  table+=`<tr data-profile="${E(r.profile)}" data-allocation="${allocation}" data-region-profile="${E(r.region_profile)}" data-agg-profile="${E(r.agg_profile)}" data-agg-width="${r.agg_attention_width}" data-agg-hidden="${r.agg_hidden_width}" data-timing="${mode}" data-period="${period}" data-d="${r.d}" data-total="${r.total}" data-active="${a}"><td title="${E(r.profile)} · 宽度 ${p.widths.join('→')}">${E(p.label)}${shared&&p.family==='fc'?' · 两共享档等价':''}</td><td title="${E(r.agg_profile)}">${E(capAggProfiles[r.agg_profile].label)}</td><td>${p.depth}</td><td>${period}</td><td title="区域描述 ${r.descriptor_width} 维，${r.region_attention_heads} 头，区域 FFN ${r.region_ffn_width} 维">${r.d}</td><td title="${r.agg_attention_heads} 个注意力头，每头 ${r.agg_attention_head_dim} 维">${r.agg_attention_width||'—'} / ${r.agg_hidden_width}</td><td>${r.nodes_D} / ${r.nodes_H}</td><td title="${r.total.toLocaleString('zh-CN')} 个参数">${capNumber(r.total)}</td><td>${capNumber(a)}</td><td>${(100*a/r.total).toFixed(2)}%</td><td>${graph?`<button data-cap-case="${E(graph)}" data-cap-timing="${mode}">查看</button>`:'—'}</td></tr>`;
 }
 $('#capacity-table').innerHTML=table+'</tbody>';
}
$('#cap-family').innerHTML=option('all','全部候选')+Object.entries(capVariants).map(([key,v])=>option(key,v.label)).join('');
$('#cap-n').innerHTML=cap.node_sizes.map(n=>option(n)).join('');$('#cap-n').value='32';
$('#cap-d').innerHTML=option('all','全部维度')+cap.dimensions.map(d=>option(d)).join('');
$('#cap-agg').innerHTML=option('all','全部接收配置')+Object.entries(capAggProfiles).map(([key,p])=>option(key,p.label)).join('');$('#cap-agg').value='source_residual';
$('#cap-allocation').innerHTML=Object.entries(capAllocations).map(([key,p])=>option(key,p.label)).join('');
$('#cap-family').addEventListener('change',()=>{capacityProfiles();renderCapacity();});
$('#cap-n').addEventListener('change',()=>{capacityBudgets();renderCapacity();});
$('#cap-h').addEventListener('change',()=>{capacityMemory();renderCapacity();});
for(const name of ['profile','d','memory','sharing','rank','k','limit','schedule','agg','allocation'])$('#cap-'+name).addEventListener('change',renderCapacity);
$('#cap-use-graph').addEventListener('click',()=>{
 $('#cap-family').value=variantKey(current.family,state.timing);capacityProfiles(`${current.profile}-${state.timing==='settle'?'settle':'timed'}`);
 $('#cap-schedule').value=state.timing==='timed-common'?'common':'native';
 $('#cap-n').value=current.ordinary;$('#cap-h').value=current.hubs;capacityMemory();capacityBudgets();renderCapacity();
});
$('#capacity-table').addEventListener('click',e=>{const button=e.target.closest('[data-cap-case]');if(button){mountCase(button.dataset.capCase,button.dataset.capTiming);$('#regions').scrollIntoView({behavior:'smooth'});}});
capacityProfiles();capacityBudgets();renderCapacity();
window.atlasCapacityDebug={renderCapacity,budgets:capacityBudgets};
