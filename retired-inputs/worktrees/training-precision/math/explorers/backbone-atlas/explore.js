'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const E=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const kinds=ATLAS.kinds, colors=ATLAS.colors;
const order=['5','5-dd4','7','9','13','15','17'];
let current, outs, ins, ids;
const state={timing:'timed',key:'9',focus:[4,3],role:'d',index:0,scope:'near',source:'3,2',target:'4,2'};
const pos=n=>`${n[0]},${n[1]}`;
const nodeName=n=>`${n[2].toUpperCase()}(${n[0]},${n[1]}; ${n[3]})`;
const hubsAt=l=>current.hub_counts[l];
const positionNodes=([l,p])=>current.nodes.flatMap((n,u)=>n[0]===l&&n[1]===p?[u]:[]);
const option=(v,label)=>`<option value="${E(v)}">${E(label===undefined?v:label)}</option>`;
const family=k=>k.startsWith('HH')?'HH':k.startsWith('HD')?'HD':k;
function svgText(x,y,t,size=16,color='#23344b',weight=400,anchor='start',attrs=''){
 return `<text x="${x}" y="${y}" font-family="Noto Sans CJK SC,Droid Sans,Microsoft YaHei,sans-serif" font-size="${size}" fill="${color}" font-weight="${weight}" text-anchor="${anchor}" ${attrs}>${E(t)}</text>`;
}
function svgStart(w,h,label){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${E(label)}"><rect width="100%" height="100%" fill="white"/>`;}
const isUnit=()=>state.timing==='timed'||state.timing==='timed-common';
const period=()=>['settle','timed-common'].includes(state.timing)?current.settle_period:current.period;
const edgeDelay=(l,m)=>isUnit()?1:m-l;
const displayDelays=(u,detail=false)=>isUnit()?(detail?current.ordinary_delays[u]:current.delays[u]):
 ((detail?current.ordinary_delays[u]:current.delays[u]).length?[current.nodes[u][0]]:[]);
const variantKey=(family,timing)=>`${family}-${timing==='settle'?'settle':'timed'}`;
const validTiming=(c,mode)=>['timed','timed-common','settle'].includes(mode)&&(c.family==='multiscale'||mode==='settle');
function profileLabel(c){return `${c.profile.includes('compact')?'窄图 · ':''}${c.depth} 层${c.profile.includes('two-inputs')?' · 双入口位置':''}${c.add_dd4?' + DD4':''}${c.family==='exchange'?` · m=${c.expansion}, R=${c.rounds}`:''} · ${c.ordinary}${c.hubs?'+4 / 独立 H 区':' / 无 hub'}`;}
function mountCase(key,mode=ATLAS.cases[key].default_timing){
 if(!validTiming(ATLAS.cases[key],mode))throw new RangeError('Unsupported timing for spatial graph: '+mode);
 state.key=key;current=ATLAS.cases[key];state.timing=mode;
 $('#family-select').value=variantKey(current.family,state.timing);
 $('#case-select').innerHTML=Object.values(ATLAS.cases).filter(c=>c.family===current.family).map(c=>option(c.key,profileLabel(c))).join('');
 $('#case-select').value=key;
 $('#timing-mode').value=state.timing;$('#timing-mode').disabled=current.family!=='multiscale';
 outs=current.nodes.map(()=>[]);ins=current.nodes.map(()=>[]);ids=new Map();
 current.nodes.forEach((n,u)=>ids.set(n.join(','),u));
 current.edges.forEach(([u,v,k])=>{outs[u].push([v,k]);ins[v].push([u,k]);});
 const n=current.nodes[current.selected_d];state.focus=n.slice(0,2);state.role=n[2];state.index=n[3];
 $('#metrics').innerHTML=`<span><b>${current.depth}</b>层</span><span><b>${current.stats.positions}</b>位置</span><span><b>${current.stats.regions}</b>区域</span><span><b>${current.stats.nodes.toLocaleString()}</b>节点</span><span><b>${current.stats.edges.toLocaleString()}</b>消息边</span>`;
 $$('[data-family]').forEach(el=>{el.checked=true;el.closest('label').hidden=!Object.keys(current.stats.edges_by_kind).some(k=>family(k)===el.dataset.family);});
 $('#example-h').disabled=!current.hubs;$('#preset-hub').disabled=!current.hubs;
 $('#region-canvas').innerHTML=current.svg;
 const positions=[];current.widths.forEach((w,l)=>{if(l<current.depth-1)for(let p=0;p<w;p++)positions.push(`${l},${p}`);});
 $('#pair-source').innerHTML=positions.map(p=>option(p,`(${p})`)).join('');
 const sl=Math.max(0,Math.floor(current.depth/2)-1),tl=sl+1,sp=Math.min(2,current.widths[sl]-1);
 state.source=`${sl},${sp}`;$('#pair-source').value=state.source;
 const sw=current.widths[sl],tw=current.widths[tl],base=tw>=sw?sp*(tw/sw):Math.floor(sp/(sw/tw));
 updateTargets(`${tl},${base}`);
 syncNodeControls();updateTimeLabels();renderNode();paintRegion();renderDepthTable();renderSettleTable();
}
function updateTimeLabels(){
 const delta=isUnit()?'δ=1':'δ=目标层−源层';
 $('#case-note').textContent=`${{multiscale:'多跨度图',lattice:'原 HB-Lattice',exchange:'Settle 位交换图'}[current.family]} · 宽度 ${current.widths.join('→')}。${delta}，输入间隔 ${period()}。`+
  (state.timing==='settle'?`D=${period()}>M=${current.depth}；同源路径对齐。`:'长短路径可在不同时间到达。')+
  ` D 最大入/出度 ${current.stats.degrees.d.join('/')}。`+(current.hubs?` H：${current.stats.degrees.h.join('/')}。`:'');
 $('#region-timing-label').textContent=`一条线汇总一个边族的真实连线；${delta}。线的交叉不是连接点。`;
 $$('#region-canvas .bundle').forEach(el=>{el.querySelector('title').textContent=el.querySelector('title').textContent.replace(/δ=\d+/,`δ=${isUnit()?1:el.dataset.span}`);});
}
function setTiming(mode){
 if(!validTiming(current,mode))throw new RangeError('Unsupported timing for spatial graph: '+mode);
 state.timing=mode;$('#timing-mode').value=mode;$('#family-select').value=variantKey(current.family,mode);
 updateTimeLabels();renderNode();renderMatrix();paintRegion();renderDepthTable();renderSettleTable();
}
function syncNodeControls(){
 $('#node-layer').innerHTML=current.widths.map((_,l)=>option(l)).join('');$('#node-layer').value=state.focus[0];
 $('#node-position').innerHTML=Array.from({length:current.widths[state.focus[0]]},(_,p)=>option(p)).join('');$('#node-position').value=state.focus[1];
 const hubCount=hubsAt(state.focus[0]);
 if(!hubCount)state.role='d';
 $('#node-role').innerHTML=option('d','普通 D')+(hubCount?option('h','Hub H'):'');$('#node-role').value=state.role;
 const size=state.role==='d'?current.ordinary:hubCount;state.index=Math.min(state.index,size-1);
 $('#node-index').innerHTML=Array.from({length:size},(_,i)=>option(i)).join('');$('#node-index').value=state.index;
}
function selectedId(){return ids.get([...state.focus,state.role,state.index].join(','));}
function chooseNode(u){const n=current.nodes[u];state.focus=n.slice(0,2);state.role=n[2];state.index=n[3];syncNodeControls();renderNode();paintRegion();}
function paintRegion(){
 const enabled=new Set($$('[data-family]:checked').map(x=>x.dataset.family));
 state.scope=$('#scope').value;const selected=state.focus.join(','),involved=new Set([selected]);let bundles=0,edges=0;
 $$('#region-canvas .bundle').forEach(path=>{
  const e=path.dataset,valid=enabled.has(family(e.kind));
  const hot=state.scope==='all'||(state.scope==='near'&&(e.src===selected||e.dst===selected))||(state.scope==='out'&&e.src===selected)||(state.scope==='in'&&e.dst===selected);
  path.style.display=valid?'':'none';path.style.opacity=hot?(state.scope==='all'?'.48':'.88'):'.035';path.style.strokeWidth=hot?(state.scope==='all'?'1.5':'2.5'):'1';
  if(valid&&hot){bundles++;edges+=+e.count;involved.add(e.src);involved.add(e.dst);}
 });
 $$('#region-canvas .position').forEach(n=>{const hit=n.dataset.pos===selected,r=n.querySelector('.position-outline');r.style.stroke=hit?'#0f766e':'#dce4ed';r.style.strokeWidth=hit?'3':'1';n.style.opacity=state.scope==='all'||involved.has(n.dataset.pos)?'1':'.48';});
 $('#region-summary').textContent=`选中位置 (${selected}) · 高亮 ${bundles} 组区域连接，对应 ${edges} 条实际消息边。`;
 const svg=$('#region-canvas svg'),v=svg.viewBox.baseVal,z=+$('#zoom').value;svg.setAttribute('width',v.width*z);svg.setAttribute('height',v.height*z);
 if(window.refreshFormulas)window.refreshFormulas();
}
function grouped(es){
 const map=new Map();for(const [v,k] of es){const n=current.nodes[v],key=[n[0],n[1],n[2],k].join(',');if(!map.has(key))map.set(key,{l:n[0],p:n[1],role:n[2],k,vs:[]});map.get(key).vs.push(v);}
 return [...map.values()].sort((a,b)=>a.l-b.l||a.p-b.p||a.role.localeCompare(b.role)||a.k-b.k);
}
function renderNode(){
 const u=selectedId(),n=current.nodes[u],left=grouped(ins[u]),right=grouped(outs[u]);
 const h=Math.max(620,220+Math.max(left.length,right.length)*83),w=1510,cy=(h+100)/2;
 let s=svgStart(w,h,`${nodeName(n)} 完整上下游`)+`<defs>`;
 kinds.forEach(k=>{s+=`<marker id="ego-${k}" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="6" markerHeight="5" orient="auto"><path d="M0 0 L10 4 L0 8 Z" fill="${colors[k]}"/></marker>`;});s+='</defs>';
 s+=svgText(25,37,`${nodeName(n)} · 全部一跳连接`,25,'#23344b',700)+svgText(25,69,'侧卡完整列出端点；数字可点击。曲线按区域、角色与边族归组，不代表额外汇总节点。',16,'#5c6d81');
 s+=svgText(260,108,`全部入边 ${ins[u].length} 条`,18,'#23344b',600,'middle')+svgText(1260,108,`全部出边 ${outs[u].length} 条`,18,'#23344b',600,'middle');
 for(const [side,groups] of [['in',left],['out',right]]){
  const gx=side==='in'?22:1020,top=145+(h-185-groups.length*83)/2;
  groups.forEach((g,row)=>{
   const y=top+row*83,k=kinds[g.k],c=colors[k];let sx,sy,tx,ty;
   if(side==='in'){sx=gx+465;sy=y+32;tx=615;ty=cy;}else{sx=895;sy=cy;tx=gx;ty=y+32;}
   s+=`<path d="M${sx} ${sy} C${sx+54} ${sy} ${tx-54} ${ty} ${tx} ${ty}" fill="none" stroke="${c}" stroke-width="2" opacity=".76" marker-end="url(#ego-${k})"/>`;
   s+=`<rect x="${gx}" y="${y}" width="465" height="66" rx="10" fill="white" stroke="${c}"/>`+svgText(gx+15,y+25,`${g.role.toUpperCase()}(${g.l},${g.p}) · ${k} · ${g.vs.length} 条`,17,c,600);
   s+=`<text x="${gx+15}" y="${y+51}" font-family="Noto Sans CJK SC,sans-serif" font-size="16" fill="#23344b">节点 `;
   g.vs.sort((a,b)=>current.nodes[a][3]-current.nodes[b][3]).forEach((v,i)=>{s+=`${i?', ':''}<tspan class="node-link" data-jump="${v}" tabindex="0" role="button" aria-label="查看 ${E(nodeName(current.nodes[v]))}">${current.nodes[v][3]}</tspan>`;});s+='</text>';
  });
 }
 const color=n[2]==='d'?'#2563eb':'#b57a13';s+=`<rect x="615" y="${cy-95}" width="280" height="190" rx="16" fill="#f5f8fd" stroke="${color}" stroke-width="2"/>`;
 s+=svgText(755,cy-48,nodeName(n),24,color,700,'middle')+svgText(755,cy-12,isUnit()?'每条消息边 δ=1':'边时延 δ=目标层−源层',16,'#23344b',400,'middle')+svgText(755,cy+20,'首层输入的潜在总时延',14,'#5c6d81',400,'middle')+svgText(755,cy+48,displayDelays(u).join(', '),18,'#23344b',600,'middle')+svgText(755,cy+77,'Full 沿固定出边广播',14,'#5c6d81',400,'middle');
 s+=svgText(24,h-22,'这里只画该节点的一跳上下游；邻居之间及邻居通往其他节点的边不在此图中。',15,'#5c6d81')+'</svg>';
 $('#node-canvas').innerHTML=s;
 $('#node-summary').textContent=`${nodeName(n)} · 入度 ${ins[u].length}，出度 ${outs[u].length}。`+(n[2]==='d'?`仅经普通路径可达的时延：${displayDelays(u,true).join('、')}。`:'Hub 仍依赖自身输入与历史；独立分区不会消除这些依赖。');
 const svg=$('#node-canvas svg');svg.setAttribute('width',w*.88);svg.setAttribute('height',h*.88);
 if(window.refreshFormulas)window.refreshFormulas();
}
function updateTargets(preferred){
 state.source=$('#pair-source').value;
 const targets=[...new Set(current.bundles.filter(b=>`${b[0]},${b[1]}`===state.source).map(b=>`${b[3]},${b[4]}`))].sort((a,b)=>{const x=a.split(',').map(Number),y=b.split(',').map(Number);return x[0]-y[0]||x[1]-y[1];});
 $('#pair-target').innerHTML=targets.map(p=>option(p,`(${p})`)).join('');
 state.target=targets.includes(preferred)?preferred:targets[0];$('#pair-target').value=state.target;renderMatrix();
}
function renderMatrix(){
 state.target=$('#pair-target').value;const src=state.source.split(',').map(Number),dst=state.target.split(',').map(Number);
 const sources=positionNodes(src),targets=positionNodes(dst),rows=sources.length,columns=targets.length;
 const ti=new Map(targets.map((u,i)=>[u,i])),matrix=new Map(),counts={};sources.forEach((u,i)=>{for(const [v,k]of outs[u])if(ti.has(v)){matrix.set(`${i},${ti.get(v)}`,[u,v,k]);counts[k]=(counts[k]||0)+1;}});
 const cell=29,x0=145,y0=145,legendX=Math.max(815,x0+columns*cell+55),w=legendX+405,h=Math.max(800,y0+rows*cell+65);
 let s=svgStart(w,h,'完整端点连接矩阵')+svgText(25,36,`位置 (${state.source}) → (${state.target}) · 全部端点`,24,'#23344b',700)+svgText(25,67,'行：源节点。列：目标节点。彩点：消息边。角色块按当前图显示；无 hub 时只有 D→D。',16,'#5c6d81');
 s+=svgText(x0+current.ordinary*cell/2,y0-48,'下游普通区 D',17,'#2563eb',600,'middle')+(hubsAt(dst[0])?svgText(x0+(current.ordinary+hubsAt(dst[0])/2)*cell,y0-48,'下游 hub 区 H',16,'#b57a13',600,'middle'):'');
 for(let j=0;j<columns;j++){
  const label=j<current.ordinary?'D'+j:'H'+(j-current.ordinary);
  s+=svgText(x0+j*cell+cell/2,y0-14,label,12,'#23344b',400,'middle',`class="node-link" data-jump="${targets[j]}"`);
 }
 for(let i=0;i<rows;i++){
  const label=i<current.ordinary?'D'+i:'H'+(i-current.ordinary);
  s+=svgText(x0-17,y0+i*cell+20,label,13,'#23344b',400,'end',`class="node-link" data-jump="${sources[i]}"`);
  for(let j=0;j<columns;j++){
   const hit=matrix.get(`${i},${j}`),x=x0+j*cell,y=y0+i*cell,fill=i<current.ordinary&&j<current.ordinary?'#f3f7fe':i>=current.ordinary&&j>=current.ordinary?'#fdf6e9':'#fafbfd';
   s+=`<g ${hit?`data-jump="${hit[1]}" style="cursor:pointer"`:''}><rect class="matrix-hit" x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${fill}" stroke="#dce4ed" stroke-width=".7"/>`;
   if(hit)s+=`<circle cx="${x+cell/2}" cy="${y+cell/2}" r="5.8" fill="${colors[kinds[hit[2]]]}"/>`;
   s+=`<title>${E(nodeName(current.nodes[sources[i]]))} → ${E(nodeName(current.nodes[targets[j]]))}${hit?' · '+kinds[hit[2]]+' · δ='+edgeDelay(src[0],dst[0]):' · 无边'}</title></g>`;
  }
 }
 let y=185;Object.entries(counts).sort((a,b)=>+a[0]-b[0]).forEach(([k,n])=>{s+=`<circle cx="${legendX+10}" cy="${y-5}" r="7" fill="${colors[kinds[k]]}"/>`+svgText(legendX+30,y,`${kinds[k]}：${n} 条`,19,colors[kinds[k]],600);y+=38;});
 ['DD：普通细节传播','DH：普通结果进入 hub','HH：hub 交换背景','HD：hub 向普通节点投递钩子','','这里只显示这一对位置。','点击编号 / 彩点进入节点图。'].forEach((t,i)=>s+=svgText(legendX,y+32+i*31,t,16,'#5c6d81'));
 s+=svgText(25,h-20,'每位置一个 D 区，非末层可另有 H 区；矩阵按两端实际节点数绘制。',15,'#5c6d81')+'</svg>';
 $('#matrix-canvas').innerHTML=s;$('#matrix-summary').textContent=`${matrix.size} 条消息边 · 跨 ${dst[0]-src[0]} 层 · 时延均为 ${edgeDelay(src[0],dst[0])} · ${rows}×${columns} 端点`;
 if(window.refreshFormulas)window.refreshFormulas();
}
function renderDepthTable(){
 let s='<thead><tr><th>候选（点击切换）</th><th>位置 / 节点</th><th>普通路径时延</th><th>普通路径每相位入口位置数</th><th>全图每相位入口位置数</th><th>普通路径首层 D 来源数</th><th>全图首层 D 来源数</th></tr></thead><tbody>';
 const keys=order.concat(Object.values(ATLAS.cases).filter(c=>c.profile.includes('compact')&&c.ordinary===current.ordinary&&c.hubs===current.hubs).map(c=>c.key));
 keys.forEach(key=>{const c=ATLAS.cases[key],st=c.stats;s+=`<tr data-case="${key}" class="${key===state.key?'current':''}"><td>${profileLabel(c)}</td><td>${st.positions} / ${st.nodes.toLocaleString()}</td><td>${st.ordinary_delays.join(', ')}</td><td>${st.ordinary_phase_sources.map(p=>p[0]).join(' / ')}（共 ${c.widths[0]}）</td><td>${st.all_phase_sources.map(p=>p[0]).join(' / ')}</td><td>${st.ordinary_phase_node_sources.map(p=>p[0]).join(' / ')}（共 ${st.source_D_nodes}）</td><td>${st.all_phase_node_sources.map(p=>p[0]).join(' / ')}</td></tr>`;});$('#depth-table').innerHTML=s+'</tbody>';
}
function renderSettleTable(){
 let s='<thead><tr><th>Settle 空间图（点击切换）</th><th>层 / 位置 / 节点</th><th>边数</th><th>D / 对齐路径时延</th><th>宽段轮数 / 覆盖源区域与节点</th></tr></thead><tbody>';
 for(const c of Object.values(ATLAS.cases).filter(c=>c.family!=='multiscale'||c.key==='9')){
  const m=c.expansion,R=c.rounds,cover=c.family==='exchange'?`${R} 轮 / ${2**Math.min(R,m)} 区、${2**(Math.min(R,m)+Math.min(R,Math.log2(c.ordinary)))} 节点`:'—';
  s+=`<tr data-case="${c.key}" class="${c.key===state.key&&state.timing==='settle'?'current':''}"><td>${{multiscale:'多跨度 Settle',lattice:'原 HB',exchange:'位交换'}[c.family]} · ${profileLabel(c)}</td><td>${c.depth} / ${c.stats.positions} / ${c.stats.nodes}</td><td>${c.stats.edges}</td><td>${c.settle_period} / ${c.depth-1}</td><td>${cover}</td></tr>`;
 }
 $('#settle-table').innerHTML=s+'</tbody>';
}
$('#family-select').addEventListener('change',e=>{
 const {family,timing}=ATLAS.capacity.variants[e.target.value];
 if(family===current.family)setTiming(timing);
 else mountCase({multiscale:'9',lattice:'lattice-7',exchange:'exchange-10'}[family],timing);
});
$('#timing-mode').addEventListener('change',e=>setTiming(e.target.value));
$('#settle-table').addEventListener('click',e=>{const row=e.target.closest('[data-case]');if(row){mountCase(row.dataset.case,'settle');$('#regions').scrollIntoView({behavior:'smooth'});}});
$('#case-select').addEventListener('change',e=>mountCase(e.target.value,state.timing));
$('#scope').addEventListener('change',paintRegion);$('#zoom').addEventListener('change',paintRegion);
$$('[data-family]').forEach(c=>c.addEventListener('change',paintRegion));
for(const [id,predicate]of [['preset-detail',x=>x.startsWith('DD')],['preset-hub',x=>!x.startsWith('DD')],['preset-all',x=>true]])$('#'+id).addEventListener('click',()=>{$$('[data-family]').forEach(c=>c.checked=predicate(c.dataset.family));paintRegion();});
function regionClick(e){
 const p=e.target.closest('.position');if(p){state.focus=p.dataset.pos.split(',').map(Number);if($('#scope').value==='all')$('#scope').value='near';syncNodeControls();renderNode();paintRegion();return;}
 const edge=e.target.closest('.bundle');if(edge){state.source=edge.dataset.src;$('#pair-source').value=state.source;updateTargets(edge.dataset.dst);$('#matrix').scrollIntoView({behavior:'smooth',block:'start'});}
}
$('#region-canvas').addEventListener('click',regionClick);$('#region-canvas').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();regionClick(e);}});
$('#node-layer').addEventListener('change',e=>{state.focus[0]=+e.target.value;state.focus[1]=Math.min(state.focus[1],current.widths[state.focus[0]]-1);syncNodeControls();renderNode();paintRegion();});
$('#node-position').addEventListener('change',e=>{state.focus[1]=+e.target.value;renderNode();paintRegion();});
$('#node-role').addEventListener('change',e=>{state.role=e.target.value;syncNodeControls();renderNode();});
$('#node-index').addEventListener('change',e=>{state.index=+e.target.value;renderNode();});
$('#example-d').addEventListener('click',()=>chooseNode(current.selected_d));$('#example-h').addEventListener('click',()=>chooseNode(current.selected_h));
for(const host of ['node-canvas','matrix-canvas']){
 const handler=e=>{const el=e.target.closest('[data-jump]');if(el){chooseNode(+el.dataset.jump);if(host==='matrix-canvas')$('#neighbors').scrollIntoView({behavior:'smooth'});}};
 $('#'+host).addEventListener('click',handler);$('#'+host).addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();handler(e);}});
}
$('#pair-source').addEventListener('change',()=>updateTargets());$('#pair-target').addEventListener('change',renderMatrix);
$('#depth-table').addEventListener('click',e=>{const row=e.target.closest('[data-case]');if(row){mountCase(row.dataset.case);$('#regions').scrollIntoView({behavior:'smooth'});}});
$$('[data-export]').forEach(b=>b.addEventListener('click',()=>{const svg=$('#'+b.dataset.export+' svg'),blob=new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`hb-${state.key}-${state.timing}-${b.dataset.export}.svg`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}));
mountCase('9');
window.atlasDebug={get case(){return current;},get state(){return state;},get keys(){return Object.keys(ATLAS.cases);},mountCase,chooseNode,setTiming,displayDelays};
