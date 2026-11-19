'use strict';
// Native MathML keeps the formula panel available offline.
const mathBlock=x=>`<div class="math-equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">${x}</math></div>`;
const mi=x=>`<mi>${E(x)}</mi>`,mn=x=>`<mn>${E(x)}</mn>`,mo=x=>`<mo>${E(x)}</mo>`;
const sub=(x,y)=>`<msub>${x}<mrow>${y}</mrow></msub>`;
const sup=(x,y)=>`<msup>${x}<mrow>${y}</mrow></msup>`;
const cal=x=>`<mi mathvariant="script">${E(x)}</mi>`;
const par=x=>`<mrow>${mo('(')}${x}${mo(')')}</mrow>`;
const setMath=xs=>mo('{')+xs.map(mn).join(mo(','))+mo('}');
const textMath=x=>`<mtext>${E(x)}</mtext>`;
function targetFormula(l,s,p,mode){
 const w=current.widths[l],v=current.widths[l+s],g=Math.min(w,v),A=w/g,B=v/g;
 let x=Math.floor(p/A);
 if(mode==='near')x=(x+1)%g;
 if(mode==='cross'&&g>1)x^=1<<((l+Math.log2(s))%Math.log2(g));
 return Array.from({length:B},(_,j)=>B*x+j);
}
function settleTargets(l,p,exchange=true){
 const w=current.widths[l],v=current.widths[l+1],g=Math.min(w,v),A=w/g,B=v/g,x=Math.floor(p/A);
 const xs=exchange&&g>1?[x,x^(1<<(l%Math.log2(g)))]:[x];
 return [...new Set(xs.flatMap(y=>Array.from({length:B},(_,j)=>B*y+j)))].sort((a,b)=>a-b);
}
function settleRegionFormulas(){
 const [l,p]=state.focus,ex=current.family==='exchange';
 let s=`<h3>${ex?'分阶段位交换':'原 HB 递归邻域'}</h3><p class="formula-state">${current.depth} 层 · 位置 (${l},${p}) · ${hubsAt(l)?'D / H 分区':'只有 D 区'}</p>`;
 s+=mathBlock(sub(mi('r'),mi('v'))+mo('=')+mi('ℓ')+mo('+')+mn(1)+mo(',')+mi('δ')+mo('=')+mn(1));
 s+=mathBlock(mi('ι')+par(mi('t'))+mo('=')+mn(period())+mi('t')+mo('+')+mn(1)+mo(',')+mi('D')+mo('>')+mi('M'));
 if(ex){
  const m=current.expansion,R=current.rounds;
  s+=mathBlock(mi('M')+mo('=')+mn(2)+mi('m')+mo('+')+mi('R')+mo('+')+mn(1));
  s+=mathBlock(mi('m')+mo('=')+mn(m)+mo(',')+mi('R')+mo('=')+mn(R));
  s+=mathBlock(sub(mi('W'),mi('ℓ'))+mo('=')+sup(mn(2),textMath('min(ℓ,m,M−1−ℓ)')));
  s+=mathBlock(sub(mi('c'),mi('ℓ'))+mo('=')+mi('ℓ')+textMath(' mod log₂ n'));
  s+=mathBlock(mi('i')+mo('↦')+mo('{')+mi('i')+mo(',')+mi('i')+mo('⊕')+sup(mn(2),sub(mi('c'),mi('ℓ')))+mo('}'));
  s+='<p>普通 DDX 边族由区域配对与节点配对取笛卡尔积。只计这一边族，等宽且宽度大于 1 时，每节点入 / 出度为 4；相邻宽度比为 2 时至多 8。Hub 的 DH、HH、HD 连线另计。展开、宽段与收拢均使用本规则。</p>';
 }else{
  s+=mathBlock(mi('L')+mo('=')+mn(2)+mi('h')+mo('+')+mn(1)+mo(',')+mi('D')+mo('=')+mi('L')+mo('+')+mn(2));
  s+=mathBlock(mi('i')+mo('↦')+mo('{')+mi('i')+mo(',')+par(mi('i')+mo('+')+mn(1))+textMath(' mod n')+mo('}'));
  s+='<p>二叉递归展开与收拢；不增加跨粗组交换。此处层号从 0 起，对应原文层号减 1。</p>';
 }
 if(l<current.depth-1){
  const w=current.widths[l],v=current.widths[l+1],g=Math.min(w,v),A=w/g,B=v/g,x=Math.floor(p/A);
  s+='<h3>当前层的完整代入</h3>';
  s+=mathBlock(mi('g')+mo('=')+mn(g)+mo(',')+mi('A')+mo('=')+mn(A)+mo(',')+mi('B')+mo('=')+mn(B));
  s+=mathBlock(mi('x')+mo('=')+mo('⌊')+`<mfrac>${mn(p)}${mn(A)}</mfrac>`+mo('⌋')+mo('=')+mn(x));
  if(ex&&g>1){
   const b=l%Math.log2(g);
   s+=mathBlock(mi('b')+mo('=')+mn(l)+textMath(' mod ')+mn(Math.log2(g))+mo('=')+mn(b));
   s+=mathBlock(mi('y')+mo('∈')+setMath([x,x^(1<<b)]));
  }else{
   s+=ex?'<p>基础配对；g=1 时不定义区域 XOR 位，也不计算模零。</p>':'<p>原图只使用递归基础配对。</p>';
   s+=mathBlock(mi('y')+mo('=')+mi('x')+mo('=')+mn(x));
  }
  s+=mathBlock(mi('p′')+mo('=')+mi('B')+mi('y')+mo('+')+mi('j')+mo(',')+mn(0)+mo('≤')+mi('j')+mo('<')+mi('B'));
  s+=mathBlock(cal('Q')+par(mn(p))+mo('=')+setMath(settleTargets(l,p,ex)));
  if(ex)s+=mathBlock(sub(mi('c'),mn(l))+mo('=')+mn(l%Math.log2(current.ordinary)));
 }else s+='<p>末层没有消息出边。</p>';
 if(ex)s+='<p>Hub 增强在非末层沿 Q 做 DH 收集与 HH 交换，沿基础配对做每 hub 一个 HD 探针；末层只有 D，因此进入末层只保留 DDX 与 HD1。源与目标的完整节点端点见矩阵。</p>';
 s+=`<p><a href="../../../topologies/${ex?'hb-exchange':'hb-lattice'}.md">完整拓扑与证明</a> · <a href="../../../instances/hb-settle-weighted-message.md">共同局部绑定</a></p>`;
 $('#region-formulas').innerHTML=s;
}
function regionFormulas(){
 if(current.family!=='multiscale'){settleRegionFormulas();return;}
 const [l,p]=state.focus;
 let s=`<h3>区域与时间</h3><p class="formula-state">当前：${current.depth} 层 · 位置 (${l},${p})</p>`;
 s+=mathBlock(sub(mi('D'),mi('ℓ')+mo(',')+mi('p'))+mo('=')+mo('{')+sub(mi('d'),mi('ℓ')+mo(',')+mi('p')+mo(',')+mi('i'))+mo(':')+mn(0)+mo('≤')+mi('i')+mo('<')+mi('n')+mo('}'));
 s+=mathBlock(sub(mi('H'),mi('ℓ')+mo(',')+mi('p'))+mo('=')+(hubsAt(l)?mo('{')+sub(mi('h'),mi('ℓ')+mo(',')+mi('p')+mo(',')+mi('a'))+mo(':')+mn(0)+mo('≤')+mi('a')+mo('<')+mi('h')+mo('}'):mo('∅')));
 s+=mathBlock(mi('n')+mo('=')+mn(current.ordinary)+mo(',')+mi('h')+mo('=')+mn(current.hubs)+mo(',')+mi('δ')+par(mi('e'))+mo('=')+(isUnit()?mn(1):mi('s')));
 const clockSymbol=state.timing==='settle'?'D':'T';
 s+=mathBlock(mi('ι')+par(mi('t'))+mo('=')+mi(clockSymbol)+mi('t')+mo('+')+mn(1)+mo(',')+mi(clockSymbol)+mo('=')+mn(period()));
 s+='<p>每条边严格增加层号。同区只有共同分配，没有隐含消息边。主图末层只有 D；H 集合为空，不设 H 区或其关联边。</p><h3>不同宽度间的配对</h3>';
 s+=mathBlock(mi('g')+mo('=')+textMath('min')+par(sub(mi('W'),mi('ℓ'))+mo(',')+sub(mi('W'),mi('ℓ')+mo('+')+mi('s'))));
 s+=mathBlock(mi('A')+mo('=')+`<mfrac>${sub(mi('W'),mi('ℓ'))}${mi('g')}</mfrac>`+mo(',')+mi('B')+mo('=')+`<mfrac>${sub(mi('W'),mi('ℓ')+mo('+')+mi('s'))}${mi('g')}</mfrac>`);
 s+=mathBlock(`<msubsup>${cal('B')}<mrow>${mi('ℓ')+mo(',')+mi('s')}</mrow>${mi('σ')}</msubsup>`+par(mi('p'))+mo('=')+mo('{')+mi('B')+mi('σ')+par(mo('⌊')+`<mfrac>${mi('p')}${mi('A')}</mfrac>`+mo('⌋'))+mo('+')+mi('j')+mo(':')+mn(0)+mo('≤')+mi('j')+mo('<')+mi('B')+mo('}'));
 s+=mathBlock(sub(mi('σ'),mn(0))+par(mi('x'))+mo('=')+mi('x')+mo(',')+sub(mi('σ'),mo('+'))+par(mi('x'))+mo('=')+par(mi('x')+mo('+')+mn(1))+textMath(' mod ')+mi('g'));
 s+=mathBlock(sub(mi('σ'),mo('×'))+par(mi('x'))+mo('=')+mi('x')+mo('⊕')+sup(mn(2),mi('k')));
 s+=mathBlock(mi('k')+mo('=')+par(mi('ℓ')+mo('+')+sub(textMath('log'),mn(2))+mi('s'))+textMath(' mod ')+sub(textMath('log'),mn(2))+mi('g'));
 s+='<p>上述 XOR 式用于 g&gt;1；g=1 时，跨组映射取 σ×(x)=x，不计算模零。</p>';
 const rows=[1,2,4].filter(d=>l+d<current.depth).map(d=>`<tr><td>${d}</td><td>${targetFormula(l,d,p,'base').join(',')}</td><td>${targetFormula(l,d,p,'near').join(',')}</td><td>${targetFormula(l,d,p,'cross').join(',')}</td></tr>`).join('');
 s+=`<p><b>代入当前位置：</b>下面是目标层 ℓ+s 中的位置编号。三个映射的具体用途由边族决定。</p>${rows?`<table class="formula-table"><thead><tr><th>s</th><th>基础 0</th><th>邻组 +</th><th>跨组 ×</th></tr></thead><tbody>${rows}</tbody></table>`:'<p>已到终层，没有后继。</p>'}`;
 s+='<details><summary>边族使用哪些配对</summary><p>DD1、DH1：基础 ∪ 邻组。DD2、额外 DD4：跨组。HH：基础 ∪ 跨组。HD1：基础；HD2、HD4：跨组。</p><p>层宽为正的二次幂，允许 1／2；边界 g=1 使用恒等跨组映射。</p></details>';
 $('#region-formulas').innerHTML=s;
}
function nodeFormulas(){
 const u=selectedId(),n=current.nodes[u],ds=displayDelays(u),dd=displayDelays(u,true),ph=[...new Set(ds.map(x=>x%period()))].sort((a,b)=>a-b);
 let s=`<h3>所选节点</h3><p class="formula-state">${E(nodeName(n))}</p>`;
 s+=mathBlock(mi('v')+mo('=')+sub(mi(n[2]),n.slice(0,2).concat(n[3]).map(mn).join(mo(','))));
 s+=mathBlock(textMath('Pred')+par(mi('v'))+mo('=')+mo('{')+mi('u')+mo(':')+par(mi('u')+mo(',')+mi('v'))+mo('∈')+mi('E')+mo('}'));
 s+=mathBlock(textMath('Succ')+par(mi('v'))+mo('=')+mo('{')+mi('w')+mo(':')+par(mi('v')+mo(',')+mi('w'))+mo('∈')+mi('E')+mo('}'));
 s+=mathBlock(mo('|')+textMath('Pred')+par(mi('v'))+mo('|')+mo('=')+mn(ins[u].length)+mo(',')+mo('|')+textMath('Succ')+par(mi('v'))+mo('|')+mo('=')+mn(outs[u].length));
 s+='<h3>潜在路径的总时延</h3>';
 s+=mathBlock(cal('D')+par(mi('v'))+mo('=')+`<munder>${mo('⋃')}<mrow>${par(mi('u')+mo(',')+mi('v'))+mo('∈')+mi('E')}</mrow></munder>`+mo('{')+mi('d')+mo('+')+mi('δ')+par(mi('u')+mo(',')+mi('v'))+mo(':')+mi('d')+mo('∈')+cal('D')+par(mi('u'))+mo('}'));
 s+='<p>首层输入节点取 D(v)={0}，随后沿 DAG 递推；这里计算的是消息路径。</p>';
 s+=mathBlock(cal('D')+par(mi('v'))+mo('=')+setMath(ds));
 if(n[2]==='d')s+=mathBlock(sub(cal('D'),textMath('普通'))+par(mi('v'))+mo('=')+setMath(dd));
 s+=mathBlock(mi('Φ')+par(mi('v'))+mo('=')+mo('{')+mi('d')+textMath(' mod ')+sub(mi('P'),textMath('in'))+mo(':')+mi('d')+mo('∈')+cal('D')+par(mi('v'))+mo('}'));
 s+=mathBlock(mi('Φ')+par(mi('v'))+mo('=')+setMath(ph));
 s+='<p>此处 P<sub>in</sub> 是当前输入周期（Timed 记 T，Settle 记 D）。Φ 是相对入口的 d mod P<sub>in</sub>；绝对时间相位为 (1+d) mod P<sub>in</sub>。Settle 图每节点只有一个潜在路径时延。</p><h3>与局部作用的对应</h3><p>同刻到达的消息进入同一时间纤维，再由 Agg 聚合；异刻消息分次作用，通过持久状态衔接。</p><p class="formula-boundary">面板只计算拓扑可达。记忆、评分和 SelStep 已在 <a href="../../../instances/hb-multiscale-weighted-message.md">三组主绑定</a>中定义，此处没有模拟 cᵥ 或实际发送。</p>';
 $('#node-formulas').innerHTML=s;
}
function pairFormulas(){
 if(!state.source||!state.target)return;
 const [l,p]=state.source.split(',').map(Number),[m,q]=state.target.split(',').map(Number),span=m-l;
 const present=current.bundles.filter(b=>b[0]===l&&b[1]===p&&b[3]===m&&b[4]===q);
 let s=`<h3>当前两位置的端点公式</h3><p class="formula-state">(${l},${p}) → (${m},${q})</p>`;
 s+=mathBlock(mi('s')+mo('=')+mn(m)+mo('−')+mn(l)+mo('=')+mn(span)+mo(',')+mi('δ')+mo('=')+mn(edgeDelay(l,m)));
 if(current.hubs)s+=mathBlock(mi('b')+mo('=')+`<mfrac>${mi('n')}${mi('h')}</mfrac>`+mo('=')+mn(current.ordinary/current.hubs));
 for(const b of present){
  const k=b[6];s+=`<h4 style="color:${colors[k]}">${k} · ${b[7]} 条</h4>`;
  if(k==='DDX')s+=mathBlock(mi('i')+mo('↦')+mo('{')+mi('i')+mo(',')+mi('i')+mo('⊕')+sup(mn(2),mn(l%Math.log2(current.ordinary)))+mo('}'));
  else if(k==='DD1'||k==='DDL')s+=mathBlock(mi('i')+mo('↦')+mo('{')+mi('i')+mo(',')+par(mi('i')+mo('+')+mn(1))+textMath(' mod ')+mi('n')+mo('}'));
  else if(k==='DD2'||k==='DD4')s+=mathBlock(mi('i')+mo('↦')+sub(mi('π'),mi('ℓ')+mo(',')+mi('p')+mo(',')+mi('q'))+par(mi('i')));
  else if(k==='DH1')s+=mathBlock(mi('i')+mo('↦')+mo('⌊')+`<mfrac><mrow>${sub(mi('ρ'),mi('ℓ')+mo(',')+mi('p')+mo(',')+mi('q'))+par(mi('i'))}</mrow>${mi('b')}</mfrac>`+mo('⌋'));
  else if(k.startsWith('HH'))s+=mathBlock(mi('a')+mo('↦')+par(mi('a')+mo('+')+mi('ℓ')+mo('+')+mi('p')+mo('+')+mi('q')+mo('+')+sub(textMath('log'),mn(2))+mi('s'))+textMath(' mod ')+mi('h'));
  else if(k.startsWith('HD')){
   s+=mathBlock(mi('c')+mo('=')+par(mi('a')+mo('+')+mi('ℓ')+mo('+')+mi('p')+mo('+')+mi('s'))+textMath(' mod ')+mi('h'));
   s+=mathBlock(mi('η')+mo('=')+par(mn(7)+mi('ℓ')+mo('+')+mn(3)+mi('p')+mo('+')+mi('q')+mo('+')+mn(5)+mi('s'))+textMath(' mod ')+mi('b'));
   s+=mathBlock(sub(mi('j'),mi('t'))+mo('=')+mi('χ')+par(mi('c')+mi('b')+mo('+')+par(par(mi('η')+mo('+')+`<mfrac><mrow>${mi('t')+mi('b')}</mrow>${sub(mi('r'),mi('s'))}</mfrac>`)+textMath(' mod ')+mi('b'))));
   s+=mathBlock(mn(0)+mo('≤')+mi('t')+mo('<')+sub(mi('r'),mi('s'))+mo(',')+sub(mi('r'),mi('s'))+mo('=')+mn(current.probes[String(span)]));
   s+='<p>χ 是目标位置的固定普通节点置换；c 是目标分组编号，不是发送控制 cᵥ。</p>';
  }
  const pairs=[];for(const [u,v,ki]of current.edges)if(kinds[ki]===k&&pos(current.nodes[u])===state.source&&pos(current.nodes[v])===state.target)pairs.push([current.nodes[u][3],current.nodes[v][3]]);
  const first=Math.min(...pairs.map(x=>x[0]));
  s+=`<p class="formula-example">端点代入：${b[2].toUpperCase()}${first} → ${pairs.filter(x=>x[0]===first).map(x=>b[5].toUpperCase()+x[1]).join('、')}</p>`;
 }
 if(current.hubs||current.family==='multiscale')s+='<details><summary>固定置换怎样确定</summary><p>π、ρ、χ 在建图时确定。对标签与索引拼接固定种子后取 SHA-256，按摘要排序得到置换；不是每次输入重新采样。</p><p>π 使用 detail 标签，额外 DD4 使用 detail4；ρ 使用 collect；χ 使用 hook。精确字节规则见 <a href="../../../reference/backbones/multiscale.py">构图源</a>与 <a href="../../../topologies/hb-multiscale.md">拓扑定义</a>。</p></details>';
 $('#pair-formulas').innerHTML=s;
}
function renderTiming(){
 const mode=state.timing,u=selectedId(),n=current.nodes[u],cycle=period(),delays=displayDelays(u);
 const arrivals=new Map();
 for(let t=0;t<3;t++)for(const d of delays){
  const theta=1+cycle*t+d;
  if(!arrivals.has(theta))arrivals.set(theta,[]);
  arrivals.get(theta).push(`输入 ${t}，路径时延 ${d}`);
 }
 $('#timing-formula').innerHTML=mathBlock(mi('δ')+par(mi('e'))+mo('=')+(isUnit()?mn(1):mi('s'))+mo(',')+mi('ι')+par(mi('t'))+mo('=')+mn(cycle)+mi('t')+mo('+')+mn(1));
 $('#timing-summary').textContent=`${nodeName(n)} · 潜在路径时延 {${delays.join(', ')}} · 输入间隔 ${cycle}。`+(mode==='settle'?`D=${cycle}>M=${current.depth}，符合 Settle 时间编码。`:mode==='timed-common'?'与 Settle 保持相同输入日程，使用单位边时延。':'长短路径在不同逻辑时刻产生机会，也可能让不同输入位置同刻汇合。');
 $('#timing-table').innerHTML='<thead><tr><th>逻辑时间 θ</th><th>同刻潜在来源</th></tr></thead><tbody>'+[...arrivals].sort((a,b)=>a[0]-b[0]).map(([theta,xs])=>`<tr><td>${theta}</td><td>${xs.map(E).join('；')}</td></tr>`).join('')+'</tbody>';
}
window.refreshFormulas=function(){if(!current)return;regionFormulas();nodeFormulas();pairFormulas();renderTiming();};
$('#formulas-toggle').addEventListener('change',e=>document.body.classList.toggle('hide-formulas',!e.target.checked));
window.atlasFormulaDebug={targetFormula,settleTargets};
window.refreshFormulas();
