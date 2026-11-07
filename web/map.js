'use strict';
const data = JSON.parse(document.getElementById('research-data').textContent);
const $ = id => document.getElementById(id);
const svg = $('map'), NS = 'http://www.w3.org/2000/svg', W = 250, H = 118;
const kinds = {goal:'目标',baseline:'起点',experiment:'实验',result:'结果位置',evidence:'已有证据'};
const statuses = {target:'待研究',unselected:'待选定',documented:'固定定义',planned:'未运行',awaiting:'待填 · 未运行',scoped:'范围限定'};
const edgeNames = {decomposes:'分解',precondition:'前提',variant:'改造',produces:'产出',comparison:'对照',transfer:'迁移待验',informs:'参考',evaluates:'检验目标',supports:'支持',challenges:'反证'};
const nodes = new Map(data.nodes.map(n => [n.id,n]));
const nodeElements = new Map(), edgeElements = new Map();
let selected = null, scale = 1;
function element(tag, attrs = {}, text = null) {
  const el = document.createElementNS(NS,tag);
  for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
  if (text !== null) el.textContent = text;
  return el;
}
function html(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function interactive(el, action) {
  el.addEventListener('click', action);
  el.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault();action();}});
}
svg.setAttribute('viewBox',`0 0 ${data.width} ${data.height}`);
$('notice').textContent = data.notice;
const defs=element('defs'),marker=element('marker',{id:'arrow',markerWidth:8,markerHeight:8,refX:7,refY:4,orient:'auto'});
marker.append(element('path',{d:'M 0 1 L 7 4 L 0 7',fill:'none',stroke:'#82959e'}));defs.append(marker);svg.append(defs);
for (const col of data.columns) {
  svg.append(element('text',{x:col.x+8,y:55,class:'column'},col.title));
  svg.append(element('line',{x1:col.x-12,x2:col.x-12,y1:80,y2:data.height-25,class:'column-line'}));
}
function geometry(edge) {
  const a=nodes.get(edge.source),b=nodes.get(edge.target);
  if(a.x===b.x){
    const sx=a.x+W,sy=a.y+H/2,ty=b.y+H/2,via=sx+16;
    return {path:`M${sx},${sy} C${via},${sy} ${via},${sy} ${via},${(sy+ty)/2} S${via},${ty} ${sx},${ty}`,
            x:via-8,y:(sy+ty)/2,anchor:'middle',rotation:-90};
  }
  const sx=a.x+W,sy=a.y+H/2,tx=b.x,ty=b.y+H/2,mid=(sx+tx)/2;
  return {path:`M${sx},${sy} C${mid},${sy} ${mid},${ty} ${tx},${ty}`,
          x:mid+(Math.abs(ty-sy)>H?12:0),y:(sy+ty)/2-(Math.abs(ty-sy)>H?0:7),anchor:'middle',rotation:Math.abs(ty-sy)>H?-90:0};
}
for (const edge of data.edges) {
  const g=element('g',{class:`edge ${edge.kind} ${edge.status}`,tabindex:0,role:'button','aria-label':`${edge.label}：${nodes.get(edge.source).title} → ${nodes.get(edge.target).title}`});
  const p=geometry(edge);
  if(p.rotation)g.classList.add('cross');
  g.append(element('path',{d:p.path,class:'hit'}),element('path',{d:p.path,class:'wire'}),
           element('text',{x:p.x,y:p.y,'text-anchor':p.anchor,transform:`rotate(${p.rotation||0} ${p.x} ${p.y})`},edge.label));
  interactive(g,()=>showEdge(edge));svg.append(g);edgeElements.set(edge.id,g);
}
for (const node of data.nodes) {
  const g=element('g',{class:`node ${node.kind} ${node.status}`,transform:`translate(${node.x},${node.y})`,tabindex:0,role:'button','aria-label':`${node.title}，${statuses[node.status]}`});
  g.append(element('rect',{width:W,height:H,rx:9}),
           element('text',{x:14,y:23,class:'kind'},`${node.id} / ${kinds[node.kind]}`),
           element('text',{x:W-14,y:23,class:'status','text-anchor':'end'},statuses[node.status]),
           element('text',{x:14,y:57,class:'title'},node.title),
           element('text',{x:14,y:88,class:'summary'},node.summary));
  interactive(g,()=>showNode(node));svg.append(g);nodeElements.set(node.id,g);
}
function section(parent, title, value) {
  if(!value)return;
  parent.append(html('h3',title),html('p',value));
}
function showNode(node) {
  selected=node.id;
  const body=$('detail-body');body.replaceChildren();
  body.append(html('div',`${node.id} · ${kinds[node.kind]} · ${statuses[node.status]}`,'meta'));
  const heading=html('h2',node.title);heading.id='detail-title';body.append(heading);
  section(body,'这一步要回答',node.question);
  section(body,'据此作什么决定',node.decision);
  section(body,'证据范围',node.scope || (node.kind==='result'?'尚无结果。这是待填位置，不计入已完成实验。':'拟议研究关系；不是已通过结论。'));
  section(body,'资源',node.budget);
  if(node.sources){
    body.append(html('h3','固定来源'));
    for(const source of node.sources){const a=html('a',source.label);a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';body.append(a,html('br'));}
  }
  body.append(html('h3','直接关联'));
  for(const edge of data.edges.filter(e=>e.source===node.id||e.target===node.id)){
    const other=nodes.get(edge.source===node.id?edge.target:edge.source);
    const b=html('button',`${edge.source===node.id?'→':'←'} ${edgeNames[edge.kind]} · ${other.title}`,'neighbor-link');
    b.addEventListener('click',()=>showNode(other));body.append(b);
  }
  $('details').hidden=false;update();
}
function showEdge(edge) {
  const body=$('detail-body');body.replaceChildren();
  body.append(html('div',`${edge.id} · ${edge.status==='planned'?'拟议 / 待验证':'关系已定义，效力按来源解释'}`,'meta'));
  const h=html('h2',edgeNames[edge.kind]);h.id='detail-title';body.append(h);
  section(body,'连接',`${nodes.get(edge.source).title} → ${nodes.get(edge.target).title}`);
  section(body,'含义与边界',edge.detail);
  $('details').hidden=false;
}
function update(){
  const query=$('search').value.trim().toLowerCase(),kind=$('kind').value,view=$('view').value;
  const visible=new Set(data.nodes.filter(n=>(kind==='all'||n.kind===kind)&&JSON.stringify(n).toLowerCase().includes(query)).map(n=>n.id));
  const focus=$('neighbors').checked&&selected;
  const near=new Set([selected]);
  for(const e of data.edges)if(e.source===selected||e.target===selected){near.add(e.source);near.add(e.target);}
  for(const [id,el] of nodeElements){
    el.classList.toggle('hidden',!visible.has(id));el.classList.toggle('selected',id===selected);
    el.classList.toggle('dim',!!focus&&!near.has(id));
    el.setAttribute('tabindex',visible.has(id)?'0':'-1');
  }
  let edgeCount=0;
  for(const e of data.edges){
    const allowed=view==='all'||(view==='route'?['variant','produces','comparison','decomposes'].includes(e.kind):['informs','transfer','evaluates','precondition','supports','challenges'].includes(e.kind));
    const shown=allowed&&visible.has(e.source)&&visible.has(e.target),el=edgeElements.get(e.id);
    el.classList.toggle('hidden',!shown);el.classList.toggle('dim',!!focus&&e.source!==selected&&e.target!==selected);
    el.setAttribute('tabindex',shown?'0':'-1');if(shown)edgeCount++;
  }
  const measured=data.nodes.filter(n=>n.kind==='result'&&n.status==='scoped').length;
  const awaiting=data.nodes.filter(n=>n.kind==='result'&&n.status==='awaiting').length;
  $('count').textContent=`显示 ${visible.size}/${data.nodes.length} 个节点，${edgeCount} 条关系 · 已录入研究结果 ${measured} 项，待填 ${awaiting} 项`;
}
function zoom(value){scale=Math.max(.45,Math.min(2.5,value));svg.style.width=`${data.width*scale}px`;svg.style.height=`${data.height*scale}px`;}
function fit(){zoom(Math.min(($('canvas').clientWidth-3)/data.width,($('canvas').clientHeight-3)/data.height));}
$('filters').addEventListener('input',update);$('filters').addEventListener('change',update);
$('reset').addEventListener('click',()=>{HTMLFormElement.prototype.reset.call($('filters'));selected=null;$('details').hidden=true;update();});
$('close').addEventListener('click',()=>{$('details').hidden=true;});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('details').hidden=true;});
$('plus').addEventListener('click',()=>zoom(scale*1.2));$('minus').addEventListener('click',()=>zoom(scale/1.2));$('fit').addEventListener('click',fit);
update();fit();
