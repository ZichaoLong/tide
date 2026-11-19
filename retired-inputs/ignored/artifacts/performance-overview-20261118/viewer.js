
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
