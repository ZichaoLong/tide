// Optional real-browser check: node --experimental-websocket tests/browser_smoke.mjs
// Uses only Node built-ins and an installed Firefox. No package/browser download.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, writeFile, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const profile=await mkdtemp(path.join(tmpdir(),'tide-map-browser-'));
const browser=spawn(process.env.FIREFOX_BINARY || 'firefox',[
  '--headless','--no-remote','--profile',profile,'--remote-debugging-port','0',
  '--width','1600','--height','1100'
],{detached:true,stdio:['ignore','pipe','pipe'],env:{...process.env,MOZ_HEADLESS_WIDTH:'1600',MOZ_HEADLESS_HEIGHT:'1100'}});
let socket,id=0,timer;
const pending=new Map(), errors=[];
function send(method,params={}){
  return new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});socket.send(JSON.stringify({id:key,method,params}));});
}
async function run(){
  const url=await new Promise((resolve,reject)=>{
    let log='';
    const capture=chunk=>{log+=chunk.toString();const m=log.match(/WebDriver BiDi listening on (ws:\/\/[^\s]+)/);if(m)resolve(m[1]);};
    browser.stdout.on('data',capture);browser.stderr.on('data',capture);
    browser.on('error',reject);browser.on('exit',code=>reject(new Error(`Firefox exited ${code}: ${log.slice(-1500)}`)));
  });
  socket=new WebSocket(url.replace(/\/$/,'')+'/session');
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  socket.addEventListener('message',event=>{
    const value=JSON.parse(event.data),p=pending.get(value.id);
    if(value.method==='log.entryAdded' && value.params.level==='error')errors.push(value.params.text);
    if(!p)return;pending.delete(value.id);
    if(value.type==='error')p.reject(new Error(JSON.stringify(value)));else p.resolve(value.result);
  });
  await send('session.new',{capabilities:{}});
  await send('session.subscribe',{events:['log.entryAdded']});
  const {contexts}=await send('browsingContext.getTree');
  const context=contexts[0].context;
  await send('browsingContext.navigate',{context,url:pathToFileURL(path.join(root,'web/index.html')).href,wait:'complete'});
  async function evaluate(expression){
    const r=await send('script.evaluate',{expression,target:{context},awaitPromise:true});
    if(r.type!=='success')throw new Error(JSON.stringify(r));
    return r.result.value;
  }
  const graph=JSON.parse(await readFile(path.join(root,'research/map.json'),'utf8'));
  assert.equal(await evaluate('document.querySelectorAll(".node").length'),graph.nodes.length);
  assert.equal(await evaluate('document.querySelectorAll(".edge").length'),graph.edges.length);
  async function range(value){
    await evaluate(`document.getElementById("range").value=${JSON.stringify(value)};document.getElementById("range").dispatchEvent(new Event("change",{bubbles:true}));`);
  }
  assert.equal(await evaluate('document.getElementById("range").value'),'current');
  assert.equal(await evaluate('document.querySelectorAll(".node:not(.hidden)").length'),graph.nodes.filter(n=>n.view_group==='current').length);
  assert.equal(await evaluate('document.querySelectorAll(".node.result.scoped:not(.hidden)").length'),graph.nodes.filter(n=>n.view_group==='current'&&n.kind==='result'&&n.status==='scoped').length);
  await range('history');
  assert.equal(await evaluate('document.querySelectorAll(".node:not(.hidden)").length'),graph.nodes.filter(n=>n.view_group==='history').length);
  assert.equal(await evaluate('document.getElementById("map").viewBox.baseVal.y'),graph.view_groups.find(g=>g.id==='history').y);
  assert.equal(await evaluate('document.querySelectorAll(".node.result.scoped:not(.hidden)").length'),graph.nodes.filter(n=>n.view_group==='history'&&n.kind==='result'&&n.status==='scoped').length);
  await range('all');
  assert.match(await evaluate('document.getElementById("count").textContent'),new RegExp(`已录入研究结果 ${graph.nodes.filter(n=>n.kind==='result'&&n.status==='scoped').length} 项`));
  await evaluate('document.getElementById("kind").value="result";document.getElementById("kind").dispatchEvent(new Event("change",{bubbles:true}));');
  assert.equal(await evaluate('document.querySelectorAll(".node:not(.hidden)").length'),graph.nodes.filter(n=>n.kind==='result').length);
  const awaiting=graph.nodes.filter(n=>n.kind==='result'&&n.status==='awaiting').length;
  assert.equal(await evaluate('document.querySelectorAll(".node.result.awaiting").length'),awaiting);
  assert.match(await evaluate('document.getElementById("count").textContent'),new RegExp(`待填 ${awaiting} 项`));
  if(awaiting){
    await evaluate('document.querySelector(".node.result.awaiting").dispatchEvent(new MouseEvent("click",{bubbles:true}));');
    assert.equal(await evaluate('document.getElementById("details").hidden'),false);
    assert.match(await evaluate('document.getElementById("detail-body").textContent'),/尚无结果/);
  }
  await evaluate('document.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape"}));document.getElementById("reset").click();');
  assert.equal(await evaluate('document.getElementById("details").hidden'),true);
  await evaluate('document.getElementById("search").value="HB";document.getElementById("search").dispatchEvent(new Event("input",{bubbles:true}));');
  const visible=await evaluate('document.querySelectorAll(".node:not(.hidden)").length');
  assert.ok(visible>0&&visible<graph.nodes.length);
  await evaluate('document.getElementById("reset").click();document.querySelector(".edge:not(.hidden)").dispatchEvent(new MouseEvent("click",{bubbles:true}));');
  assert.match(await evaluate('document.getElementById("detail-body").textContent'),/含义与边界/);
  await evaluate('document.getElementById("reset").click();');
  const before=await evaluate('document.getElementById("map").style.width');
  await evaluate('document.getElementById("plus").click();');
  const after=await evaluate('document.getElementById("map").style.width');
  assert.ok(parseFloat(after)>parseFloat(before));
  await evaluate('document.getElementById("fit").click();');
  assert.deepEqual(errors,[],'Browser console errors');
  await evaluate('document.querySelector(".node[aria-label^=\\"B：语言能力与局部替换\\"]").dispatchEvent(new MouseEvent("click",{bubbles:true}));Array.from(document.querySelectorAll(".neighbor-link")).find(b=>b.textContent.includes("非零接入与能力保持结果")).click();');
  assert.equal(await evaluate('document.getElementById("range").value'),'history');
  assert.match(await evaluate('document.getElementById("detail-body").textContent'),/尚未区分的解释/);
  assert.equal(await evaluate('Array.from(document.querySelectorAll(".node .summary")).every(e=>e.getComputedTextLength()<=222)'),true);
  await evaluate('document.getElementById("reset").click();');
  const screenshot=await send('browsingContext.captureScreenshot',{context,origin:'document',format:{type:'image/png'}});
  await mkdir(path.join(root,'artifacts'),{recursive:true});
  await writeFile(path.join(root,'artifacts/research-map.png'),Buffer.from(screenshot.data,'base64'));
  console.log('Firefox: current/history/all views, cross-view evidence navigation, result filter, pending-result disclosure, search, details, Escape and zoom passed.');
  await send('session.end');
}
try{
  await Promise.race([run(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Browser check exceeded 40s')),40000);})]);
} finally {
  clearTimeout(timer);if(socket)socket.close();
  try{process.kill(-browser.pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')throw e;}
  await new Promise(resolve=>setTimeout(resolve,600));
  try{process.kill(-browser.pid,'SIGKILL');}catch(e){if(e.code!=='ESRCH')throw e;}
  await rm(profile,{recursive:true,force:true});
}
