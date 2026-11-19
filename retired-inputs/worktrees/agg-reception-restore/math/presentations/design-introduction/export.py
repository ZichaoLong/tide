#!/usr/bin/env python3
"""Render all speaker-controlled stops to PDF and a six-frame preview.

Requires playwright-cli, Chrome and Pillow. A loopback HTTP server and an
isolated browser session exist only during export; there are no remote assets.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import json
import subprocess
import tempfile
import threading
import os

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass


def main():
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(BASE)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    session=f'tide-export-{os.getpid()}'
    with tempfile.TemporaryDirectory(prefix='tide-presentation-') as work:
        work=Path(work)
        def cli(*args):
            result=subprocess.run(['playwright-cli',f'-s={session}',*args],cwd=work,capture_output=True,text=True)
            if result.returncode or '### Error' in result.stdout:
                raise RuntimeError(result.stdout+result.stderr)
            return result.stdout
        try:
            cli('open',f'http://127.0.0.1:{server.server_port}/presentations/design-introduction/index.html','--browser=chrome',f'--profile={work / "browser-profile"}')
            script='''async page => {
              const errors=[];
              const frames=await page.evaluate(()=>window.tideDeck.data);
              const targetChapter=frames.find(d=>d.id==='history-first').slide;
              page.on('pageerror',err=>errors.push(err.message));
              await page.keyboard.press('Home');
              await page.keyboard.press('ArrowRight');
              const forward=await page.evaluate(()=>window.tideDeck.index);
              await page.keyboard.press('ArrowLeft');
              const backward=await page.evaluate(()=>window.tideDeck.index);
              await page.keyboard.press('n');
              const notes=await page.locator('#notes-dialog').evaluate(el=>el.open);
              await page.keyboard.press('Escape');
              await page.keyboard.press('g');
              await page.locator('#toc-list button[data-stop-id="history-first"]').click();
              const chapter=await page.evaluate(()=>window.tideDeck.data[window.tideDeck.index].slide);
              await page.keyboard.press('ArrowRight');
              await page.locator('#replay').click();
              const replay=await page.evaluate(()=>window.tideDeck.data[window.tideDeck.index].step);
              await page.keyboard.press('End');
              const last=await page.locator('#next').isDisabled();
              await page.setViewportSize({width:390,height:844});
              const mobileFits=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
              const interactions={forward,backward,notes,chapter,replay,last,mobileFits};
              if(forward!==1||backward!==0||!notes||chapter!==targetChapter||replay!==0||!last||!mobileFits)throw new Error(JSON.stringify(interactions));
              await page.setViewportSize({width:1600,height:900});
              await page.addStyleTag({content:'#viewport{height:900px!important}#stage{transform:none!important}.controls{display:none!important}'});
              await page.evaluate(()=>document.fonts.ready);
              const layout=[];
              for(let i=0;i<frames.length;i++){
                await page.evaluate(n=>window.tideDeck.show(n),i);
                const result=await page.evaluate(()=>{
                  const f=document.querySelector('.frame.active');
                  const footer=f.querySelector('.footer').getBoundingClientRect();
                  const root=f.getBoundingClientRect();
                  const issues=[];
                  for(const el of f.querySelectorAll('h1,.lede,.content *')){
                    if(el.closest('svg'))continue;
                    const rect=el.getBoundingClientRect();
                    if(rect.width===0||rect.height===0)continue;
                    if(rect.left<root.left-1||rect.right>root.right+1||rect.bottom>footer.top-8)issues.push({tag:el.tagName,text:el.textContent.slice(0,60),bottom:rect.bottom,footer:footer.top,right:rect.right});
                  }
                  for(const el of f.querySelectorAll('svg text')){
                    const r=el.getBoundingClientRect(), bounds=el.closest('svg').getBoundingClientRect();
                    if(r.left<bounds.left-1||r.right>bounds.right+1||r.top<bounds.top-1||r.bottom>bounds.bottom+1)issues.push({tag:'svg-text',text:el.textContent});
                  }
                  const example=f.querySelector('g[data-example]');
                  const graph=f.querySelector('g[data-candidates]');
                  return {issues,example:example?{binding:example.dataset.example,expertFull:Number(example.dataset.expertFull),output:example.dataset.output}:null,graph:graph?{candidates:Number(graph.dataset.candidates),full:Number(graph.dataset.full),messages:Number(graph.dataset.messages)}:null};
                });
                layout.push({id:frames[i].id,preview:frames[i].preview,stop:`${frames[i].slide+1}.${frames[i].step+1}`,title:frames[i].title,...result});
                await page.screenshot({path:SCREEN_DIR+'/frame-'+String(i).padStart(2,'0')+'.png',animations:'disabled'});
              }
              await page.pdf({path:PDF_PATH,width:'1600px',height:'900px',preferCSSPageSize:true,printBackground:true,displayHeaderFooter:false});
              const report={frames:frames.length,errors,layout,interactions,browserVersion:page.context().browser().version()};
              await page.evaluate(report=>{document.body.dataset.exportReport=JSON.stringify(report);},report);
              return report;
            }'''.replace('SCREEN_DIR',json.dumps(str(work))).replace('PDF_PATH',json.dumps(str(HERE/'tide-design-introduction.pdf')))
            (work/'render.js').write_text(script)
            cli('run-code',f'--filename={work / "render.js"}')
            raw=cli('--raw','eval','document.body.dataset.exportReport').strip()
            report=json.loads(raw)
            if isinstance(report,str):report=json.loads(report)
            report['html_sha256']=hashlib.sha256((HERE/'index.html').read_bytes()).hexdigest()
            report['pdf_sha256']=hashlib.sha256((HERE/'tide-design-introduction.pdf').read_bytes()).hexdigest()
            report['viewport']=[1600,900]
            report['browser']='Chrome via playwright-cli'
            (HERE/'render-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            selected=[i for i,entry in enumerate(report['layout']) if entry['preview']]
            if len(selected)!=6:raise ValueError('Select exactly six preview stops in content.py')
            canvas=Image.new('RGB',(1648,1508),'#e3e7e2')
            draw=ImageDraw.Draw(canvas)
            font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',28,index=2)
            draw.text((24,20),'Tide · Transformer、MoE 与局部网络',font=font,fill='#193d46')
            for k,n in enumerate(selected):
                frame=Image.open(work/f'frame-{n:02d}.png').convert('RGB').resize((792,446),Image.Resampling.LANCZOS)
                x=24+(k%2)*808;y=80+(k//2)*470
                canvas.paste(frame,(x,y))
            canvas.save(HERE/'preview.png',optimize=True)
            issues=[entry for entry in report['layout'] if entry['issues']]
            print(f'Exported {report["frames"]} PDF pages; six-frame preview; browser errors={len(report["errors"])}, layout issues={len(issues)}')
            for entry in issues:print(json.dumps(entry,ensure_ascii=False))
            if report['errors'] or issues:raise SystemExit(1)
        finally:
            try:cli('close')
            finally:server.shutdown();server.server_close()


if __name__=='__main__':
    main()
