"""Exercise the generated offline page in an isolated headless Firefox."""
import base64
import csv
from decimal import Decimal
from itertools import zip_longest
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import tempfile
import argparse

ROOT = Path(__file__).resolve().parent


def check_capacity_export():
    """Cross-check human units, time bindings and the page's exact counts."""
    data = json.loads((ROOT/'generated'/'data.json').read_text())['capacity']
    seen = set()
    spatial_rows = {}
    reception_rows = {}
    count = 0
    with (ROOT/'generated'/'capacity.csv').open(encoding='utf-8', newline='') as f:
        expanded = ((values, allocation) for values in data['rows']
                    for allocation in ('quad','shannon','pool'))
        for exported, expected_row in zip_longest(csv.DictReader(f), expanded):
            assert exported is not None and expected_row is not None, 'CSV/page allocation expansion differs'
            values, allocation = expected_row
            row = dict(zip(data['fields'], values))
            p = data['profiles'][row['profile']]
            identity = tuple(row[name] for name in ('profile','n','h','d','memory','sharing','rank','agg_profile'))
            full_identity = identity+(allocation,)
            assert full_identity not in seen, full_identity
            seen.add(full_identity)
            assert exported['profile'] == row['profile']
            assert exported['spatial_profile'] == p['spatial_profile']
            assert exported['timing'] == p['timing']
            assert row['profile'] == p['spatial_profile']+'-'+p['timing']
            widths = list(map(int, exported['widths'].split(';')))
            assert widths == p['widths'] and len(widths) == int(exported['depth'])
            assert int(exported['positions']) == sum(widths)
            assert int(exported['max_width']) == max(widths)
            expected_period = 4 if p['timing']=='timed' else len(widths)+(2 if p['family'] in ('fc','lattice') else 1)
            assert int(exported['input_period']) == expected_period
            assert row['nodes_D'] == row['n']*sum(widths)
            assert row['nodes_H'] == row['h']*(sum(widths)-widths[-1])
            for field,column in (('total','total_M'),('base','base_M'),('D','private_D_M'),('H','private_H_M')):
                assert Decimal(exported[column])*1_000_000 == row[field], (identity,column)
            assert exported['within_256M'] == str(row['total'] <= 256_000_000)
            a = data['agg_profiles'][row['agg_profile']]
            assert exported['agg_profile'] == row['agg_profile']
            assert exported['agg_mode'] == row['agg_mode'] == a['mode']
            width = row['d'] if a['attention_width']=='d' else a['attention_width']
            expected_sizes = dict(agg_hidden_width=row['d']//a['hidden_divisor'],
                                  agg_attention_width=width,
                                  agg_attention_heads=width//32,
                                  agg_attention_head_dim=32 if width else 0)
            for field, expected in expected_sizes.items():
                assert int(exported[field]) == row[field] == expected, (identity,field)
            for field, expected in dict(descriptor_width=row['d']//2,
                    region_attention_heads=row['d']//64,region_attention_head_dim=32,
                    region_ffn_width=row['d']).items():
                assert int(exported[field]) == row[field] == expected, (identity,field)
            for field,expected in dict(region_profile='encoded_self_attention',
                    local_score='encoded',region_score='self_attention').items():
                assert exported[field] == row[field] == expected
            assert exported['allocation_profile'] == allocation
            assert exported['selection_mode_D'] == {
                'quad':'role_bounded_quad','shannon':'role_bounded_shannon',
                'pool':'pooled_attention'}[allocation]
            assert exported['selection_mode_H'] == ('role_bounded_quad' if row['h'] else '')
            assert exported['lower_base_D'] == ('received' if allocation!='pool' else '')
            assert exported['total_mode_D'] == ('interval' if allocation!='pool' else '')
            assert exported['lower_base_H'] == ('received' if row['h'] else '')
            assert exported['total_mode_H'] == ('interval' if row['h'] else '')
            reception_rows.setdefault(identity[:-1], set()).add(row['agg_profile'])
            for k in data['budgets']:
                value = exported[f'active_k{k}_M']
                if k > row['n'] or allocation == 'pool':
                    assert value == ''
                else:
                    expected = (row['base']+Decimal(k)*row['D']/row['n']+row['H']
                                if allocation=='quad' else row['total'])
                    assert Decimal(value)*1_000_000 == expected, (identity,k)
            assert exported['active_pool_M'] == (exported['total_M'] if allocation=='pool' else '')
            spatial_identity = (p['spatial_profile'],)+identity[1:]
            numeric = tuple(row[name] for name in ('nodes_D','nodes_H','total','base','D','H'))
            bindings = spatial_rows.setdefault(spatial_identity, {})
            bindings[p['timing']] = numeric
            count += 1
    for identity, bindings in spatial_rows.items():
        if identity[0].startswith('multiscale-'):
            assert set(bindings) == {'timed','settle'}
            assert bindings['timed'] == bindings['settle'], identity
        else:
            assert set(bindings) == {'settle'}
    assert set(data['agg_profiles']) == {'source_residual','incoming_light_64','incoming_light_128','incoming_wide'}
    assert all(profiles == set(data['agg_profiles']) for profiles in reception_rows.values())
    assert count == 676200 and len(data['rows']) == 225400 and len(reception_rows) == 56350
    return dict(rows=count, spatial_configurations=len(spatial_rows),
                bound_profiles=len(data['profiles']), reception_profiles=list(data['agg_profiles']))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
export_checks = check_capacity_export()
OUT = args.output_dir or Path(tempfile.mkdtemp(prefix='backbone-atlas-browser-'))
OUT.mkdir(parents=True, exist_ok=True)
profile = OUT / 'browser-profile'
profile.mkdir(exist_ok=True)
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
(profile/'user.js').write_text(
    'user_pref("marionette.port", %d);\n'
    'user_pref("browser.shell.checkDefaultBrowser", false);\n'
    'user_pref("browser.startup.page", 0);\n' % port)
log=(OUT/'browser.log').open('w')
process=subprocess.Popen(['firefox','--headless','--no-remote','--profile',str(profile),'--marionette','about:blank'],stdout=log,stderr=log)
sock=None
try:
    for attempt in range(100):
        try:
            sock=socket.create_connection(('127.0.0.1',port),timeout=.3)
            break
        except OSError:
            if process.poll() is not None:raise RuntimeError('Firefox exited; see browser.log')
            time.sleep(.15)
    if sock is None:raise RuntimeError('Marionette did not start')
    sock.settimeout(30)
    def receive():
        header=b''
        while not header.endswith(b':'):
            part=sock.recv(1)
            if not part:raise EOFError('Firefox disconnected')
            header+=part
        size=int(header[:-1]);raw=b''
        while len(raw)<size:
            part=sock.recv(size-len(raw))
            if not part:raise EOFError('Firefox disconnected')
            raw+=part
        return json.loads(raw)
    greeting=receive();counter=0
    def command(name,parameters=None):
        global counter
        counter+=1
        raw=json.dumps([0,counter,name,parameters or {}]).encode()
        sock.sendall(str(len(raw)).encode()+b':'+raw)
        response=receive()
        assert response[0]==1 and response[1]==counter,response
        if response[2] is not None:raise RuntimeError(response[2])
        return response[3]
    def script(code):
        code=code.replace('window.atlasDebug','window.wrappedJSObject.atlasDebug')
        code=code.replace('window.atlasFormulaDebug','window.wrappedJSObject.atlasFormulaDebug')
        r=command('WebDriver:ExecuteScript',{'script':code,'args':[], 'newSandbox':False,'sandbox':'default'})
        return r['value'] if isinstance(r,dict) and 'value' in r else r
    command('WebDriver:NewSession',{'capabilities':{'alwaysMatch':{'acceptInsecureCerts':True}}})
    command('WebDriver:SetWindowRect',{'width':1510,'height':1150})
    command('WebDriver:Navigate',{'url':(ROOT/'generated'/'index.html').as_uri()})
    initial=script('return {title:document.title,metrics:document.querySelector("#metrics").textContent,graphs:document.querySelectorAll(".canvas svg").length,debug:typeof window.atlasDebug,sharing:document.querySelector("#cap-sharing").value};')
    assert initial['graphs']==3 and initial['debug']=='object',initial
    assert initial['sharing']=='region_shared',initial
    script('window.atlasErrors=[];window.addEventListener("error",e=>window.atlasErrors.push(e.message));return true;')
    records=[]
    for key in ['5','5-dd4','7','9','13','15','17']:
        code='''
        window.atlasDebug.mountCase(%s);
        const c=window.atlasDebug.case;
        const choose=(s,v)=>{const el=document.querySelector(s);el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}));};
        choose('#scope','all');document.querySelector('#preset-detail').click();
        const visible=[...document.querySelectorAll('#region-canvas .bundle')].filter(e=>e.style.display!=='none');
        const visibleKinds=[...new Set(visible.map(e=>e.dataset.kind))].sort();
        document.querySelector('#example-h').click();
        const hSummary=document.querySelector('#node-summary').textContent;
        document.querySelector('#example-d').click();
        const dSummary=document.querySelector('#node-summary').textContent;
        const point=document.querySelector('#matrix-canvas [data-jump]');point.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        return {key:c.key,nodes:c.nodes.length,edges:c.edges.length,visibleKinds,hSummary,dSummary,
                dd4Visible:getComputedStyle(document.querySelector('#dd4-label')).display!=='none',
                mathCount:document.querySelectorAll('.formula-side math').length,
                nodeFormula:document.querySelector('#node-formulas .formula-state').textContent,
                matrices:document.querySelectorAll('#matrix-canvas .matrix-hit').length,
                matrixCircles:document.querySelectorAll('#matrix-canvas g circle').length,
                phaseRows:document.querySelectorAll('#depth-table tbody tr').length,
                errors:window.atlasErrors.slice()};
        ''' % json.dumps(key)
        result=script(code)
        assert result['matrices']==400 and result['phaseRows']==12 and not result['errors'],result
        assert result['visibleKinds']==(['DD1','DD2','DD4'] if key=='5-dd4' else ['DD1','DD2']),result
        assert result['dd4Visible']==(key=='5-dd4'),result
        assert result['mathCount']>=20 and result['nodeFormula'].startswith('D('),result
        records.append(result)
    # Check actual region-click and edge-click navigation, then restore the main view.
    interactions=script('''
      window.atlasDebug.mountCase('9');document.querySelector('#preset-all').click();
      document.querySelector('#scope').value='near';document.querySelector('#scope').dispatchEvent(new Event('change'));
      document.querySelector('#region-canvas [data-pos="3,2"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
      const selected=window.atlasDebug.state.focus.join(',');
      const edge=[...document.querySelectorAll('#region-canvas .bundle')].find(e=>e.dataset.src==='3,2'&&e.dataset.dst==='4,2');
      edge.dispatchEvent(new MouseEvent('click',{bubbles:true}));
      const pair=document.querySelector('#matrix-summary').textContent;
      document.querySelector('#pair-source').value='4,3';document.querySelector('#pair-source').dispatchEvent(new Event('change'));
      document.querySelector('#pair-target').value='8,0';document.querySelector('#pair-target').dispatchEvent(new Event('change'));
      const skipPair=document.querySelector('#matrix-summary').textContent;
      window.atlasDebug.mountCase('9');window.scrollTo({top:0,behavior:'instant'});
      return {selected,pair,skipPair,errors:window.atlasErrors.slice(),horizontalOverflow:document.documentElement.scrollWidth>window.innerWidth};
    ''')
    assert interactions['selected']=='3,2' and interactions['pair'].startswith('56 条') and not interactions['errors'],interactions
    assert not interactions['horizontalOverflow'],interactions
    formula_checks=script('''
      let checked=0;const failures=[];
      for(const key of ['5','5-dd4','7','9','13','15','17']){
        window.atlasDebug.mountCase(key);const c=window.atlasDebug.case;
        for(const b of c.bundles){
          const [l,p,r,m,q,t,k,count]=b,s=m-l,f=window.atlasFormulaDebug.targetFormula;
          let modes;
          if(k==='DD1'||k==='DH1')modes=['base','near'];
          else if(k.startsWith('DD'))modes=['cross'];
          else if(k.startsWith('HH'))modes=['base','cross'];
          else modes=s===1?['base']:['cross'];
          const valid=modes.some(mode=>Array.from(f(l,s,p,mode)).includes(q));
          if(!valid)failures.push([key,Array.from(b)]);checked++;
        }
      }
      window.atlasDebug.mountCase('9');
      const toggle=document.querySelector('#formulas-toggle');toggle.click();
      const hidden=getComputedStyle(document.querySelector('#region-formulas')).display==='none';toggle.click();
      const visible=getComputedStyle(document.querySelector('#region-formulas')).display!=='none';
      const math=document.querySelector('#region-formulas math').getBoundingClientRect();
      return {checked,failures,hidden,visible,mathWidth:math.width,mathHeight:math.height,errors:window.atlasErrors.slice()};
    ''')
    assert formula_checks['checked']>1000 and not formula_checks['failures'] and not formula_checks['errors'],formula_checks
    assert formula_checks['hidden'] and formula_checks['visible'] and formula_checks['mathWidth']>0 and formula_checks['mathHeight']>0,formula_checks
    timing_checks=script("""
      window.atlasDebug.mountCase('9');
      const el=document.querySelector('#timing-mode'),result=[];
      const options=[...el.options].map(o=>o.value);
      for(const mode of options){
        el.value=mode;el.dispatchEvent(new Event('change',{bubbles:true}));
        result.push({mode,rows:document.querySelectorAll('#timing-table tbody tr').length,
          summary:document.querySelector('#timing-summary').textContent});
      }
      el.value='timed';el.dispatchEvent(new Event('change',{bubbles:true}));
      let rejected=false;
      try{window.atlasDebug.setTiming('unsupported');}catch(e){rejected=e.name==='RangeError';}
      return {options,cases:result,rejected,mode:window.atlasDebug.state.timing};
    """)
    assert timing_checks['options']==['timed','timed-common','settle'],timing_checks
    timing_cases={item['mode']:item for item in timing_checks['cases']}
    assert timing_cases['timed']['rows']>3 and timing_cases['timed-common']['rows']>3,timing_checks
    assert timing_cases['settle']['rows']==3,timing_checks
    assert timing_checks['rejected'] and timing_checks['mode']=='timed',timing_checks
    mixing_check=script("""
      const c=window.atlasDebug.case;
      const u=c.nodes.findIndex((n,i)=>n[0]===8&&n[2]==='d'&&c.delays[i].includes(2)&&c.delays[i].includes(6));
      window.atlasDebug.chooseNode(u);
      const row=[...document.querySelectorAll('#timing-table tbody tr')].find(r=>r.children[0].textContent==='7');
      const contents=row.children[1].textContent;
      window.atlasDebug.mountCase('9');return contents;
    """)
    assert '输入 0，路径时延 6' in mixing_check and '输入 1，路径时延 2' in mixing_check, mixing_check
    # Actual Settle spatial candidates, including the g=1 boundary and no-hub UI.
    settle_checks=script("""
      const results=[],failures=[];
      const choose=(s,v)=>{const el=document.querySelector(s);el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}));};
      for(const key of ['lattice-5','lattice-7','lattice-9','exchange-7','exchange-7-h','exchange-10','exchange-10-h','exchange-13','exchange-13-h']){
        window.atlasDebug.mountCase(key);const c=window.atlasDebug.case;
        const size=c.ordinary+c.hubs;
        const cells=document.querySelectorAll('#matrix-canvas .matrix-hit').length;
        const circles=document.querySelectorAll('#matrix-canvas g circle').length;
        const pair=window.atlasDebug.state.source+'>'+window.atlasDebug.state.target;
        const expected=c.edges.filter(([u,v])=>c.nodes[u].slice(0,2).join(',')+'>'+c.nodes[v].slice(0,2).join(',')===pair).length;
        if(cells!==size*size||circles!==expected)failures.push(['matrix',key,cells,circles,expected]);
        if(document.querySelector('#node-role').options.length!==(c.hubs?2:1))failures.push(['roles',key]);
        if(document.querySelector('#timing-mode').value!=='settle'||!document.querySelector('#timing-mode').disabled)failures.push(['timing',key]);
        let checked=0;
        for(const b of c.bundles){
          const [l,p,r,m,q,t,k,count]=b;
          const exchange=c.family==='exchange'&&k!=='HD1';
          if(!Array.from(window.atlasFormulaDebug.settleTargets(l,p,exchange)).includes(q))failures.push(['region',key,Array.from(b)]);
          checked++;
        }
        if(c.family==='exchange')for(const [u,v,k] of c.edges){
          const a=c.nodes[u],b=c.nodes[v];
          if(a[2]==='d'&&b[2]==='d'&&b[3]!==a[3]&&b[3]!==(a[3]^(1<<(a[0]%4))))failures.push(['node-bit',key,u,v]);
        }
        choose('#node-layer','0');choose('#node-position','0');
        const atBoundary=document.querySelector('#region-formulas').textContent;
        if(/NaN|Infinity|undefined/.test(atBoundary))failures.push(['boundary',key]);
        const edge=document.querySelector('#region-canvas .bundle');
        edge.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        const matrixSummary=document.querySelector('#matrix-summary').textContent;
        document.querySelector('#preset-detail').click();
        if(window.atlasDebug.case.nodes.length!==c.nodes.length)failures.push(['filter-mutated-graph',key]);
        results.push({key,cells,circles,checked,period:c.settle_period,nodes:c.nodes.length,matrixSummary});
      }
      // Family select changes the actual model, not only the displayed edge families.
      choose('#family-select','lattice-settle');const latticeKey=window.atlasDebug.case.key;
      choose('#family-select','exchange-settle');const exchangeKey=window.atlasDebug.case.key;
      choose('#family-select','multiscale-timed');const multiscaleKey=window.atlasDebug.case.key;
      return {results,failures,latticeKey,exchangeKey,multiscaleKey,errors:window.atlasErrors.slice()};
    """)
    assert not settle_checks['failures'] and not settle_checks['errors'],settle_checks
    assert (settle_checks['latticeKey'],settle_checks['exchangeKey'],settle_checks['multiscaleKey']) == ('lattice-7','exchange-10','9'),settle_checks
    terminal_checks=script("""
      const results=[],failures=[];
      const choose=(s,v)=>{const el=document.querySelector(s);el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}));};
      const keys=['5','5-dd4','7','9','13','15','17','lattice-5','lattice-7','lattice-9',
                  'exchange-7','exchange-7-h','exchange-10','exchange-10-h','exchange-13','exchange-13-h'];
      for(const key of keys){
        window.atlasDebug.mountCase(key);const c=window.atlasDebug.case,last=c.depth-1;
        if(c.hubs)document.querySelector('#example-h').click();
        choose('#node-layer',String(last));choose('#node-position','0');
        if(document.querySelector('#node-role').options.length!==1||window.atlasDebug.state.role!=='d')failures.push(['terminal-role',key]);
        if(c.nodes.some(n=>n[0]===last&&n[2]==='h'))failures.push(['terminal-hub',key]);
        const box=document.querySelector(`#region-canvas [data-pos="${last},0"]`);
        if(box.textContent.includes('H4'))failures.push(['drawn-hub',key]);
        choose('#pair-source',`${last-1},0`);choose('#pair-target',`${last},0`);
        const cells=document.querySelectorAll('#matrix-canvas .matrix-hit').length;
        const circles=document.querySelectorAll('#matrix-canvas g circle').length;
        const rows=c.nodes.filter(n=>n[0]===last-1&&n[1]===0).length;
        const columns=c.nodes.filter(n=>n[0]===last&&n[1]===0).length;
        const expected=c.edges.filter(([u,v])=>c.nodes[u][0]===last-1&&c.nodes[u][1]===0&&c.nodes[v][0]===last&&c.nodes[v][1]===0).length;
        if(cells!==rows*columns||circles!==expected||columns!==16)failures.push(['terminal-matrix',key,cells,rows,columns,circles,expected]);
        if(/NaN|Infinity|undefined/.test(document.querySelector('#matrix-canvas').textContent+document.querySelector('#region-formulas').textContent))failures.push(['terminal-text',key]);
        const point=document.querySelector('#matrix-canvas g[data-jump]');
        point.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        if(window.atlasDebug.state.focus[0]!==last)failures.push(['terminal-navigation',key]);
        results.push({key,rows,columns,edges:circles});
      }
      return {results,failures,errors:window.atlasErrors.slice()};
    """)
    assert not terminal_checks['failures'] and not terminal_checks['errors'],terminal_checks
    # Exercise every newly registered width / node-count / optional-hub case.
    compact_checks=[]
    all_keys=list(json.loads((ROOT/'generated'/'checks.json').read_text()))
    assert len(all_keys)==58,all_keys
    for key in [k for k in all_keys if k.startswith('compact-') or k.startswith('exchange-m')]:
        result=script('''
          window.atlasDebug.mountCase(%s);const c=window.atlasDebug.case,failures=[];
          const choose=(s,v)=>{const el=document.querySelector(s);el.value=String(v);el.dispatchEvent(new Event('change',{bubbles:true}));};
          const pair=window.atlasDebug.state.source+'>'+window.atlasDebug.state.target;
          const count=c.edges.filter(([u,v])=>c.nodes[u].slice(0,2).join(',')+'>'+c.nodes[v].slice(0,2).join(',')===pair).length;
          const cells=document.querySelectorAll('#matrix-canvas .matrix-hit').length;
          if(cells!==(c.ordinary+c.hubs)**2||document.querySelectorAll('#matrix-canvas g circle').length!==count)failures.push('matrix');
          for(const b of c.bundles){
            const [l,p,r,m,q,t,k]=b,s=m-l;
            if(c.family==='multiscale'){
              const modes=(k==='DD1'||k==='DH1')?['base','near']:k.startsWith('DD')?['cross']:k.startsWith('HH')?['base','cross']:s===1?['base']:['cross'];
              if(!modes.some(mode=>Array.from(window.atlasFormulaDebug.targetFormula(l,s,p,mode)).includes(q)))failures.push(['mapping',Array.from(b)]);
            }else if(!Array.from(window.atlasFormulaDebug.settleTargets(l,p,k!=='HD1')).includes(q))failures.push(['mapping',Array.from(b)]);
          }
          choose('#node-layer',0);choose('#node-position',0);
          if(/NaN|Infinity|undefined/.test(document.querySelector('#region-formulas').textContent))failures.push('boundary');
          if(c.family==='multiscale'&&Array.from(window.atlasFormulaDebug.targetFormula(0,1,0,'cross')).some(p=>p>=c.widths[1]))failures.push('cross-range');
          const last=c.depth-1;choose('#node-layer',last);choose('#node-position',0);
          if(document.querySelector('#node-role').options.length!==1)failures.push('terminal-role');
          choose('#pair-source',`${last-1},0`);choose('#pair-target',`${last},0`);
          if(document.querySelectorAll('#matrix-canvas .matrix-hit').length!==(c.ordinary+c.hubs)*c.ordinary)failures.push('terminal-matrix');
          return {key:c.key,n:c.ordinary,h:c.hubs,cells,failures,errors:window.atlasErrors.slice()};
        ''' % json.dumps(key))
        assert not result['failures'] and not result['errors'],result
        compact_checks.append(result)
    capacity_checks=script('''
      const choose=(s,v)=>{const el=document.querySelector(s);el.value=String(v);el.dispatchEvent(new Event('change',{bubbles:true}));};
      const original=window.atlasDebug.case.key;
      choose('#cap-family','multiscale-timed');choose('#cap-profile','multiscale-compact-9-timed');
      choose('#cap-n',8);choose('#cap-h',0);choose('#cap-d',64);choose('#cap-memory','K');choose('#cap-sharing','independent');
      const smallBudgets=[...document.querySelector('#cap-k').options].map(o=>o.value);
      const row=document.querySelector('#capacity-table tbody tr'),total=+row.dataset.total;
      choose('#cap-k',8);const dense=+document.querySelector('#capacity-table tbody tr').dataset.active;
      choose('#cap-k',1);const sparse=+document.querySelector('#capacity-table tbody tr').dataset.active;
      choose('#cap-allocation','shannon');const shannon=+document.querySelector('#capacity-table tbody tr').dataset.active;
      const shannonBudget=!document.querySelector('#cap-k').disabled;
      const regionMath=document.querySelector('#capacity-region-formula').textContent;
      choose('#cap-allocation','pool');const pool=+document.querySelector('#capacity-table tbody tr').dataset.active;
      const poolBudget=document.querySelector('#cap-k').disabled;
      const unchanged=window.atlasDebug.case.key===original;
      document.querySelector('#capacity-table [data-cap-case]').click();const linked=window.atlasDebug.case.key;
      document.querySelector('#cap-use-graph').click();
      const allocationPreserved=document.querySelector('#cap-allocation').value;
      choose('#cap-allocation','quad');
      choose('#cap-n',128);const largeBudgets=[...document.querySelector('#cap-k').options].map(o=>o.value);
      choose('#cap-n',32);choose('#cap-h',4);choose('#cap-memory','X');choose('#cap-profile','multiscale-17-timed');choose('#cap-d',128);
      choose('#cap-limit','all');const large=+document.querySelector('#capacity-table tbody tr').dataset.total;
      choose('#cap-limit','256');const hiddenLarge=document.querySelectorAll('#capacity-table tbody tr').length===0;
      choose('#cap-family','fc-settle');const noHub=document.querySelector('#cap-h').disabled&&document.querySelector('#cap-h').value==='0';
      choose('#cap-profile','fc-4-settle');choose('#cap-sharing','region_shared');choose('#cap-rank',4);
      const p4=+document.querySelector('#capacity-table tbody tr').dataset.total;
      const sharingOptions=[...document.querySelector('#cap-sharing').options].map(o=>o.value);
      const fcEquivalence=document.querySelector('#capacity-table').textContent.includes('两共享档等价');
      choose('#cap-sharing','layer_shared');
      const layerP4=+document.querySelector('#capacity-table tbody tr').dataset.total;
      choose('#cap-rank',16);const p16=+document.querySelector('#capacity-table tbody tr').dataset.total;
      choose('#cap-sharing','region_shared');
      const regionP16=+document.querySelector('#capacity-table tbody tr').dataset.total;
      choose('#cap-sharing','independent');const rankDisabled=document.querySelector('#cap-rank').disabled;
      choose('#cap-family','multiscale-timed');choose('#cap-profile','multiscale-compact-9-timed');choose('#cap-h',4);choose('#cap-memory','X');choose('#cap-rank',8);choose('#cap-k',1);
      const hbIndependent=+document.querySelector('#capacity-table tbody tr').dataset.total;
      choose('#cap-sharing','layer_shared');const hbLayer=+document.querySelector('#capacity-table tbody tr').dataset.total;
      const layerScope=document.querySelector('#capacity-sharing-summary').textContent;
      choose('#cap-sharing','region_shared');const hbRegion=+document.querySelector('#capacity-table tbody tr').dataset.total;
      const regionScope=document.querySelector('#capacity-sharing-summary').textContent;
      const regionFormula=document.querySelector('#capacity-formula').textContent;
      const regionRankEnabled=!document.querySelector('#cap-rank').disabled;
      return {smallBudgets,largeBudgets,total,dense,pool,sparse,shannon,shannonBudget,poolBudget,regionMath,allocationPreserved,unchanged,linked,large,hiddenLarge,noHub,p4,p16,layerP4,regionP16,sharingOptions,fcEquivalence,hbIndependent,hbLayer,hbRegion,layerScope,regionScope,regionFormula,regionRankEnabled,rankDisabled,errors:window.atlasErrors.slice()};
    ''')
    assert capacity_checks['smallBudgets']==['1','2','4','8'],capacity_checks
    assert capacity_checks['largeBudgets']==['1','2','4','8','16','32','64'],capacity_checks
    assert round(capacity_checks['total']/1e6,3)==9.331,capacity_checks
    assert capacity_checks['dense']==capacity_checks['pool']==capacity_checks['shannon']==capacity_checks['total'],capacity_checks
    assert capacity_checks['sparse']<capacity_checks['total'] and capacity_checks['shannonBudget'] and capacity_checks['poolBudget'],capacity_checks
    assert capacity_checks['allocationPreserved']=='pool' and '32' in capacity_checks['regionMath'],capacity_checks
    assert capacity_checks['unchanged'] and capacity_checks['linked']=='compact-9-n8-h0',capacity_checks
    assert capacity_checks['large']>256e6 and capacity_checks['hiddenLarge'] and capacity_checks['noHub'],capacity_checks
    assert capacity_checks['p4']<capacity_checks['p16'] and capacity_checks['rankDisabled'] and not capacity_checks['errors'],capacity_checks
    assert capacity_checks['sharingOptions']==['independent','region_shared','layer_shared'],capacity_checks
    assert capacity_checks['p4']==capacity_checks['layerP4'] and capacity_checks['p16']==capacity_checks['regionP16'] and capacity_checks['fcEquivalence'],capacity_checks
    assert round(capacity_checks['hbIndependent']/1e6,3)==165.084,capacity_checks
    assert capacity_checks['hbRegion']>capacity_checks['hbLayer'] and capacity_checks['regionRankEnabled'],capacity_checks
    assert '实际区域' in capacity_checks['regionScope'] and '跨区域共享' in capacity_checks['layerScope'],capacity_checks
    assert 'g(v)' in capacity_checks['regionFormula'] and not any(x in capacity_checks['regionFormula'] for x in ('NaN','undefined')),capacity_checks
    capacity_reception=script('''
      const choose=(s,v)=>{const el=document.querySelector(s);el.value=String(v);el.dispatchEvent(new Event('change',{bubbles:true}));};
      const rows=()=>[...document.querySelectorAll('#capacity-table tbody tr')].map(r=>({
        profile:r.dataset.aggProfile,total:+r.dataset.total,width:+r.dataset.aggWidth,hidden:+r.dataset.aggHidden}));
      choose('#cap-family','fc-settle');choose('#cap-profile','fc-4-settle');
      choose('#cap-n',8);choose('#cap-h',0);choose('#cap-d',128);choose('#cap-memory','K');
      choose('#cap-sharing','independent');choose('#cap-limit','all');choose('#cap-agg','all');
      const options=[...document.querySelector('#cap-agg').options].map(o=>o.value),comparison=rows();
      const formulas=[];
      for(const agg of options.filter(v=>v!=='all')){
        choose('#cap-agg',agg);
        formulas.push({profile:agg,rows:rows(),math:document.querySelectorAll('#capacity-agg-formula math').length,
          text:document.querySelector('#capacity-agg-formula').textContent});
      }
      choose('#cap-d','all');
      const symbolic=[];
      for(const agg of options.filter(v=>v!=='all')){
        choose('#cap-agg',agg);symbolic.push({profile:agg,text:document.querySelector('#capacity-agg-formula').textContent});
      }
      choose('#cap-d',64);choose('#cap-agg','incoming_wide');
      const shared=[];
      for(const sharing of ['region_shared','layer_shared']){
        choose('#cap-sharing',sharing);
        for(const rank of [4,16]){choose('#cap-rank',rank);shared.push({sharing,rank,total:rows()[0].total});}
      }
      choose('#cap-family','multiscale-timed');choose('#cap-profile','multiscale-compact-9-two-inputs-timed');
      choose('#cap-n',32);choose('#cap-h',4);choose('#cap-d',128);choose('#cap-memory','X');choose('#cap-sharing','independent');
      choose('#cap-agg','all');const unfiltered=rows();choose('#cap-limit','256');const filtered=rows();
      choose('#cap-agg','incoming_wide');const hiddenWide=rows().length===0;
      choose('#cap-limit','all');document.querySelector('#capacity-table [data-cap-case]').click();
      const linked=window.atlasDebug.case.key;document.querySelector('#cap-use-graph').click();
      const preserved=document.querySelector('#cap-agg').value;
      choose('#cap-agg','all');document.querySelector('#capacity').scrollIntoView({behavior:'instant'});
      return {options,comparison,formulas,symbolic,shared,unfiltered,filtered,hiddenWide,linked,preserved,errors:window.atlasErrors.slice()};
    ''')
    assert capacity_reception['options']==['all','source_residual','incoming_light_64','incoming_light_128','incoming_wide'],capacity_reception
    # Independent FC-4, n=8, d=128: closed-form K nodes, one source class per
    # node and one regional head per layer, independent of the CSV generator.
    d,n,depth=128,8,4
    baseline=n*depth*(17.5*d*d+2.5*d+1+d//8)+depth*(2*d*d+2*d)
    expected={
        'source_residual':(baseline,0,16),
        'incoming_light_64':(baseline+n*depth*(4*d*64+d),64,16),
        'incoming_light_128':(baseline+n*depth*(4*d*128+d),128,16),
        'incoming_wide':(baseline+n*depth*(23*d*d/4+15*d/8),128,128),
    }
    assert len(capacity_reception['comparison'])==4,capacity_reception
    for row in capacity_reception['comparison']:
        assert (row['total'],row['width'],row['hidden'])==expected[row['profile']],row
    for formula in capacity_reception['formulas']:
        assert len(formula['rows'])==1 and formula['rows'][0]['profile']==formula['profile'],formula
        assert formula['math']==(1 if formula['profile']=='source_residual' else 2),formula
        assert not any(x in formula['text'] for x in ('NaN','undefined','Infinity')),formula
    for formula in capacity_reception['symbolic']:
        assert 'd' in formula['text'] and not any(x in formula['text'] for x in ('NaN','undefined','Infinity')),formula
    shared=capacity_reception['shared']
    assert shared[0]['total']==shared[2]['total']<shared[1]['total']==shared[3]['total'],shared
    assert len(capacity_reception['unfiltered'])==4,capacity_reception
    assert {r['profile'] for r in capacity_reception['filtered']}=={'source_residual','incoming_light_64','incoming_light_128'},capacity_reception
    assert all(r['total']<=256e6 for r in capacity_reception['filtered']),capacity_reception
    assert capacity_reception['hiddenWide'] and capacity_reception['linked']=='compact-9-two-inputs-n32-h4',capacity_reception
    assert capacity_reception['preserved']=='incoming_wide' and not capacity_reception['errors'],capacity_reception
    shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
    (OUT/'reception-comparison-preview.png').write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    script("const el=document.querySelector('#cap-agg');el.value='source_residual';el.dispatchEvent(new Event('change'));return true;")
    capacity_timing=script('''
      const choose=(s,v)=>{const el=document.querySelector(s);el.value=String(v);el.dispatchEvent(new Event('change',{bubbles:true}));};
      choose('#cap-family','multiscale-settle');choose('#cap-profile','multiscale-compact-9-settle');
      choose('#cap-n',8);choose('#cap-h',0);choose('#cap-d',64);choose('#cap-memory','K');choose('#cap-sharing','independent');
      const settleRow=document.querySelector('#capacity-table tbody tr');
      const settleTotal=+settleRow.dataset.total,settlePeriod=+settleRow.dataset.period;
      document.querySelector('#capacity-table [data-cap-case]').click();
      const settle={key:window.atlasDebug.case.key,mode:window.atlasDebug.state.timing,variant:document.querySelector('#family-select').value};
      document.querySelector('#cap-use-graph').click();
      const settleProfile=document.querySelector('#cap-profile').value;
      choose('#family-select','multiscale-timed');
      const sameGraph=window.atlasDebug.case.key===settle.key;
      choose('#timing-mode','timed-common');document.querySelector('#cap-use-graph').click();
      const common={profile:document.querySelector('#cap-profile').value,schedule:document.querySelector('#cap-schedule').value,
        period:+document.querySelector('#capacity-table tbody tr').dataset.period};
      document.querySelector('#capacity-table [data-cap-case]').click();common.mode=window.atlasDebug.state.timing;
      choose('#case-select','compact-7-n8-h0');document.querySelector('#cap-use-graph').click();
      const changed={profile:document.querySelector('#cap-profile').value,period:+document.querySelector('#capacity-table tbody tr').dataset.period};
      choose('#cap-schedule','native');document.querySelector('#capacity-table [data-cap-case]').click();
      const native={mode:window.atlasDebug.state.timing,period:+document.querySelector('#capacity-table tbody tr').dataset.period};
      choose('#cap-family','all');choose('#cap-limit','all');
      const pair=[...document.querySelectorAll('#capacity-table tbody tr')]
        .filter(r=>r.dataset.profile.startsWith('multiscale-compact-9-')&&!r.dataset.profile.includes('two-inputs'))
        .map(r=>({profile:r.dataset.profile,total:+r.dataset.total,period:+r.dataset.period}));
      const variants=[...document.querySelector('#cap-family').options].map(o=>o.value);
      return {settle,settleProfile,settleTotal,settlePeriod,sameGraph,common,changed,native,pair,variants,errors:window.atlasErrors.slice()};
    ''')
    assert capacity_timing['settle']==dict(key='compact-9-n8-h0',mode='settle',variant='multiscale-settle'),capacity_timing
    assert capacity_timing['settleProfile']=='multiscale-compact-9-settle' and capacity_timing['settlePeriod']==10,capacity_timing
    assert capacity_timing['sameGraph'] and capacity_timing['common']==dict(profile='multiscale-compact-9-timed',schedule='common',period=10,mode='timed-common'),capacity_timing
    assert capacity_timing['changed']==dict(profile='multiscale-compact-7-timed',period=8),capacity_timing
    assert capacity_timing['native']==dict(mode='timed',period=4),capacity_timing
    assert len(capacity_timing['pair'])==2 and {r['period'] for r in capacity_timing['pair']}=={4,10},capacity_timing
    assert all(r['total']==capacity_timing['settleTotal'] for r in capacity_timing['pair']),capacity_timing
    assert capacity_timing['variants']==['all','fc-settle','lattice-settle','exchange-settle','multiscale-timed','multiscale-settle'],capacity_timing
    assert not capacity_timing['errors'],capacity_timing
    script("document.querySelector('#matrix').scrollIntoView({behavior:'instant'});return true;")
    shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
    (OUT/'terminal-matrix-preview.png').write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    timing_sync=script("""
      window.atlasDebug.mountCase('9');
      const select=document.querySelector('#pair-source');select.value='4,3';select.dispatchEvent(new Event('change'));
      const target=document.querySelector('#pair-target');target.value='8,0';target.dispatchEvent(new Event('change'));
      const rows=[];
      for(const mode of ['timed','timed-common','settle']){
        window.atlasDebug.setTiming(mode);
        const edge=[...document.querySelectorAll('#region-canvas .bundle')].find(e=>e.dataset.src==='4,3'&&e.dataset.dst==='8,0');
        rows.push({mode,title:edge.querySelector('title').textContent,
          matrix:document.querySelector('#matrix-summary').textContent,
          formula:document.querySelector('#pair-formulas math').textContent,
          timing:document.querySelector('#timing-summary').textContent});
      }
      return rows;
    """)
    for item in timing_sync:
        delta=1 if item['mode'] in ('timed','timed-common') else 4
        assert f'δ={delta}' in item['title'] and f'时延均为 {delta}' in item['matrix'],item
        assert f'δ={delta}' in item['formula'],item
    assert '输入间隔 4' in timing_sync[0]['timing'],timing_sync
    assert '输入间隔 10' in timing_sync[1]['timing'] and '输入间隔 10' in timing_sync[2]['timing'],timing_sync
    preserved_timing=script("""
      const el=document.querySelector('#case-select');el.value='17';el.dispatchEvent(new Event('change'));
      return {key:window.atlasDebug.case.key,mode:window.atlasDebug.state.timing,summary:document.querySelector('#timing-summary').textContent};
    """)
    assert preserved_timing['key']=='17' and preserved_timing['mode']=='settle' and 'D=18>M=17' in preserved_timing['summary'],preserved_timing
    # Inspect each new family visually; external temporary evidence, not tracked outputs.
    for key in ('lattice-7','exchange-10','exchange-10-h','compact-9-n8-h0','compact-9-two-inputs-n32-h4'):
        script("window.atlasDebug.mountCase(%s);document.querySelector('#regions').scrollIntoView({behavior:'instant'});return true;" % json.dumps(key))
        shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
        (OUT/(key+'-preview.png')).write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    command('WebDriver:SetWindowRect',{'width':850,'height':1100})
    small_layout=script("window.atlasDebug.mountCase('exchange-10');return {overflow:document.documentElement.scrollWidth>window.innerWidth,math:document.querySelector('#region-formulas math').getBoundingClientRect().width,errors:window.atlasErrors.slice()};")
    assert not small_layout['overflow'] and small_layout['math']>0 and not small_layout['errors'],small_layout
    capacity_layout=script("document.querySelector('#capacity').scrollIntoView({behavior:'instant'});return {overflow:document.documentElement.scrollWidth>window.innerWidth};")
    assert not capacity_layout['overflow'],capacity_layout
    shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
    (OUT/'capacity-preview.png').write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    command('WebDriver:SetWindowRect',{'width':1510,'height':1150})
    script("window.atlasDebug.mountCase('9');window.scrollTo({top:0,behavior:'instant'});return true;")
    # Check all local document/download targets after formula rendering.
    from urllib.parse import urlparse, unquote
    hrefs=script('return [...document.querySelectorAll("a[href]")].map(a=>a.href);')
    for href in hrefs:
        url=urlparse(href)
        if url.scheme == 'file':
            assert Path(unquote(url.path)).is_file(), href
    # Ensure the embedded ladder image decoded successfully.
    image_status=script('return {complete:document.querySelector("img.ladder").complete,width:document.querySelector("img.ladder").naturalWidth};')
    assert image_status['complete'] and image_status['width']==1640,image_status
    shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
    (OUT/'browser-preview.png').write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    script('document.querySelector("#regions").scrollIntoView({behavior:"instant"});return true;')
    shot=command('WebDriver:TakeScreenshot',{'id':None,'full':False,'hash':False,'scroll':False})
    (OUT/'formula-preview.png').write_bytes(base64.b64decode(shot['value'] if isinstance(shot,dict) else shot))
    result={'initial':initial,'cases':records,'interactions':interactions,'formulas':formula_checks,'timing':timing_checks,'mixing':mixing_check,'settle':settle_checks,'terminal':terminal_checks,'compact':compact_checks,'capacity':capacity_checks,'capacity_reception':capacity_reception,'capacity_export':export_checks,'capacity_timing':capacity_timing,'timing_sync':timing_sync,'preserved_timing':preserved_timing,'small_layout':small_layout,'capacity_layout':capacity_layout,'embedded_image':image_status}
    (OUT/'browser-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    command('Marionette:Quit',{'flags':['eForceQuit']})
finally:
    print('Browser evidence:', OUT)
    if sock:sock.close()
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    log.close()
