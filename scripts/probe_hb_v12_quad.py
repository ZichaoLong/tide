#!/usr/bin/env python3
"""Bounded allocator stress or disposable replay of an original failed trajectory."""
import argparse,json,time,sys
from pathlib import Path

def main(a):
    # Replay imports the exact original model/optimizer source; this new observer
    # is outside that frozen directory and never saves a scientific child.
    source=a.parent_source if a.action=='replay' else Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source))
    import torch
    from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
    from tide_lab.train import runtime,cpu_tree,source_identity,installed_identity
    from tide_lab.direct_semantic_ops import allocation
    a.output.mkdir(parents=True,exist_ok=False);run='tl-v12-quad-'+a.action
    report=dict(schema='tide-hb-v12-quad-diagnosis-v1',state='running',started_at=now(),action=a.action,
        original_threshold=2e-5,source=source_identity(),installation=installed_identity(),learning='not_a_training_trajectory',rows=[])
    atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=run,name=run,group='TL-HB-NUMERICAL-v12',study_id='TL-HB-NUMERICAL-v12'))
    metrics=Metrics(a.output,run)
    try:
        if a.action=='stress':
            runtime('cpu',4,11);g=torch.Generator().manual_seed(20261115)
            for offset in (0.,100.,1e4,1e6):
                for scale in (.1,1.,10.,100.):
                    t=torch.randn(128,64,generator=g)*scale+offset;mu=torch.rand(128,64,generator=g)
                    b=mu.sum(-1,keepdim=True).clamp_max(2)*.75
                    original,er=allocation(t,mu,b,b,1.)
                    centered,ec=allocation(t-t.amax(-1,keepdim=True),mu,b,b,1.)
                    reference,_=allocation(t.double(),mu.double(),b.double(),b.double(),1.)
                    item=dict(offset=offset,scale=scale,original_max_residual=float(er.max()),centered_max_residual=float(ec.max()),
                        original_fail_rows=int((er>2e-5).sum()),centered_fail_rows=int((ec>2e-5).sum()),
                        original_max_error=float((original-reference).abs().max()),centered_max_error=float((centered-reference).abs().max()))
                    report['rows'].append(item)
            report['centered_solver_check']='passed' if all(r['centered_fail_rows']==0 for r in report['rows']) else 'failed'
            report['scope']='16 named score offset/scale conditions,128x64 rows each; same represented FP32 inputs compared with FP64; not a general robustness proof'
        else:
            from tide_lab.campaign_runtime import make_model,optimizer_for,restore,logical_update
            from tide_lab.campaign_data import Corpus
            from tide_lab.v10_repair import training_batches,advance
            import tide_lab.direct_semantic as semantic
            c=json.loads(a.config.read_text());device=runtime(c['device'],c['threads'],c['seed'])
            saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
            if saved['identity']['config_sha256']!=digest(a.config) or saved['identity']['source_commit']!=source_identity()['commit']:
                raise ValueError('original source/config identity mismatch')
            model=make_model(c,device);opt=optimizer_for(model,c);progress=restore(a.checkpoint,model,opt,saved['identity'],device)
            report.update(parent_sha256=digest(a.checkpoint),parent_identity=saved['identity'],parent_progress=dict(progress));del saved
            corpus=Corpus(a.data,256);caught=[]
            original=semantic.allocation
            def observe(t,mu,lower,upper,tau=.25,kind='quad',alpha=1.):
                c,res=original(t,mu,lower,upper,tau,kind,alpha)
                worst=float(res.max())
                if worst>2e-5 and not caught:
                    index=int(res.flatten().argmax());n=t.shape[-1]
                    get=lambda x:x.detach().reshape(-1,x.shape[-1])[index:index+1].cpu()
                    witness=dict(t=get(t),mu=get(mu),lower=get(lower),upper=get(upper),tau=tau,kind=kind,alpha=alpha,
                        original_residual=worst,progress=dict(progress))
                    torch.save(witness,a.output/'witness.pt');caught.append(witness)
                return c,res
            semantic.allocation=observe
            report['reproduced']=False
            for step in range(1024):
                if time.time()>a.deadline_epoch-90:break
                batches,cursor,raw=training_batches(corpus,c,progress,device,128*2**20)
                try:row=logical_update(model,opt,batches,c,progress['tokens'],device)
                except RuntimeError as error:
                    report.update(replay_error=repr(error),reproduced=bool(caught));break
                advance(progress,row,cursor,raw,c)
                if step%16==0:
                    metrics.log(step,{'replay/tokens':progress['tokens'],'replay/loss':row['loss']})
                    atomic_json(a.output/'progress.json',dict(progress))
            report['last_progress']=progress
            if caught:
                w=caught[0]
                for backend,dtype,center in [('cpu',torch.float64,False),('cpu',torch.float32,False),('cpu',torch.float32,True),('npu:0',torch.float32,True)]:
                    t,mu,lo,hi=[w[k].to(device=backend,dtype=dtype) for k in ('t','mu','lower','upper')]
                    if center:t=t-t.amax(-1,keepdim=True)
                    c,res=allocation(t,mu,lo,hi,w['tau'],w['kind'],w['alpha'])
                    report['rows'].append(dict(device=backend,dtype=str(dtype),centered=center,max_residual=float(res.max()),values=c.detach().cpu().tolist()))
                report['witness_sha256']=digest(a.output/'witness.pt')
            report['scope']='Original frozen source/parent RNG+Adam/cursor; at most1024 disposable updates or1h; observer synchronizes scalar residuals and captures first failure. No scientific continuation, no threshold change.'
        metrics.log(1025,{'diagnostic/rows':len(report['rows']),'diagnostic/completed':1})
        report['state']='completed'
    except BaseException as e:report.update(state='failed',error=repr(e));raise
    finally:
        report['finished_at']=now();atomic_json(a.output/'result.json',report);metrics.close()
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=report['state']));project_to_lens(a.output,a.lens_cli)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stress','replay'))
    for n in ('output','config','checkpoint','parent-source','data'):p.add_argument('--'+n,type=Path,required=n=='output')
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
