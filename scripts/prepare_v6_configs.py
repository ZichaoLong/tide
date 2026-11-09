#!/usr/bin/env python3
"""Generate auditable, bounded candidate configurations without model execution."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.campaign_config import anchor,local_candidates,config_for
from tide_lab.records import atomic_json
from tidegraph.planning.recipes import model_recipe


def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];rejected=[]
    def emit(c,group):
        try:
            if c['kind']=='transformer':
                c['planned_inventory']=dict(lm_trainable=c['planned_transformer_parameters'],
                    transformer=c['transformer'])
            else:
                m,inv=model_recipe(**c['core']['recipe'])
                c['planned_inventory']=dict(core_trainable=inv['totals']['trainable'],
                    lm_trainable=inv['totals']['trainable']+4096*c['core']['width']+4096,
                    nodes=m['graph']['nodes'],regions=m['graph']['regions'],
                    resolved_memories=m['memories'],resolved_regions=m['regions'])
        except (ValueError,TypeError) as error:
            rejected.append(dict(id=c['id'],group=group,recipe=c['core']['recipe'],error=str(error)));return
        c['coverage_group']=group
        atomic_json(output/(c['id']+'.json'),c)
        rows.append(dict(id=c['id'],line=c['line'],kind=c['kind'],group=group,
            config=str(output/(c['id']+'.json')),parameters=c['planned_inventory']['lm_trainable']))
    for topology in ('fc','hb'):
        for item in local_candidates():
            r=copy.deepcopy(item['recipe']);r.update(topology=topology,depth=4 if topology=='fc' else 2)
            # Mixed FC needs both GDN and SSD center regions; keep this named
            # structural difference visible instead of claiming empty coverage.
            if r['family']=='mixed' and topology=='fc':r['depth']=8
            emit(config_for(f'cover-{topology}-{item["id"]}',r),item['group'])
    for name,layers in [('fc-budget',1),('hb-budget',7)]:
        c=config_for('cover-transformer-'+name,anchor(),kind='transformer',
            transformer=dict(layers=layers,width=128,heads=4,ffn=256))
        c['planned_transformer_parameters']=4096*128+4096+layers*(4*128**2+2*128*256+9*128+256)+2*128
        emit(c,'transformer_control')
    for name,topology,depth,width,family in [
        ('fc16-kv','fc',4,128,'kv'),('fc32-kv','fc',8,128,'kv'),('fc64-kv','fc',16,128,'kv'),
        ('hb40-kv','hb',2,128,'kv'),('hb88-kv','hb',3,128,'kv'),
        ('fc16-ssd','fc',4,128,'ssd'),('fc32-ssd','fc',8,128,'ssd'),
        ('fc16-gdn','fc',4,128,'gated_delta')]:
        emit(config_for('scale-'+name+'-quad',anchor(topology,width,depth,family),line='scale'),'scale')
    for name,topology,depth in [('fc16','fc',4),('fc32','fc',8),('hb40','hb',2)]:
        r=anchor(topology,128,depth);r.update(regularizer='ent',ent_alpha=1.)
        r['region_overrides']['support_policy']='positive-floor-v1'
        emit(config_for('scale-'+name+'-shannon',r,line='scale'),'dense_control')
    for name,layers,width,ffn in [('5m',5,256,1024),('10m',11,256,1024),('19m',10,384,1536)]:
        c=config_for('scale-transformer-'+name,anchor(),line='scale',kind='transformer',
            transformer=dict(layers=layers,width=width,heads=4 if width==256 else 6,ffn=ffn))
        # Exact task-side parameter formula, tied embedding/head, pre/post norms.
        c['planned_transformer_parameters']=4096*width+4096+layers*(4*width*width+2*width*ffn+9*width+ffn)+2*width
        emit(c,'transformer_control')
    assert sum(r['line']=='coverage' for r in rows)<=256
    assert sum(r['line']=='scale' for r in rows)<=16
    manifest=dict(schema='tide-language-campaign-catalog-v1',launchable=False,
        description='Configuration union; device/batch resolved after finite calibration, not automatic all-platform matrix',
        cases=rows,rejected=rejected,counts={line:sum(r['line']==line for r in rows) for line in ('coverage','scale')})
    atomic_json(output/'catalog.json',manifest)
    print(json.dumps(dict(counts=manifest['counts'],rejected=rejected),ensure_ascii=False))


if __name__=='__main__':generate(sys.argv[1])
