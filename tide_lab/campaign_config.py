"""Finite v6 union of local candidates; public planning owns validity/math."""
import copy


def anchor(topology='fc', width=32, depth=None, family='kv'):
    return dict(topology=topology, family=family, width=width,
        depth=depth or (4 if topology=='fc' else 2), region_width=4,
        branches=2, offsets=[0,1], profile='baseline', sparse_denominator=2,
        selection='regional_total', regularizer='quad', history=512,
        source_mode='region', memory_overrides={'memory_read':'qry' if family=='kv' else 'state'},
        region_overrides={'allocation_temperature':.25})


def local_candidates():
    """Each named choice has a witness; no Cartesian product of all fields."""
    rows=[]
    def add(name,group,recipe):rows.append(dict(id=name,group=group,recipe=recipe))
    add('anchor','anchor',anchor())
    for family in ('kv','affine','delta','gated_delta','ssd'):
        for read in ('sum','qry') if family=='kv' else ('sum','state'):
            r=anchor(family=family);r['memory_overrides']['memory_read']=read
            if family=='kv' and read=='qry':continue
            add(f'mem-{family}-{read}','memory',r)
    for mode in ('0','-'):
        r=anchor();r['region_overrides']['read_mode']=mode
        add('read-'+('zero' if mode=='0' else 'old'),'read_snapshot',r)
    for name,changes in [
        ('local-match',{'local_score':'match'}),
        ('region-att-stats',{'region_score':'att','region_key':'content_stats'}),
        ('region-att-content',{'region_score':'att','region_key':'content'}),
        ('evidence-grouped',{'region_evidence':'grouped'}),
        ('match-att-grouped',{'local_score':'match','region_score':'att','region_key':'content','region_evidence':'grouped'}),
        ('fixed-total25',{'region_control_mode':'fixed','total_fraction':.25}),
        ('fixed-total50',{'region_control_mode':'fixed','total_fraction':.5}),
        ('fixed-total100',{'region_control_mode':'fixed','total_fraction':1.}),
        ('gamma1',{'gamma_max':1.}),('gamma4',{'gamma_max':4.}),
        ('temperature1',{'allocation_temperature':1.}),
        ('score-width8',{'hidden_width':8}),
        ('evidence-width8',{'region_control_width':8}),
        ('match-width8',{'local_score':'match','selection_width':8}),
        ('match-residual-off',{'local_score':'match','alpha_res':0.}),
        ('match-inner-off',{'local_score':'match','alpha_match':0.}),
        ('att-local-off',{'region_score':'att','alpha_loc':0.}),
        ('att-regional-off',{'region_score':'att','alpha_sel':0.})]:
        r=anchor();r['region_overrides'].update(changes);add(name,'scoring_control',r)
    modes=('regional_total','regional_threshold','constrained','regional_total_zero',
        'regional_threshold_hard','hard_selection','direct_attention','unit_attention',
        'pooled_attention','proportional')
    for mode in modes:
        regs=('quad','bern','ent1','ent15','ent2') if mode in ('regional_total','regional_threshold','constrained') else (None,)
        for reg in regs:
            if mode=='regional_total' and reg=='quad':continue
            r=anchor();r['selection']=mode;r['region_overrides']={};r.pop('regularizer')
            if reg is not None:
                r['regularizer']='ent' if reg.startswith('ent') else reg
                r['region_overrides']['allocation_temperature']=.25
                if reg.startswith('ent'):r['ent_alpha']={'ent1':1.,'ent15':1.5,'ent2':2.}[reg]
                if reg=='ent1':r['region_overrides']['support_policy']='positive-floor-v1'
            add('send-'+mode+('-'+reg if reg else ''),'sending',r)
    for name,mode,changes in [
        ('direct-received','direct_attention',{'allocation_base':'received'}),
        ('unit-mu','unit_attention',{'allocation_base':'mu'}),
        ('constrained-context-gain','constrained',{'send_gain':'context'}),
        ('constrained-learned-gain','constrained',{'send_gain':'fixed','gain_logit':0.}),
        ('gain-att-received','constrained',{'region_score':'att','query_weight':'received'}),
        ('gain-att-mu','constrained',{'region_score':'att','query_weight':'mu'}),
        ('threshold-fixed','regional_threshold',{'region_control_mode':'fixed','eta':0.}),
        ('zero-fixed','regional_total_zero',{'region_control_mode':'fixed','total_fraction':.5})]:
        r=anchor();r['selection']=mode;r['region_overrides']=changes
        if mode not in ('constrained','regional_threshold'):r.pop('regularizer')
        add(name,'sending_control',r)
    for mode in ('node','source','adapted'):
        r=anchor();r['source_mode']=mode;add('source-'+mode,'source_sharing',r)
    for name,changes in [('window16',{'window':16}),('window128',{'window':128}),
        ('summary8',{'summary_width':8}),('adapter8',{'adapter_rank':8}),
        ('ffn2x',{'ffn_width':64}),('ffn8x',{'ffn_width':256}),
        ('full-memory-off',{'alpha_att':0.}),('full-ffn-off',{'alpha_ff':0.}),
        ('heads2',{'heads':2}),('decay-slow',{'beta_short':.001}),
        ('rope-half',{'full_frequencies':[[10000.**(-2*j/16) for j in range(8)]]}),
        ('rope-full',{'full_frequencies':[[10000.**(-2*j/32) for j in range(16)]]}),
        ('rope-read',{'read_frequencies':[10000.**(-2*j/32) for j in range(16)]}),
        ('distance-slope',{'distance_slopes':[.01]})]:
        r=anchor();r['memory_overrides'].update(changes);add(name,'memory_shape_position',r)
    for family in ('gated_delta','ssd'):
        for name,changes in [('conv1',{'conv_size':1}),('conv8',{'conv_size':8}),
            ('heads2',{'heads':2}),('mem-off',{'alpha_mem':0.}),('ffn-off',{'alpha_ff':0.})]:
            r=anchor(family=family);r['memory_overrides'].update(changes)
            add(f'{family}-{name}','recurrent_shape',r)
    r=anchor(family='ssd');r['memory_overrides'].update(heads=2,groups=2)
    add('ssd-groups2','recurrent_shape',r)
    for divisor in (1,4):
        r=anchor();r['sparse_denominator']=divisor;add(f'budget-div{divisor}','budget',r)
    for family in ('kv','mixed'):
        r=anchor(family=family);r['profile']='textbook';r['memory_overrides']={}
        add('textbook-'+family,'composed',r)
    for row in rows:
        region=row['recipe'].get('region_overrides',{})
        if region.get('region_control_mode')=='fixed':region['region_evidence']=None
    return rows


def config_for(case_id, recipe, *, line='coverage', kind='tide', transformer=None):
    return dict(schema='tide-language-campaign-config-v1',campaign='language-v6',id=case_id,
        study_id='TL-COVER-v6' if line=='coverage' else 'TL-SCALE-v6',line=line,
        kind=kind,seed=11,dtype='float32',threads=1,device='cpu',
        core=dict(width=recipe['width'],recipe=copy.deepcopy(recipe),max_materialize_bytes=2**30),
        transformer=transformer,training=dict(target_tokens=65536 if line=='coverage' else 4194304,
            milestones=[65536] if line=='coverage' else [262144,1048576,4194304],
            length=32 if line=='coverage' else 128,batch=8 if line=='coverage' else 8,
            lr=.003 if line=='coverage' else .001,lr_floor_fraction=.1,warmup_tokens=2048 if line=='coverage' else 65536,
            clip_norm=1.,eps=1e-6,weight_decay=.01,eval_tokens=8192 if line=='coverage' else 32768,
            checkpoint_seconds=600,validation_every_tokens=16384 if line=='coverage' else 131072,
            timeout_seconds=8*3600 if line=='coverage' else 36*3600,log_every=8),
        checks=dict(core_atol=1e-6,core_rtol=1e-5,finite_exploration=True),
        evaluation=dict(test_policy='sealed; separate frozen finalist review only',inference_length=16),
        resources=dict(max_rss_gib=8 if line=='coverage' else 48),
        revisions=dict(mathematics='0770593838a37c790a568226930567ed60f99b83',
            foundation='74e7bbd62341d2ccd382194a2c52ad4fea033665'))
