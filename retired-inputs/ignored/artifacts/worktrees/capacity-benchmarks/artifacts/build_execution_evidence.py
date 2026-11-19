"""Read existing terminal execution observations; never run models or device jobs."""
from collections import Counter
from pathlib import Path
import copy
import json
import math
import statistics
import subprocess
from benchmarks.references import file_identity
from benchmarks.search_report import verify_metric_files, write_text_atomic
from scripts.job import atomic

ROOT=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts')
PERF_SOURCE='36c42a7a3ef765fe6257c42c7c1404d28857b77c'
SEARCH_SOURCE='8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee'
MATH='4f5b9f20a4030d68ee66ae2d26e34659a5e37234'
VARIANTS={'baseline':dict(local_attention_backend=None,region_batch_size=1,checkpoint_regions=False),
          'local_math':dict(local_attention_backend='math',region_batch_size=1,checkpoint_regions=False),
          'batch4':dict(local_attention_backend=None,region_batch_size=4,checkpoint_regions=False),
          'recompute':dict(local_attention_backend=None,region_batch_size=1,checkpoint_regions=True),
          'combined':dict(local_attention_backend='math',region_batch_size=4,checkpoint_regions=True)}
PERFS=sorted(ROOT.glob('execution-perf-*-v1/results/variants.json'))
SEARCHES=sorted(ROOT.glob('execution-capacity-settle-*-v1/results/search.json'))
assert len(PERFS)==4 and len(SEARCHES)==2
IDENTITIES={}
runs,normal_rows,normal_raw,ledger_rows,ledger_raw,capacity_rows,capacity_raw,search_records=[],[],[],[],[],[],[],[]


def identity(path):
    result=file_identity(path)
    IDENTITIES[result['path']]=result
    return result


def read(path):
    result=identity(path)
    return result,json.loads(Path(result['path']).read_text())


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def verify(reference):
    actual=identity(reference['path'])
    assert actual['sha256']==reference['sha256'] and actual['bytes']==reference['bytes']


def run_metadata(run,source):
    refs={};values={}
    for kind in ('launch','status','queue','environment'):
        refs[kind],values[kind]=read(run/(kind+'.json'))
    upstream_id,upstream=read(run/'source/docs/upstream.json')
    assert upstream['revision']==MATH
    launch,status,queue,env=(values[k] for k in ('launch','status','queue','environment'))
    assert launch['source']==status['source']==source
    assert status['state'] not in ('running','planned') and queue['status'] not in ('running','queued')
    result=dict(name=run.name,source=source,support_files=refs,upstream_manifest=upstream_id,
                launch_arguments=launch['arguments'],service=launch['unit'],device_policy=launch['device_policy'],
                resource_enforcement=launch['resource_enforcement'],status={k:status[k] for k in ('state','pid','started','finished','elapsed','max_rss_bytes','peak_rss_bytes','exit_code')},
                queue=queue,module_origins=env['module_origins'],loaded_modules=env['environment']['LOADEDMODULES'])
    runs.append(result)
    return result


def audit_compact(audit):
    result={k:v for k,v in audit.items() if k not in ('work_by_sample_time','effective_touched','stages')}
    result['effective_touched']={k:v for k,v in audit['effective_touched'].items() if k!='by_sample_time'}
    result['stages']={key:{k:v for k,v in value.items() if k!='by_sample_time'} for key,value in audit['stages'].items()}
    close(result['effective_touched']['mean'],result['effective_touched']['sum']/result['sample_time_slots'])
    result['query_padding']={key:dict(calls=value['calls'],selected_rows=value['selected_rows'],executed_rows=value['executed_rows'],
                                    padding_rows=value['executed_rows']-value['selected_rows']) for key,value in audit['query_packing'].items()}
    result['omitted_arrays']='effective_touched/stages by_sample_time and work_by_sample_time remain in the hashed original result'
    return result


def normal_compact(raw):
    result=copy.deepcopy({k:v for k,v in raw.items() if k not in ('initial_audits','calibration_audits','observations')})
    result['initial_audits']={k:audit_compact(v) for k,v in raw['initial_audits'].items()}
    observations=[]
    for obs in raw['observations']:
        row={k:v for k,v in obs.items() if k!='post_timing_audit'}
        if obs.get('post_timing_audit') is not None:
            row['post_timing_audit']=audit_compact(obs['post_timing_audit'])
        observations.append(row)
    result['observations']=observations
    return result


def semantic_delta(candidate,baseline):
    return dict(effective_touched_mean=candidate['effective_touched']['mean']-baseline['effective_touched']['mean'],
                work={key:candidate['work'].get(key,0)-baseline['work'].get(key,0) for key in sorted(set(candidate['work'])|set(baseline['work']))},
                packing={kind:{key:candidate['query_padding'][kind][key]-baseline['query_padding'][kind][key] for key in ('calls','selected_rows','executed_rows','padding_rows')}
                         for kind in candidate['query_padding']})


for path in PERFS:
    run=path.parent.parent
    manifest_id,manifest=read(path)
    assert manifest['source']==manifest['harness_source']==PERF_SOURCE and manifest['state']=='completed'
    metadata=run_metadata(run,PERF_SOURCE)
    metadata['manifest_file']=manifest_id
    metadata['arguments']=manifest['arguments']
    metadata['children']=manifest['children']
    args=manifest['arguments'];kind='Timed' if 'timed' in run.name else 'Settle'
    assert args['mode']=='train' and args['warmup']==2 and args['steps']==5 and args['ledger_segments']==1
    assert args['ledger_variants']==['baseline','recompute','combined'] and args['variants']==list(VARIANTS)
    assert args['threads']==4 and args['seed']==1107 and args['activity']=='natural'
    assert (args['batch'],args['length'])==((4,128) if kind=='Timed' else (8,256))
    assert args['device_budget_gib']==48 and args['rss_gib']==72
    verify(manifest['config_file'])
    original=json.loads(Path(manifest['config_file']['path']).read_text())
    assert [c['kind'] for c in manifest['children']]==['normal']*5+['ledger']*3
    assert len({c['pid'] for c in manifest['children']})==8
    assert str(metadata['queue']['physical_id'])==manifest['physical_devices']
    baseline=None;normal_by_variant={}
    for child in manifest['children']:
        assert child['state']=='completed' and child['exit_code']==0
        result_id,raw=read(child['output']);verify(child['config_file'])
        assert child['result_sha256']==result_id['sha256']
        assert raw['source']==raw['harness_source']==PERF_SOURCE and raw['state']=='completed_finite'
        assert raw['physical_devices']==manifest['physical_devices']
        config=json.loads(Path(child['config_file']['path']).read_text())
        assert all(config[k]==v for k,v in VARIANTS[child['variant']].items())
        assert {k:v for k,v in config.items() if k not in VARIANTS['baseline']}==original
        if child['kind']=='normal':
            obs=raw['observations'][0]
            assert obs['subject']=='tide' and obs['mode']=='train' and obs['state']=='completed_finite'
            assert obs['cpu_affinity']==manifest['cpu_affinity'] and obs['torch_threads']==4
            assert len(obs['seconds'])==5 and len(obs['losses'])==5
            assert all(math.isfinite(v) for v in obs['seconds']+obs['losses'])
            close(obs['median_seconds'],statistics.median(obs['seconds']))
            close(obs['input_positions_per_second'],args['batch']*args['length']/obs['median_seconds'])
            close(obs['sample_range_over_median'],(max(obs['seconds'])-min(obs['seconds']))/obs['median_seconds'])
            assert obs['peak_device_bytes']==obs['device_memory']['peak_allocated_bytes']
            assert all(key.endswith(':torch.float32') for group in obs['actual_storage'].values() for key in group)
            assert obs['numerical_equivalence_qualification'] is False
            compact=normal_compact(raw);initial=compact['initial_audits']['train'];post=compact['observations'][0]['post_timing_audit']
            if baseline is None:
                baseline=(obs,initial,post)
            bobs,bi,bp=baseline
            assert obs['initial_parameter_sha256']==bobs['initial_parameter_sha256'] and obs['input_sha256']==bobs['input_sha256']
            computed=dict(run=run.name,kind=kind,precision=args['precision'],variant=child['variant'],batch=args['batch'],length=args['length'],
                          result_file=result_id,seconds=obs['seconds'],median_seconds=obs['median_seconds'],input_positions_per_second=obs['input_positions_per_second'],
                          sample_range_over_median=obs['sample_range_over_median'],timed_peak_device_bytes=obs['peak_device_bytes'],
                          candidate_over_baseline_time=obs['median_seconds']/bobs['median_seconds'],candidate_over_baseline_peak=obs['peak_device_bytes']/bobs['peak_device_bytes'],
                          initial_parameter_input_equal=True,initial_work_summary_equal=initial['work']==bi['work'],post_work_summary_equal=post['work']==bp['work'],
                          initial_active_mean=initial['effective_touched']['mean'],post_seven_updates_active_mean=post['effective_touched']['mean'],
                          initial_query_padding=initial['query_padding'],post_query_padding=post['query_padding'],
                          initial_minus_baseline=semantic_delta(initial,bi),post_minus_baseline=semantic_delta(post,bp),
                          numerical_equivalence_qualification=False)
            original_summary=next(x for x in manifest['summaries'] if x['variant']==child['variant'])
            for key in ('median_seconds','candidate_over_baseline_time','candidate_over_baseline_peak','sample_range_over_median'):
                close(computed[key],original_summary[key])
            assert original_summary['initial_parameter_input_equal'] and original_summary['initial_work_summary_equal']==computed['initial_work_summary_equal']
            normal_rows.append(computed);normal_by_variant[child['variant']]=raw
            normal_raw.append(dict(run=run.name,variant=child['variant'],result_file=result_id,record=compact))
        else:
            obs=raw['observations'][0]
            assert len(raw['observations'])==1 and obs['finite'] and obs['actual_work'] is None
            assert args['warmup']==raw['arguments']['warmup']==2 and not raw['arguments']['continue_state']
            normal=normal_by_variant[child['variant']]
            assert raw['initial_parameter_sha256']==normal['observations'][0]['initial_parameter_sha256']
            assert raw['input_sha256']==normal['observations'][0]['input_sha256']
            phases=obs['phases'];saved=obs['saved_for_backward']
            assert set(phases)=={'boundary','forward','backward','update','released_outputs_and_graph'}
            assert saved['current_live_unique_storage_bytes']==0 and saved['current_live_unique_storage_count']==0
            assert saved['peak_live_additional_storage_bytes']<=saved['peak_live_unique_storage_bytes']
            for phase_name,phase in phases.items():
                close(sum(v['exclusive_unique_storage_bytes'] for v in phase['categories'].values()),phase['unique_storage_union_bytes'])
                assert sum(phase['unique_storage_union_by_device'].values())==phase['unique_storage_union_bytes']
                assert phase['allocator']['allocated_bytes']-phase['unique_storage_union_by_device'].get('npu:0',0)==phase['live_allocator_minus_visible_union_bytes']
            released=phases['released_outputs_and_graph']['categories']
            persistent=sum(released[k]['unique_storage_bytes'] for k in ('parameters','gradients','optimizer'))
            assert persistent==16*raw['parameters']+4*released['parameters']['tensor_count']
            assert all(released[k]['dtypes']==['torch.float32'] for k in ('parameters','gradients','optimizer'))
            if args['precision']=='bf16':assert obs['runtime']['bf16_matrix_observed']
            ledger_rows.append(dict(run=run.name,kind=kind,precision=args['precision'],variant=child['variant'],result_file=result_id,
                                    saved_peak_unique_storage_bytes=saved['peak_live_unique_storage_bytes'],saved_peak_additional_storage_bytes=saved['peak_live_additional_storage_bytes'],
                                    saved_payload_event_bytes=saved['saved_payload_event_bytes'],persistent_parameter_gradient_optimizer_bytes=persistent,
                                    detached_private_state_storage_bytes=released['private_state']['unique_storage_bytes'],
                                    diagnostic_peak_allocated_bytes=max(phase['allocator']['peak_allocated_bytes'] for name,phase in phases.items() if name!='boundary'),
                                    boundary_prior_peak_allocated_bytes=phases['boundary']['allocator']['peak_allocated_bytes'],
                                    forward_visible_union_bytes=phases['forward']['unique_storage_union_bytes'],
                                    forward_unclassified_live_bytes=phases['forward']['live_allocator_minus_visible_union_bytes'],
                                    released_visible_union_bytes=phases['released_outputs_and_graph']['unique_storage_union_bytes'],
                                    initial_parameter_input_pairing=True,post_warmup_parameter_sha256=raw['post_warmup_parameter_sha256'],actual_work=None))
            ledger_raw.append(dict(run=run.name,variant=child['variant'],result_file=result_id,record=raw))

for path in SEARCHES:
    run=path.parent.parent
    manifest_id,manifest=read(path)
    if manifest['state'] in ('running','planned'):
        raise SystemExit('normal/ledger verification passed; pending: '+str(path))
    assert manifest['source']==manifest['harness_source']==SEARCH_SOURCE
    assert manifest['state'] not in ('running','planned')
    verify_metric_files(manifest)
    metadata=run_metadata(run,SEARCH_SOURCE)
    metadata['manifest_file']=manifest_id
    metadata['arguments']=manifest['arguments']
    metadata['children']=manifest['children']
    args=manifest['arguments'];variant='combined' if 'combined' in run.name else 'baseline'
    assert args['subjects']==['tide'] and args['table']=='common' and args['axis']=='batch'
    assert args['precision']=='bf16' and args['mode']=='train' and args['length']==256
    assert args['device_budget_gib']==32 and args['warmup']==2 and args['steps']==5 and args['rss_gib']==88
    assert args['config_overrides']==(VARIANTS['combined'] if variant=='combined' else {})
    verify(manifest['config_file']);verify(manifest['frozen_config'])
    for audit in manifest['audits']:
        if audit.get('result_file'):verify(audit['result_file'])
    metrics={x['child']:x for line in manifest['lines'] for x in line['observations']}
    for child in manifest['children']:
        p=Path(child['output'])
        if not p.exists():continue
        result_id,raw=read(p)
        if child.get('result_sha256'):assert result_id['sha256']==child['result_sha256']
        if child['identity'] not in metrics:continue
        row=metrics[child['identity']]
        if row['admitted']:
            obs=raw['observations'][0]
            assert len(obs['seconds'])==5 and obs['seconds']==row['seconds']
            close(row['median_seconds'],statistics.median(obs['seconds']))
            close(row['input_positions_per_second'],row['batch']*256/row['median_seconds'])
            assert row['timed_peak_device_bytes']==obs['peak_device_bytes']
            assert row['stable']==(row['sample_range_over_median']<=args['maximum_noise'])
        capacity_rows.append(dict(run=run.name,variant=variant,**row))
        capacity_raw.append(dict(run=run.name,variant=variant,result_file=result_id,record=normal_compact(raw)))
    assert all(not point['comparisons'] for point in manifest['common_shape_table'])
    search_records.append(dict(run=run.name,variant=variant,manifest_file=manifest_id,state=manifest['state'],arguments=args,
                               axis_values=manifest['axis_values'],child_states=manifest['child_states'],
                               stop_reasons=[line['stop_reason'] for line in manifest['lines']],audits=manifest['audits'],
                               best_observed=max((row for row in capacity_rows if row['run']==run.name and row['admitted'] and row['stable']),
                                                 key=lambda row:row['input_positions_per_second'],default=None)))

rounding_report=identity(ROOT.parent/'docs/evidence/capacity-rounding-20261118.json')
best_by_variant={row['variant']:row['best_observed'] for row in search_records}
budget_throughput_ratio=best_by_variant['combined']['input_positions_per_second']/best_by_variant['baseline']['input_positions_per_second']
shared_batch_rows=[row for row in capacity_rows if row['batch']==16]
assert len(shared_batch_rows)==2 and len({row['initial_parameter_sha256'] for row in shared_batch_rows})==len({row['input_sha256'] for row in shared_batch_rows})==1
numerical_refs=[]
for name in ('execution-candidates-fp64-v3','execution-candidates-npu-v3','execution-batched-npu-v4','execution-timed-math-npu-v4','execution-timed-recompute-npu-v4'):
    run=ROOT/name
    reference,record=read(run/'results/suite.json')
    entry=dict(run=name,suite_file=reference,state=record['state'],source=record['source'],qualification_reinterpreted_here=False)
    if (run/'superseded.json').exists():
        entry['superseded_file'],entry['superseded']=read(run/'superseded.json')
    numerical_refs.append(entry)

assert len(normal_rows)==20 and len(ledger_rows)==12
assert len({c['pid'] for run in runs[:4] for c in run['children']})==32
assert all(row['sample_range_over_median']<=.25 for row in normal_rows)
evidence=dict(schema='tide-capacity-execution-evidence-v1',date='2026-11-18',math_revision=MATH,
    fixed_shape_source=PERF_SOURCE,tide_batch_search_source=SEARCH_SOURCE,reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    scope='four same-device five-candidate training observations, twelve separate storage ledgers, and two selected Tide-only batch searches; no experiments rerun',
    qualification=dict(finite_execution=True,numerical_equivalence=False,work_counts_prove_trajectory_equivalence=False,
                        global_optimum=False,transformer_capacity_comparison=False),
    candidates=VARIANTS,
    normal_boundary=dict(warmup_updates=2,timed_updates=5,post_semantic_audit_after_updates=7,
                         state='zero at every normal prefill step; complete required Timed flush included',
                         includes='zero_grad/preparation/packing/forward/readout/loss/backward/AdamW foreach/synchronization',
                         excludes='construction/input transfer/initial and post semantic/runtime audits/checkpoint/JSON writes/sample memory probes',
                         peak='normal allocated peak since post-warmup reset, read before intrusive post audit',
                         ratio='candidate/baseline elapsed or peak at fixed B/L, precision and inherited physical card; not trajectory-equivalent speedup',
                         order='fixed serial order; no randomization or independent repeat sessions; host/time drift remains'),
    semantic_boundary=dict(initial='zero-state forward semantic audit before updates',post='same input/zero state after two warmup and five measured updates',
                           activity='unique semantically effective Parameter slices per sample/logical time; not physical reads/FLOPs',
                           timed='mean over logical ticks including necessary flush; per-input work normalization retained separately',
                           work_count_equality='does not prove identical support sets, outputs, gradients or update trajectory',
                           padding='executed_rows minus selected_rows for K/S packing; calls and raw rows retained'),
    ledger_boundary=dict(warmup_updates=2,diagnostic_updates=1,independent_process=True,actual_work=None,
                         persistent='unique parameter+gradient+AdamW storage at released_outputs_and_graph, including CPU FP32 step scalars',
                         private='detached returned private-state storage after diagnostic output/graph release',
                         saved='unique simultaneous storage held by outer saved_tensors_hooks; repeated payload events are separate; nested checkpoint hooks/custom opaque allocations may not be visible',
                         additional='saved storage excluding parameter/slot/input/incoming-state ownership at segment boundary; outgoing state can still overlap',
                         overlap='saved peaks, state/category storage, allocator peaks and the foreach allowance are overlapping measurements, not additive components',
                         unclassified='same-instant live allocator minus visible storage union; not operator scratch alone or hardware bandwidth',
                         normal_pairing='same initial parameter/input hashes and two warmups; ledger has one diagnostic update versus five normal samples; normal post-warmup parameter hash is not recorded'),
    original_schema_notes=[dict(field='ledger.runtime.instrumentation',original='separate ordinary forward without semantic counting hooks; excluded from timing',
                                interpretation='ledger RuntimeAudit context actually includes forward/readout/loss/backward/optimizer and phase snapshots; its gradient operators confirm the broader diagnostic scope'),
                           dict(field='ledger.phases.boundary.allocator.peak_allocated_bytes',interpretation='sampled before diagnostic reset, includes previous warmup/history; exclude this value when reporting diagnostic-segment peak'),
                           dict(field='ledger later phase allocator peaks',interpretation='cumulative from reset immediately after boundary, not per-phase increments'),
                           dict(future_label_fix='3543825aa61618b174efe915c3a36a8246acc6fe',raw_evidence_rewritten=False)],
    tide_best_observed_throughput_comparison=dict(combined_over_baseline=budget_throughput_ratio,scope='different selected B and different physical cards; no same-workload speedup or global optimum claim'),
    tide_batch_search_boundary='BF16 Settle L256 Tide only, explicit32GiB budget, selected baselineB8/16 and combinedB16/32/64 candidates; different cards and B cannot define same-workload speedup or A-D budget table',
    numerical_evidence=dict(report='capacity-rounding-20261118.md',report_json=rounding_report,report_commit='48538d2e4dee00fc6224d4841d516b7462a4e0d2',raw_suites=numerical_refs,
                            scope='retain original failures/cancelled identities; numerical qualification is evaluated separately and is not inferred from these performance results'),
    validation=dict(verified_json_identity_files=len(IDENTITIES),normal_children=20,normal_timed_samples=100,separate_ledger_children=12,
                    tide_batch_searches=2,batch_search_observations=len(capacity_rows),batch_search_timed_samples=sum(len(row.get('seconds',[])) for row in capacity_rows),
                    batch_search_initial_audits=sum(len(row['audits']) for row in search_records),batch16_input_initial_parameter_hashes_paired=True,
                    all_normal_initial_parameter_input_hashes_paired=True,normal_observations_under_25_percent_range_over_median=True,
                    checks=['all child result/config hashes, sources and math pins','terminal launcher/queue records and physical mappings','five samples, medians, throughput, noise and candidate/baseline ratios',
                            'initial/post effective touches, work and padding differences','normal/ledger separate PIDs and initial parameter/input hashes',
                            'ledger per-category exclusive union, device union and live unclassified arithmetic','saved storage release and fixed FP32 parameter/gradient/optimizer bytes',
                            'diagnostic peak excludes pre-reset boundary','terminal Tide-only search identities, original limits/failures and no Transformer comparisons']),
    runs=runs,verified_json_files=list(IDENTITIES.values()),normal_comparisons=normal_rows,normal_raw_compact=normal_raw,
    ledger_summary=ledger_rows,ledger_raw_records=ledger_raw,tide_batch_searches=search_records,tide_batch_observations=capacity_rows,tide_batch_raw_compact=capacity_raw)
atomic('docs/evidence/capacity-execution-20261118.json',evidence)

lines=['# 执行候选性能、存储与有限扩批量观察 · 2026-11-18','',
'四组同卡固定形状训练比较完成五个候选、共100个正常计时样本；每组另有 baseline/recompute/combined 三个独立内存账本进程。',
'另两组是统一32 GiB预算的 Tide-only 扩批量观察，没有 Transformer 调优或 A–D 容量比较。',
'本页只读既有终态结果。样本、工作量、padding、活动变化、分阶段存储与原始身份分别保存在 [JSON](capacity-execution-20261118.json)。','',
'固定形状源码/设施为 `'+PERF_SOURCE+'`；扩批量为 `'+SEARCH_SOURCE+'`；数学为 `'+MATH+'`。',
'数值资格见 [有限舍入调查](capacity-rounding-20261118.md) 及下列 v3/v4 原始记录。完成、工作计数相等或初始哈希相等均不构成轨迹等价资格。','',
'## 同卡五候选正常计时','',
'候选明确固定三个执行选项：baseline=(默认局部 Attention, region_batch1, 无重计算)；local_math 仅改局部 Attention 为 math；',
'batch4 仅把区域批量设为4；recompute 仅开启区域重计算；combined 同时启用 math/batch4/重计算。数学配置不变。',
'每组各自固定同一物理卡、输入和初始参数哈希，候选按上述顺序串行独立进程运行；Settle B8/L256，Timed B4/L128，FP32/BF16 分组。',
'每候选两次预热更新、五个正常训练样本，每步状态从零开始；训练包括必要 Timed flush、loss、反向、完整 AdamW foreach 与同步。',
'初态/后更新语义及 runtime 审计在计时外，内存账本全部另起进程。正常峰值在预热后重置、后审计前读取。',
'设备 allocated 预算48 GiB，子进程树采样RSS72 GiB。比值为候选/baseline，小于1表示对应耗时或峰值较小。','',
'| 形态 / 精度 | 候选 | 中位秒 | 正常峰值 GiB | 时间比 | 峰值比 |',
'|---|---|---:|---:|---:|---:|']
for row in normal_rows:
    lines.append('| '+' | '.join([row['kind']+' / '+row['precision'].upper(),row['variant'],f"{row['median_seconds']:.3f}",f"{row['timed_peak_device_bytes']/2**30:.3f}",f"{row['candidate_over_baseline_time']:.3f}",f"{row['candidate_over_baseline_peak']:.3f}"])+' |')
lines += ['',
'本次 batch4 的完整训练时间比为0.695–0.792，峰值比为1.009–1.038；recompute 的峰值降至0.148–0.202，但耗时增至1.738–1.813。',
'combined 的峰值为0.213–0.271、耗时为1.235–1.354。这显示当前固定形状下，区域批量与重计算分别偏向吞吐和内存，组合没有消除重计算成本。',
'local_math 的时间差较小，不能凭一次固定顺序的五样本结果宣称稳定通用收益。全部观察的样本range/median低于25%，这仅是有限噪声描述，不是置信区间或随机交叉重复实验。','',
'## 实际工作与活动漂移','',
'初态参数/输入哈希均相同。语义审计的初态和两次预热加五次计时更新后的状态分列，后审计也从零私有状态执行同一输入。',
'表内活动差是候选减同组baseline的有效触及均值；K padding是executed_rows−selected_rows。',
'完整正消息、候选更新、Full、区域/KV有效对数、S packing和矩阵行数，以及各项差值均保留在JSON。','',
'| 形态 / 精度 | 候选 | 初态触及差 | 更新后触及差 | K padding 初态 / 更新后 |',
'|---|---|---:|---:|---|']
for row in normal_rows:
    if row['variant']=='baseline':continue
    lines.append('| '+' | '.join([row['kind']+' / '+row['precision'].upper(),row['variant'],f"{row['initial_minus_baseline']['effective_touched_mean']:.3f}",f"{row['post_minus_baseline']['effective_touched_mean']:.3f}",str(row['initial_query_padding']['K']['padding_rows'])+' / '+str(row['post_query_padding']['K']['padding_rows'])])+' |')
lines += ['',
'例如Settle FP32初态batch4的K调用从32降至22，selected rows仍为150,803，executed rows从199,872升至205,952；',
'它减少调用但增加padding，这与本次更短时间、略高峰值同时出现。计数不提供关键路径时间分摊，不能将全部收益归因到某一个算子。',
'FP32初态汇总工作相同，七次更新后已存在工作差异；BF16的local_math/batch4/combined连初态汇总就有差异。',
'BF16 recompute在本组初态及后更新汇总相同，也仍不能据此证明support集合、输出、梯度和参数轨迹相等。有效触及量不是FLOPs或物理读取字节。','',
'## 三候选独立内存账本','',
'账本使用同初始参数/输入、两轮预热，再执行一个诊断更新；正常计时是五个更新，后审计不能与账本终态混同。',
'`actual_work=null` 明确表示账本不做语义WorkAudit。RuntimeAudit与saved_tensors_hooks属于侵入式诊断，其峰值不能替换正常峰值。',
'持久参数/梯度/优化器列为释放输出和图后的唯一存储；私有状态列是同阶段保留的detached状态。',
'saved列是外层hooks持有的同时存活唯一storage峰值。诊断峰值排除boundary，后续各阶段allocator peak从同一次reset累计。','',
'| 形态 / 精度 | 候选 | saved唯一峰值 GiB | 参数+梯度+优化器 GiB | 私有状态 GiB | 诊断峰值 GiB |',
'|---|---|---:|---:|---:|---:|']
for row in ledger_rows:
    lines.append('| '+' | '.join([row['kind']+' / '+row['precision'].upper(),row['variant'],*[f"{row[key]/2**30:.3f}" for key in ('saved_peak_unique_storage_bytes','persistent_parameter_gradient_optimizer_bytes','detached_private_state_storage_bytes','diagnostic_peak_allocated_bytes')]])+' |')
lines += ['',
'这些配置实际持有15,694,224参数。参数、梯度与AdamW存储在两精度中均为FP32，合计约0.234 GiB；正常十余GiB峰值不能归结为参数持有量。',
'重计算的外层saved storage明显降低，正常峰值也降低；账本不保证看见checkpoint内嵌hooks或后端不透明分配。',
'各类别可能共享storage，saved峰值、私有状态、allocator峰值及foreach规划allowance不能相加。JSON同时保留exclusive归属、可见union、',
'逐阶段live allocated减device-visible union的余量；该余量包含未追踪张量和allocator/后端因素，不是测得的算子scratch或硬件传输量。','',
'原36c42账本有两处标签/边界需按实际执行解读：内嵌runtime.instrumentation固定写“ordinary forward”，但诊断context实际覆盖forward/readout/loss/backward/optimizer；',
'boundary的allocator peak在reset之前，含此前预热历史；reset紧随boundary，其他阶段是本诊断段累计峰值，非单阶段增量。',
'旧raw保持不变。后续仅标签修复 `3543825aa61618b174efe915c3a36a8246acc6fe` 明确了这两点，没有重跑或更改本次数据身份。','',
'## 32 GiB预算的选定Tide扩批量观察','',
'以下均为BF16 Settle、L256、两预热/五采样：baseline预设B8/16，combined预设B16/32/64；各自独立卡运行。',
'没有调优Transformer，也没有把这些观察添加为A–D容量表。不同B的吞吐用于比较选定生产负载，不是同输入量加速比。','',
'| 候选 | B | 中位秒 | 输入位置/s | 正常峰值 GiB | 稳定/准入 |',
'|---|---:|---:|---:|---:|---|']
for row in capacity_rows:
    value=lambda key,digits=3:'—' if row.get(key) is None else f"{row[key]:.{digits}f}"
    peak='—' if row.get('timed_peak_device_bytes') is None else f"{row['timed_peak_device_bytes']/2**30:.3f}"
    lines.append('| '+' | '.join([row['variant'],str(row['batch']),value('median_seconds'),value('input_positions_per_second',2),peak,str(row['stable'])+'/'+str(row['admitted'])])+' |')
for record in search_records:
    best=record['best_observed']
    lines += ['',record['variant']+'的停止记录为 `'+', '.join(record['stop_reasons'])+'`；'+('没有稳定准入候选。' if best is None else '最佳已测稳定点为B'+str(best['batch'])+'，'+f"{best['input_positions_per_second']:.2f}"+'输入位置/s。')]
lines += ['',
'按各自最佳已测点，combined吞吐为baseline的'+f'{budget_throughput_ratio:.3f}'+'倍；这是不同B、不同物理卡下的预算观察。'+('本矩阵中更大的已完成B没有转化为更高的最佳吞吐。' if budget_throughput_ratio<1 else '该收益仅限这些选定负载与卡的观察。'),
'这是预设候选内的有限观察，未证明全局最佳配置或最大可行B；尤其baseline没有尝试B32，不能把未试候选写成OOM失败。',
'32GiB是同步检查的allocated预算，子进程树采样RSS88GiB；原失败/停止状态和实际活动变化保留。','',
'## 资格链接、资源与复现','',
'数值比较与有限舍入诊断由 [capacity-rounding-20261118](capacity-rounding-20261118.md) 单列。既存v3/v4原始入口如下：','']
for ref in numerical_refs:
    lines.append('- ['+ref['run']+'](../../artifacts/'+ref['run']+'/results/suite.json)：原suite状态 `'+ref['state']+'`，源码 `'+ref['source']+'`。')
lines += ['',
'v3 NPU候选因batch4/combined的BF16 bias epilogue实现问题中止，superseded记录和旧失败保留；v4部分suite仍为failed。',
'本页不依据后续性能完成把这些失败改判通过，也不把CPU FP64诊断升级为NPU FP32/BF16数值资格。','',
'每进程四线程，固定形状每组候选共用其队列准入的物理卡；实际身份均为协作队列独占、外部任务不受控制。',
'候选前后host load、初始参数/输入hash、软件版本、算子dtype审计和全部命令留在JSON。同卡固定顺序仍有主机争用和时间漂移。','',
'| 运行身份 | 物理NPU | CPU affinity | 外层时限s / RSS GiB |',
'|---|---:|---|---|']
for run in runs:
    args=run['launch_arguments']
    lines.append('| '+run['name']+' | '+str(run['queue']['physical_id'])+' | '+args['affinity']+' | '+str(args['seconds'])+' / '+str(args['rss_gib'])+' |')
lines += ['',
'只读核验本页所有引用JSON，不调用模型：','',
'```sh',"python - <<'PY'",'from pathlib import Path','import hashlib, json',
"r = json.loads(Path('docs/evidence/capacity-execution-20261118.json').read_text())",
"for item in r['verified_json_files']:","    data = Path(item['path']).read_bytes()",
"    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']",
"print('verified', len(r['verified_json_files']), 'JSON files')",'PY','```','',
'重新执行时使用JSON中的冻结源码及`runs[].launch_arguments`/`children`，选择新输出身份并按当前资源重新准入。',
'正常性能、侵入式存储诊断、选定扩批量和独立数值资格各自保留边界。','']
write_text_atomic('docs/evidence/capacity-execution-20261118.md','\n'.join(lines))
print(json.dumps(evidence['validation'],indent=2))
print('JSON bytes',Path('docs/evidence/capacity-execution-20261118.json').stat().st_size)
print('Markdown bytes',Path('docs/evidence/capacity-execution-20261118.md').stat().st_size)
