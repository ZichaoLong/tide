"""Consolidate completed JSON only; never open or stat checkpoint payloads."""
from collections import Counter
from pathlib import Path
import json
import math
import statistics
import subprocess
from benchmarks.references import file_identity
from benchmarks.search_report import write_text_atomic
from scripts.job import atomic

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts')
CONTINUOUS_SOURCE = 'efca7198570e8ee7a21ddccd81165eb36bdb269a'
RESTORE_SOURCE = '062a8ab6d6ba0faa191d8900420b697711b7d21d'
MATH = '4f5b9f20a4030d68ee66ae2d26e34659a5e37234'
IDENTITIES = {}
runs, continuous, restores, summary, checkpoint_summary, restore_summary = [], [], [], [], [], []


def identity(path):
    path = Path(path)
    assert path.suffix == '.json', 'Only JSON reads are admitted in this evidence pass'
    result = file_identity(path)
    IDENTITIES[result['path']] = result
    return result


def read(path):
    result = identity(path)
    return result, json.loads(Path(result['path']).read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a,b)


def run_metadata(run, source):
    values = {}
    refs = {}
    for kind in ('launch','status','queue','environment'):
        refs[kind], values[kind] = read(run/(kind+'.json'))
    upstream_id, upstream = read(run/'source/docs/upstream.json')
    assert upstream['revision'] == MATH
    launch,status,queue,env = (values[key] for key in ('launch','status','queue','environment'))
    assert launch['source'] == status['source'] == source
    assert status['state'] == 'passed' and status['exit_code'] == 0
    assert queue['status'] == 'completed' and queue['return_code'] == 0
    metadata = dict(name=run.name,source=source,support_files=refs,upstream_manifest=upstream_id,
                    launch_arguments=launch['arguments'],service=launch['unit'],
                    device_policy=launch['device_policy'],resource_enforcement=launch['resource_enforcement'],
                    status={k:status[k] for k in ('state','pid','started','finished','elapsed','max_rss_bytes','peak_rss_bytes','exit_code')},
                    queue=queue,module_origins=env['module_origins'],loaded_modules=env['environment']['LOADEDMODULES'])
    runs.append(metadata)
    return metadata


for name,label in (('capacity-continuous-hb05b-v3','HB 0.5B'),('capacity-continuous-hb1b-v3','HB 1B')):
    run = ROOT/name
    metadata = run_metadata(run,CONTINUOUS_SOURCE)
    programme_id, programme = read(run/'results/programme.json')
    metadata['programme_file'] = programme_id
    metadata['programme_arguments'] = programme['arguments']
    metadata['children'] = programme['children']
    assert programme['source'] == CONTINUOUS_SOURCE and programme['state'] == 'completed'
    assert len(programme['children']) == 6
    assert programme['arguments']['warmup'] == 0 and programme['arguments']['steps'] == 3
    assert str(metadata['queue']['physical_id']) == programme['physical_devices']
    for child in programme['children']:
        result_id,raw = read(child['output'])
        config_id,config = read(child['config'])
        assert child['result_sha256'] == result_id['sha256'] and child['config_sha256'] == config_id['sha256']
        assert child['state'] == 'completed' and child['exit_code'] == 0
        assert raw['source'] == raw['harness_source'] == CONTINUOUS_SOURCE and raw['state'] == 'completed'
        args=raw['arguments'];obs=raw['observations']
        assert args['batch'] == 1 and args['prefill'] == 128 and args['prefill_chunk'] == 64 and args['decode'] == 8
        assert args['updates'] == 3 and args['warmup'] == 0 and args['threads'] == 4
        assert raw['physical_devices'] == programme['physical_devices'] and raw['cpu_affinity'] == programme['cpu_affinity']
        assert raw['spec']['topology']['timing'] == 'settle'
        assert len(obs) == 3 and raw['input_shape'] == [1,136,raw['spec']['d']]
        for index,row in enumerate(obs):
            assert row['index'] == index and row['state_position'] == 136*(index+1)
            assert row['input_positions'] == row['output_slots'] == 136
            assert row['finite_output'] and row['finite_parameters'] and row['finite_gradients']
            assert row['flush_seconds'] == 0 and len(row['calls']) == 10 and len(row['decode_seconds']) == 8
            assert [call['inputs'] for call in row['calls']] == [64,64]+[1]*8
            assert [call['phase'] for call in row['calls']] == ['prefill']*2+['decode']*8
            close(row['prefill_seconds'],sum(call['seconds'] for call in row['calls'][:2]))
            close(row['prefill_positions_per_second'],128/row['prefill_seconds'])
            assert row['decode_seconds'] == [call['seconds'] for call in row['calls'][2:]]
            close(row['stable_decode_median_seconds'],statistics.median(row['decode_seconds'][1:]))
            close(row['stable_decode_positions_per_second'],7/sum(row['decode_seconds'][1:]))
            assert row['complete_seconds']+1e-8 >= row['input_plus_flush_seconds']+row['backward_seconds']+row['optimizer_seconds']
            assert row['peak_device_bytes'] > 0
            if args['mode'] == 'train':
                assert math.isfinite(row['loss']) and row['backward_seconds'] > 0 and row['optimizer_seconds'] > 0
            else:
                assert row['loss'] is None and row['backward_seconds'] == row['optimizer_seconds'] == 0
        times = [row['complete_seconds'] for row in obs]
        peaks = [row['peak_device_bytes'] for row in obs]
        record = dict(identity=child['identity'],run=name,instance=label,result_file=result_id,config_file=config_id,record=raw)
        continuous.append(record)
        summary.append(dict(identity=child['identity'],run=name,instance=label,precision=args['precision'],mode=args['mode'],parameters=raw['parameters'],
                            input_shape=raw['input_shape'],initial_parameters_sha256=raw['initial_parameters_sha256'],input_sha256=raw['input_sha256'],
                            complete_seconds=times,median_complete_seconds=statistics.median(times),total_complete_seconds=sum(times),
                            aggregate_input_positions_per_second=408/sum(times),peak_device_bytes=peaks,max_reported_peak_device_bytes=max(peaks),
                            state_positions=[row['state_position'] for row in obs],checkpoint_comparison='not_run' if args['mode'] != 'train' else raw['checkpoint']['state']))
        if args['mode'] == 'train':
            cp=raw['checkpoint']
            assert cp['state'] == 'within_threshold'
            checks={k:v for k,v in cp.items() if isinstance(v,dict) and 'within_threshold' in v}
            assert set(checks) == {'saved_data','next_forward','next_state','input_gradients','parameter_gradients','updated_parameters','updated_optimizer','updated_state'}
            for key,check in checks.items():
                assert check['within_threshold'] and check['finite']
                assert not check['metadata_mismatches'] and not check['support_mismatches'] and not check['numerical_failures']
                expected = (0,0) if key in ('saved_data','next_forward','next_state') else (1e-6,.01 if key == 'input_gradients' else .002)
                assert (check['atol'],check['rtol']) == expected
                if check['exact']:
                    assert check['max_abs'] == check['max_scaled_error'] == 0
                else:
                    assert check['max_abs'] > 0 and 0 < check['max_scaled_error'] <= 1
            assert len(cp['update_windows']) == 2
            assert all(window == dict(input_positions=64,window_output_slots=64,flush_output_slots=0,final_state_position=472) for window in cp['update_windows'])
            assert cp['losses'][0] == cp['losses'][1]
            checkpoint_summary.append(dict(run=name,identity=child['identity'],instance=label,precision=args['precision'],
                                           checkpoint_path=args['checkpoint'],checkpoint_bytes=cp['file_bytes'],
                                           saved_state_position=408,comparison_input_positions_per_owner=64,comparison_final_state_position=472,
                                           input_hash_scope='original record hashes the full 136-input tensor; no separate 64-input witness hash is recorded',
                                           save_seconds=cp['save_seconds'],same_process_load_seconds=cp['load_seconds'],state=cp['state'],checks=checks))

for path in sorted(ROOT.glob('capacity-restore-*-v1/restore.json')):
    run=path.parent
    metadata=run_metadata(run,RESTORE_SOURCE)
    result_id,raw=read(path)
    args=raw['arguments']
    assert raw['source'] == raw['harness_source'] == RESTORE_SOURCE and raw['state'] == 'completed_finite'
    assert raw['checkpoint_source'] == CONTINUOUS_SOURCE
    assert args['batch'] == 1 and args['length'] == 128 and args['prefill_chunk'] == 64 and args['decode'] == 8
    assert args['threads'] == 4 and args['device_budget_gib'] == 52 and args['expected_checkpoint_sha256'] is None
    assert raw['physical_devices'] == str(metadata['queue']['physical_id'])
    producer=next(row for row in continuous if row['record']['arguments']['checkpoint'] == args['checkpoint'])
    assert producer['record']['arguments']['mode'] == 'train'
    assert producer['record']['spec'] == raw['config'] and producer['record']['parameters'] == raw['parameters']
    assert producer['record']['precision'] == raw['precision']
    checkpoint=raw['checkpoint_file']
    assert checkpoint['path'] == args['checkpoint'] and checkpoint['bytes'] == producer['record']['checkpoint']['file_bytes']
    assert len(checkpoint['sha256']) == 64
    assert all(checkpoint[key] == raw['checkpoint_stat_before_load'][key] for key in ('bytes','mtime_ns','device','inode'))
    assert raw['loaded_state_position'] == 408 and raw['restored_state_position'] == 544
    assert raw['input_positions'] == 136 and raw['input_shape'] == [1,136,raw['config']['d']]
    assert raw['output_windows'] == dict(window_output_slots=136,flush_output_slots=0)
    assert raw['private_state_detached'] and not raw['optimizer_step_increment_errors']
    assert all(group['finite'] for key in ('loaded_finite','updated_finite') for group in raw[key].values())
    before,after=raw['optimizer_before'],raw['optimizer_after']
    assert set(before['steps']) == set(after['steps'])
    assert len(before['steps']) == before['state_parameter_tensors'] == raw['parameter_tensors_with_gradients']
    assert all(before['steps'][key] == 3 and after['steps'][key] == 4 for key in before['steps'])
    assert before['groups'] == after['groups'] and before['storage'] == after['storage']
    assert raw['parameter_storage']['dtypes'] == raw['gradient_storage']['dtypes'] == before['storage']['dtypes'] == ['torch.float32']
    assert raw['parameter_storage']['payload_bytes'] == raw['gradient_storage']['payload_bytes'] == 4*raw['parameters']
    assert before['storage']['payload_bytes'] == 8*raw['parameters']+4*before['state_parameter_tensors']
    update_sample=raw['parameter_update_sample']
    assert update_sample['changed_scalars'] == sum(update_sample['changed_by_parameter'].values())
    assert 0 < update_sample['changed_scalars'] <= update_sample['sampled_scalars']
    phases=raw['phases']
    assert all(value['state'] == 'completed' for value in phases.values())
    load_keys=('load_model_state','load_optimizer')
    update_keys=('zero_grad','forward_loss_flush','backward','optimizer','detach')
    for interval,keys,peak_key in (('load_complete_seconds',load_keys,'load_peak_device_bytes'),('next_update_complete_seconds',update_keys,'next_update_peak_device_bytes')):
        assert raw[interval] >= sum(phases[key]['seconds'] for key in keys)
        assert raw[peak_key]['npu:0'] == max(phases[key]['device_memory']['npu:0']['peak_allocated_bytes'] for key in keys)
    close(raw['next_update_input_positions_per_second'],136/raw['next_update_complete_seconds'])
    assert raw['numerical_equivalence_qualification'] is False and math.isfinite(raw['loss'])
    restores.append(dict(run=run.name,instance=producer['instance'],result_file=result_id,producing_result_file=producer['result_file'],record=raw))
    restore_summary.append(dict(run=run.name,instance=producer['instance'],precision=args['precision'],result_file=result_id,
                                producing_result_file=producer['result_file'],checkpoint_file_as_recorded=checkpoint,
                                checkpoint_hash_recomputed_in_this_review=False,expected_checkpoint_sha256=None,
                                checkpoint_source_is_caller_provenance=True,input_sha256=raw['input_sha256'],input_shape=raw['input_shape'],
                                load_complete_seconds=raw['load_complete_seconds'],load_peak_device_bytes=raw['load_peak_device_bytes'],
                                next_update_complete_seconds=raw['next_update_complete_seconds'],next_update_peak_device_bytes=raw['next_update_peak_device_bytes'],
                                next_update_input_positions_per_second=raw['next_update_input_positions_per_second'],
                                loaded_state_position=408,restored_state_position=544,optimizer_step_before=3,optimizer_step_after=4,
                                optimizer_parameter_tensors=before['state_parameter_tensors'],sampled_scalars=update_sample['sampled_scalars'],
                                changed_sampled_scalars=update_sample['changed_scalars'],finite=True,private_state_detached=True))
    paired=next(row for row in checkpoint_summary if row['checkpoint_path'] == checkpoint['path'])
    paired['checkpoint_file_as_recorded_by_restore']=checkpoint
    paired['checkpoint_hash_record_source']=result_id

assert len(continuous)==12 and len(restores)==4 and len(checkpoint_summary)==4 and len(IDENTITIES)==50
assert len({child['pid'] for run in runs if 'children' in run for child in run['children']}) == 12
for label in ('HB 0.5B','HB 1B'):
    members=[row['record'] for row in continuous if row['instance'] == label]
    assert len({row['initial_parameters_sha256'] for row in members}) == len({row['input_sha256'] for row in members}) == 1
    fresh=[row['record'] for row in restores if row['instance']==label]
    assert len({row['input_sha256'] for row in fresh}) == 1 and fresh[0]['input_sha256'] != members[0]['input_sha256']

checkpoint_summary.sort(key=lambda row:(row['instance'],row['precision']=='bf16'))
restore_summary.sort(key=lambda row:(row['instance'],row['precision']=='bf16'))
evidence=dict(schema='tide-capacity-continuous-evidence-v1',date='2026-11-18',math_revision=MATH,
    continuous_source=CONTINUOUS_SOURCE,continuous_harness_source=CONTINUOUS_SOURCE,
    fresh_restore_source=RESTORE_SOURCE,fresh_restore_harness_source=RESTORE_SOURCE,
    reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    scope='read-only consolidation of twelve completed continuous children and four fresh-process restoration observations; only JSON files opened',
    qualification=dict(continuous_finite_execution='both model sizes, both precisions, three modes, three segments',
                       paired_checkpoint_resume='four train witnesses within original thresholds; comparison is same precision and two owners in one process',
                       fresh_process_restore='four single-owner load/first-update observations with finite/storage/step/detach checks; no paired trajectory equivalence',
                       cross_precision_numerical_equivalence=False,operator_dtype_qualification=False,steady_state_performance_qualification=False),
    workload=dict(batch=1,prefill=128,prefill_chunk=64,decode=8,input_positions_per_segment=136,continuous_segments=3,warmup=0,
                  continuous_positions=[136,272,408],paired_checkpoint_resume_input_positions_per_owner=64,
                  paired_checkpoint_resume_positions=[408,472],fresh_restore_input_positions=136,fresh_restore_positions=[408,544],
                  topology='both are Settle, zero flush output slots; 0.5B and 1B have different topology/width, not a pure size sweep'),
    timing=dict(continuous='per-record complete_seconds includes selected-mode entire 136-input stream, and train adds full within-segment BPTT, loss, backward, AdamW, synchronization and state detach; first segment has no warmup',
                boundaries='grad_forward retains graph within a segment and detaches between observations; train detaches at optimizer boundaries; subsequent segments continue private state',
                decode='recorded stable_decode drops the first of eight decode calls; throughput is 7/sum(remaining seconds), median is separate; not a statistical stationarity claim',
                aggregation='three complete times are sequential stateful segments, not IID repetitions; median is descriptive only; displayed memory maximum is max of three reported peaks',
                fresh_restore='one cold first update per independent process; complete load/update includes phase allocator probes/bookkeeping, excludes on-disk records, checkpoint SHA pass, input setup and finite/storage validation',
                checkpoint_resume='save_seconds/load_seconds belong to the in-process paired witness; no timed paired next-update sample recorded',
                filesystem_cache='uncontrolled; fresh process does not imply cold filesystem cache'),
    checkpoint_identity_scope='SHA256 and stat values are inherited from completed restore JSON after its measured load; expected hash arguments were null; this evidence pass neither opens nor stats checkpoint files',
    paired_input_hash_scope='continuous records hash the full 136-input tensor; the 64-input paired witness has no separate input hash field',
    precision_scope='declared FP32/BF16 policies are preserved; fresh records confirm actual FP32 parameters/gradients/AdamW storage and retained state dtypes; timed operator auditing is explicitly absent',
    resource_scope='cooperative exclusive queue for each job, external jobs uncontrolled; four CPU threads per process; actual physical mapping and host load retained',
    validation=dict(json_identity_files=50,continuous_children=12,continuous_segments=36,paired_checkpoint_witnesses=4,paired_checks=32,fresh_restore_observations=4,
                    checkpoints_read_or_statted=0,all_continuous_children_completed=True,all_paired_checks_within_original_threshold=True,
                    all_fresh_restore_checks_completed_finite=True,
                    checks=['programme result/config hashes and frozen math/source identity','six durable service and queue terminal states/resource mappings',
                            'all segment input/call counts, clock progression, prefill/decode arithmetic and finite flags','original eight checkpoint thresholds and mismatch sets',
                            'checkpoint path/recorded byte identity and before/after stat fields only','fresh optimizer state names/steps 3->4/groups/storage',
                            'fresh actual FP32 parameter/gradient/moment bytes','fresh finite scans, detached state and sampled updates',
                            'fresh complete intervals versus phases, combined peaks and input throughput','within-model initial parameter/input hash pairing']),
    runs=runs,verified_json_files=list(IDENTITIES.values()),continuous_summary=summary,
    paired_checkpoint_resume_summary=checkpoint_summary,fresh_process_restore_summary=restore_summary,
    raw_continuous_records=continuous,raw_fresh_restore_records=restores)
atomic('docs/evidence/capacity-continuous-20261118.json',evidence)

lines=['# 大实例连续运行与恢复 · 2026-11-18','',
'0.5B/1B 在 FP32、BF16 的推理、含梯度前向及完整训练共 12 个子任务、36 个连续段均完成。四组训练 checkpoint 的成对续接检查均在原阈值内；',
'另外四个独立新进程完成加载和第一次真实更新。两组恢复证据的输入、终态和计时边界不同，分别列出。',
'全部原始分段字段、阈值、命令、哈希及资源记录保留于 [JSON 证据](capacity-continuous-20261118.json)。本轮只读 50 个 JSON，没有打开或 stat 大 checkpoint，也没有启动实验。','',
'连续运行/成对检查源码与设施：`'+CONTINUOUS_SOURCE+'`。新进程恢复源码与设施：`'+RESTORE_SOURCE+'`。',
'两者数学均为 `'+MATH+'`。模型实际唯一参数分别为 494,110,704 和 976,768,720。','',
'## 三段连续执行','',
'每段 B1，prefill128 由两个 chunk64 执行，随后真实历史 decode8，合计 136 个输入；三段时钟为 136→272→408。',
'本次两个规模均为 Settle，flush 秒数/额外输出槽均为零，不能将这份结果扩展为 Timed flush 资格。',
'预热为零，第一段可能包含首次调用成本；后两段延续前段私有状态。三段是有序连续观察，不是独立重复或长期稳态样本。','',
'`grad_forward` 在段内保留完整图、段间 detach，没有反向与参数更新。`train` 在段内完整 BPTT，整段时间包括',
'prefill/decode、返回结果与 loss、反向、完整 AdamW、同步及更新后 detach。输入传输、构造、checkpoint 和检查不计入三段完整时间。',
'表中峰值直接对应三段原始 `peak_device_bytes`；CPU RSS 是进程生命周期高水位，另存 JSON，不能混同这些 NPU 峰值。','',
'| 模型 | 精度 | 模式 | 三段完整时间 s | 三段 NPU 峰值 GiB |',
'|---|---|---|---|---|']
for row in summary:
    lines.append('| '+' | '.join([row['instance'],row['precision'].upper(),row['mode'],' / '.join(f'{v:.3f}' for v in row['complete_seconds']),' / '.join(f'{v/2**30:.3f}' for v in row['peak_device_bytes'])])+' |')
lines += ['',
'训练最大已报峰值：0.5B 为 FP32 25.211 / BF16 18.969 GiB；1B 为 40.299 / 30.366 GiB。',
'这些观察中 BF16 省内存，但没有表现为更短的完整训练时间。两种精度的活动轨迹可能随更新分化，不能宣称严格同轨迹加速。',
'0.5B 与 1B 的拓扑和宽度不同，较大参数模型在这里更快不构成一般规模规律。','',
'原记录还保留每个 prefill/decode 调用、首个主干输出、输入窗口、flush、反向、优化器和完整时间，以及私有状态存储量。',
'`stable_decode` 字段仅按脚本规则排除八个 decode 中的第一个：中位数与 `7/其余七次秒数总和` 吞吐分别报告，不能把该名字当作统计稳定证明。',
'首主干输出不是含任务头的语言模型 TTFT。','',
'## 原进程内 checkpoint 成对续接检查','',
'只对四个 `train` 子任务执行：在三次更新后的时钟408保存并加载，原 owner 与恢复 owner 分别对 **64 个输入**做下一前向/反向/AdamW，',
'两个终态均为472。这里没有记录该成对下一步的性能时间，`save_seconds`/`load_seconds` 仅属于原检查的保存/加载边界。',
'原输入哈希对应完整136输入张量，64输入 witness 没有单独哈希字段；完整输入哈希和冻结源码仍保留。','',
'| 检查 | 原 atol | 原 rtol | 四组结果 |',
'|---|---:|---:|---|',
'| saved_data / next_forward / next_state | 0 | 0 | 全部 exact，最大绝对差 0 |',
'| input_gradients | 1e-6 | 1e-2 | 全部 exact，最大绝对差 0 |',
'| parameter_gradients | 1e-6 | 2e-3 | 均非 exact，最大绝对差 3.7253e-9 |',
'| updated_parameters | 1e-6 | 2e-3 | 均非 exact，最大绝对差 2.9104e-11 |',
'| updated_optimizer | 1e-6 | 2e-3 | 均非 exact，最大绝对差 4.6566e-10 |',
'| updated_state | 1e-6 | 2e-3 | 全部 exact，最大绝对差 0 |','',
'全部32项检查为 finite、within_threshold，metadata/support/numerical failure 列表为空；两 owner 的 loss 相同。',
'参数梯度/参数更新/优化器的非零差异未抹去、未改阈值，也未改称 exact。JSON 保留各组 max_abs、max_scaled_error、浮点张量数与原阈值。','',
'| 模型 / 精度 | 保存 s | 原检查加载 s | checkpoint GiB |',
'|---|---:|---:|---:|']
for row in checkpoint_summary:
    lines.append('| '+' | '.join([row['instance']+' / '+row['precision'].upper(),f"{row['save_seconds']:.3f}",f"{row['same_process_load_seconds']:.3f}",f"{row['checkpoint_bytes']/2**30:.3f}"])+' |')
lines += ['',
'## 新进程加载及首个真实更新','',
'每个观察只有一个恢复 owner，无模型/更新预热。恢复原 checkpoint 的时钟408、AdamW step3，然后执行',
'**prefill128/chunk64 + decode8，共136输入**，终态544、step4。它使用新输入哈希，既不是上面的64输入成对检查，也不是同一条后续轨迹。','',
'| 模型 / 精度 | 完整加载 s | 首次完整更新 s | 加载峰值 GiB | 更新峰值 GiB | 更新输入位置/s |',
'|---|---:|---:|---:|---:|---:|']
for row in restore_summary:
    lines.append('| '+' | '.join([row['instance']+' / '+row['precision'].upper(),f"{row['load_complete_seconds']:.3f}",f"{row['next_update_complete_seconds']:.3f}",f"{row['load_peak_device_bytes']['npu:0']/2**30:.3f}",f"{row['next_update_peak_device_bytes']['npu:0']/2**30:.3f}",f"{row['next_update_input_positions_per_second']:.3f}"])+' |')
lines += ['',
'加载完整时间包括 model/state 与优化器恢复及同步；更新包括 zero_grad、前向/readout/loss、反向、AdamW、detach 和同步。',
'两段完整时间包含各阶段 allocator 探针和 Python bookkeeping，阶段秒数则不包含其后内存读取/重置。',
'模块/设备上下文建立、输入生成/传输/哈希、checkpoint SHA 扫描、finite/storage/slot/sample 检查和磁盘记录写入不在上述完整加载/更新时间内。',
'原始分阶段峰值与合并峰值均在 JSON；文件系统缓存未控制，新进程不等于冷文件缓存。每格只有一次观察，不是稳态吞吐估计。','',
'加载及更新后的参数、优化器和私有状态 finite；更新后梯度、输出 finite，私有状态已 detach。',
'0.5B 的860个、1B 的432个优化器参数槽全部 step3→4，增量错误列表为空，原 AdamW 组配置恢复并保留。',
'实际参数、梯度、一二阶矩及 step 张量均为 FP32；BF16 私有状态混用 BF16/FP32/int64。',
'每个 Parameter 前最多16个标量的更新抽样分别改变 13,756/13,760（0.5B FP32）、13,759/13,760（0.5B BF16）、',
'6,910/6,912（1B FP32）、6,912/6,912（1B BF16）。这只验证抽样发生更新，不是完整轨迹比较。',
'计时恢复过程未加入算子 dtype 诊断；本页不据此扩大 BF16 算子或跨精度数值资格。','',
'## 哈希、执行身份与复现','',
'四份 checkpoint SHA 来自各新进程在加载后完成的流式哈希记录；四次 `expected_checkpoint_sha256` 实参均为 null。',
'加载前后 size/mtime/device/inode 记录一致，且字节数与原保存记录一致；本轮继承这些 SHA，没有重读 checkpoint 或宣称进行了预先期望哈希对照。',
'`checkpoint_source` 是调用方提供的生产源码身份，checkpoint 格式本身没有编码源码提交。','']
for row in restore_summary:
    lines.append('- '+row['instance']+' '+row['precision'].upper()+'：`'+row['checkpoint_file_as_recorded']['sha256']+'`。')
lines += ['',
'全部作业使用协作队列独占准入，记录为 `exclusive_cooperative_queue; external jobs not controlled`，不能据此假定整机无争用。',
'连续作业在 UTC07:30–07:40、新恢复作业在07:40–07:42执行，候选前后 host load、服务终态与 PID 保留。',
'每进程四线程；Torch 标签 `2.12.0+cpu`，实际设备为 NPU，模块 `cann/9.2.0:ascend/dev-workspace-9.2.0`。','',
'| 作业 | 物理 NPU | CPU affinity | 外层时限 s / 采样 RSS GiB |',
'|---|---:|---|---|']
for run in runs:
    args=run['launch_arguments']
    lines.append('| '+run['name']+' | '+str(run['queue']['physical_id'])+' | '+args['affinity']+' | '+str(args['seconds'])+' / '+str(args['rss_gib'])+' |')
lines += ['',
'连续作业单子进程限时900秒，外层队列等待上限600秒；新恢复单卡 allocator 预算52 GiB，队列等待600秒。',
'RSS 是采样进程树监督，非 cgroup 硬内存；连续原记录没有显式单卡 allocator 预算字段，不补造预算。','',
'仅复核 JSON 身份，不接触 checkpoint：','',
'```sh',
"python - <<'PY'",
'from pathlib import Path',
'import hashlib, json',
"record = json.loads(Path('docs/evidence/capacity-continuous-20261118.json').read_text())",
"for item in record['verified_json_files']:",
"    path = Path(item['path'])",
"    assert path.suffix == '.json'",
'    payload = path.read_bytes()',
"    assert len(payload) == item['bytes']",
"    assert hashlib.sha256(payload).hexdigest() == item['sha256']",
"print('verified', len(record['verified_json_files']), 'JSON files; checkpoint payloads untouched')",
'PY',
'```','',
'若需要新实验，原始冻结源码与精确命令在 JSON 的 `runs[].launch_arguments`，连续子进程另保留在 `runs[].children`。',
'使用新输出身份并按当时资源情况准入。源码身份、两精度输入/初始参数哈希、checkpoint 哈希和所有原阈值都应继续保留；',
'本页的有限连续运行、同精度恢复检查和单次恢复性能，均不替代长期训练质量或其他配置的数学/数值资格。','']
write_text_atomic('docs/evidence/capacity-continuous-20261118.md','\n'.join(lines))
print(json.dumps(evidence['validation'],indent=2))
print('JSON bytes',Path('docs/evidence/capacity-continuous-20261118.json').stat().st_size)
print('Markdown bytes',Path('docs/evidence/capacity-continuous-20261118.md').stat().st_size)
