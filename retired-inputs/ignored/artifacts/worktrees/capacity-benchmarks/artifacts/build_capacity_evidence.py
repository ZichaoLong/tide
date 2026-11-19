"""Read-only consolidation of six completed v2 searches; no model imports."""
from pathlib import Path
import json
import math
import statistics
import subprocess

from benchmarks.references import file_identity
from benchmarks.search_report import render, verify_metric_files, write_text_atomic
from scripts.job import atomic

ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts')
MATH = '4f5b9f20a4030d68ee66ae2d26e34659a5e37234'
SOURCE = 'f486a56ed7b6f68e2aff2e7100673bf2a2fe20be'
PATHS = sorted(ROOT.glob('capacity-*-v2/results/search.json'))
assert len(PATHS) == 6
LABELS = {'capacity-fck-budget-bf16-v2': 'FC-K', 'capacity-fck-length-fp32-v2': 'FC-K',
          'capacity-fcs-length-bf16-v2': 'FC-S', 'capacity-hb1b-length-fp32-v2': 'HB 1B',
          'capacity-settle-length-fp32-v2': 'HB Settle', 'capacity-timed-length-bf16-v2': 'HB Timed'}

runs, raw_rows, common_rows, capacity_rows, audit_rows, unstable, records = [], [], [], [], [], [], []
verified = 0
matched = 0
selection_checked = set()
parameter_errors = []
software = None
policies = {}


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def select_check(match):
    """Enumerate all legal widths which can improve the recorded parameter error."""
    global matched
    spec, target = match['constructor'], match['targets']
    d, blocks, ffn = spec['d'], spec['layers'], spec['ffn_width']
    actual = blocks * (4*d*d + 3*d*ffn + 2*d) + d
    assert actual == match['actual']['parameters']
    assert blocks == target['blocks'] == match['actual']['blocks']
    assert d >= 32 and d % 16 == 0 and ffn % 8 == 0 and 2*d <= ffn <= 6*d
    assert spec['head_dim'] % 2 == 0 and d % spec['head_dim'] == 0 and spec['window'] == 128
    close(match['relative_deviation']['parameters'], (actual-target['parameters'])/target['parameters'])
    assert match['relative_deviation']['blocks'] == 0
    key = (blocks, target['parameters'], d, ffn)
    if key not in selection_checked:
        ceiling = target['parameters'] + abs(actual-target['parameters'])
        best = None
        candidate_d = 32
        while blocks*(10*candidate_d*candidate_d+2*candidate_d)+candidate_d <= ceiling or candidate_d <= d:
            for candidate_ffn in range(2*candidate_d, 6*candidate_d+1, 8):
                count = blocks*(4*candidate_d*candidate_d+3*candidate_d*candidate_ffn+2*candidate_d)+candidate_d
                candidate = (abs(count-target['parameters']), abs(candidate_ffn/candidate_d-4), candidate_d, candidate_ffn)
                if best is None or candidate < best:
                    best = candidate
            candidate_d += 16
        assert best[2:] == (d, ffn), (key, best)
        selection_checked.add(key)
    matched += 1
    parameter_errors.append(abs(match['relative_deviation']['parameters']))


def compact_audit(a):
    return {k: a[k] for k in ('parameters', 'definition', 'deduplication', 'input_positions',
                              'sample_time_slots', 'logical_axis', 'work', 'work_per_input_position',
                              'query_packing', 'node_matrices', 'matrix_cost_definition')} | {
        'effective_touched': {k: a['effective_touched'][k] for k in ('sum', 'mean', 'min', 'max')}}


for path in PATHS:
    record = json.loads(path.read_text())
    records.append((file_identity(path), record))
    count = verify_metric_files(record)
    verified += count
    run = path.parent.parent
    name = run.name
    args = record['arguments']
    assert record['state'] == 'completed'
    assert record['source'] == record['harness_source'] == SOURCE
    assert args['mode'] == 'train' and args['activity'] == 'natural' and args['steps'] == 3 and args['warmup'] == 1
    assert args['threads'] == 4 and args['seed'] == 1107 and args['optimizer_foreach']
    assert args['device_budget_gib'] == 48 and args['rss_gib'] == 112
    assert args['minimum_gain'] == .03 and args['flat_patience'] == 2 and args['maximum_noise'] == .25
    assert all(line['stop_reason'] == 'candidate_limit' for line in record['lines'])
    assert all(child['state'] == 'completed' for child in record['children'])
    assert len(set(child['pid'] for child in record['children'])) == len(record['children'])
    assert file_identity(record['config_file']['path']) == record['config_file']
    upstream_path = run/'source/docs/upstream.json'
    assert json.loads(upstream_path.read_text())['revision'] == MATH
    launch, status, queue, env = (json.loads((run/(kind+'.json')).read_text()) for kind in ('launch','status','queue','environment'))
    assert launch['source'] == status['source'] == SOURCE and status['state'] == 'passed' and queue['status'] == 'completed'
    assert status['exit_code'] == queue['return_code'] == 0
    assert str(queue['physical_id']) == record['physical_devices']
    assert launch['arguments']['affinity'] == f"{record['cpu_affinity'][0]}-{record['cpu_affinity'][-1]}"
    audits = {}
    for audit in record['audits']:
        raw = json.loads(Path(audit['result_file']['path']).read_text())
        assert raw['source'] == SOURCE and raw['state'] == 'audit_complete'
        shape = (raw['arguments']['batch'],raw['arguments']['length'])
        audits[shape] = raw
        a = raw['initial_audits']['train']
        assert a['input_positions'] == shape[0]*shape[1]
        close(a['effective_touched']['sum']/a['sample_time_slots'], a['effective_touched']['mean'])
        audit_rows.append(dict(run=name, batch=shape[0], length=shape[1], result_file=audit['result_file'],
                               initial_parameter_sha256=raw['initial_tide_parameter_sha256'],
                               input_sha256=raw['audit_input_sha256'], topology=raw['topology'],
                               audit=compact_audit(a), initial_runtime=raw['initial_runtime_audits']['train']))
    frozen_match = (audits[(args['batch'],args['length'])]['joint_matches_by_mode']['train']
                    if args['table'] == 'capacity' else None)
    children = {child['identity']: child for child in record['children']}
    run_rows = []
    for line in record['lines']:
        for row in line['observations']:
            raw = json.loads(Path(row['result_file']['path']).read_text())
            obs = raw['observations'][0]
            child = children[row['child']]
            assert child['result_sha256'] == row['result_file']['sha256']
            assert raw['source'] == raw['harness_source'] == SOURCE and raw['state'] == 'completed_finite'
            assert obs['state'] == 'completed_finite' and row['admitted']
            assert obs['numerical_equivalence_qualification'] is False
            assert raw['physical_devices'] == record['physical_devices']
            assert obs['cpu_affinity'] == record['cpu_affinity'] and obs['torch_threads'] == 4
            assert obs['seconds'] == row['seconds'] and len(obs['seconds']) == 3
            assert all(math.isfinite(value) for value in obs['seconds']+obs['losses'])
            median = statistics.median(obs['seconds'])
            close(row['median_seconds'],median)
            close(row['input_positions_per_second'],row['batch']*row['length']/median)
            noise = (max(obs['seconds'])-min(obs['seconds']))/median
            close(row['sample_range_over_median'],noise)
            assert row['stable'] == (noise <= args['maximum_noise'])
            assert row['timed_peak_device_bytes'] == obs['peak_device_bytes'] == obs['device_memory']['peak_allocated_bytes']
            assert row['parameters'] == obs['parameters']
            assert row['initial_parameter_sha256'] == obs['initial_parameter_sha256'] and row['input_sha256'] == obs['input_sha256']
            assert row['actual_storage'] == obs['actual_storage']
            for group in obs['actual_storage'].values():
                assert all(key.endswith(':torch.float32') for key in group)
            observed_software = {k:raw[k] for k in ('torch','torch_npu','python','executable','loaded_modules','allow_hf32')}
            if software is None:
                software = observed_software
            assert observed_software == software
            policies[args['precision']] = raw['precision']
            reference = audits[(row['batch'],row['length'])]
            assert raw['initial_audits']['train'] == reference['initial_audits']['train']
            if row['subject'] == 'tide':
                assert obs['initial_parameter_sha256'] == reference['initial_tide_parameter_sha256']
                assert obs['input_sha256'] == reference['audit_input_sha256']
                assert obs['post_timing_audit']['work'] == row['activity']['post_update_work']
                runtime = raw['initial_runtime_audits']['train']
            else:
                match = row['joint_match']
                assert match == obs['spec']['joint_match']
                expected = (frozen_match if frozen_match is not None else reference['joint_matches_by_mode']['train'])[row['subject'][-1]]
                assert match == expected
                a = (audits[(args['batch'],args['length'])] if frozen_match else reference)['initial_audits']['train']
                target = a['parameters']['unique_parameters'] if row['subject'][-1] in 'AC' else a['effective_touched']['mean']
                close(match['targets']['parameters'],target)
                assert match['targets']['blocks'] == reference['topology']['nodes' if row['subject'][-1] in 'AB' else 'layers']
                select_check(match)
                assert match['actual']['parameters'] == row['parameters']
                runtime = obs['post_timing_runtime_audit']
            if args['precision'] == 'bf16':
                assert runtime['bf16_matrix_observed']
            raw_row = {key: row[key] for key in ('subject','batch','length','result_file','reference_record','calibration_record',
                        'seconds','initial_parameter_sha256','input_sha256','parameters','joint_match','actual_storage')}
            raw_row.update(run=name, identity=row['child'], losses=obs['losses'],
                           child={key:child[key] for key in ('pid','started','ended','elapsed_seconds','sampled_process_tree_peak_rss_bytes','exit_code')},
                           sample_device_memory=obs['sample_device_memory'], host_load_before=obs['host_load_before'],
                           host_load_after=obs['host_load_after'], runtime_operator_outputs=sorted({value for operator in runtime['operators'] for value in operator['outputs']}),
                           bf16_matrix_observed=runtime['bf16_matrix_observed'])
            if row['subject'] == 'tide':
                raw_row['post_update_audit'] = compact_audit(obs['post_timing_audit'])
            raw_rows.append(raw_row)
            if not row['stable']:
                unstable.append(dict(run=name,subject=row['subject'],batch=row['batch'],length=row['length'],
                                     sample_range_over_median=noise,seconds=obs['seconds'],result_file=row['result_file']))
            run_rows.append(row)
    for point in record['common_shape_table']:
        tide = next(row for row in point['subjects'] if row['subject'] == 'tide')
        lookup = {row['subject']:row for row in point['subjects']}
        comparisons = {}
        for ratio in point['comparisons']:
            baseline = lookup['transformer_'+ratio['joint_class']]
            close(ratio['tide_over_transformer_time'],tide['median_seconds']/baseline['median_seconds'])
            close(ratio['tide_over_transformer_peak'],tide['timed_peak_device_bytes']/baseline['timed_peak_device_bytes'])
            assert ratio['both_stable'] == (tide['stable'] and baseline['stable'])
            comparisons[ratio['joint_class']] = {key:ratio[key] for key in ('tide_over_transformer_time','tide_over_transformer_peak','both_stable')}
        common_rows.append(dict(run=name,instance=LABELS[name],precision=args['precision'],batch=point['batch'],length=point['length'],
                                tide_median_seconds=tide['median_seconds'],input_positions_per_second=tide['input_positions_per_second'],
                                tide_timed_peak_device_bytes=tide['timed_peak_device_bytes'], comparisons=comparisons,
                                stable_subjects=[row['subject'] for row in point['subjects'] if row['stable']],
                                subjects=[{key:row[key] for key in ('subject','child','median_seconds','input_positions_per_second','timed_peak_device_bytes','stable','sample_range_over_median')} for row in point['subjects']]))
    for item in record['capacity_table']:
        best = item['best']
        candidates = [row for row in run_rows if row['subject'] == item['subject'] and row['stable'] and row['admitted']]
        assert best == max(candidates,key=lambda row:row['input_positions_per_second'])
        capacity_rows.append(dict(run=name,instance=LABELS[name],precision=args['precision'],subject=item['subject'],fixed_length=item['fixed_length'],
                                  best_observed_batch=best['batch'],median_seconds=best['median_seconds'],
                                  input_positions_per_second=best['input_positions_per_second'],timed_peak_device_bytes=best['timed_peak_device_bytes'],
                                  initial_tide_active_minus_calibration=best['activity']['initial_minus_calibration'],
                                  stop_reason=item['stop_reason'],result_file=best['result_file'],
                                  observations=[{key:row[key] for key in ('batch','child','median_seconds','input_positions_per_second','timed_peak_device_bytes','stable','sample_range_over_median')} for row in candidates]))
    runs.append(dict(name=name,label=LABELS[name],search_file=file_identity(path),arguments=args,source=record['source'],harness_source=record['harness_source'],
                     source_config=record['config_file'],frozen_config=record['frozen_config'],config=json.loads(Path(record['frozen_config']['path']).read_text()),
                     upstream_manifest=file_identity(upstream_path),support_files={kind:file_identity(run/(kind+'.json')) for kind in ('launch','status','queue','environment')},
                     state=record['state'],child_states=record['child_states'],axis_values=record['axis_values'],
                     started=record['started'],ended=record['ended'],physical_devices=queue['physical_ids'],cpu_affinity=record['cpu_affinity'],
                     resource_identity=record['resource_identity'],launcher_device_policy=launch['device_policy'],
                     service=launch['unit'],outer_limits={key:launch['arguments'][key] for key in ('seconds','rss_gib','queue_seconds')},
                     outer_peak_rss_bytes=status['peak_rss_bytes'],outer_elapsed_seconds=status['elapsed'],
                     module_origins=env['module_origins'],calibration_record=record.get('calibration_record'),
                     stop_reasons=[line['stop_reason'] for line in record['lines']],verified_metric_files=count,
                     launch_arguments=launch['arguments'],
                     child_command_scope='exact independent child commands and PID/log identities retained in the hashed search_file children array'))

assert len(raw_rows) == 95 and len(audit_rows) == 19 and len(common_rows) == 15 and len(capacity_rows) == 5
assert verified == 120 and len(unstable) == 6 and matched == 76
assert len({row['child']['pid'] for row in raw_rows}) == len(raw_rows)
assert all(row['tide_median_seconds'] > 0 for row in common_rows)
write_text_atomic('artifacts/capacity-search-rendered-20261118.md',render(records))

evidence = dict(schema='tide-capacity-search-evidence-v1',date='2026-11-18',source=SOURCE,harness_source=SOURCE,math_revision=MATH,
    reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    scope='six existing finite NPU prefill-from-zero training searches; read-only evidence consolidation; no experiments rerun',
    qualifications=dict(finite_training_observations=True,numerical_equivalence=False,maximum_capacity_found=False,
                        continuous_training=False,fresh_process_restoration=False,model_quality_equivalence=False),
    timing=dict(kind='prefill_from_zero',includes=['zero_grad','preparation','packing','forward','readout','required Timed flush','scalar loss','backward','AdamW foreach','device synchronization','scalar loss extraction'],
                excludes=['model construction','input transfer','initial audits','post-update audits','checkpoint','JSON writes','sample memory probes'],
                reset_state_each_sample=True,warmup_updates=1,timed_updates=3,post_update_audit_after_updates=4),
    statistics=dict(central='median of three raw seconds',throughput='actual B*L input positions / median seconds',
                    stability='sample range / median <= 0.25; finite heuristic, not confidence interval',
                    search_stop='predeclared doubling axis candidates exhausted in every line; no OOM/budget/time/plateau stop',
                    plateau_rule=dict(minimum_gain=.03,consecutive_stable_points=2),
                    performance_ratio='Tide / Transformer elapsed time at the same B/L; > 1 means Tide takes longer'),
    matching=dict(classes={'A':['nodes','total'],'B':['nodes','effective_touched'],'C':['depth','total'],'D':['depth','effective_touched']},
                  parameter_scope='Transformer main blocks and final RMSNorm, without embedding/head/artificial unused parameters',
                  formula='blocks * (4*d*d + 3*d*ffn_width + 2*d) + d',
                  selection_rule='exact block count; nearest parameters over d=16*j>=32, FFN=8*j in [2d,6d]; ties prefer FFN=4d',
                  common='per-shape initial audit rematches A-D before updates',
                  capacity='constructors fixed from B32/L128 calibration; no rematching at higher B or after updates',
                  active_definition='unique Parameter scalars in semantically effective owner/group/source slices; not nonzero gradients or physical loads',
                  active_deduplication='sample/input step for Settle/FC; sample/logical tick for Timed with repeated ticks accumulated; no output causal attribution',
                  fixed_history='Transformer window128; Tide logical KV128*period unchanged; L changes stream/BPTT extent',
                  max_absolute_parameter_relative_deviation=max(parameter_errors),all_block_targets_exact=True),
    memory=dict(unit='bytes; displayed GiB=2**30 bytes',peak='per-process cumulative allocated peak since post-warmup reset, before intrusive post-audit',
                device_budget_gib=48,child_sampled_rss_budget_gib=112,
                enforcement='allocator budget checked after synchronized construction/audit/warmup/sample, plus sampled process-tree RSS; no hard reservation',
                diagnostics='initial semantic/runtime audits and post-update audits excluded from timing; their memory checks still affect admission'),
    software=software,precision_policies=policies,
    resource_scope='six overlapping searches, distinct queue-admitted physical NPUs and disjoint four-CPU affinity; cooperative exclusive queue does not control external jobs or host contention',
    input_pairing='same seed1107; CPU feature-wise input seed1108; common sample/time/feature prefixes across widths, with actual hashes retained',
    validation=dict(search_records=6,verified_referenced_metric_config_files=120,search_identity_files=6,initial_audit_children=19,timed_children=95,
                    all_children_completed=114,timed_samples=285,stable_timed_observations=89,unstable_timed_observations=6,stable_tide_observations=19,
                    verified_transformer_observations=76,exhaustively_checked_distinct_parameter_selections=len(selection_checked),
                    checks=['raw SHA256 and byte identity','source/harness and math pin consistency','launcher/queue terminal and resource mapping',
                            'three finite samples/losses, median, input throughput, range/median and stability','normal allocated peak versus worker record',
                            'initial Tide parameter/input hashes versus per-shape audit','FP32 parameter/gradient/moment storage keys and BF16 operator observation',
                            'all A-D targets, counts, deviations and legal nearest constructors','all common time/peak ratios','capacity fixed constructors and best stable observations']),
    known_summary_omission='v2 search activity initial_packing/post_update_packing are null due to pre-fix summary key; query_packing is recovered from the unchanged raw audit records below',
    runs=runs,common_shape_table=common_rows,independent_capacity_table=capacity_rows,unstable_observations=unstable,
    raw_observations=raw_rows,initial_workload_audits=audit_rows)
atomic('docs/evidence/capacity-search-20261118.json',evidence)

lines = ['# 有界吞吐与规模搜索 · 2026-11-18','',
'六组既有 v2 搜索共完成 19 个初态审计和 95 个训练计时子进程，保留 285 个原始样本。',
'下表覆盖较长输入窗口和 FC-K 独立批量搜索；所有搜索都因预设候选耗尽停止，未测得显存或全局吞吐极限。',
'本页由 `benchmarks.search_report` 核验原始引用并生成基础双表，再逐项复核计时、匹配与资源口径；没有重跑实验。',
'完整样本、构造器、活动变化、执行命令及哈希分列于 [JSON 证据](capacity-search-20261118.json)。','',
'源码与测量设施均为 `'+SOURCE+'`；数学固定 `'+MATH+'`。',
'资格仅为这些训练负载有限完成，不扩展为数值等价、持续训练、恢复或模型质量资格。','',
'## 口径与停止边界','',
'- 每个候选使用一个独立进程、一轮预热更新、三次正常训练采样；`train` 包括 zero_grad、准备与 packing、前向/readout、必要的 Timed flush、loss、反向、完整 AdamW foreach、设备同步及标量 loss 读取。每步从零状态 prefill，参数累计更新；后审计位于共四次更新之后。',
'- 模型构造、输入传输、初态/后更新审计、checkpoint、JSON 写入和样本后内存探针不计入上述秒数。峰值为预热后清零、后审计前的累计 allocated peak，包含当时存活的参数、梯度、优化器状态和执行临时量。',
'- 吞吐分母为实际 B×L 输入位置。Transformer 历史窗口固定 128；Tide 既定逻辑 KV 窗口 128×period 不变。扩大 L 是扩大输入流及反向图范围，没有修改历史语义。',
'- 两表均为同精度比较。BF16 是 autocast 矩阵计算；实际参数、梯度、AdamW 一二阶矩和主权重仍为 FP32，没有另设一份主权重。原始独立 runtime audit 记录了 BF16 矩阵输出；FP32 的 matmul/conv HF32 均关闭。',
'- 每卡 allocated 预算 48 GiB，单子进程树采样 RSS 预算 112 GiB；外层服务 128 GiB/5400 s，队列等待最多 600 s。搜索总限时 4800 s、单子进程 900 s。预算是同步点检查及采样监督，不是硬显存隔离。',
'- 候选按单轴倍增；稳定条件为三样本 `(max−min)/median ≤ 25%`。预设平坦规则是连续两个稳定候选相对最佳吞吐增益不超过 3%；本次所有线都先到 `candidate_limit`，没有 OOM、预算、超时或平坦停止。','',
'## 同 B/L 的四类联合匹配','',
'A/B 固定接近 Tide 节点数的 Transformer block 数，分别匹配总参数/初态实际有效触及量；C/D 同理固定层数/深度。',
'每个 B/L 单独做初态匹配，训练后不重新匹配。所有 block 数目标精确满足，参数相对误差最大 '+f'{max(parameter_errors)*100:.3f}%'+ '；构造器、目标、实际值及偏差均在 JSON。',
'有效触及量统计真实有效 owner/group/source 切片内的唯一 Parameter 标量：FC/Settle 按样本及输入步跨层去重，Timed 按样本及逻辑 tick 跨区域去重。',
'Timed 的均值涵盖输入和必要 flush 的逻辑时刻，重复 tick 的工作累计，另列每实际输入位置的工作；不推断输出因果归属。该口径不等同于非零梯度、物理读取量或 FLOPs。',
'参数数量独立按 `blocks×(4d²+3d×FFN+2d)+d` 复核，并枚举合法宽度检查最近点。',
'表内 A–D 是 **Tide/Transformer 时间比**，大于 1 表示 Tide 耗时更多。`†` 表示该 Transformer 观察超出波动阈值，保留数值但不作为稳定比较。','',
'| 实例 / 精度 | B/L | Tide 秒 | 输入位置/s | Tide 峰值 GiB | A | B | C | D |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for row in common_rows:
    ratios = [f"{row['comparisons'][name]['tide_over_transformer_time']:.3f}"+('' if row['comparisons'][name]['both_stable'] else '†') for name in 'ABCD']
    cells = [row['instance']+' / '+row['precision'].upper(),f"{row['batch']}/{row['length']}",f"{row['tide_median_seconds']:.3f}",f"{row['input_positions_per_second']:.2f}",f"{row['tide_timed_peak_device_bytes']/2**30:.3f}",*ratios]
    lines.append('| '+' | '.join(cells)+' |')
lines += ['',
'HB 1B 的实际唯一参数为 976,768,720，节点数 848、区域层数 5。B1/L128 时，Tide 的 1.763 s 比 A/B 的 848-block 构造耗时少，',
'耗时是 C/D 的 5-block 构造的 13.93/40.02 倍；这两个比较维度不能互相替代。',
'同点初态有效触及均值 277,431,539，四次更新后为 254,190,352；活动均值没有替代参数持有量，也不表示同等 FLOPs。','',
'FC-K B8 的 L128→1024 吞吐从 7,803 增至 18,401 输入位置/s，峰值从 1.129 增至 7.976 GiB。',
'HB Settle B8/L512 已测至 30.515 GiB；HB Timed B4/L256 为 27.386 GiB、153.46 输入位置/s。',
'Timed 在该点的 A/B 时间比约 7.8、C/D 约 145–151，仍有明显差距。单凭这些计时不能把差距分配给算子、调度或反向图；',
'JSON 保留初态/后更新的工作计数、query packing 和矩阵行数，具体热点与执行变体证据另列。','',
'19 个 Tide 观察均满足有限稳定条件；其余六个超差观察为 FC-K FP32 B8/L256 的 C/D、FC-S BF16 B8/L512 的 D、',
'HB Settle FP32 B8/L128 的 C 和 B8/L512 的 D、HB Timed BF16 B4/L128 的 D。原始秒数与波动比保留在 JSON，未筛掉或重测。','',
'## 固定构造器的预算内独立批量搜索','',
'本表为 FC-K BF16 训练，L=128，B 候选为 32、64、128、256。A–D 构造器全部冻结于共同校准点 B32/L128，',
'初态活动目标为 4,194,736；较大 B 的活动变化单独记录。表中是每个模型在已测稳定候选内的最佳吞吐，',
'本次恰好均为 B256。该搜索没有在每个 B 重新匹配，不能替代上表的逐形状联合匹配。','',
'| 模型 | 最佳已测 B | 秒 | 输入位置/s | 峰值 GiB | 停止 |',
'|---|---:|---:|---:|---:|---|']
for row in capacity_rows:
    lines.append('| '+' | '.join([row['subject'],str(row['best_observed_batch']),f"{row['median_seconds']:.3f}",f"{row['input_positions_per_second']:.2f}",f"{row['timed_peak_device_bytes']/2**30:.3f}",row['stop_reason']])+' |')
lines += ['',
'Tide B32→B256 的吞吐从 24,438.54 增至 44,446.54 输入位置/s，峰值从 3.106 增至 23.738 GiB。',
'B128 已达 44,156.77 输入位置/s，B256 只再增 0.656%；仅一个平坦增量不足以触发两点停止规则，下一候选不在该冻结搜索矩阵中。',
'B256 的初态 Tide 有效触及均值 4,203,444，比校准点多 8,708（约 0.208%）；四次更新后为 4,339,062。',
'预算尚有余量不等于更大 B 一定适用，当前证据没有确定最大可行批量。','',
'## 身份、资源与复现','',
'这些作业在 UTC 07:23–07:31 期间重叠运行，各自使用一张协作队列独占准入的 NPU，CPU affinity 互不重叠；',
'记录身份为 `exclusive_cooperative_queue; external jobs not controlled`。这是队列内独占身份，不代表整机无其他任务；',
'原始记录保留每个候选前后 host load，短样本仍受主机争用、串行顺序和时间漂移影响。','',
'运行软件为 PyTorch `2.12.0+cpu` + `torch_npu 2.12.0.post3`，模块 `cann/9.2.0:ascend/dev-workspace-9.2.0`；',
'`+cpu` 是 Torch 包版本标签，实际矩阵输出/设备由运行审计确认是 NPU。私有解释器、模块来源、服务终态、队列映射及精确命令均保留。','',
'| 搜索身份 | 物理 NPU | CPU affinity | 候选范围 |',
'|---|---:|---|---|']
for run in runs:
    args=run['arguments']
    lines.append('| '+run['name']+' | '+','.join(map(str,run['physical_devices']))+' | '+str(run['cpu_affinity'][0])+'–'+str(run['cpu_affinity'][-1])+' | '+args['axis']+' '+','.join(map(str,run['axis_values']))+' |')
lines += ['',
'只读复现汇总，无需设备或 Torch 导入：','',
'```sh',
'python -m benchmarks.search_report \\',
'  --inputs artifacts/capacity-fck-budget-bf16-v2/results/search.json \\',
'  artifacts/capacity-fck-length-fp32-v2/results/search.json \\',
'  artifacts/capacity-fcs-length-bf16-v2/results/search.json \\',
'  artifacts/capacity-hb1b-length-fp32-v2/results/search.json \\',
'  artifacts/capacity-settle-length-fp32-v2/results/search.json \\',
'  artifacts/capacity-timed-length-bf16-v2/results/search.json \\',
'  --output artifacts/NEW_CAPACITY_REPORT.md',
'```','',
'该入口核验配置、审计、校准和计时文件的 SHA256/字节数后生成基础双表，报告稳定候选数量；逐项波动状态以原始 JSON 的 `stable`/`both_stable` 为准。',
'本页另外独立复核 95 个观察的中位数、吞吐、波动、峰值及全部 76 个 Transformer 构造的目标/公式/最近合法点，',
'并核对容量表的固定构造与最佳候选。所有 120 个引用配置/结果文件身份一致；六份搜索文件身份如下。','']
for run in runs:
    lines.append('- ['+run['name']+'](../../artifacts/'+run['name']+'/results/search.json)：`'+run['search_file']['sha256']+'`。')
lines += ['',
'若重新执行实验，使用 JSON 中的冻结源码与 `runs[].launch_arguments`；精确子进程命令留在对应搜索文件的 `children`。重新核对当前资源并选择新输出身份；',
'不能把已有路径当新运行目的地。原始 v2 汇总的 packing 字段存在旧键名遗漏，本 JSON 从未改动的 raw `query_packing` 取回；',
'原始文件及其空字段保留原样。FP32/BF16 的具体数值超差、持续状态训练、恢复后更新和后续优化资格仍以各自证据为准。','']
write_text_atomic('docs/evidence/capacity-search-20261118.md','\n'.join(lines))
print(json.dumps(evidence['validation'],ensure_ascii=False,indent=2))
print('JSON bytes',Path('docs/evidence/capacity-search-20261118.json').stat().st_size)
print('Markdown bytes',Path('docs/evidence/capacity-search-20261118.md').stat().st_size)
