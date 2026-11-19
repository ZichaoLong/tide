"""Read existing qualification JSON only; no model, Torch or device execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
DESTINATION = Path(__file__).resolve().parent
PROFILES = ('light64', 'light128', 'wide')
FLAGS = ('metadata_failures', 'dtype_differences', 'nonfinite', 'support_differences', 'work_differences')


def write_text_atomic(path, text):
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def family(field):
    if field.startswith('root.forward.') or field.startswith('root.segments.'):
        return 'forward_state' if '.state.' in field else 'forward_output'
    if field.startswith('root.gradients.'):
        return 'parameter_gradient'
    if field == 'root.input_gradient':
        return 'input_gradient'
    if field.startswith('root.parameters.'):
        return 'updated_parameter'
    if field.startswith('root.optimizer.'):
        return 'optimizer_state'
    if field.startswith('root.state.'):
        return 'continued_state'
    return 'other'


def reduce_check(phase, check):
    comparison = check['comparison']
    a, b = check['screen_actual'], check['screen_expected']
    threshold = comparison['threshold_failures']
    rows = comparison['largest_errors']
    return dict(phase=phase, exact=comparison['exact'], tolerance=comparison['tolerance'],
        threshold_failure_count=len(threshold), threshold_failure_fields=threshold,
        threshold_failure_families=dict(Counter(family(name) for name in threshold)),
        largest_scaled_error=max([row['scaled_error'] for row in rows] or [0]),
        largest_scaled_rows=rows,
        largest_rows_scope='up to eight rows ranked by scaled error; their max_abs is per tensor, not the global maximum absolute error',
        **{name: comparison[name] for name in FLAGS},
        none_fields_only_actual=sorted(set(a['none_fields'])-set(b['none_fields'])),
        none_fields_only_expected=sorted(set(b['none_fields'])-set(a['none_fields'])),
        discrete_hash_equal=a['discrete_sha256'] == b['discrete_sha256'],
        support_hash_equal=a['support_sha256'] == b['support_sha256'],
        first_returned_differences=check['first_differences'],
        first_scope=check['first_scope'])


memo = dict(schema='reception-npu-paths-readonly-summary-v1',
    generated_at_utc=datetime.now(timezone.utc).isoformat(),
    method='read existing result.json snapshots only; no experiments, tensor replays or device use',
    source_field_scope='original source/source_dirty fields retained; actual package and harness hashes identify measured files; current HEAD is not substituted',
    profiles=[], context={
        'independent_cpu_formula': dict(implementation_commit='6907f3d5', tests='tests/test_reception.py',
            scope='40 CPU FP64 formula/VJP tests across source_residual and all three incoming profiles, separate/fused and structural-deletion versus genuine-zero derivative cases',
            establishes='selected algebra, all input/parameter VJPs and structural-versus-real-zero domains at the tested CPU scope',
            does_not_establish='these integrated NPU trajectories, arbitrary NPU operators, or every later optimizer coordinate'),
        'cpu_decimal_adamw': dict(path=str(ROOT/'artifacts/worktrees/agg-inference-reception/artifacts/reception-fp32-adamw-witness.json'),
            scope='CPU FP32 compact-5-dd4 mixed n2 h1 d64 incoming_light_64 Settle; regional.4.second.bias coordinate 20',
            established='step-2 observed parameter delta 2.0191073417663574e-05 versus Decimal replay 2.0195122628671736e-05; error 4.0492110081613575e-09; step-3 error 4.097399267100066e-09',
            mechanism='small gradient differences can be amplified by AdamW epsilon/normalization and then retained in moments',
            limitation='the present NPU largest update failures often concern receive.q/k left or weight; no scalar replay of those NPU coordinates was performed, so the CPU witness cannot close their causal explanation'),
        'cpu_thread_repeat': dict(path=str(DESTINATION/'restore-threads-b4/report.json'),
            scope='CPU FP32 incoming_light_128 Timed, 4-thread versus 1-thread bounded restore/repeated-VJP diagnosis',
            established='same-owner repeated VJPs can differ without checkpoint loading; exact test thread scope was isolated in 7901ab75 while original failures were retained',
            limitation='this is not proof of the mechanism behind the NPU embedding-gradient repeats'),
    }, interpretations=[
        'Same-profile inference, grad-enabled forward, continued training and restored update are separate claims. A successful candidate-local restore does not erase cross-candidate divergence.',
        'Fixed-start and inference equality do not require subsequent gradients, moments or optimizer trajectories to be bitwise equal.',
        'FP32 attention auto permits gradient-dependent prior representations. This is a plausible site to inspect, but these records do not capture actual per-call dispatch, and no NPU root cause is asserted.',
        'The optimized variant combines compatible region batching, fused QKV/static mask, and SSD doubling. Its differences are not attributable to any one component from this combined comparison.',
        'First returned fields are deterministic lexicographic tree-order witnesses, not earliest execution or causal boundaries. Internal capture-limit records establish no internal first divergence.',
        'Threshold-failure counts count tensor-field occurrences; repeated segments and forward/continued state can repeat related quantities. They are not counts of wrong nodes, scalars or independent bugs.',
        'No new support, integer-time/index, dtype, None-connectivity or nonfinite difference in these records is a finite-scope observation, not a universal correctness proof.'
    ])
all_checks = []
for profile in PROFILES:
    path = ROOT/f'artifacts/agg-paths-{profile}-fp32-v1/paths/result.json'
    content = path.read_bytes()
    original = json.loads(content)
    result = original['result']
    base = result['candidates'][0]
    p = dict(profile=profile, input_file=str(path), input_sha256=hashlib.sha256(content).hexdigest(),
        input_bytes=len(content), input_mtime_ns=path.stat().st_mtime_ns,
        source=original['source'], source_dirty=original['source_dirty'],
        harness_sha256=original['harness_sha256'], package_identity=original['package_identity'],
        physical_devices=original['physical_devices'], resource_identity=original['resource_identity'],
        torch=original['torch'], policy=original['policy'], arguments=original['arguments'],
        outer_state=original['state'], state=result['state'], tolerance=result['tolerance'],
        restore_tolerance=result['restore_tolerance'], boundary=result['boundary'],
        observation=result['observation'], anchor_optimizer=result['anchor_optimizer'],
        resolved_candidates=result['resolved_candidates'], input_hashes=result['input_hashes'],
        fixed_start=result['fixed_start']['sha256'], candidates=[])
    for candidate in result['candidates']:
        checks = [reduce_check(name, value) for name, value in candidate['checks'].items()]
        updates = [dict(index=step['index']+1, work=step['work'],
                        baseline_work_equal=step['work'] == base['updates'][step['index']]['work'],
                        comparison=None if 'vs_baseline' not in step else
                                   reduce_check(f"update.{step['index']+1}", step['vs_baseline']))
                   for step in candidate['updates']]
        restoration = candidate.get('restore', {})
        restore_checks = [reduce_check('restore.'+name, value) for name,value in restoration.get('checks',{}).items()]
        actual_update = restoration.get('optimizer_update', {})
        update_summary = dict(checked=actual_update.get('checked'),
            active_parameter_count=len(actual_update.get('active_original_parameters', [])),
            before_steps=sorted(set(actual_update.get('before', {}).values())),
            after_steps=sorted(set(actual_update.get('after', {}).values())))
        c = dict(name=candidate['name'], state=candidate['state'], config=candidate['config'],
            checks=checks, updates=updates,
            work_by_mode={name: observation['work'] for name, observation in candidate['observations'].items()},
            baseline_work_equal_by_mode={name: observation['work'] == base['observations'][name]['work']
                                        for name,observation in candidate['observations'].items()},
            actual_none_parameters={name: observation.get('actual_none_parameters')
                                    for name, observation in candidate['observations'].items() if name.startswith('vjp')},
            first_internal_witness=candidate.get('first_divergence_witness', dict(state='not_collected',
                reason='the finite checker attempts only its first cross-candidate internal witness; no causal inference from absence')),
            restore=dict(state=restoration.get('state'), actual_update=restoration.get('actual_update'),
                new_process=restoration.get('new_process'), config_exact=restoration.get('config_exact'),
                source_exact=restoration.get('source_exact'), exit_code=restoration.get('exit_code'),
                checkpoint_sha256=restoration.get('checkpoint_sha256'), bundle_sha256=restoration.get('bundle_sha256'),
                optimizer_update=update_summary,
                original_update_work_equal=restoration.get('work') == restoration.get('original_update_work'),
                checks=restore_checks))
        p['candidates'].append(c)
        all_checks += [dict(profile=profile,candidate=c['name'],**check) for check in
                       checks + [step['comparison'] for step in updates if step['comparison'] is not None] + restore_checks]
    memo['profiles'].append(p)

memo['aggregate'] = dict(profile_count=len(memo['profiles']), candidate_count=sum(len(p['candidates']) for p in memo['profiles']),
    candidate_states=dict(Counter(c['state'] for p in memo['profiles'] for c in p['candidates'])),
    restored_states=dict(Counter(c['restore']['state'] for p in memo['profiles'] for c in p['candidates'])),
    comparison_count=len(all_checks), exact_comparison_count=sum(row['exact'] for row in all_checks),
    threshold_failed_comparison_count=sum(bool(row['threshold_failure_count']) for row in all_checks),
    threshold_failed_field_occurrences=sum(row['threshold_failure_count'] for row in all_checks),
    structural_flag_occurrences={flag: sum(len(row[flag]) for row in all_checks) for flag in FLAGS},
    none_mismatch_checks=sum(bool(row['none_fields_only_actual'] or row['none_fields_only_expected']) for row in all_checks),
    unequal_discrete_hash_checks=sum(not row['discrete_hash_equal'] for row in all_checks),
    unequal_support_hash_checks=sum(not row['support_hash_equal'] for row in all_checks),
    all_recorded_mode_work_equal=all(all(c['baseline_work_equal_by_mode'].values()) for p in memo['profiles'] for c in p['candidates']),
    all_recorded_update_work_equal=all(step['baseline_work_equal'] for p in memo['profiles'] for c in p['candidates'] for step in c['updates']),
    internal_witness_states=dict(Counter(c['first_internal_witness']['state'] for p in memo['profiles'] for c in p['candidates'])))
write_text_atomic(DESTINATION/'npu-reception-paths-memo.json', json.dumps(memo, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(memo['aggregate'],ensure_ascii=False,indent=2))

lines = [
    '# NPU Reception 路径有限核对备忘', '',
    '仅只读汇总三份正式 result.json；wide 已终态。没有启动模型、NPU、重跑或扩大捕获。完整原阈值失败字段、返回差异坐标和输入文件哈希见同目录 `npu-reception-paths-memo.json`。', '',
    '## 身份与范围', '',
    '- 三组均为 compact-9 Settle，n=8、h=4、d=128、X、region_shared、FP32、B1/L4、chunk2；有一次真实 anchor 更新，随后三个连续更新，再新进程真实恢复更新。',
    '- baseline 使用单区域、separate QKV、不启用 padding mask，Reception 重算和 SSD prefix 已启用；qkv_mask 只添加 fused QKV 与 padding mask；optimized 再添加 region_batch_size=4 与 SSD doubling。',
    '- 记录源码标签为 `28941994880a277495b315f0fec3213239ef0b49`，三份 `source_dirty=true` 原样保留。实测 Python 包 SHA256 为 `9cd45f321a7d700021d1fb8a0d31a5c55ebde47c8fa248d7866c6a3c3682cf62`；本备忘不以当前 HEAD 替代实测身份。',
    '- 设备映射分别为 light64→物理1、light128→物理7、wide→物理5，进程内均 npu:0；协作队列独占准入，外部任务不受控、共享主机。',
    '- 原阈值统一 `atol=1e-6, rtol=2e-3`。计数是超差张量字段次数，可能重复包含 segment/state，不能当成超差标量数或独立故障数。', '',
    '## 数值状态', '',
    '| Profile | baseline | qkv_mask | optimized | 三候选各自新进程恢复更新 |',
    '|---|---|---|---|---|',
]
for p in memo['profiles']:
    cs=p['candidates'];lines.append('| '+p['profile']+' | '+' | '.join(c['state'] for c in cs)+' | 全部 within_threshold |')
lines += ['',
    '共 99 个比较中 39 个 exact；41 个比较包含原阈值超差，合计 1459 次字段超差。9 个候选均无 nonfinite、dtype、support、整数元数据或 None 连接差异；全部显式参数在 VJP 中有梯度。按模式及各次更新单独核对的 candidates/full/positive_messages/candidate_slots 工作计数也一致。',
    '九次恢复均：保存数据和下一前向 exact；实际更新在原阈值内但非 exact；364 个活动参数的 AdamW 计数真实从 4 到 5。恢复结果只说明各自候选的恢复，不消除候选间训练超差。', '',
    '## 推理与含图前向', '',
    '| Profile | baseline 超差字段数 / 最大 scaled | qkv_mask | optimized |',
    '|---|---:|---:|---:|',
]
for p in memo['profiles']:
    entries=[]
    for c in p['candidates']:
        check=next(v for v in c['checks'] if v['phase']=='inference_vs_grad_forward')
        entries.append(f"{check['threshold_failure_count']} / {check['largest_scaled_error']:.3f}")
    lines.append('| '+p['profile']+' | '+' | '.join(entries)+' |')
lines += ['',
    '这里的失败字段均是 KV keys/values 等浮点状态；未出现活动支持或原始整数时间变化。三组 qkv_mask 的无梯度推理与 baseline exact，但首次带梯度完整 VJP 分别有 3、4、1 个字段超差，全部仍是前向状态字段；初次参数/input 梯度均未超过原阈值。optimized 无梯度推理已有 1、4、2 个字段超差。',
    '同一 owner 的重复 VJP 九组均非 exact、均在原阈值内。baseline/qkv_mask 首个返回差异是 receive.embedding.weight 梯度，optimized 是 describe.encode.bias 梯度；该事实区分重复反向的有限变化与 checkpoint 故障，但不能据此确定具体 NPU 归约算子。', '',
    '## 跨候选三个更新', '',
    '| Profile / candidate | 超差字段数：更新1→2→3 | 第3步最大 scaled 所属字段 | 该张量 max_abs |',
    '|---|---|---|---:|',
]
for p in memo['profiles']:
    for c in p['candidates'][1:]:
        updates=[u['comparison'] for u in c['updates']]
        top=updates[-1]['largest_scaled_rows'][0]
        lines.append(f"| {p['profile']} / {c['name']} | "+' → '.join(str(u['threshold_failure_count']) for u in updates)+
                     f" | {top['tensor']} | {top['max_abs']:.6g} |")
lines += ['',
    '最大 scaled 行主要落在 receive.q/k 的 private left 或 base weight。部分第2/3次更新还出现参数梯度、输入梯度及更广泛的浮点状态超差；完整字段分类与列表保留在 JSON。支持、整数时间和计数仍一致，不能把这些有限数值差异说成已经观察到不同路由。', '',
    '## 最早见证与解释边界', '',
    '三个 qkv_mask 内部见证都返回 `capture_limit: 262,144 captured elements per path`；未建立内部因果起点。optimized 没有第二份内部捕获，是检查器只尝试第一个候选的有限策略，并非内部没有差异。',
    '初始跨候选和 inference↔grad 比较的首个返回浮点差异都是 `segments.0.state.memories.10.K.features`（跨候选前有 `root.forward`）。更新2/3首个返回字段变为 `segments.0.state.memories.0.K.features`。这些是字典树排序中的首个不等字段；memory 10 排在 memory 2 之前，不能据此定位首次发生计算分歧的 region。',
    '已建立：独立 CPU FP64 Reception 公式/VJP/真实零导数检查覆盖选定代数；独立 Decimal AdamW 见证解释了另一个 CPU 小图的 regional.4.second.bias 坐标更新放大。它们支持局部实现语义，但没有重放本次 NPU q/k 最大差异坐标。',
    '未建立：这些 NPU 初始含图前向差异的具体核/表示来源；QKV 融合与静态 mask 的各自贡献；optimized 中合批与 SSD doubling 的各自贡献；全部后续状态/梯度超差的完整传播链。FP32 auto Attention 的梯度相关先验表示是可讨论机制，本记录缺少对应实际调用输入/dispatch 见证，不能直接归因。',
    '结论边界：九个候选执行、连通性、状态元数据与各自恢复检查完成，八个仍标 finite_deviation。不扩大为所有拓扑、批量、长序列、精度、设备或后续训练轨迹的数值等价。', '',
    '## 原始记录', '',
]
lines += [f"- {p['profile']}: `{p['input_file']}`；SHA256 `{p['input_sha256']}`" for p in memo['profiles']]
write_text_atomic(DESTINATION/'npu-reception-paths-memo.md','\n'.join(lines)+'\n')
