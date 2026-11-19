#!/usr/bin/env python3
"""Read existing finite evidence; no model, device, or workload execution."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
ROOT = Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
sys.path.insert(0, str(WORKTREE))
from scripts.job import atomic

inputs = {}
cache = {}


def load(relative):
    path = ROOT / relative
    if relative not in cache:
        raw = path.read_bytes()
        inputs[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        cache[relative] = json.loads(raw)
    return cache[relative]


def subset(obj, keys):
    return {key: obj[key] for key in keys if key in obj}


def work(audit):
    if 'work' not in audit:
        return audit
    matrices = audit.get('node_matrices', {})
    return dict(work=audit['work'], effective_touched=audit.get('effective_touched'),
                query_packing=audit.get('query_packing'),
                node_matrix_forward_multiply_adds=sum(x['dense_multiply_adds'] for x in matrices.values()),
                node_matrix_cost_definition=audit.get('matrix_cost_definition'),
                node_matrix_top=sorted((dict(site=k, **v) for k, v in matrices.items()),
                                       key=lambda x: x['dense_multiply_adds'], reverse=True)[:6])


def stages(report):
    rows = report.get('rows', [])
    inclusive = {x['path']: x['inclusive_seconds'] for x in rows
                 if x['path'].count('/') <= 1}
    cats = defaultdict(lambda: dict(calls=0, exclusive_seconds=0.0))
    for row in rows:
        if row['path'].startswith('training/forward'):
            cat = row['path'].split('/')[-1]
            cats[cat]['calls'] += row['calls']
            cats[cat]['exclusive_seconds'] += row['exclusive_seconds']
    forward = inclusive.get('training/forward', 0)
    for value in cats.values():
        value['fraction_of_intrusive_forward'] = value['exclusive_seconds']/forward if forward else None
    return dict(inclusive_seconds=inclusive,
                forward_exclusive_by_leaf=dict(sorted(cats.items(), key=lambda item: -item[1]['exclusive_seconds'])),
                backward_reception=[x for x in rows if x['path'].startswith('training/backward/reception')],
                partition_check=sum(x['exclusive_seconds'] for x in cats.values())-forward,
                boundary=report.get('boundary'), interpretation=report.get('interpretation'))


def runtime(report):
    counts = defaultdict(Counter)
    for row in report.get('operators', []):
        phase = row['path'].split('/')[1] if '/' in row['path'] else row['path']
        counts[phase][row['operator']] += row['calls']
    selected = ('bmm', 'matmul', 'mm.', 'cumprod', 'fusion_attention', 'index', 'copy_',
                'nonzero', 'cat.', 'slice', 'mul.', 'where', 'local_scalar')
    counts = {phase: {name: count for name, count in value.items() if any(s in name for s in selected)}
              for phase, value in counts.items()}
    matrix = [x for x in report.get('matrix_groups', []) if x['path'].startswith('training/forward')]
    # Shape groups remain operator observations, not kernel/FLOP counts.
    shapes = []
    for row in sorted(matrix, key=lambda x: -x['calls'])[:12]:
        shapes.append(dict(path=row['path'], operator=row['operator'], calls=row['calls'],
                           inputs=[subset(x, ('shape', 'dtype', 'device')) for x in row['inputs']],
                           outputs=[subset(x, ('shape', 'dtype', 'device')) for x in row['outputs']]))
    mats = defaultdict(lambda: dict(calls=0, logical_output_bytes=0))
    for row in report.get('materializations', []):
        key = row['operator']
        mats[key]['calls'] += row['calls']
        mats[key]['logical_output_bytes'] += row['logical_output_bytes']
    return dict(dispatched_calls=report.get('dispatched_calls'),
                selected_operator_counts_by_phase=counts,
                forward_matrix_shape_groups_top12_by_calls=shapes,
                materialization_events=dict(mats), interpretation=report.get('interpretation'),
                caveat='operator wrappers can overlap; logical materialization bytes are repeated events, not peak memory or measured bandwidth')


def trace(diag):
    summary = diag.get('summary', {})
    selected = ('Cumprod', 'IndexBackward0', 'FlashAttentionScore', 'IndexPutV2',
                'BatchMatMul', 'MatMul', '_Constrained', '_TiledAttention')
    events = [x for values in summary.get('top_by_category', {}).values() for x in values
              if any(name in x['name'] for name in selected)]
    return dict(intrusive_elapsed_seconds=diag.get('intrusive_elapsed_seconds'),
                trace=diag.get('trace'), categories=summary.get('categories'),
                selected_events_from_top50_per_category=events,
                boundary=diag.get('boundary'), interpretation=summary.get('interpretation'),
                caveat='raw categories retained; inclusive durations overlap and cannot be added or normalized by normal wall time; absence in top50 is not absence from full trace')


def cost_record(name):
    base = f'artifacts/agg-cost-{name}-fp32-v1'
    status = load(base+'/status.json')
    data = load(base+'/costs.json')
    record = dict(status=subset(status, ('state', 'exit_code', 'elapsed', 'finished', 'peak_rss_bytes')),
                  **subset(data, ('state', 'source', 'harness_source', 'physical_devices', 'resource_identity',
                                  'cpu_affinity', 'config', 'timing', 'joint_matches')))
    record['arguments'] = subset(data['arguments'], ('batch', 'length', 'warmup', 'steps', 'threads', 'precision', 'seed'))
    record['initial_parameter_sha256'] = data.get('initial_tide_parameter_sha256')
    record['initial_work'] = work(data.get('initial_tide_audit', {}))
    record['observations'] = []
    for obs in data.get('observations', []):
        out = subset(obs, ('subject', 'candidate', 'state', 'parameters', 'input_sha256'))
        if obs['subject'] != 'tide':
            out['spec'] = obs.get('spec')
        out['normal'] = subset(obs.get('normal', {}), ('state', 'seconds', 'losses', 'median_seconds',
                   'input_positions_per_second', 'device_memory', 'sample_range_over_median',
                   'start_identity', 'actual_storage'))
        out['endpoint_work'] = work(obs.get('normal_endpoint_work', {}))
        out['diagnostics'] = {}
        for kind, diag in obs.get('diagnostics', {}).items():
            dest = subset(diag, ('state', 'intrusive_elapsed_seconds'))
            report = diag.get('report')
            if report is not None:
                if kind == 'stages': dest.update(stages(report))
                elif kind == 'runtime': dest.update(runtime(report))
                elif kind == 'work': dest['work'] = work(report)
                elif kind == 'saved': dest['total'] = {k:v for k,v in report['total'].items() if k != 'top_saved_shapes'}
            elif kind == 'trace': dest.update(trace(diag))
            out['diagnostics'][kind] = dest
        record['observations'].append(out)
    record['suite_complete'] = status.get('state') == 'passed' and data.get('state') == 'completed_finite'
    return record


def reception_record(precision):
    base = f'artifacts/agg-reception-sites-{precision}-v1'
    status = load(base+'/status.json')
    data = load(base+'/reception.json')
    out = subset(data, ('source', 'harness_source', 'physical_devices', 'resource_identity', 'cpu_affinity',
                        'arguments', 'boundary', 'numerical_boundary', 'memory_scope', 'precision_policies'))
    out['status'] = subset(status, ('state', 'exit_code', 'elapsed', 'finished'))
    out['row_states'] = dict(Counter(x['state'] for x in data['rows']))
    signatures = []
    index = {}
    groups = {}
    failures = []
    for row in data['rows']:
        key = '/'.join(row[k] for k in ('profile', 'case', 'mode'))
        group = groups.setdefault(key, dict(shape=row['shape'], canonical_fixture_sha256=row['canonical_fixture_sha256'], candidates={}))
        assert group['canonical_fixture_sha256'] == row['canonical_fixture_sha256']
        audit = row['audit']
        signature = dict(attention=audit['attention'], operators=audit['operators'])
        sign = json.dumps(signature, sort_keys=True)
        if sign not in index:
            index[sign] = len(signatures)
            signatures.append(signature)
        comparison = row['vs_separate']
        group['candidates'][row['candidate']] = dict(
            state=row['state'], median_seconds=row['median_seconds'],
            sample_range_over_median=(max(row['seconds'])-min(row['seconds']))/row['median_seconds'],
            peak_allocated_bytes=row['peak_device_allocated_bytes'], peak_reserved_bytes=row['peak_device_reserved_bytes'],
            effective_qkv_backend=row['effective_qkv_backend'],
            actual_operator_signature=index[sign], exact=comparison['exact'], finite=comparison['finite'],
            max_abs=comparison['max_abs'], max_scaled_error=comparison['max_scaled_error'],
            metadata_mismatches=len(comparison['metadata_mismatches']),
            support_mismatches=len(comparison['support_mismatches']),
            numerical_failures=len(comparison['numerical_failures']))
        if comparison['numerical_failures']:
            failures.append(dict(group=key, candidate=row['candidate'], failures=comparison['numerical_failures']))
    out['threshold'] = subset(data['rows'][0]['vs_separate'], ('atol', 'rtol'))
    out['groups'] = groups
    out['actual_operator_signatures'] = signatures
    out['numerical_failures'] = failures
    out['limits'] = ['audit is one separate post-timing forward/backward/update; not all timed steps',
                     'timed training uses candidate-specific updates; compare same fixture/config only',
                     'throughput is receiver-node x sample x position per second, not backbone tokens',
                     'BF16 and FP32 ran on different physical NPUs and source revisions; no cross-dtype speed ranking',
                     'hardware accumulation is not inferred from tensor dtypes']
    return out


costs = {name: cost_record(name) for name in ('source', 'doubling', 'light64', 'half')}
source, doubling = (costs[key]['observations'][0] for key in ('source', 'doubling'))
sn, dn = source['normal'], doubling['normal']
delta = dict(latency_reduction_fraction=1-dn['median_seconds']/sn['median_seconds'],
             throughput_increase_fraction=dn['input_positions_per_second']/sn['input_positions_per_second']-1,
             peak_allocated_change_bytes=dn['device_memory']['peak_allocated_bytes']-sn['device_memory']['peak_allocated_bytes'],
             peak_allocated_change_fraction=dn['device_memory']['peak_allocated_bytes']/sn['device_memory']['peak_allocated_bytes']-1,
             initial_parameter_equal=costs['source']['initial_parameter_sha256']==costs['doubling']['initial_parameter_sha256'],
             input_equal=source['input_sha256']==doubling['input_sha256'],
             initial_work_equal=costs['source']['initial_work']==costs['doubling']['initial_work'],
             normal_start_identity_equal={key:sn['start_identity'][key]==dn['start_identity'][key] for key in sn['start_identity']},
             endpoint_work_delta={key:doubling['endpoint_work']['work'][key]-value for key,value in source['endpoint_work']['work'].items()},
             limits=['same initial parameters/input/work but candidate-specific warmup changes the normal-start anchor',
                     'initial/endpoint work is not cumulative timed work or support equality',
                     'same physical card at different times; external jobs not controlled',
                     'no numerical-equivalence qualification inferred from performance'])
variants = load('artifacts/agg-light64-variants-fp32-v1/variants/variants.json')
variant_result = subset(variants, ('source', 'harness_source', 'state', 'physical_devices', 'resource_identity', 'cpu_affinity',
                                  'scope', 'timing_scope', 'pairing_scope', 'order_scope'))
variant_result['rows'] = []
for row in variants['summaries']:
    item = dict(row)
    path = str(Path(row['result_file']['path']).relative_to(ROOT))
    child = load(path)
    item['device_memory'] = child['observations'][0]['device_memory']
    item['configuration'] = subset(child['config'], ('reception_qkv_backend', 'reception_mask_padding', 'checkpoint_reception', 'ssd_scan_backend'))
    item['throughput_increase_fraction'] = 1/row['candidate_over_baseline_time']-1
    variant_result['rows'].append(item)

analysis = dict(schema='tide.agg_cost_reception_readonly_analysis.v1',
                snapshot_utc=datetime.now(timezone.utc).isoformat(),
                scope='read-only extraction of existing finite records; no workload execution',
                costs=costs, source_to_doubling=delta, light64_variants=variant_result,
                reception={precision:reception_record(precision) for precision in ('fp32', 'bf16')}, inputs=inputs)
atomic(HERE/'agg-cost-reception-analysis.json', analysis)


def atomic_text(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.analysis.')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normal_table(rows):
    lines = ['| 对象 | 参数数 | 中位秒/更新 | 输入位置/秒 | allocated / reserved 峰 GiB | range/median |',
             '|---|---:|---:|---:|---:|---:|']
    for name, obs in rows:
        n = obs['normal']
        if 'median_seconds' not in n:
            lines.append(f'| {name} | — | 尚未完成 | — | — | — |')
            continue
        mem = n['device_memory']
        lines.append(f"| {name} | {obs['parameters']:,} | {n['median_seconds']:.6f} | {n['input_positions_per_second']:.2f} | "
                     f"{mem['peak_allocated_bytes']/2**30:.3f} / {mem['peak_reserved_bytes']/2**30:.3f} | {n['sample_range_over_median']*100:.2f}% |")
    return '\n'.join(lines)


def stage_table(run):
    lines = ['| 对象 | 整次更新 s | 前向 s | 反向 s | AdamW s |', '|---|---:|---:|---:|---:|']
    for name, obs in run:
        s = obs['diagnostics'].get('stages', {}).get('inclusive_seconds', {})
        if not s:
            continue
        lines.append(f"| {name} | {s['training']:.6f} | {s['training/forward']:.6f} | {s['training/backward']:.6f} | {s['training/optimizer']:.6f} |")
    return '\n'.join(lines)


small_rows = [(name, costs[name]['observations'][0]) for name in ('source', 'doubling', 'light64')]
variant_lines = ['| Light64 执行候选 | 中位 s | 输入位置/s | allocated / reserved GiB | 吞吐相对 baseline |',
                 '|---|---:|---:|---:|---:|']
for row in variant_result['rows']:
    mem = row['device_memory']
    variant_lines.append(f"| {row['variant']} | {row['median_seconds']:.6f} | {row['input_positions_per_second']:.2f} | "
                         f"{mem['peak_allocated_bytes']/2**30:.3f} / {mem['peak_reserved_bytes']/2**30:.3f} | {row['throughput_increase_fraction']*100:+.2f}% |")
reception_lines = ['| FP32 profile / case / mode | separate ms | fused ms | masked ms | fused_masked ms |',
                   '|---|---:|---:|---:|---:|']
for group, value in analysis['reception']['fp32']['groups'].items():
    reception_lines.append('| '+group.replace('incoming_', '')+' | '+' | '.join(
        f"{value['candidates'][candidate]['median_seconds']*1000:.3f}"
        for candidate in ('separate', 'fused', 'masked', 'fused_masked'))+' |')
half = costs['half']
half_label = '终态已完成' if half['suite_complete'] else '运行中快照；已完成行仅作为阶段记录'
half_rows = [(x['subject'], x) for x in half['observations']]
half_status = json.dumps(half['status'], ensure_ascii=False)
memo = f"""# Agg 与 SSD 成本证据只读汇总

快照：{analysis['snapshot_utc']}。本分析只读取既有 JSON；没有启动、等待或重跑设备任务。
输入文件及 SHA256 在同目录 `agg-cost-reception-analysis.json` 的 `inputs`；提取脚本为
`agg-cost-reception-analysis.py`。所有时间、峰值和算子归属原记录源码，不替换成当前 HEAD。

## 1. 小规模完整训练：收益与代价

共同负载为 compact-9 HB Settle、n8/hubs4、d128、X memory、region_shared，B8/L256，
FP32、region batch4、Reception checkpoint、SSD prefix，2 次热身后 5 次完整 AdamW 更新。
每次更新从零状态 prefill；这是顺序参数更新，不是带缓存历史的连续训练。普通计时包括
packing/forward/loss/backward/AdamW/sync，排除构造、输入传输、审计、anchor IO 和 checkpoint IO。

{normal_table(small_rows)}

source 与 doubling 在同一物理 NPU1，源码分别 `95465188`、`b4cf1514`；Light64 在 NPU7，
源码 `95465188`。准入身份均为 `exclusive_cooperative_queue; external jobs not controlled`。
不能称整机独占。源码/环境/设备完整值保留在 JSON；HF32 关闭。

doubling 普通时间减少 **16.15%**、吞吐增加 **19.25%**；allocated 峰增加
194,885,120 B，即 185.86 MiB / 1.47%。初始参数、输入、初始工作量都相同；各自热身后
normal-start 的参数、optimizer、gradient hash 已不同，RNG hash 相同。第一步诊断工作摘要
相同；末端正消息差 -32、Full 动作差 -5、KV 有效 pair 差 -1180。没有把这些端点摘要称为
全步骤等算量、全支持相同或数值资格。收益不是通过放宽阈值得来。

### 侵入式阶段和设备 trace

下表来自各自第一步 anchor 的单次同步阶段诊断。包含逐 scope 同步开销；
不能代替上表正常吞吐。父/子 inclusive 时间不可相加。

{stage_table(small_rows)}

source→doubling 的前向 exclusive：SSD matrix **252.224→52.275 ms**，占这次侵入式
前向的 **29.77%→8.22%**；prefix 已在两边启用，为 10.200→10.093 ms。
分配 104.097→93.339 ms，gather 70.087→70.150 ms，query packing 57.523→55.830 ms，
所有 attention site 合计 69.272→69.519 ms。反向 578.968→609.266 ms，确有增加。

完整 runtime 审计中，旧 SSD 的 forward/backward 各有 128 次 `aten.cumprod`，doubling 为 0；
乘法计数 forward 1839→2059、backward 2567→2951。外层保存 hook 的峰 unique storage
12,331,957,617→12,520,701,297 B，与实际 allocator 峰是不同量。hook 不保证覆盖嵌套 hook
及后端 opaque 保存，也不把重复 saved payload 当 unique memory。

独立 profiler 记录的 raw `aclnnCumprod_Cumprod_Cumprod` 旧路径 128 次，累计 inclusive
duration 233.062 ms；新 trace 的 CPU_op 207,880→223,364，enqueue 37,943→40,177。
trace 整体 15.929→18.073 s，远慢于正常计时；这不推翻普通计时收益，也不能用其总时长
计算正常占比。`IndexBackward0` 两边各 1153 次、136.448/136.376 ms；FlashAttentionScoreV2
38 次、11.589/11.762 ms，GradV2 38 次、35.539/35.791 ms。raw category 未被擅自改判为
某硬件执行单元；top50 缺席也不是全 trace 零调用的证明。

Light64 是另一数学实例，参数 15.694M→17.880M，支持/动作及设备也变了，不能把总时间
差称为同函数的实现开销。其 Reception 前向 inclusive 145.882 ms（自身 exclusive
71.809 ms，attention 子项 74.074 ms），原 source 为 33.614 ms；反向 Reception 重算
347.803 ms，原 source 为 122.635 ms。这描述新算子的成本，不表示该语义应被取消。

## 2. 与 Transformer 的具体差别

以下 A/C 只来自 source 成本任务的相同 B8/L256 FP32 正常训练；该成本任务没有测 B/D，
不能替代完整 A–D 联合匹配主表。A 188 blocks/d80/FFN240/head-dim16，15.672M；
C 9 blocks/d368/FFN1088/head-dim16，15.693M。窗口都为128。

{normal_table([(x['subject']+'/'+x['candidate'], x) for x in costs['source']['observations'][1:]])}

source 的第一步 anchor 实际 NodeLinear 前向含 padding/adapter 的乘加对为 **123.319G**；
A 线性层为 **32.034G**。Tide 该计数不含 attention、normalization、非 NodeLinear 运算，
Transformer 该计数也不含 attention、norm/RoPE/activation，不能宣称完整 FLOPs 对齐。
它至少说明参数接近并不保证物理线性工作接近。source 的 K query 选择147,804行，执行
205,312行，执行/选择约1.389；消息正条目1,949,957，而物理 message slots2,850,816。

Tide 前向有 1606 次 `aten.index`、641 次 bmm，反向有1153次 `index_put`；
A 没有对应的动态节点索引链，前向为752次 matmul（188块各四类矩阵投影）及376次
融合 attention，chunk256 则 attention 减为188次。其 Q/K/V 形状主要是
[8,5,128,16]、[8,5,255,16]；chunk256 为[8,5,256,16]。Tide 的区域/节点分组、可变
query packing、独立参数 bmm、SSD 32×32 block 和接收/分配有各自的小算子与索引链。
两者功能和图不同，不能由某个 bmm/attention 个数直接推断总效率。

A chunk256 的 attention operand rectangle 反而从368.660M增至492.831M pair，
普通更新却从0.759582降至0.584350s：减少调用与改善形状可以比减少矩形 pair 更有利。
C 的宽矩阵只串行9层，runtime dispatched calls 2170（chunk256为1774），
A为44593（chunk256为36321），Tide source为57321、doubling为56867。
这些是对应审计观察的 dispatch 数，不等同设备 kernel 数，也不应与 profiler 事件数混加。

## 3. Light64 执行候选保留价值

独立 serial 子进程、相同 NPU7、相同初始参数/输入/工作摘要，源码 `b4cf1514`。
固定候选顺序且每项仅5次，不消除时间/主机争用影响；后续活动与工作摘要略有不同。
这一组 baseline 与上一成本任务 Light64 是不同计时记录，不能混为同一次测量。

{chr(10).join(variant_lines)}

qkv_mask_scan（融合QKV+结构mask+doubling）是本组最快观察值，中位时间下降11.64%，
allocated峰仅增1.34%。取消 Reception checkpoint 也有收益，但显存峰增82.57%；与
qkv_mask 本身比较，取消checkpoint时间下降6.76%、峰增约82.77%。因此重算候选与
不重算候选都值得保留，是否选默认要同时满足吞吐、内存余量和数值资格；不能一概关闭。

## 4. 接收 self-attention：局部真实算子与性能

N8/B8/L128/E8/d128，shared，warmup2+steps5，无Reception重算。普通训练计时含AdamW；
输入和 routed mass resident。吞吐单位为 receiver-node×sample×position/s，不能当主干token/s。
算子审计是另一次 post-timing 前向/反向/更新，并不覆盖所有 timed step。

{chr(10).join(reception_lines)}

FP32 在NPU1、源码`d635a982`，**48/48 within_threshold**，原阈值atol1e-6/rtol.002。
padded training 的separate/fused实际走tiled；masked/fused_masked能走NPU融合Attention，
却更慢。light64 separate→masked为11.412→13.214ms、峰371.37→435.69MiB；
light128为15.621→19.440ms、510.03→708.35MiB；wide为16.017→19.682ms、
510.70→709.01MiB。融合路径不是普适收益。light64的native QKV在推理为
[8192,2,8,32]、FP32、带PSE；结构mask训练为[8192,2,8,48]、FP32、无PSE。
真实零质量case训练保留tiled。结构padding与真实零根没有被混为同一个删除规则。

QKV合并的局部算子证据：light64训练bmm36→24、mm18→12；推理bmm12→8、mm6→4。
FP32 light64训练有约2.3–2.4%的观察收益，light128近乎持平，wide约0.5–0.7%；
宽profile推理反而下降。端到端合并收益与局部时间不能线性相加。

BF16在NPU4、源码`f3e6661c`，完整48行，**27 within / 21 finite_deviation**；
exit1来自脚本对非within行的返回条件，并非任务异常中断。21个失败全部是
`root.result.1`（Agg输出）；全有限，metadata/support mismatch均0，其余被比较的梯度、
参数、optimizer条目没有超差。原阈值atol.002/rtol.05保留，最大scaled error8.9582。
每个profile失败模式一致：padded inference masked/fused_masked；padded training
fused/masked/fused_masked；zero training fused/fused_masked。不能把这些结果列为已合格
默认候选，也不能仅由其有限性断言全是正常舍入。BF16个别融合候选有较大速度/显存收益，
但本备忘不以未通过数值项作取优依据；FP32/BF16卡与源码不同也不作跨dtype速度排名。

## 5. 0.5B 成本：{half_label}

状态快照：`{half_status}`。源码`{half['source']}`，NPU7，
CPU128–131，FP32 B1/L128，n32/profile9/independent、region batch4、Reception checkpoint、
prefix+doubling。A联合匹配为1712×d128/FFN584，496.562M；C为9×d2032/FFN6296，
494.107M。B/D构造也在记录中，但此成本任务不测B/D。若总任务仍运行，以下只能是
已完成行的阶段快照，不能宣布整个成本任务通过。

{normal_table(half_rows)}

独立阶段诊断（intrusive）：

{stage_table(half_rows)}

Tide初始总参数494.111M，实际触及均值145.753M；第一步anchor为141.547M，
不是用k/n静态代理代替实际活跃参数。anchor K packing选择18,322行、执行96,160行，
约5.248倍；正消息328,220，物理槽1,714,176。Tide前向exclusive allocation215.266ms
（该侵入式前向27.69%），gather69.760ms、packing64.506ms、attention64.014ms、
SSD matrix36.554ms。SSD已优化后，分配和稀疏到矩形的packing成为更明确的后续方向。

该anchor Tide NodeLinear前向45.875G乘加对；A线性层63.504G，且A需要1712层串行。
A审计前向6848次matmul、1712次融合Attention，Q/K/V为[1,4,128,32]；
Tide前向448次bmm、13次native Attention且另有tiled，1839次index，反向1126次index_put。
这次大模型可能出现Tide胜A、小模型Tide负A并不矛盾：物理工作、层数和batch形状都变了。
A普通5样本range/median20.57%，保留抖动，不将单次比值泛化为稳健硬件上限。

## 6. 可直接引用的边界

本证据支持“有限候选实测、完整训练边界、对应anchor成本与形状分析”，不支持训练全程
等轨迹、普适最优、全尺度资格或任务质量。正常计时、同步阶段、runtime hook、saved hook、
profiler各自单独测量，不相加。大模型首token/持续inference、正式多步延续和恢复的结论
应引用各自专门记录；这些成本任务每步重置模型状态，不能替代那些记录。
"""
atomic_text(HERE/'agg-cost-reception-analysis.zh.md', memo)
print(json.dumps(dict(output=str(HERE/'agg-cost-reception-analysis.json'),
                      bytes=(HERE/'agg-cost-reception-analysis.json').stat().st_size,
                      half_status=costs['half']['status'], source_to_doubling=delta,
                      reception_states={p:r['row_states'] for p,r in analysis['reception'].items()}), ensure_ascii=False, indent=2))
