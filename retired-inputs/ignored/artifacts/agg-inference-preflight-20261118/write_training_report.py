import json
from pathlib import Path
from scripts.job import atomic
from benchmarks.search_report import write_text_atomic

data=json.loads(Path('artifacts/agg-inference-preflight-20261118/training-summary.json').read_text())
destination=Path('docs/evidence/reception-training-20261118')
atomic(destination.with_suffix('.json'),data)
labels={'fc':'FC-K 9.33M','settle':'HB source 15.69M Settle','timed':'HB source 15.69M Timed',
        'light64':'HB light64 17.88M Settle','half':'HB source 494.11M Settle'}
def name(row,prefix='train'):
    stem=Path(row['record']['path']).parts[1]
    return labels[stem.split(f'agg-{prefix}-')[1].split('-fp32')[0]]
lines=['# Reception：FP32 连续训练、容量与 A–D 三模式对照','',
'本页采用真实 continuing private state 的完整 AdamW 更新，训练以 FP32 为主。',
'吞吐分子均为实际输入位置；Timed 空 tick 和 drain 输出不充作输入。',
'[精简 JSON](reception-training-20261118.json)保存源码、命令、SHA256、资源、',
'原阈值恢复结果、容量及三模式主表。局部优化和双方成本见[性能归因](reception-performance-20261118.md)，',
'跨实现数值资格见[Reception 数值报告](reception-numerics-20261118.md)。','',
'## 一百次连续更新','',
'稳定吞吐取第3–100次更新的输入位置总数除以完整时间总和。计入准备、packing、',
'forward/readout/loss、必要 Timed flush、backward、AdamW、detach 和同步；',
'构造、输入生成/传输、热身、finite 审计及 checkpoint 保存/比较另计。',
'每个窗口保留真实历史，更新间 detach，不把一百次零状态启动称作连续训练。',
'Timed 训练的每个有限 loss 窗口包含 flush；它与不在中途 flush 的连续推理是不同负载。','',
'| 配置 | B/L | 输入位置/s | 峰 allocated/reserved GiB | 配对恢复更新 | 源码 |',
'|---|---|---:|---:|---|---|']
for row in data['continuous']:
    a=row['arguments'];lines.append(f"| {name(row)} | {a['batch']}/{a['prefill']} | {row['steady_positions_per_second']:.2f} | {row['peak_allocated_bytes']/2**30:.3f}/{row['peak_reserved_bytes']/2**30:.3f} | {row['checkpoint']['state']} | `{row['source'][:8]}` |")
lines+=['','全部表内更新的输出、参数、梯度、optimizer、私有状态有限，窗口之间状态已 detach；',
'各自 final_continuation 保存 optimizer 实际 step、KV retained slots 和 slot limits。',
'保存数据和下一前向的 exact 检查，与原阈值内的下一实际更新分别记录；',
'within_threshold 不等于逐位相同，更不消除跨候选有限超差。',
'参数和 AdamW 槽保持 FP32。64 GiB NPU 的 sampled allocated 预算为48 GiB；',
'reserved 另列，运行时仍有 Torch 外的占用。此预算不是硬隔离，也不包含下游任务头。',
'各作业通过协作队列独占准入，仍共享主机且不控制外部任务；精简 JSON 保留物理卡、',
'请求 affinity 及原记录提供的实际 affinity；缺失的观测保持 null，不以请求值冒充实测。','',
'## 新进程真实恢复','',
'下表从上述第100次更新保存点，在新解释器加载同一 checkpoint 并执行第101次更新。',
'加载和首个恢复更新均包含阶段同步，排除运行时初始化和 checkpoint SHA pass；',
'没有新建原 owner 参照，所以这些时间不是额外的数值等价通过。',
'文件系统缓存未控制；新进程不等于冷磁盘。','',
'| 配置 | 加载秒 | 第101次更新秒 | 更新峰 GiB | 状态 |',
'|---|---:|---:|---:|---|']
for row in data['restore']:
    if row['state']!='completed_finite': continue
    peaks=row['next_update_peak_device_bytes'] or {}
    lines.append(f"| {name(row,'restore')} | {row['load_complete_seconds']:.3f} | {row['next_update_complete_seconds']:.3f} | {max(peaks.values(),default=0)/2**30:.3f} | {row['state']} |")
lines+=['','各记录保留 checkpoint 生产源码、文件 SHA256、恢复者源码、状态位置、',
'更新前后 optimizer step 和 increment_errors。冷恢复较稳定更新慢时，不能把其吞吐替代稳定训练。','',
'## 有限 batch 搜索与内存余量','',
'这些是每步从零私有状态开始的五样本搜索，不能与上面的持续吞吐直接混成同一个测点。',
'增加 batch 以真实 sample 轴为单位；没有用节点数或 sequence 长度替代 batch。','',
'| 配置 / 接收重算 | B/L | 输入位置/s | allocated/reserved GiB |',
'|---|---|---:|---:|']
for row in data['capacity']:
    if row['state']!='completed_finite': continue
    a=row['arguments'];m=row['device_memory'];c=row['config'];top=c['topology']['kind']
    label=('FC' if top=='fc' else ('HB 0.5B' if row['parameters']>100_000_000 else 'HB light64'))
    lines.append(f"| {label} / {c.get('checkpoint_reception',False)} | {a['batch']}/{a['length']} | {row['positions_per_second']:.2f} | {m['peak_allocated_bytes']/2**30:.3f}/{m['peak_reserved_bytes']/2**30:.3f} |")
lines+=['','Light64 不重算 B16 的短测吞吐较高，但 allocated 已44.1GiB；',
'其余量小于 B24 接收重算。选主力默认时优先采用通过连续100更新的后者；',
'这不是宣称任何任务下都应重算。两种候选、整区域重算和原 block 路径全部保留。',
'0.5B B1→2→4 的短测持续增长，在本轮预定 B4 边界停止；不声称已经找到全局最优 batch。','',
'## 三种接收的四类联合匹配','',
'共同 FP32 B8/L128，compact-9 n8/h4、d128、X、region_shared，Settle；',
'batch4、Reception 重算、prefix+doubling、fused QKV+结构 padding mask。',
'每种接收及每种模式独立审计实际初态活动后冻结 A–D；Transformer query chunk256，',
'原生/auto Attention。A/B 匹配188个 block，C/D匹配9层；参数量分别匹配总量和实际有效触及均值。',
'相同输入特征前缀、固定种子；每模式 warmup2、计时5。训练的后续活动另记，',
'计时过程中不按新活动重建对照。Transformer 是不同数学函数，不据此比较模型质量。','',
'| 接收 | 总参数 M | 实际触及 M（训练初态） | Full/候选槽位 |',
'|---|---:|---:|---:|']
for row in data['modes']:
    a=row['initial_audits']['train'];w=a['work']
    lines.append(f"| {row['config']['aggregation']} | {a['parameters']['unique_parameters']/1e6:.3f} | {a['effective_touched']['mean']/1e6:.3f} | {w['full_actions']}/{w['candidate_slots']} |")
lines+=['','下表单位全部是输入位置/s；无梯度推理、含梯度前向、完整训练分别计时。',
'含梯度前向不包含 backward/optimizer；本表均从零私有状态开始，不能替代缓存 decode。','',
'| 接收 | 模式 | Tide | A | B | C | D |',
'|---|---|---:|---:|---:|---:|---:|']
for row in data['modes']:
    for mode in ('inference','grad_forward','train'):
        observations={x['subject']:x for x in row['observations'] if x['mode']==mode}
        values=[f"{observations[s]['input_positions_per_second']:.2f}" for s in ('tide','transformer_A','transformer_B','transformer_C','transformer_D')]
        lines.append('| '+row['config']['aggregation']+' | '+mode+' | '+' | '.join(values)+' |')
lines+=['','所有实际 constructor、block/参数目标与相对偏差、allocated/reserved、五个原始样本和',
'range/median 均在配套 JSON。A/B 的差距没有因 Tide 合批而消失；',
'相近参数和 block 数仍不约束乘加、动态索引、SSD、接收 Attention、padding 或串行深度。',
'C/D 宽浅矩阵的吞吐更高；这也是其对照意义，不能视为同一数学函数的替代实现。',
'具体成本证据见[双边归因](reception-performance-20261118.md)。','']
for row in data.get('selected_comparisons', []):
    a=row['initial_audit']; w=a['work']
    lines += ['## 0.5B 主力 B4 的四类完整训练对照','',
        'FP32 B4/L128、HB profile9/n32+h4、X、independent、source_residual；',
        'region batch4、Reception 重算、prefix+doubling，与上面所选 0.5B 训练配置一致。',
        '本表每次从零私有状态开始，warmup2、完整训练5样本；不同于100步真实历史吞吐。',
        'Transformer chunk128，原生/auto Attention；每个对象在独立子进程计时。',
        f"总参数 {a['parameters']['unique_parameters']/1e6:.3f}M，初态实际有效触及均值 {a['effective_touched']['mean']/1e6:.3f}M，",
        f"Full/候选槽位 {w['full_actions']}/{w['candidate_slots']}。四类同用本负载初态审计，计时中不重匹配。",'',
        '| 对象 | block数 / 宽度 / FFN | 实际参数 M | 目标参数偏差 | 输入位置/s | allocated/reserved GiB | range/median |',
        '|---|---|---:|---:|---:|---:|---:|']
    for subject in ('tide', 'transformer_A', 'transformer_B', 'transformer_C', 'transformer_D'):
        observed=next((x for x in row['observations'] if x['subject']==subject), None)
        if observed is None:
            lines.append(f'| {subject} | — | — | — | 未完成 | — | — |')
            continue
        if subject == 'tide':
            shape=f"{row['topology']['nodes']} 节点 / d128"
            delta='—'
        else:
            match=row['joint_matches']['train'][subject[-1]]; c=match['constructor']
            shape=f"{c['layers']} / {c['d']} / {c['ffn_width']}"
            delta=f"{match['relative_deviation']['parameters']:+.4%}"
        m=observed.get('device_memory') or {}
        if observed['state']=='completed_finite':
            rate=f"{observed['input_positions_per_second']:.2f}"
            peaks=f"{m['peak_allocated_bytes']/2**30:.3f}/{m['peak_reserved_bytes']/2**30:.3f}"
            noise=f"{observed['sample_range_over_median']:.2%}"
        else:
            rate=observed['state'];peaks=noise='—'
        params=observed.get('parameters')
        lines.append(f"| {subject} | {shape} | {params/1e6 if params else 0:.3f} | {delta} | {rate} | {peaks} | {noise} |")
    lines+=['','A/B 均1712层，C/D均9层；总量与实际触及的目标分别匹配。',
        '这张表比较同 B/L 与精度的不同数学主干，不是同函数精度验证，也不认证 Transformer 全局最优。',
        '原始样本、初态活动、后续训练工作、参考记录和有限搜索停止原因保存在精简 JSON；',
        '正常计时与 0.5B B1 的侵入式成本诊断分开解释。','']
lines += [
'## 复现','',
'使用每行记录的冻结源码及 `launch.json` 命令，而不是把新 HEAD 追认为旧测量。',
'命令入口分别为 `benchmarks.search`、`benchmarks.continuous`、`benchmarks.restore` 和',
'`benchmarks.run`，输出需要新的目录；按[执行契约](../execution-flows.md)在 detached',
'background.slice 内完成资源准入。真实记录位于 `artifacts/agg-{train,restore,modes,...}-*`，',
'精简 JSON 给出逐文件路径和 SHA256。加载配置入口见[训练建议](../execution-recommendations.md)。','']
write_text_atomic(destination.with_suffix('.md'),'\n'.join(lines))
