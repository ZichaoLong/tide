#!/usr/bin/env python3
"""Build the scoped Markdown/JSON evidence from completed local records."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
WORKTREE=HERE.parent
ROOT=Path('/mi/data2T/zlong/home/llm/graph-execution-foundation')
data=json.loads((HERE/'agg-cost-reception-analysis.json').read_text())
assert all(x['suite_complete'] for x in data['costs'].values())
manifest=dict(data['inputs'])


def read(relative):
    raw=(ROOT/relative).read_bytes()
    manifest[relative]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return json.loads(raw)


def atomic_text(path,text):
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.evidence.')
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


primitives={}
primitive_lines=['| L / 模式 | block ms | incremental ms | doubling ms | block→doubling allocated 峰 MiB |',
                 '|---|---:|---:|---:|---:|']
for name in ('single','scan'):
    base=f'artifacts/agg-ssd-{name}-fp32-v1'
    raw=read(base+'/scan.json');status=read(base+'/status.json')
    assert status['state']=='passed' and raw['within_threshold']
    p={key:raw[key] for key in ('source','arguments','dtype','device_name','physical_devices','resource_identity',
       'cpu_affinity','boundary','operator_observation','fixture','cotangent_sha256','scan_argument_strides',
       'resident_input_and_cotangent_bytes','analytical_storage_components','bounded_reference_shapes',
       'torch','torch_npu','python_executable','reduced_precision_matmul')}
    p['state']=status['state'];p['rows']=raw['rows'];p['candidates']=[]
    for row in raw['candidates']:
        c=dict(backend=row['backend'],operators=row['operators'],arithmetic=row['arithmetic'])
        for key in ('same_fixture_vs_block','bounded_vs_literal_fp64'):
            comparison=row[key]
            c[key]={k:v for k,v in comparison.items() if k!='tensors'}
            c[key]['max_absolute']=max(x['max_absolute'] for x in comparison['tensors'])
            c[key]['max_relative_l2']=max(x['relative_l2'] for x in comparison['tensors'])
            c[key]['checked_tensors']=[x['name'] for x in comparison['tensors']]
        p['candidates'].append(c)
    primitives[name]=p
    for mode in ('inference','grad_forward','backward'):
        group={x['backend']:x for x in raw['rows'] if x['mode']==mode}
        primitive_lines.append(f"| {raw['arguments']['length']} / {mode} | "+' | '.join(
           f"{group[backend]['median_seconds']*1000:.4f}" for backend in ('block','incremental','doubling'))+
           f" | {group['block']['memory_after']['peak_allocated_bytes']/2**20:.3f}→{group['doubling']['memory_after']['peak_allocated_bytes']/2**20:.3f} |")

trace_results={name:json.loads((HERE/f'agg-trace-{name}-analysis.json').read_text()) for name in ('source','doubling','half')}
for run in trace_results.values():
    assert all(x['sha256_verified'] and x['hardware_processes'] for x in run.values())

compact_costs={}
for name,run in data['costs'].items():
    raw=read(f'artifacts/agg-cost-{name}-fp32-v1/costs.json')
    out={k:v for k,v in run.items() if k!='observations'}
    out['precision']=raw['precision'];out['observations']=[]
    out['software']={k:raw[k] for k in ('torch','torch_npu','allow_hf32','python','executable') if k in raw}
    out.update({k:raw[k] for k in ('comparisons_equivalent','numerical_equivalence_qualification')})
    for row in run['observations']:
        item={k:v for k,v in row.items() if k!='diagnostics'}
        item['diagnostics']={k:v for k,v in row['diagnostics'].items() if k not in ('runtime','trace')}
        runtime=row['diagnostics'].get('runtime',{})
        item['runtime']={k:runtime[k] for k in ('dispatched_calls','selected_operator_counts_by_phase','materialization_events','caveat') if k in runtime}
        item['trace_key']=name+'/'+row['subject']+'/'+row['candidate'] if name in trace_results else None
        out['observations'].append(item)
    compact_costs[name]=out

receptions={}
columns=['profile','case','mode','candidate','median_seconds','sample_range_over_median','peak_allocated_bytes',
         'peak_reserved_bytes','state','effective_qkv_backend','actual_operator_signature','exact','finite','max_abs',
         'max_scaled_error','metadata_mismatches','support_mismatches','numerical_failures']
for precision,raw in data['reception'].items():
    out={k:v for k,v in raw.items() if k!='groups'}
    original=read(f'artifacts/agg-reception-sites-{precision}-v1/reception.json')
    out['software']={k:original[k] for k in ('torch','torch_npu','allow_hf32')}
    out['row_columns']=columns;out['rows']=[];out['fixtures']={}
    for group,values in raw['groups'].items():
        profile,case,mode=group.split('/')
        out['fixtures'][group]=dict(shape=values['shape'],sha256=values['canonical_fixture_sha256'])
        for candidate,row in values['candidates'].items():
            record=dict(profile=profile,case=case,mode=mode,candidate=candidate,**row)
            out['rows'].append([record[k] for k in columns])
    receptions[precision]=out

traces={}
for run,rows in trace_results.items():
    for key,value in rows.items():
        traces[run+'/'+key]={k:v for k,v in value.items() if k not in ('process_names','index_hardware_tasks')}
        descriptor=value['trace'];manifest[str(Path(descriptor['path']).relative_to(ROOT))]={k:descriptor[k] for k in ('sha256','bytes')}
output=dict(schema='tide.reception_performance_evidence.v1',snapshot_utc=data['snapshot_utc'],
            fixed_math='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
            scope='FP32 finite complete-training costs, SSD primitives, Reception candidates; BF16 retained deviations; not continuation or inference qualification',
            costs=compact_costs,source_to_doubling=data['source_to_doubling'],ssd_primitives=primitives,
            light64_variants=data['light64_variants'],reception=receptions,device_traces=traces,inputs=manifest)
output['analysis_tools']={str(path.relative_to(ROOT)):dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
                         for path in (HERE/'agg-cost-reception-analysis.py',HERE/'agg-trace-readonly.py',Path(__file__).resolve())}
destination=WORKTREE/'docs/evidence/reception-performance-20261118.json'
atomic_text(destination,json.dumps(output,ensure_ascii=False,separators=(',',':'))+'\n')

memo=(HERE/'agg-cost-reception-analysis.zh.md').read_text()
memo=memo.replace('# Agg 与 SSD 成本证据只读汇总','# Reception、SSD 与完整训练成本 · 2026-11-18')
start=memo.index('快照：');end=memo.index('## 1.')
memo=memo[:start]+f"""FP32 是本轮主力。SSD doubling 在 compact-9 完整训练将吞吐提高约19.25%、allocated峰提高1.47%；
Light64 六候选中融合QKV+结构mask+doubling观察到最高吞吐。接收微测FP32 48/48在原阈值内，
BF16保留21个有限超差。0.5B在本B/L下快于极深窄A、慢于宽浅C，具体计算、packing和实际设备事件如下。
原始样本、资源/源码/输入身份及设备摘要见[精简JSON](reception-performance-20261118.json)。

快照：{data['snapshot_utc']}。固定数学`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。
本页是测量证据，不修改[精度](../precision.md)、[验证](../verification.md)或
[A–D联合匹配](../benchmark-contract.md)契约。测量提交保留各自身份，不能用整理HEAD替代。

## SSD 基元：直接递推与块乘积

单头矩阵状态与读出为

\\[
M_t=a_t M_{{t-1}}+u_t B_t^\\top,\\qquad r_t=M_t C_t.
\\]

在一个块内，令\\(P_{{ij}}=\\prod_{{k=j+1}}^i a_k\\)、\\(P_i=\\prod_{{k=0}}^i a_k\\)，则

\\[
r_i=P_i M_{{\\rm in}} C_i+\\sum_{{j\\le i}}P_{{ij}}u_j(B_j^\\top C_i).
\\]

这是带真实矩阵状态的SSD递推，与接收边self-attention和历史KV Attention是不同计算。
旧block对32×32局部传播乘积使用cumprod；doubling用有界乘法树重排同一乘积，不除以decay，
保留零decay的导数。L=1直接更新矩阵并读出，省去pairwise tile；incremental只改变L=1，
L>1回到block，doubling则同时提供单步直接式和长块乘积树。块边界仍保留矩阵状态和完整BPTT。

N8/B8/H4、value/state width32、chunk32，FP32 resident冻结输入与cotangent，3次热身/9次计时。
`backward`计入前向和五个输入根的VJP，不含optimizer；返回图/结果销毁与同步计入。

{chr(10).join(primitive_lines)}

L1在NPU7，L256在NPU1，两组不作跨卡速度比。L1两候选是同一个直接式，微小时间差不代表
不同算法。L256 doubling相对block：inference约6.43倍、含图前向5.28倍、前向+VJP4.92倍；
含图前向allocated峰多约25MiB，前向+VJP峰多约4.06MiB。局部倍数不能直接当完整训练倍数。
原阈值atol=rtol=3e-5下，全尺寸同fixture输出/末态/五根梯度以及有界CPU literal FP64参照均通过；
FP64参照长度为L1和L65的小形状，不冒充全尺寸FP64实测。

primitive记录源码commit为`57eb12c3`，且保留当时dirty名单，不能声称完整clean提交身份。
冻结`src/tide/ssd.py` SHA256为`42e0e9c2de11f7c266af93c7bf43bf3e53e63efd6ed1eb5b86ef18aa1c982af3`；
JSON保留实现文件hash、输入和cotangent hash及实测FP32算子dtype。此处不推断芯片内部累加精度。

"""+memo[end:]
memo=memo.replace('raw category 未被擅自改判为\n某硬件执行单元；top50 缺席也不是全 trace 零调用的证明。',
                  '完整trace的process_name与Task Type复核见下文；top50缺席本身不是零调用的证明。')
memo=memo.replace('0.5B 成本：终态已完成','0.5B 成本：终态通过')
memo=memo.replace('但本备忘不以未通过数值项作取优依据','但本页不以未通过数值项作取优依据')
memo=memo.replace('B/D构造也在记录中，但此成本任务不测B/D。若总任务仍运行，以下只能是\n已完成行的阶段快照，不能宣布整个成本任务通过。',
                  'B/D构造也在记录中，但此成本任务不测B/D。三对象正常计时、阶段/runtime/work/trace均终态completed_finite。')
memo=memo.replace('这次大模型可能出现Tide胜A、小模型Tide负A并不矛盾：物理工作、层数和batch形状都变了。',
                  'Tide/A吞吐比约9.24，C/Tide约14.31。这次大模型Tide胜A、小模型Tide负A并不矛盾：物理工作、层数和batch形状都变了。')
bf16_lines=['| BF16 padded training | separate ms / 峰MiB | fused_masked ms / 峰MiB | 数值状态 |',
            '|---|---:|---:|---|']
for profile in ('incoming_light_64','incoming_light_128','incoming_wide'):
    group=data['reception']['bf16']['groups'][profile+'/padded/training']['candidates']
    separate,fused=(group[k] for k in ('separate','fused_masked'))
    bf16_lines.append(f"| {profile} | {separate['median_seconds']*1000:.3f} / {separate['peak_allocated_bytes']/2**20:.2f} | "
                      f"{fused['median_seconds']*1000:.3f} / {fused['peak_allocated_bytes']/2**20:.2f} | finite_deviation |")
memo=memo.replace('## 5. 0.5B',chr(10).join(bf16_lines)+'\n\n上述保留了潜在收益候选，但超差结论不变；完整96行数值与时间在JSON中。\n\n## 5. 0.5B')

trace_lines=['| 实际trace | CPU_op次数 / inclusive总s | 硬件任务 / union s | KERNEL任务 / union s | 硬件首末跨度s |',
             '|---|---:|---:|---:|---:|']
for run,key,label in [('source','tide/standard','small source'),('doubling','tide/standard','small doubling'),
                      ('source','transformer_A/standard','small A'),('source','transformer_A/chunk256','small A chunk256'),
                      ('half','tide/standard','0.5B Tide'),('half','transformer_A/standard','0.5B A'),('half','transformer_C/standard','0.5B C')]:
    value=trace_results[run][key]
    trace_lines.append(f"| {label} | {value['host_cpu_op_events']:,} / {value['host_cpu_op_inclusive_duration_sum_seconds']:.6f} | "
       f"{value['hardware_events']:,} / {value['hardware_interval_union_seconds']:.6f} | "
       f"{value['kernel_events']:,} / {value['kernel_interval_union_seconds']:.6f} | {value['hardware_span_seconds']:.6f} |")
trace_text=f"""## 实际设备轨迹：双边证据

按完整trace的`process_name=Ascend Hardware`识别硬件，再按`Task Type`以`KERNEL`开头识别kernel。
逐文件SHA256已核对原成本记录；没有把全部无cat事件（含Overlap Analysis）当设备kernel。

{chr(10).join(trace_lines)}

union是这次带profiler观察的时间戳区间并集，不是正常运行利用率；CPU事件嵌套、设备事件可重叠，
各列不相加，也不除以普通更新中位数制造“正常耗时占比”。trace外层elapsed还含导出/分析。

small source的Cumprod确为128次`KERNEL_AICPU`、duration总233.062ms；这是Ascend设备任务，
不是host CPU fallback。doubling完整硬件任务表中Cumprod为0；kernel union约1.157→0.931s，
同时kernel次数54256→58404。更多小任务仍可以换来更低昂贵算子成本；正常训练收益以上表为准。

0.5B Tide设备任务可见Add 5654次/112.702ms、Mul 5087/61.248ms、Index 2022/51.317ms，
BatchMatMul 1376/25.256ms。A则MatMul 20544/118.773ms，另有StridedSlice 23968/262.042ms、
Mul 59924/249.302ms、Transpose 15408/185.180ms，AttentionGrad 1712/66.899ms。
C只有108次MatMul/18.541ms，且optimizer设备任务成为显眼部分。
这些duration是各任务名的和，仍非可加的完整墙钟分摊。

0.5B A的硬件首末跨度14.960s而kernel union1.778s，结合1712层和1.45M个host CPU_op，
支持它在本B1形状有显著串行launch/调度间隙，不能把其12.421s普通时间全部解释为Attention算术。
这不是对空隙逐段归因的证明，也不是Transformer全局最佳实现声明。C宽浅结构以更大矩阵、更少层
避开这一重复调度，仍是更高吞吐参照。Tide更少层/相容区域合批的收益和自身分配/索引成本同时存在。

## 对剩余成本的解释与有限后续方向

接收的gather把不同源状态按边/接收者排成矩形，保留真实质量和结构support；对共享源的反向需要
索引梯度归并。packing把实际Full时刻按owner/相容组装成矩形，组内短行补齐，因此选择行数和
执行行数不同。0.5B约5.248倍的K padding比小模型约1.389倍更值得检查：缩小或按有效行数分桶
可降低padding，但会增加矩阵/launch数。本页没有未测候选的加速承诺。

quad allocation对收到质量给定上下界与总质量约束，求分段线性的
\\(c_i=\\operatorname{{clip}}(s_i-\\lambda,\\ell_i,u_i)\\)。当前实现排序断点、前缀sum、判定所处面，
必要时使用受保护的二分回退，并为边界导数保存面信息；不是一层普通softmax。0.5B阶段里的
215ms allocation是整站点同步观察，尚没有逐次记录fallback是否发生，不能全部归给sort或fallback。
可以有界研究减轻分配小算子/同步、减少矩形padding及合并索引梯度；这些改动需要保留质量约束、
零质量导数、原时刻KV可见性和私有状态。增加区域并发本身不能消除这些数学/布局工作。

"""
memo=memo.replace('## 6. 可直接引用的边界',trace_text+'## 6. 可直接引用的边界')
memo+='''
## 复现入口与原记录

冻结源码和完整命令由各作业的`launch.json`、`run.sh`、`status.json`保存；不要以当前HEAD替换冻结源码复测。
示例均需按[执行契约](../execution-flows.md)先完成配额/任务/设备准入；下面列的是入口，不请求自动启动。

- `python -m benchmarks.ssd_scan_candidates`：`agg-ssd-single-fp32-v1/scan.json`与`agg-ssd-scan-fp32-v1/scan.json`。
- `python -m benchmarks.costs`：`agg-cost-source-fp32-v1`、`agg-cost-doubling-fp32-v1`、`agg-cost-light64-fp32-v1`、`agg-cost-half-fp32-v1`的`costs.json`。
- `python -m benchmarks.variants`：`agg-light64-variants-fp32-v1/variants/variants.json`及六份独立normal记录。
- `python -m benchmarks.reception`：`agg-reception-sites-fp32-v1/reception.json`和`agg-reception-sites-bf16-v1/reception.json`。

这些路径均位于仓库`artifacts/`；相对路径、字节数和SHA256完整列于配套JSON。
本页数据归纳脚本保留在`artifacts/worktrees/agg-inference-ssd/artifacts/`，包括
`agg-cost-reception-analysis.py`、`agg-trace-readonly.py`和`build-reception-performance-evidence.py`；
其行为仅只读解析既有记录。全部实际trace保留，未清理复现依赖。
'''
atomic_text(WORKTREE/'docs/evidence/reception-performance-20261118.md',memo)
print(json.dumps(dict(json=str(destination),bytes=destination.stat().st_size,
                      markdown=str(WORKTREE/'docs/evidence/reception-performance-20261118.md'),
                      reception_rows={k:len(v['rows']) for k,v in receptions.items()},traces=len(traces)),ensure_ascii=False))
