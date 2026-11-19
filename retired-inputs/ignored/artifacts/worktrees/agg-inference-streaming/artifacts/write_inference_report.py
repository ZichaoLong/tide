"""Render current-cycle inference evidence without modifying parent artifacts."""
from pathlib import Path
import json
import os
import tempfile

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE/'inference-summary.json').read_text())
runs = {r['run']:r for r in summary['runs']}
short = [
    ('FC 0.499B','agg-inference-fc-half-fp32-v1'),
    ('HB 0.494B','agg-inference-hb-half-fp32-v1'),
    ('HB 8.778B','agg-inference-hb9b-fp32-v1'),
    ('HB 16.707B','agg-inference-hb17b-fp32-v1'),
]
long = [
    ('FC 0.999B','agg-inference-fc1b-sustain-fp32-v1'),
    ('HB 0.977B','agg-inference-hb1b-stream-fp32-v1'),
    ('HB 8.778B','agg-inference-hb9b-sustain-fp32-v1'),
    ('HB 16.707B','agg-inference-hb17b-sustain-fp32-v1'),
    ('Timed Light64','agg-inference-timed-long-fp32-v1'),
    ('HB 8.778B B2','agg-inference-hb9b-batch2-fp32-v1'),
]


def fmt(value, decimals=3):
    return '—' if value is None else f'{value:.{decimals}f}'


def metric(subject, key):
    return subject['metrics'][key]['median']


lines=[]
def add(text=''):
    lines.extend(text.splitlines())


add('''# 2026-11-18：连续推理与大模型 A–D 对照

本页记录主干推理，输出不包含词表头、采样或服务传输。大参数量 FC/HB 测点使用
`source_residual` 接收绑定；它们提供大模型基线，并不是 incoming self-attention 的
性能。Timed 长流使用 `incoming_light_64`。接收候选的训练选择另见
[连续训练与三模式对照](reception-training-20261118.md)；
[最终数值与后端验证](reception-validation-20261118.md)保留各路径原有有限超差。
本页的样本、配置、哈希与检查保留在[机器可读摘要](reception-inference-20261118.json)。

## 身份与计时

精度均为 FP32，正常样本 2 次、warmup 1 次。数学固定
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。设备为本机 Ascend910、每逻辑卡
64 GiB HBM；使用 PyTorch `2.12.0+cpu` 加 torch_npu 的
`ascend/dev-workspace-9.2.0` 私有环境及协作队列。队列身份为
`exclusive_cooperative_queue; external jobs not controlled`，不同实验可能落在不同
物理卡，同机其他作业与主机争用不受本测量控制。每组实际设备、CPU affinity、
源码/参数/输入哈希、构造器与原始文件 SHA256 保存在同名 JSON 摘要中。

下表是每个指标的正常样本中位数。decode 为排除第一次增量调用后的
`B × 输入位置 / 稳定调用总时间`；不是平均时延倒数。显存为各样本、各参与卡的
最大峰值，未将多卡相加。KV/SSD/pending 在请求内真实续接，输出逐窗口生成、
同步后释放；Timed 只有最后一次 drain。诊断、构造、传参和 warmup 不计时。
所有参与卡均同步。返回值为 FP32，逻辑时间为 int64；完整精度政策在摘要记录。

“首输出”是第一个有正输出强度的完整调用结束；“prompt 就绪”要求返回窗口覆盖
最后 prompt 注入加完整图深度延迟。两者由独立诊断映射到计时调用边界。
这些不是语言模型服务 TTFT，不能从一个 chunk 的完整调用时间推断更细的逐 tick
时延。多 subjects 共用进程时，后续 subject 的 cold 会受到前序内核缓存影响，
因此本表使用 warmup 后正常样本；cold 原始记录仅作诊断。
''')
pending=[r['run'] for r in runs.values() if r['result_state'] in ('running','not_started_or_no_record')]
pending += [r['run'] for r in summary.get('cache_diagnostics',[]) if r['result_state'] in ('running','not_started_or_no_record')]
add(f'快照：`{summary["captured_utc"]}`。')
if pending:
    add('\n**草稿：仍有作业未终态；只列已完成的正常样本，不能将整个比较视为已完成。**')
    add('\n待终态：'+', '.join('`'+r+'`' for r in pending)+'.')
add('''
## 128 prompt / 16 decode：规模与四类联合匹配

各组 B=1；0.5B 两组 prefill chunk=128，9B/17B 两组 chunk=32。
FC 为 K、独立参数、12 层；0.5B HB 为 profile9、X、n32+h4；9B/17B 为
profile5-dd4、X、n32/n64+h4、d768。HB 使用区域合批4、SSD doubling/prefix。
17B Tide 是两卡 region placement/per_device，Transformer 是同两卡连续层所有权；
host-stage 传输保持 owner 本地 KV，无复制副本。其余为单卡。

| Tide 配置 | 唯一参数/B | 实际活跃均值/B | 均值/总量 | source | 物理卡 |
|---|---:|---:|---:|---|---|
''')
for label,name in short:
    r=runs[name]; a=r['audit']; total=a['parameters']['unique_parameters']; active=a['effective_touched']['mean']
    add(f'| {label} | {total/1e9:.6f} | {active/1e9:.6f} | {100*active/total:.2f}% | `{r["source"]["source_commit"][:8]}` | {r["physical_devices"]} |')
add('''
A/B 固定近节点总数的 Transformer block 数，C/D 固定近图层数；A/C 匹配实际唯一
总参数，B/D 匹配本条完整真实请求的有效触及均值。均值不是 Full 比例，也不是
整条流的参数并集。A/B 因而往往非常深且窄，C/D 则较浅且宽。四组目的不同，
不能合并成单一胜负，也没有数学、任务质量或全局最优实现的等价声明。
Timed 的均值按 sample/tick 去重并含唯一最终 drain；每输入位置累计触及另外记录，
不能把这个累计量误当作平均活跃参数。

| Tide 配置 | 对照 | blocks | d / FFN | 实参/B | 参数相对误差 |
|---|---|---:|---:|---:|---:|
''')
for label,name in short:
    for key,match in runs[name]['reference']['matches'].items():
        c=match['constructor']
        add(f'| {label} | {key} | {c["layers"]} | {c["d"]} / {c["ffn_width"]} | {match["actual"]["parameters"]/1e9:.6f} | {match["relative_deviation"]["parameters"]*100:.5f}% |')
add('''
| Tide 配置 | 主干 | 首输出/s | prompt 就绪/s | 稳定 decode/位置·s⁻¹ | 峰 allocated / reserved GiB |
|---|---|---:|---:|---:|---:|
''')
for label,name in short:
    for key,s in runs[name]['subjects'].items():
        add(f'| {label} | {key} | {fmt(metric(s,"first_any_seconds"))} | {fmt(metric(s,"prompt_ready_seconds"))} | {fmt(metric(s,"stable_decode_positions_per_second"))} | {fmt(s["max_per_device_peak_allocated_gib"])} / {fmt(s["max_per_device_peak_reserved_gib"])} |')
add('''
这些点中 Tide 的单位置吞吐高于深窄的 A/B，低于浅宽的 C/D。不能据此断言 Tide
已达可达性能：Tide 的小形状、多 owner 与路由执行、参数行选择、接收/记忆/区域
网络产生不同工作量；Transformer 的矩阵形状与串行层数也不同。这里没有将本来
不同的活动比例或计算量作为实现缺陷，也没有把 A/B 当作执行效率上限。

## 较长续接、分块流与大 batch

表中 decode256 是持续主测；B2 9B 使用 decode16 和 stream512，是独立的有限
批吞吐点，不能代替 decode256。改变输入总长度会改变本次生成的完整输入身份；
相同 B/总长度时，不同模型宽度才保证共同 feature 前缀逐元素相同。
9B/17B 的256步续接与9B B2点只测Tide；其已测 A–D 属于上面的128/16请求。
每个长负载仍重新记录实际活动与冻结构造器，但没有把短负载的 A–D 时间冒充为
长负载正式比较。完整长流 A–D 覆盖的是本节 FC1B、HB1B 和 Timed 代表点。

| 配置 | B | prompt/chunk | decode | stream/chunk | 已完成 subjects | 状态 |
|---|---:|---:|---:|---:|---|---|
''')
for label,name in long:
    r=runs[name]
    if 'schedule' not in r:
        add(f'| {label} | — | — | — | — | — | {r["result_state"]} |'); continue
    sc=r['schedule']; done=','.join(k for k,v in r['subjects'].items() if v['state']=='completed') or '—'
    add(f'| {label} | {r["arguments"]["batch"]} | {sc["prefill"]}/{sc["prefill_chunk"]} | {sc["decode_steps"]} | {sc["stream_positions"]}/{sc["stream_chunk"]} | {done} | {r["result_state"]} |')
add('''
| 配置 | 主干 | 首输出/s | prompt 就绪/s | 稳定 decode/位置·s⁻¹ | 流吞吐/位置·s⁻¹ | 完整请求/s | 峰 allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|
''')
for label,name in long:
    for key,s in runs[name].get('subjects',{}).items():
        if s['state']!='completed':continue
        values=[metric(s,k) for k in ('first_any_seconds','prompt_ready_seconds','stable_decode_positions_per_second','chunked_stream_positions_per_second','complete_seconds')]
        add('| '+label+' | '+key+' | '+' | '.join(fmt(v) for v in values)+f' | {fmt(s["max_per_device_peak_allocated_gib"])} / {fmt(s["max_per_device_peak_reserved_gib"])} |')
add('''
较短点与较长点改变了 prompt/chunk、输入和缓存历程，不能把两点差异全部归因于
模型规模或缓存是否填满。分块流吞吐使用真正的 chunked continuation；它与
单位置 decode 是不同服务目标。现有 B/L 是有限、留出显存余量的代表点，
并非声称逐配置搜索到显存上限或最大可行吞吐。

Timed Light64 在这个点的 decode 与分块流均低于 A，不能把 Settle 的相对优势
泛化到 Timed。它每输入位置推进4个逻辑 tick，实际有效参数均值为每 sample/tick
12.753M，每输入位置累计51.047M；后者不是 FLOPs，却说明只看均值会遗漏重复
执行的规模。Tide 的一次B4增量调用中位约0.583秒，因此6.856位置/秒不是每个
样本0.146秒时延。prompt 输入完成13.731秒，而完整路径窗口在后两次增量调用后
才就绪于14.894秒；最后唯一 drain 约0.503秒。

## 17B 区域合批与跨卡并发

同源 `28941994`、同参数/输入、相同请求与精度；region placement 的分区保持一致。
这里的 batch1/batch4 指 `region_batch_size`，输入 B 都为1。不同测点实际物理卡
和运行时争用不同，因此性能变化是这些有限运行的观察。

| 策略 | 物理卡 | 首输出/s | prompt 就绪/s | 稳定 decode/位置·s⁻¹ | 峰 allocated / reserved GiB |
|---|---|---:|---:|---:|---:|
''')
for label,name in [('serial / 区域合批1','agg-inference-hb17b-serial-b1-fp32-v1'),('serial / 区域合批4','agg-inference-hb17b-serial-b4-fp32-v1'),('per_device / 区域合批4','agg-inference-hb17b-fp32-v1')]:
    r=runs[name];s=r.get('subjects',{}).get('tide')
    if not s or s['state']!='completed':
        add(f'| {label} | {r.get("physical_devices","—")} | 待终态 | — | — | — |');continue
    add(f'| {label} | {r["physical_devices"]} | {fmt(metric(s,"first_any_seconds"))} | {fmt(metric(s,"prompt_ready_seconds"))} | {fmt(metric(s,"stable_decode_positions_per_second"))} | {fmt(s["max_per_device_peak_allocated_gib"])} / {fmt(s["max_per_device_peak_reserved_gib"])} |')
add('''
区域合批1与合批4/per_device 的实际选中 K 查询总数为19,942与19,941，Full 总数
也相差1；正消息、候选节点及区域有效 attention pair 总数相同。输出时间摘要一致，
输出 support 摘要不同；保留该差异，不将性能记录当作具体17B轨迹的数值等价证明。
合批4执行的 K 查询行反而更多（77,049 对70,586，包含 padding），但调用更少，
因此不能只用选中行数推断吞吐。合批1→4的串行 decode 吞吐提高33.08%；保持
合批4后，serial→per_device 提高64.42%。后两者实际活跃、work、packing、输出
support 与输出时间摘要全部一致；这个17B点没有观察到并发开关造成的活动轨迹差异。

## 参数切片候选、缓存诊断与剩余边界

`static_view` 保留为可选实现，只在已知、连续且有序 owner IDs 上使用参数视图；
动态 Full 和不规则 owner 仍 gather。它也缓存静态 ID；下列收益属于这个候选整体，
不能只归因于参数复制。9B 点使用同参数、同输入、同调度与精度，但不同物理卡。
基线 source=`1553b301`，候选=`1fbe15c9`；两者 `src/tide` 差分只有候选涉及的
5个文件，benchmark 差分只有未进入计时的 `parameter_gathers.py` 新增文件。

| 9B 参数索引 | 物理卡 | 首输出/s | prompt 就绪/s | 稳定 decode/位置·s⁻¹ | 完整请求/s | 峰 allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|
''')
for label,name in [('gather','agg-inference-hb9b-sustain-fp32-v1'),('static_view','agg-inference-hb9b-static-fp32-v1')]:
    r=runs[name];s=r.get('subjects',{}).get('tide',{})
    if s.get('state')!='completed':
        add(f'| {label} | {r.get("physical_devices","—")} | 待终态 | — | — | — | — |');continue
    values=[metric(s,k) for k in ('first_any_seconds','prompt_ready_seconds','stable_decode_positions_per_second','complete_seconds')]
    add('| '+label+f' | {r["physical_devices"]} | '+' | '.join(fmt(v) for v in values)+f' | {fmt(s["max_per_device_peak_allocated_gib"])} / {fmt(s["max_per_device_peak_reserved_gib"])} |')
if summary['pairwise_checks']['parameter_indexing']['state']=='completed':
    add('''
该点 decode 提高9.496%，完整请求时间下降8.543%；峰 allocated 基本不变，
reserved 略升约0.322 GiB。参数选择的累计物化字节与整个请求峰值存储不同，
不能将视图减少复制直接写成“峰显存下降”。实际活跃、work、packing、逐时有效
参数计数摘要、输出 support 与时间摘要一致；没有从这些汇总推断逐值或梯度等价。
当前通用参数索引默认仍为 gather；static_view 保留供适用配置选择，CPU 公式、
更新与恢复检查见 [parameter indexing](../parameter-indexing.md)。
NPU FP32 static_view 的跨更新原超差在
[最终验证](reception-validation-20261118.md#fp32-static_view-路径)中另行保留；
这里的相同活动汇总与性能收益不将那些超差改记为通过。
''')
add('''

性能记录本身包含输出有限性、原始输出时间 bounds 与状态字节。汇总器独立核对
每个已有正常样本的调用续接、输入分母和唯一末尾 drain，并核对 Tide 时间 bounds；
Transformer 的最终 FP32 KV payload 对照
`2 × B × layers × min(position, window−1) × d × 4 bytes`。这些核对不能证明每个
Tide KV owner 的实际占用和淘汰。

独立 `agg-cache-{fc,settle,timed}-fp32-v1` 诊断采用真实正接收写入的有限队列，
抽查每个 owner/sample 的容量、正条目数、原始 int64 首末时间与最后正写入年龄；
完整保留时间集合与独立写入队列逐项比较。最早保留时间前移且旧条目已过窗口，
才是验证过的淘汰证据。空/不活跃 owner 合法，不要求所有 bank 填满。SSD/Conv
与 pending/drain 也按各自时钟契约检查。诊断数据与耗时不计入性能样本。
不据小模型结果追溯认证所有大模型 bank。
这三点固定 source=`db00c3cd`，B1、prefill128/chunk32、decode256，每64输入位置
抽查并包含阶段边界。FC 为 n8×4层、d128、K、source_residual；Settle/Timed 为
n8+h4、compact-9、d128、X、region_shared、incoming_light_64。
''')
add('''
| 缓存诊断 | 物理卡 | KV owner/sample：观测 / 曾满 / 已证淘汰 | 正写入总量 | 原始输出轴检查 | 诊断峰 allocated / reserved GiB | 状态 |
|---|---|---:|---:|---:|---:|---|
''')
for c in summary.get('cache_diagnostics',[]):
    d=c.get('diagnosis',{}); mem=c.get('device_memory',{})
    peak=lambda key:max((m[key]/2**30 for m in mem.values()),default=None)
    keys=('kv_owner_samples_observed','kv_owner_samples_ever_full','kv_owner_samples_with_verified_eviction')
    counts=' / '.join(str(d.get(k,'—')) for k in keys)
    add(f'| `{c["run"]}` | {c.get("physical_devices","—")} | {counts} | {d.get("positive_writes","—")} | {d.get("output_time_checks","—")} | {fmt(peak("peak_allocated_bytes"))} / {fmt(peak("peak_reserved_bytes"))} | {c["result_state"]} |')
if all(c['result_state']=='passed' for c in summary.get('cache_diagnostics',[])):
    add('''
“曾满”指正条目数达到容量。FC 32/32、Settle 125/128、Timed 48/128 个 owner/sample
曾满；三组所有已观察 owner/sample 均有真实淘汰证据。例：第129个输入后，
node0 的最早保留时间分别从1前移到7、11、5；旧条目相对最后正写入的年龄分别
达到768、1280、512，恰好是各自窗口。最终 Timed drain 的非终端 pending 正强度
计数归零。所有结构、写入队列和原始输出时间轴检查均无失败；SSD 原始时间与
Conv保留集合也全部通过独立写入队列核对。
''')
add('''

本页没有做任意长序列、任意拓扑、BF16 或 CUDA 性能认证；只有记录中的配置。
后端记录的是实际设备、精度与请求 backend 政策，没有为每个性能样本记录逐算子
dispatch trace，因此不凭 `auto` 标签宣称所有 attention 均走某个融合算子。
双卡放置使用 host-stage，并未宣称 peer DMA/HCCL 或线性多卡扩展。

## 复现入口

入口与时间定义见 [inference benchmark](../inference-benchmark.md)，联合匹配见
[benchmark contract](../benchmark-contract.md)。本页同名 JSON 保存准确调度、配置、
subjects、构造器、源码身份和 raw `launch.json` 路径；复现使用新的输出目录。
在相同固定源码的冻结目录，按该 launch 的 `arguments.command` 重建有限作业；
硬件准入、RSS/时间上限和共享身份继续遵循本机协作与执行契约。

例如 9B 持续点的实际核心参数为：

```sh
python -m benchmarks.inference --config configs/scale/hb-9b.json \\
  --config-overrides '{"region_batch_size":4,"ssd_scan_backend":"doubling","ssd_summary_backend":"prefix"}' \\
  --subjects tide --devices npu:0 --precision fp32 --batch 1 \\
  --prefill 512 --prefill-chunk 128 --decode 256 --samples 2 --warmup 1 --no-cold \\
  --threads 4 --allocated-budget-gib 48 --reserved-budget-gib 56 \\
  --reference-output NEW-reference.json --output NEW-inference.json
```
''')

def atomic_text(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as handle:
        staged=Path(handle.name);handle.write(text);handle.flush();os.fsync(handle.fileno())
    os.replace(staged,path)
    descriptor=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


atomic_text(HERE.parent/'docs/evidence/reception-inference-20261118.md','\n'.join(lines)+'\n')
atomic_text(HERE.parent/'docs/evidence/reception-inference-20261118.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('wrote inference evidence draft')
