"""Finalize this finite follow-up after every job has reached terminal state."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

from benchmarks.search_report import write_text_atomic
from scripts.job import atomic

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
data=json.loads((BASE/'summary.json').read_text())
assert data['all_complete']
assert len(data['runs']) == 12
preflight=json.loads((BASE/'preflight.json').read_text())
source=data['runs'][0]['launch']['source']
assert all(x['launch']['source']==source for x in data['runs'])
archive_bytes=subprocess.check_output(['git','archive',source],cwd=ROOT)
with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
    source_files={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive if m.isfile()}
lifecycle=[]
for entry in data['runs']:
    run=ROOT/'artifacts'/entry['name']
    status=json.loads((run/'status.json').read_text())
    unit='tide-'+entry['name']+'.service'
    observed=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID,ActiveState,SubState,Slice,ControlGroup,Result'],text=True)
    properties=dict(s.split('=',1) for s in observed.splitlines() if '=' in s)
    assert properties['MainPID']=='0' and properties['ActiveState']=='inactive'
    live=[]
    for process in status.get('descendant_identities',[]):
        stat=Path(f"/proc/{process['pid']}/stat")
        try: fields=stat.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError: continue
        if int(fields[19])==process['start_ticks'] and fields[0] not in ('Z','X'): live.append(process)
    assert not live
    mismatches=[name for name,sha in source_files.items() if not (run/'source'/name).exists() or hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()!=sha]
    assert not mismatches
    lifecycle.append(dict(name=entry['name'],state=status['state'],unit=unit,properties=properties,
                          live_recorded_descendants=live,frozen_regular_files=len(source_files),source_mismatches=mismatches))
protected=[]
for original in preflight['user_files']:
    content=(ROOT/original['path']).read_bytes()
    assert len(content)==original['bytes'] and hashlib.sha256(content).hexdigest()==original['sha256']
    protected.append(dict(original,unchanged=True))
for name,sha in preflight['checks']['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
upstream=subprocess.check_output(['python','scripts/check_upstream.py'],cwd=ROOT,text=True)
assert json.loads(upstream)['state']=='passed'
data['release_checks']=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),source=source,
    tested_files=preflight['checks'],protected_user_files=protected,upstream=json.loads(upstream),
    lifecycle=lifecycle,quota=subprocess.check_output(['quota','-s'],text=True),
    production_tree=subprocess.check_output(['git','rev-parse',f'{source}:src/tide'],text=True).strip())
data['bounded_extension']=json.loads((BASE/'extensions.json').read_text())

rows={x['name']:x['result'] for x in data['runs']}
old={x['result']['name']:x['result'] for x in data['original_comparisons']}
depth_names=[f'depth-{scale}-l{layers}-fp32-v1' for scale in ('9b','17b') for layers in (32,48)]
def memory(r):
    m=r['metrics']['peak_bytes'];return f"{m['allocated']/2**30:.3f}/{m['reserved']/2**30:.3f}"
def depth_row(name):
    r=rows[name];s=r['spec'];m=r['metrics'];err=r['comparison']['match']['relative_deviation']['parameters']*100
    return f"| {'9B' if '-9b-' in name else '17B'} | {s['layers']} | {s['d']}/{s['ffn_width']}/{s['head_dim']} | {r['parameters']/1e9:.6f} | {err:+.6f}% | {m['stable_decode_positions_per_second']:.4f} | {m['mean_decode_call_seconds']*1000:.2f} | {memory(r)} | {r['physical_devices']} |"
lines=["# 2026-11-18：固定32/48层Transformer与Tide batch/多卡补测",'',
"本页回答交付后的两个追加问题：约9B/17B的常规深度Transformer decode如何，以及Tide能否用较大batch或更多卡提高吞吐。四个深度点和八个Tide点均完成，不替代原A–D。B2→B4仍有吞吐收益与容量余量后，按记录中的有限扩展理由补9B单卡B8及17B两卡/四卡B8；本轮batch阶梯止于8。",'',
f"冻结源码 `{source}`；生产src/tide树未改变。原始配置、样本、参数/输入哈希、生命周期和复现命令保存在[机器可读记录](transformer-depth-batch-20261118.json)。",'',
"## 共同口径",'',
"全部FP32、prompt128/chunk32/decode16、KV窗口按原配置；新点两次warmup、五个正常样本。每条请求从零态开始，请求内真实续接KV/SSD；主干输出逐调用同步后释放。稳定decode排除第一次增量，分子为其余15次的实际B×输入位置。表中吞吐为五个样本各自吞吐的中位数，平均步时延为每样本稳定耗时/15的中位数。",'',
"构造、哈希、传参、诊断和热身不计时。所有参与卡均同步，跨NPU仍为host-stage；不含任务头、采样、网络或服务队列。64GiB Ascend910按协作队列独占准入，主机和外部任务不受控制；不同测点物理卡不同。allocated/reserved是正常样本各卡最大峰值，未跨卡求和；48/56GiB采样预算不是硬隔离。",'',
"## 固定深度、近总参数Transformer",'',
"沿用原Transformer和近参数构造器：固定层数后，搜索合法d和FFN以最小化参数误差。它没有搜索全局最快结构；头宽、FFN比例和矩阵形状随匹配变化。使用QKV及gate/up合并投影、真实KV缓存和auto Attention，正常计时函数与原A–D相同。9B单卡；17B两卡连续层放置。目标参考和共同768维输入前缀与原Tide逐字节匹配。",'',
"| 目标 | 层数 | d/FFN/头宽 | 实际参数B | 参数偏差 | 稳定decode 位置/s | 平均步ms | 峰allocated/reserved GiB | 物理卡 |",
"|---|---:|---|---:|---:|---:|---:|---:|---|"]
lines += [depth_row(n) for n in depth_names]
lines += ['',"原B1短请求的Tide与5层C保留原身份：source=`28941994`、warmup1/samples2，非与新点同轮交错配对重测。所有值都是该主干的实测，不能从层数单独外推吞吐。",'',
"| 规模 | 原Tide | 原5层C | 新32层 | 新48层 |","|---|---:|---:|---:|---:|"]
for scale in ('9b','17b'):
    vals=[old[f'original-{scale}-{s}']['metrics']['stable_decode_positions_per_second'] for s in ('tide','C')]
    vals += [rows[f'depth-{scale}-l{l}-fp32-v1']['metrics']['stable_decode_positions_per_second'] for l in (32,48)]
    lines.append('| '+scale+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines += ['',"9B的32层点高于旧5层C，17B的48层点高于32层；这同时改变了宽度、头宽和FFN，且物理卡/运行时间不同，不归因成纯层数效应。五样本range/median在JSON逐点保留。",'',
"## Tide batch与多卡",'',
"仍使用source_residual、X、profile5-dd4、d768、独立参数、区域batch4、SSD doubling/prefix、gather索引。9B为n32+h4，17B为n64+h4；多卡使用参数平衡的完整区域放置及per_device并发。输入Batch与region_batch_size是两个不同的轴。较大batch增加独立样本，总吞吐改善不等于每条请求的decode延迟减少。",'',
"| 规模 | 输入B | 卡数 | 稳定decode 位置/s | 平均步ms | 每卡位置/s | 峰allocated/reserved GiB | 物理卡 |",
"|---|---:|---:|---:|---:|---:|---:|---|"]
for scale in ('9b','17b'):
    group=[old[f'original-{scale}-tide'],*[rows[f'batch-hb{scale}-b{b}-fp32-v1'] for b in (2,4)]]
    if scale=='9b':group.append(rows['batch-hb9b-b8-fp32-v1'])
    if scale=='17b':group.extend([rows['batch-hb17b-b8-fp32-v1'],rows['batch-hb17b-b4-four-fp32-v1'],rows['batch-hb17b-b8-four-fp32-v1']])
    for r in group:
        m=r['metrics']
        lines.append(f"| {scale} | {r['arguments']['batch']} | {m['participating_devices']} | {m['stable_decode_positions_per_second']:.4f} | {m['mean_decode_call_seconds']*1000:.2f} | {m['per_card_decode_positions_per_second']:.4f} | {memory(r)} | {r['physical_devices']} |")
lines += ['',"B1沿用原记录；新B2/B4/B8均与原模型的初始参数哈希一致。batch改变输入样本数，也会改变平均活动，不能把总吞吐比称为逐活动轨迹的严格加速。新点的实际触及参数、Full、padding和原样本都保留在JSON。表中只搜索到B8，未搜索全部可行batch或最大吞吐点；短请求的容量不能直接转移到更长prompt/历史。",'',
"B2→B4的全请求实际触及参数均值只变化约−0.053%/−0.300%（9B/17B），没有通过大幅降低活动量换吞吐；K packing执行/选择比分别由4.66→5.48、4.82→5.50。这些是全请求诊断，不是稳定decode阶段的精确成本占比。",'',
"17B B4/B8两卡与四卡的参数、输入和轨迹摘要核对如下。B4的Full次数为125638/125639，K执行行数437424/414368，K调用300/480，输出支持摘要不同、原时间轴一致；分区同时改变合批与packing，不能称为严格同轨迹的纯并发加速。摘要一致也不等于逐值/梯度等价，没有新增四卡训练或恢复资格：",'',
"```json",json.dumps({k:v for k,v in data['pairwise_checks'].items() if 'two_vs_four_devices' in k},ensure_ascii=False,indent=2),"```",'',
"B4分区后的S调用也由160增至320；静态前驱运输计划的远程条目从16增至54，按目的分区计的远程源节点成员数从2432增至3496（+43.75%）。这些计划计数不是实测复制次数、字节或通信耗时，但说明四卡同时增加分组与跨分区连接；不能只根据每卡参数减少就推断更快。本次没有新增通信/调度分段trace，未把吞吐差精确分摊给其中某项。",'',
"多卡参数分布与副本并行是不同策略。本页实际测的是一个模型跨卡；两个完整副本各处理一组请求的并发吞吐未测，不能把两个孤立吞吐简单相加充作测量结果。比较效率应同时看每卡吞吐和单步延迟，也应为Transformer选择相同batch/卡数后再比较；本页没有把Tide B4与Transformer B1排列成同负载胜负。",'',
"## 检查与复现",'',
"新入口及原推理回归20项通过（11.33秒）；四个大构造器的实际meta参数数目独立核对，CPU小例覆盖真实缓存、位置、分母、放置和拒绝覆盖输出。四个NPU深度点另核对原参考/输入前缀、全部请求位置、稳定吞吐分母及最终FP32 KV payload。目标参考损坏等拒绝分支未被新增单测穷举；成功运行不等于覆盖所有错误输入。",'',
"十二个作业全部passed，原始服务MainPID与已记录PID/start后代均终态，冻结归档常规文件与各自Git提交一致，四份用户文件哈希未改变。原CPU/BF16等历史数值失败没有修改；本页有限推理完成不认证训练或任意后端。",'',
"使用同名JSON内每项`launch.arguments.command`在冻结提交下重新启动，换用新输出目录；设备准入、RSS和时间预算沿用[执行契约](../execution-flows.md)。Transformer入口例如：",'',
"```sh","python -m benchmarks.transformer_depth --parameters 8778358224 --layers 32 \\",
"  --devices npu:0 --precision fp32 --batch 1 --prefill 128 --prefill-chunk 32 \\",
"  --decode 16 --samples 5 --warmup 2 --threads 4 --chunk 256 \\",
"  --allocated-budget-gib 48 --reserved-budget-gib 56 --output NEW-depth.json","```",'',
"Tide使用原`benchmarks.inference`，设定`--batch 2/4/8`；17B分别使用两卡或四卡的`--placement region --region-concurrency per_device --transport host_stage`。完整构造、设备映射和输出路径以记录为准。"]
atomic(ROOT/'docs/evidence/transformer-depth-batch-20261118.json',data)
write_text_atomic(ROOT/'docs/evidence/transformer-depth-batch-20261118.md','\n'.join(lines)+'\n')
print('Final evidence written after terminal checks')
