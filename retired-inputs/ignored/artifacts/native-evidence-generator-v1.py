import hashlib
import json
import os
from pathlib import Path
import statistics

from scripts.job import atomic

root = Path.cwd()
def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):
    return json.loads(Path(path).read_text())
def text_atomic(path, value):
    stage = path.with_suffix(path.suffix+'.staging')
    with stage.open('w') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(stage,path)
    fd=os.open(path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)

record = dict(schema='tide-native-performance-evidence-v1',
              native_source='89ce08ff858724648152790502f8c65d8d8b1222',
              math='4f5b9f20a4030d68ee66ae2d26e34659a5e37234',
              scope='standalone regional scoring slice; no complete native Tide or native-memory profiling qualification',
              numerical_threshold=dict(atol=2e-6,rtol=2e-5), runs=[], observations=[], comparisons=[])
lookup={}
for label in ('cpu','inner4-cpu','npu'):
    directory=root/'artifacts'/('native-regional-perf-'+label+'-v1')
    manifest=read(directory/'results/programme.json')
    assert manifest['state']=='completed' and read(directory/'status.json')['state']=='passed'
    record['runs'].append(dict(name=directory.name,manifest=identity(directory/'results/programme.json'),
                              source=manifest['source'],binary_sha256=manifest['binary_sha256'],
                              driver_sha256=manifest['driver_sha256'],
                              launch=read(directory/'launch.json'),status=read(directory/'status.json'),
                              queue=read(directory/'queue.json') if (directory/'queue.json').exists() else None,
                              fixture_hashes={name:read(directory/'results'/(name+'-fixture')/'sha256.json')
                                              for name in ('medium','large')}))
    for child in manifest['children']:
        assert child['state']=='completed'
        row=read(child['output']); samples=row['samples_ms']; median=statistics.median(samples)
        assert len(samples)==5 and all(value>0 for value in samples)
        if row['client']=='standalone-libtorch':
            assert row['build_revision']==record['native_source'] and not row['build_dirty']
        result=dict(experiment=label,identity=child['identity'],result_file=identity(child['output']),
                    metadata=row,median_ms=median,range_over_median=(max(samples)-min(samples))/median,
                    sampled_process_tree_peak_rss_bytes=child['sampled_process_tree_peak_rss_bytes'])
        record['observations'].append(result);lookup[label,child['identity']]=result
    for comparison in manifest['comparisons']:
        record['comparisons'].append(dict(experiment=label,**comparison))
for item in record['runs'][1:]:
    assert item['fixture_hashes']==record['runs'][0]['fixture_hashes']
assert len(record['observations'])==120 and len(record['comparisons'])==84

same=root/'artifacts/native-regional-npu-same-v1'
pool=root/'artifacts/native-regional-npu-pool-v1'
record['same_stream_qualification']=dict(file=identity(same/'checks/status.json'),record=read(same/'checks/status.json'),
                                       status=read(same/'status.json'),queue=read(same/'queue.json'))
record['pool_stream_qualification']=dict(file=identity(pool/'programme.json'),record=read(pool/'programme.json'),
                                       status=read(pool/'status.json'),queue=read(pool/'queue.json'))
record['final_parameter_inspection']=dict(file=identity(root/'artifacts/native-parameter-inspection-v1.json'),
                                         record=read(root/'artifacts/native-parameter-inspection-v1.json'))
record['summary']=dict(normal_observations=120,normal_samples=600,
    normal_comparisons=84,normal_comparisons_passed=sum(row['state']=='passed' for row in record['comparisons']),
    finite_failed_comparisons=[{k:row[k] for k in ('experiment','shape','dtype','mode','workers')}
                              for row in record['comparisons'] if row['state']!='passed'],
    samples_range_over_quarter_median=sum(row['range_over_median']>.25 for row in record['observations']))
atomic(root/'docs/evidence/native-regional-performance-20261118.json',record)

lines=['# LibTorch 区域切片：NPU 资格与正常性能 · 2026-11-18','',
'[JSON](native-regional-performance-20261118.json) 保留 120 个正常计时观察、600 个样本及 84 个末步数值比较；',
'三组作业均完成，80/84 个末步比较在原阈值内。完整 Tide 仍由 PyTorch 实现；本页仅涉及区域评分 Attention+FFN。',
'独立二进制固定到 `89ce08ff858724648152790502f8c65d8d8b1222`，CPU 构建/公式见[既有记录](native-regional-cpu-20261118.md)。','',
'## 硬件与数值边界','',
'NPU 同 stream 检查 28/29 通过：两精度、三模式、1/2/4 workers、三步更新、新进程恢复及第四步。',
'唯一失败为 BF16 四 worker 恢复后的 `v_weight` 梯度，最大差 `7.62939453125e-6`；原 `atol=2e-6, rtol=2e-5` 未变。',
'另外八个 pool-stream 训练/恢复观察全部在原阈值内，覆盖 FP32/BF16、2/4 workers、三步后恢复第四步。',
'pool 的对照是已检查的同布局 Python 三步及 native 串行恢复结果；这不是多 stream 的完整 Tide 资格，也不覆盖跨 NPU transport。',
'原同 stream 失败没有因 pool 通过而改标；有限 CPU 重放未隔离该梯度超差，见[舍入调查](capacity-rounding-20261118.md)。','',
'较大正常性能形状在五次更新后另出现四个严格比较失败：CPU BF16/4 workers，NPU BF16/1、2、4 workers。',
'CPU 最大输出差 `0.00814360`，另四个参数梯度张量超差；NPU 最大输出差分别 `0.00374132`、`0.00374132`、`0.00434518`，',
'各参数/梯度/AdamW moments 本身仍在原阈值内。这些都是有限值，执行完成与数值通过分列；不能称这些 BF16 训练吞吐为严格等价加速。',
'只读末步参数检查发现 CPU 最多约 `1.90e-6`、NPU 最多约 `8.95e-7` 的 FP32 参数差，并观察到若干参数 BF16 cast 不同。',
'这是五步后的状态，不能倒推第五次前向的首因；未进一步隔离每条训练轨迹，不能把全部超差归为同一个原因。','',
'## 负载和计时','',
'所有任务共享一组区域参数；相容节点在区域内批量处理，外层 worker 执行独立区域任务。两形状为：','',
'| 名称 | 区域任务数 | 节点/区域 | B | L | 描述宽度 |','|---|---:|---:|---:|---:|---:|',
'| medium | 4 | 8 | 8 | 128 | 64 |','| large | 4 | 16 | 16 | 128 | 128 |','',
'三组实验的 fixture SHA256 完全一致。每个观察是独立进程；两次预热使用丢弃的 owner，随后新的 owner 记录五次同步采样。',
'训练含 zero_grad、四任务 forward/loss/backward、一次共享 AdamW 和最终设备同步；首个采样仍含该 owner 的首次 slot 建立。',
'不含 fixture/model 加载、线程池创建、输出序列化及事后数值比较。训练样本连续改变参数，不能当 IID；无梯度推理与含梯度前向分别测量。',
'下面是五次样本中位数，单位毫秒。`†` 表示 range/median>25%，保留而不用于稳定排序；它也可能反映首次 slot 成本，非统计置信区间。','']
for device in ('cpu','npu'):
    lines += ['### '+device.upper(),'',
              '| 形状 / 精度 / 模式 | Python串行 inner1 | C++串行 inner1 | C++2任务 inner1 | C++4任务 inner1 |'+(' Python串行 inner4 | C++串行 inner4 |' if device=='cpu' else ''),
              '|---|---:|---:|---:|---:|'+('---:|---:|' if device=='cpu' else '')]
    for shape in ('medium','large'):
        for dtype in ('float32','bfloat16'):
            for mode in ('inference','grad_forward','train'):
                keys=[(device,f'{shape}-{dtype}-{mode}-{client}{workers}') for client,workers in
                      [('python',1),('cpp',1),('cpp',2),('cpp',4)]]
                if device=='cpu':
                    keys += [('inner4-cpu',f'{shape}-{dtype}-{mode}-{client}1') for client in ('python','cpp')]
                cells=[]
                for key in keys:
                    row=lookup[key]; cells.append(f"{row['median_ms']:.3f}"+('†' if row['range_over_median']>.25 else ''))
                lines.append('| '+f'{shape} / {dtype} / {mode}'+' | '+' | '.join(cells)+' |')
    lines.append('')
lines += ['CPU 相同 inner1 下，C++ 串行与 Python 串行的大形状训练时间接近；外层区域并发能利用更多 CPU。',
'同样四 CPU 预算，大形状 FP32 的 C++ 四任务约 870 ms，Python 串行 inner4 约 1276 ms；没有测 Python 线程池。',
'较小形状的 inner4 多个观察波动很大，不给语言或线程策略的普遍排序。',
'NPU medium/FP32 训练 Python/C++串行/四任务约 13.882/8.498/7.561 ms；large/FP32 则约 14.825/14.413/15.594 ms。',
'这支持在小算子形状研究提交成本；并发没有随形状增大而持续获益，不能据局部结果替换整个模型。','',
'## 资源与复现','',
'CPU 使用 affinity 76–79，inner1 与 inner4 两组串行衔接；NPU 正常性能在物理1、host affinity120–123，',
'pool 也是独占队列准入的物理1、host affinity140–143。CPU/内存/文件系统仍是共享主机资源。',
'每个 C++ 子进程保留实际编译源、ABI、库加载路径、current/pool stream IDs，明确无 Python runtime 库依赖。',
'BF16 projection/FFN dtype 由执行 guard 检查；参数/梯度/AdamW、Attention乘积和标量 readout 为 FP32。',
'没有原生逐算子 placement trace 或设备 allocator 峰值；RSS 由外围进程树采样，不能当显存。没有 CUDA 构建或真机资格。','',
'本阶段保留 PyTorch 主实现与可选 LibTorch 区域实验；同 stream 默认、pool 单列实验，没有自动选择语言或完整 native backend。',
'复现先按[native入口](../native-regional.md)构建固定源码，使用新输出目录和当次准入。原始命令在 JSON `runs[].launch.command`；',
'冻结 driver 为 `artifacts/native-performance-driver-v1.py`、`artifacts/native-performance-inner4-driver-v1.py`，其完整字节另随本证据提交到',
'`native-regional-performance-driver-v1.py` 与 `native-regional-performance-inner4-driver-v1.py`；运行时必须位于记录源码根并由 launcher 传入新输出。',
'同 stream/pool 的逐案例命令、日志、输入和对照位置在相应 JSON 中，CPU只读末步参数检查不构成新模型资格。','']
text_atomic(root/'docs/evidence/native-regional-performance-20261118.md','\n'.join(lines))
for name in ('native-performance-driver-v1.py','native-performance-inner4-driver-v1.py'):
    target='native-regional-performance-'+('inner4-driver-v1.py' if 'inner4' in name else 'driver-v1.py')
    text_atomic(root/'docs/evidence'/target,(root/'artifacts'/name).read_text())
print(json.dumps(record['summary']))
