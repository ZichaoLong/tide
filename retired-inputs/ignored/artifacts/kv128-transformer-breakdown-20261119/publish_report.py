"""Publish reviewed, bounded diagnostic evidence and atomically update navigation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic


def read(name):
    return json.loads((HERE / name).read_text())


def file_identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    summary, costs, audit = read('summary.json'), read('cost-review.json'), read('release-audit.json')
    qualification, compact = read('qualification-review.json'), read('compact-evidence.json')
    assert summary['state'] == 'complete_snapshot'
    assert costs['state'] == qualification['state'] == 'completed_review'
    assert audit['state'] == 'passed'
    jobs = {x['name']: x for x in summary['jobs']}
    for name, job in jobs.items():
        raw = ROOT / f'artifacts/breakdown-{name}-v1/breakdown.json'
        assert file_identity(raw)['sha256'] == job['file']['sha256']
        assert job['normal']['timing_complete']

    text = (HERE / 'report-draft.md').read_text()
    def broad(name, phase):
        return {x['module']: x['seconds'] for x in costs['cases'][name]['stages']['nonoverlapping_broad_cut'] if x['phase'] == phase}
    def subtree(name, phase, label):
        return sum(x['inclusive_seconds'] for x in costs['cases'][name]['stages']['selected_subtree_views'] if x['phase'] == phase and x['label'] == label)
    t = broad('train-kv128', 'training.forward')
    ffn = subtree('train-kv128', 'training.forward', 'ffn')
    rows = [
        ('消息收集和接收', t['messages.gather'] + t['reception']),
        ('描述及区域评分', t['description'] + t['regional.scoring']),
        ('分配求解', t['allocation']),
        ('KV准备/历史整理', t['memory.K.prepare']),
        ('SSD准备', t['memory.S.prepare']),
        ('Full中的FFN（含归一化）', ffn),
        ('Full其余（扣除FFN）', t['memory.K.full'] + t['memory.S.full'] - ffn),
        ('状态join/split', t['state.join'] + t['state.split']),
    ]
    rows.append(('输入、分组及未细分余项', sum(t.values()) - sum(v for _, v in rows)))
    rows.append(('合计', sum(t.values())))
    text = text.replace('{{TRAIN_FORWARD}}', table(['Tide同步前向模块', '每更新ms'], [(k, f'{v*1000:,.3f}') for k, v in rows]))

    normal = {n: jobs[n]['normal']['computed'] for n in ('infer-kv128', 'infer-tf32')}
    rows = []
    for label, key, scale in [
        ('首主干输出 ms', 'first_backbone_output_seconds', 1000),
        ('prefill128总计 ms', 'prefill_seconds', 1000),
        ('prefill输入位置/s', 'prefill_positions_per_second', 1),
        ('稳定decode每步平均 ms', 'stable_decode_seconds', 1000/15),
        ('稳定decode输入位置/s', 'stable_decode_positions_per_second', 1),
        ('完整请求 s', 'complete_seconds', 1),
        ('独立完整序列/s', 'independent_complete_sequences_per_second', 1),
    ]:
        rows.append([label] + [f'{normal[n][key]["median"]*scale:,.3f}' for n in normal])
    text = text.replace('{{INFERENCE_NORMAL}}', table(['正常B8推理', 'Tide KV128', 'Transformer32'], rows))

    phase_values = {}
    for phase in ('prefill', 'decode'):
        a, b = broad('infer-kv128', phase), broad('infer-tf32', phase)
        af = subtree('infer-kv128', phase, 'ffn')
        bf = b['transformer.ffn'] + b['transformer.ffn_norm']
        cut = [
            ('FFN，含自身归一化', af, bf),
            ('KV准备和历史整理', a['memory.K.prepare'], 0.),
            ('SSD准备', a['memory.S.prepare'], 0.),
            ('KV/SSD Full扣除FFN', a['memory.K.full'] + a['memory.S.full'] - af, 0.),
            ('区域分配求解', a['allocation'], 0.),
            ('描述与区域评分', a['description'] + a['regional.scoring'], 0.),
            ('消息收集与接收', a['messages.gather'] + a['reception'], 0.),
            ('状态join/split', a['state.join'] + a['state.split'], 0.),
            ('Transformer QKV/输出投影', 0., b['transformer.qkv'] + b['transformer.output']),
            ('Transformer Attention', 0., b['transformer.attention']),
            ('Transformer RoPE', 0., b['transformer.rope']),
        ]
        cut.append(('其他计算、归一化及未细分调度', sum(a.values()) - sum(x[1] for x in cut), sum(b.values()) - sum(x[2] for x in cut)))
        cut.append(('合计', sum(a.values()), sum(b.values())))
        phase_values[phase] = {label: (av * 1000 / (16 if phase == 'decode' else 1), bv * 1000 / (16 if phase == 'decode' else 1)) for label, av, bv in cut}
    rows = []
    for label in phase_values['prefill']:
        values = (*phase_values['prefill'][label], *phase_values['decode'][label])
        rows.append([label] + [f'{v:,.3f}' if v else '—' for v in values])
    text = text.replace('{{INFERENCE_STAGES}}', table(['同步模块，单位ms', 'Tide prefill总计', 'TF prefill总计', 'Tide decode每步', 'TF decode每步'], rows))
    text = text.replace('{{MATRIX_SHAPES}}', '''实际prefill FFN示例：Tide为`[128,160,768] × [128,768,3072]`的独立节点BMM，
Transformer为`[8,32,4352] × [4352,30416]`的二维gate/up权重投影。
decode示例：Tide SSD FFN为`[16,8,768] × [16,768,3072]`，
Transformer为`[8,1,4352] × [4352,30416]`，以及`[8,1,15208] × [15208,4352]`。
这只是代表操作数，不是所有算子唯一形状，也没有从dispatcher形状推断原生kernel的具体tiling。

窄KV仍保留摘要宽192；decode有`[128,8,1,129] × [128,8,129,192]`的历史摘要计算。
16步中640次历史字段gather产生21.141GiB重复逻辑输出，1,200次state.join的cat产生21.817GiB；
所有参数索引作用域产生261.385GiB重复逻辑输出，其中FFN为193.641GiB。
这些是把实际输出张量逻辑字节按调用数相加，**不是总线流量、峰值显存或带宽测量**。''')
    assert '{{' not in text
    text = text.replace('4.026×', '4.027×')
    text = text.replace('旧`run_stream`的backward区间', '前向窗口包含zero_grad、chunk调度及readout。旧`run_stream`的backward区间')
    text = text.replace('缓存最终大小不等于上述峰值。', '缓存最终大小不等于上述峰值。训练allocator采样含run_stream末尾有限性检查，\n检查不计入complete时间，但其临时分配可能计入峰值。')

    plan = read('plan.json')
    plan['state'] = 'completed_reviewed'
    plan['updated'] = datetime.now(timezone.utc).isoformat()
    plan['jobs'] = audit['jobs']
    plan['release_audit'] = file_identity(HERE / 'release-audit.json')
    atomic(HERE / 'plan.json', plan)
    compact['publication'] = dict(source='7196fcabd7ab3b401c1ebdb6107442450a99996c', production_tree='5dba5b15c0c71ce729ff82c054e9468916acd7e5',
        upstream='4f5b9f20a4030d68ee66ae2d26e34659a5e37234', release_audit=audit, development_checks=read('development-checks.json'), final_plan=plan,
        files=[file_identity(HERE / name) for name in ['publish_report.py', 'report-draft.md', 'compact-evidence.json', 'preflight.json', 'plan.json', 'development-checks.json', 'release-audit.json', 'audit_release.py']])
    target = ROOT / 'docs/evidence/kv128-transformer-breakdown-20261119'
    atomic(target.with_suffix('.json'), compact)
    atomic_text(target.with_suffix('.md'), text)

    path = ROOT / 'docs/STATUS.md'
    previous = path.read_text()
    previous = previous[previous.index('# 当前交接：五类记忆/参数绑定有限计划已完成'):]
    prefix = '''# 当前交接：KV128与32层Transformer训练/推理细拆已完成

追加四点固定工具`7196fcab`（已推送），生产src/tide未变；正常吞吐、同步模块和逐算子诊断分列。
完整阶段表、公式/实际矩阵、数值边界与复现入口见[本次证据](evidence/kv128-transformer-breakdown-20261119.md)。
中尺度d192/KV32、B16/L128/chunk32的继续训练：Tide966.043、Transformer32 4963.322位置/s，差5.138倍；
正常均值更新2.119989/0.412627s，allocated峰值22.739/6.644GiB。这不是9B训练结果。
9B图同B8：prefill335.586/1963.709位置/s；稳定decode18.231/137.153，差7.523倍；
首主干输出0.749359/0.129075s，不含任务头。双方分别在同物理NPU顺序测量，均FP32、5样本。
独立同步诊断识别出FFN权重索引、KV历史整理、Full padding、区域分配和大量非矩阵反向；
所报算子时间含同步，不是正常wall占比。Normal/diagnostic、15步stable/16步诊断分母不可混用。

34项观察器CPU检查通过。四次进程内CPU锚点恢复后真实更新均在原阈值内；KV有限非exact仍保留，
不自动归因舍入，不是新的磁盘checkpoint/新解释器资格。推理两诊断完整输出SHA一致；
normal未保留输出SHA，20个SSD卷积时钟仅有shape/dtype等元数据，不能称全部私有状态逐值一致。
4个background.slice服务passed、26个记录后代均终态，冻结源各461文件一致；产物约0.41/8GiB。
无新活动作业、无共享占用卡、无生产默认变更、无新增9B训练/多NPU/其他backend资格；四份用户文档原字节保留。
本轮追加有限计划已完成，后续优化机会留在ROADMAP；五类候选既有事实如下。

---

'''
    atomic_text(path, prefix + previous)
    path = ROOT / 'docs/ROADMAP.md'
    previous = path.read_text()
    previous = previous[previous.index('# 当前周期后续机会'):]
    previous = previous.replace('1. **共享执行分组。**', '''1. **独立权重与反向组织。** [KV128/Transformer细拆](evidence/kv128-transformer-breakdown-20261119.md)
   已完成四点；优先按收益选择节点权重索引/布局、Full query数分组、历史bank整理及反向索引汇总，
   中尺度分配求解亦是热点。任何新候选需保留私有状态/原始时间/梯度，并以正常全路径计时确认收益；
   同步算子时间不能直接当作可回收wall时间。本条是后续机会，本轮没有待运行测点。
2. **共享执行分组。**''').replace('2. **历史KV与padding。**', '3. **历史KV与padding。**').replace('3. **数值因果与新范围。**', '4. **数值因果与新范围。**').replace('4. **下游效果。**', '5. **下游效果。**')
    atomic_text(path, previous)
    path = ROOT / 'docs/evidence/README.md'
    previous = path.read_text()
    insert = '''The [KV128 / fixed32 Transformer breakdown](kv128-transformer-breakdown-20261119.md)
separates ordinary medium continuing training and large inference from intrusive
stage/operator diagnostics. It includes weight indexing, actual packing, backward
costs, restored-update checks and their limits; it adds no 9B training qualification.

'''
    if 'kv128-transformer-breakdown-20261119.md' not in previous:
        atomic_text(path, previous.replace('# Current evidence\n\n', '# Current evidence\n\n' + insert, 1))
    path = ROOT / 'docs/architecture.md'
    previous = path.read_text()
    insert = '| paired ordinary training/inference and separate synchronized module/operator diagnosis | benchmarks/training_breakdown.py, benchmarks/inference_breakdown.py, benchmarks/detailed_stages.py | test_training_breakdown.py, test_inference_breakdown.py, test_detailed_stages.py |\n'
    if 'benchmarks/training_breakdown.py' not in previous:
        atomic_text(path, previous.replace('| intrusive decode stages/shapes', insert + '| intrusive decode stages/shapes', 1))
    path = ROOT / 'docs/performance-lessons.md'
    previous = path.read_text()
    if 'kv128-transformer-breakdown-20261119.md' not in previous:
        atomic_text(path, previous + '''

## Narrow KV paired cost diagnosis, 2026-11-19

The [paired diagnosis](evidence/kv128-transformer-breakdown-20261119.md) separates
ordinary timing, synchronized modules and synchronized dispatches. Fixed-FFN KV128
still holds 6.002B independent FFN parameters. In the large B8 decode operator pass,
FFN weight indexing takes much more observed time than its BMMs; narrowing memory
does not remove those indexed weight copies. Actual publish indices and FFN operands
show about 5x executed/selected KV rows in that pass, with separate prefill/training ratios.
Storage sharing, mathematical sparsity and actual rectangular work remain different facts.

The medium continuing-training gap is mostly forward and backward, with a small
AdamW difference. Its backward includes many index writes, gather backwards,
tensor creations and pointwise operations; matrices are not the only cost.
Unscoped autograd operators cannot be assigned to forward modules from shape alone.
Entry synchronization must finish before a child timer starts; parent remainders
still include unwrapped work and waiting. These diagnostics perturb scheduling,
so neither their shares nor their logical output bytes identify ordinary wall-time
shares, bus traffic or recoverable speedup. Optimization requires a new full-path
measurement and unchanged numerical/continuation checks at the intended scale.
''')
    print(json.dumps(dict(state='published', markdown=str(target.with_suffix('.md').relative_to(ROOT)), json_bytes=target.with_suffix('.json').stat().st_size)))


if __name__ == '__main__':
    main()
