"""Update current navigation only after the reviewed capacity evidence exists."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmarks.search_report import write_text_atomic


def main():
    data = json.loads((BASE / 'summary.json').read_text())
    release = json.loads((BASE / 'release-audit.json').read_text())
    assert data['all_terminal'] and data['all_series_have_confirmed_winner']
    assert (ROOT / 'docs/evidence/decode-capacity-20261118.md').exists()
    winners = {r['series']: r['confirmed_winner'] for r in data['series']}
    def metric(series):
        row = winners[series]
        return row, row['metrics']
    table = ['| 规模/设备 | Tide B/位置每秒 | 五层C B/位置每秒 | 32层 B/位置每秒 | 48层 B/位置每秒 |',
             '|---|---:|---:|---:|---:|']
    for scale, cards in [('9b', 1), ('17b', 2), ('17b', 4)]:
        cells = []
        for kind in ('tide', 'tf5', 'tf32', 'tf48'):
            row, m = metric(f'{scale}/{kind}/{cards}')
            cells.append(f"{row['batch']} / {m['median_stable_decode_positions_per_second']:.2f}")
        table.append(f"| {scale.upper()}/{cards}卡 | " + ' | '.join(cells) + ' |')
    counts = data['counts']['classifications']
    text = '\n'.join([
        '## 追加完成：固定预算下的 decode batch 搜索', '',
        '已完成获准的12系列：9B单卡、17B双卡/四卡的Tide与近总参数Transformer C/32/48层。',
        '全部FP32、prompt128/chunk32/decode16，每卡48/56GiB allocated/reserved采样预算。',
        '下面列各自预算内正式确认的最佳实测点，B为独立输入batch；吞吐是整个batch的主干位置/s。',
        '五层C沿用原C的形状；32/48层是额外近总参数参考，不替代原A/B/C/D。', '',
        *table, '',
        '较大batch改善总吞吐，同时增加每条序列的步时延。四卡的容量收益与同batch效率分开比较；',
        '这次是固定模型、执行候选和短请求内的有限搜索，不是硬件最大容量、全整数batch或全局最快实现证明。',
        '各组已测预算边界、停止原因、未通过点、活动/packing、同B8和原B1身份，均保留在',
        '[完整证据](evidence/decode-capacity-20261118.md)及其JSON；[分模式建议](execution-recommendations.md)已更新。', '',
        f"冻结源码`98210ae3`，生产`src/tide`未变。本轮{data['counts']['current_runs']}个测点中",
        f"{counts.get('completed_within_budget',0)}个预算内完成，{counts.get('sampled_budget_refusal',0)}个主动预算拒绝；",
        '没有把预算拒绝当通过，也没有把它改称硬件OOM。正式点warmup2/samples5，筛选点warmup1/samples2。',
        '独立审计核对输入重建/共同特征前缀、源码与参数、物理卡映射、计时分母和逐卡预算；',
        '全部测点及调度/CPU审计服务的MainPID和已记录PID/start后代均终态。',
        f"{release['raw_file_count']}份原始文件指纹、冻结源核对和四份用户文件保护记录随证据保留。",
        '`artifacts/decode-capacity-20261118`保留preflight、提交、搜索决策、审计、汇总及终态检查。',
        '共享主机上的协作队列独占准入不保证没有外部争用；没有新增训练/恢复、BF16或其他后端资格。',
        '原有限数值失败、A–D及连续训练结论保持各自身份。', '',
    ])
    path = ROOT / 'docs/STATUS.md'
    content = path.read_text()
    begin = content.index('## 当前执行：固定预算下的 decode batch 搜索')
    end = content.index('## 追加测量已完成：', begin)
    write_text_atomic(path, content[:begin] + text + '\n' + content[end:])

    path = ROOT / 'docs/ROADMAP.md'
    content = path.read_text()
    begin = content.index('## 获准追加：固定预算的 batch 吞吐搜索')
    end = content.index('## 本轮交付索引', begin)
    replacement = '''## 追加交付：固定预算的 batch 吞吐搜索

12系列固定模型/FP32短请求搜索已经完成：9B单卡、17B双卡/四卡的Tide与
近总参数Transformer C/32/48层。各自48/56GiB预算内正式最佳、同B8对比、
预算拒绝/有限停止理由和原B1身份见[搜索证据](evidence/decode-capacity-20261118.md)。
全部作业及独立审计已终态；[STATUS](STATUS.md)保留当前交接。
这是有限已测最佳，不证明全整数batch、所有执行候选或硬件容量的全局极限。
没有重启原A–D矩阵，也没有增加训练/恢复或其他后端资格。

'''
    content = content[:begin] + replacement + content[end:]
    content = content.replace('该次补测已终态；下面的新搜索另有授权，不重新启动原矩阵。',
                              '该次补测与下面另行授权的搜索均已终态，不重新启动原矩阵。')
    write_text_atomic(path, content)

    path = ROOT / 'docs/evidence/README.md'
    content = path.read_text()
    intro = '''The subsequent [fixed-budget FP32 decode batch search](decode-capacity-20261118.md)
completes twelve Tide/Transformer series across one, two and four NPUs. It separates
same-batch comparisons from independently selected budget-best points, retains
sampled-budget refusals, and audits input reconstruction plus terminal ownership.

'''
    write_text_atomic(path, content.replace('# Current evidence\n\n', '# Current evidence\n\n' + intro, 1))

    path = ROOT / 'docs/execution-recommendations.md'
    content = path.read_text()
    tide9, m9 = metric('9b/tide/1')
    tide2, m2 = metric('17b/tide/2')
    tide4, m4 = metric('17b/tide/4')
    update = f'''The later [fixed-budget decode search](evidence/decode-capacity-20261118.md) formally
confirmed Tide B{tide9['batch']} at {m9['median_stable_decode_positions_per_second']:.2f} positions/s for 9B on one NPU,
B{tide2['batch']} at {m2['median_stable_decode_positions_per_second']:.2f} for 17B on two, and B{tide4['batch']} at
{m4['median_stable_decode_positions_per_second']:.2f} for 17B on four. These are total batch throughput choices
within the fixed 48/56 GiB allocated/reserved budget and the same short request.
Four devices enable a larger batch and higher total throughput here; the earlier
same-B8 comparison still favors two devices. Keep B1 for the measured low-latency
objective and reserve more memory when adding a downstream head or changing context.
All points keep the measured gather/doubling/prefix/region-batch4 candidates; the search
did not tune all operators, Transformer shapes or replicas. The report gives each
point's latency, per-card rate, budget boundary and retained-state memory separately.

'''
    anchor = '`benchmarks.inference` keeps real state across prefill, decode and stream chunks,'
    assert content.count(anchor) == 1
    write_text_atomic(path, content.replace(anchor, update + anchor, 1))
    path = ROOT / 'docs/performance-lessons.md'
    content = path.read_text()
    addition = f'''
## Per-budget decode batch search, 2026-11-18

The [twelve-series follow-up](evidence/decode-capacity-20261118.md) keeps model,
precision, request and per-card allocated/reserved budgets fixed. Same-batch and
independently selected budget-best throughput answer different questions. At 17B,
four-card Tide B{tide4['batch']} reached {m4['median_stable_decode_positions_per_second']:.2f} positions/s versus
two-card B{tide2['batch']} at {m2['median_stable_decode_positions_per_second']:.2f}, even though four cards were slower
at the previously measured common B8. Capacity can outweigh poorer same-batch
efficiency; this is not a same-work concurrency speedup.

Track both allocator quantities. A reserved-budget refusal is not hardware OOM,
and observed reserved peaks need not increase monotonically with batch: the 17B
two-card 48-layer Transformer fell from 47.314 GiB at B32 to 43.098 GiB at B49,
while allocated grew from 38.205 to 41.988 GiB. Keep physical-device/allocator
history identities and avoid assigning the change to an unprofiled kernel.
An adjacent refused batch is a finite stopping observation, not proof that every
larger batch fails. Larger-batch holes, other prefill chunks and allocator choices
were not exhaustively searched. Final private-state storage is an all-device
snapshot and cannot replace the measured whole-request per-device peaks.
'''
    assert '## Per-budget decode batch search, 2026-11-18' not in content
    write_text_atomic(path, content + addition)
    print('Updated STATUS, ROADMAP, evidence index, execution recommendations and performance lessons')


if __name__ == '__main__':
    main()
