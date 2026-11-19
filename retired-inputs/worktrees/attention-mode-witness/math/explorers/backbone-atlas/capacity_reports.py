"""Serialize backbone counts for the atlas, CSV and Markdown range table."""
import csv
from decimal import Decimal

from backbones.capacity import (
    AGG_PROFILES, ALLOCATION_PROFILES, BOUND_PROFILES, BUDGETS, DIMENSIONS, NODE_SIZES, PROFILES, SHARING,
    VARIANTS, active, aggregation_config, budgets, count, regional_config,
)


AGG_FIELDS = ['agg_profile','agg_mode','agg_hidden_width','agg_attention_width',
              'agg_attention_heads','agg_attention_head_dim']
REGION_FIELDS = list(regional_config(128))
ALLOCATION_FIELDS = ['allocation_profile','selection_mode_D','selection_mode_H',
                     'lower_base_D','lower_base_H','total_mode_D','total_mode_H']
FIELDS = ['profile','n','h','d','memory','sharing','rank',
          'nodes_D','nodes_H'] + AGG_FIELDS + REGION_FIELDS + ['total','base','D','H']
PROFILE_FIELDS = ['spatial_profile','family','timing','depth','widths',
                  'positions','max_width','expansion','rounds','input_period']
PARAMETER_COLUMNS = {'total':'total_M', 'base':'base_M',
                     'D':'private_D_M', 'H':'private_H_M'}


def millions(value):
    """Keep exact scalar counts and power-of-two budget fractions in decimal M."""
    scaled = Decimal(value) / 1_000_000
    return format(scaled, '.6f' if value == int(value) else 'f')


def build(out):
    """Write reproducible tables; the page uses the same compact row data."""
    rows = []
    csv_fields = ['profile'] + PROFILE_FIELDS + FIELDS[1:9] + AGG_FIELDS + REGION_FIELDS + ALLOCATION_FIELDS + list(PARAMETER_COLUMNS.values())
    csv_fields += ['within_256M'] + [f'active_k{k}_M' for k in BUDGETS] + ['active_pool_M']
    with (out/'capacity.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=csv_fields)
        writer.writeheader()
        for key,p in PROFILES.items():
            bindings = [b for b in BOUND_PROFILES.values() if b['spatial_profile'] == key]
            for n in NODE_SIZES:
                for h in ((0,4) if p['family'] in ('exchange','multiscale') else (0,)):
                    for d in DIMENSIONS:
                        for memory in (('K','S','X') if h else ('K','S')):
                            for sharing,rank in SHARING:
                                for agg_profile in AGG_PROFILES:
                                    x = count(key,n,h,d,memory,sharing,rank,agg_profile)
                                    values = dict(n=n,h=h,d=d,memory=memory,
                                                  sharing=sharing,rank=rank,**x,
                                                  **aggregation_config(d,agg_profile),
                                                  **regional_config(d))
                                    for binding in bindings:
                                        row = dict(profile=binding['key'], **values)
                                        # Embedded JSON retains raw counts for exact filtering.
                                        rows.append([row[name] for name in FIELDS])
                                        exported = {name:row[name] for name in FIELDS if name not in PARAMETER_COLUMNS}
                                        exported.update({name:binding.get(name,'') for name in PROFILE_FIELDS})
                                        exported['widths'] = ';'.join(map(str,binding['widths']))
                                        exported.update({column:millions(x[name]) for name,column in PARAMETER_COLUMNS.items()})
                                        exported['within_256M'] = x['total'] <= 256_000_000
                                        for allocation, config in ALLOCATION_PROFILES.items():
                                            exported.update(allocation_profile=allocation,
                                                selection_mode_D=config['selection_mode_D'],
                                                selection_mode_H=config['selection_mode_H'] if h else '',
                                                lower_base_D=config['lower_base_D'],
                                                lower_base_H='received' if h else '',
                                                total_mode_D='interval' if allocation!='pool' else '',
                                                total_mode_H='interval' if h else '')
                                            exported.update({f'active_k{k}_M':
                                                millions(active(x,n,k,allocation)) if k<=n and allocation!='pool' else ''
                                                for k in BUDGETS})
                                            exported['active_pool_M'] = millions(x['total']) if allocation=='pool' else ''
                                            writer.writerow(exported)
    write_summary(out)
    return dict(fields=FIELDS,rows=rows,profiles=BOUND_PROFILES,variants=VARIANTS,
                parameter_unit='scalar',agg_profiles=AGG_PROFILES,
                allocation_profiles=ALLOCATION_PROFILES,node_sizes=NODE_SIZES,
                dimensions=DIMENSIONS,budgets=BUDGETS)


def write_summary(out):
    lines = ['# 主干参数与尺寸阶梯（生成表）','',
             '维护定义：[统一比较绑定](../../../instances/backbone-comparison.md)。主干可训练参数；M=10^6，不计边界、任务头或状态。',
             '分别列出来源残差、两种轻量自注意力和加宽自注意力；每行总参已包含该接收配置的全部参数，无需另加增量。接收配置及准备成本见[比较绑定 §5.1](../../../instances/backbone-comparison.md#51-接收自注意力的参数与准备成本)。',
             '', '完整 CSV 的参数列以 _M 结尾，单位同为百万参数；空的 active_k 列表示 k>n 或该行为 pool，不能作为数值预算使用。active_pool_M 仅在 pool 行有值。',
             'profile 同时标识空间轮廓与时间绑定；spatial_profile 保留同端点对照关系。多跨度 Timed／Settle 分别列出，参数计数相同。',
             'CSV 使用各候选主周期；Timed 与 Settle 同周期的设置可在面板中另选。候选取舍与新变体的参考见[主干记录](../../../instances/backbone-design.md)，跨层意图见[研究意图与探索路径](../../../memos/design/design-requirements-and-architecture.md)。',
             '', '以下为 independent；普通 K，有 hub 时取 X。每格为 d=64 / 128 / 192 / 256 的参数 M。更大维度、region_shared／layer_shared 和 S 均在 CSV 及面板中。','',
             '| 拓扑与时间 | 接收配置 | 层数 | n | D / H 节点 | 参数 M |',
             '|---|---|---:|---:|---:|---|']
    for p in BOUND_PROFILES.values():
        key = p['spatial_profile']
        for n in (8,16,32):
            for h in ((0,4) if p['family'] in ('exchange','multiscale') else (0,)):
                for agg_profile, agg in AGG_PROFILES.items():
                    xs = [count(key,n,h,d,'X' if h else 'K',agg_profile=agg_profile)
                          for d in (64,128,192,256)]
                    lines.append(f"| {p['label']} | {agg['label']} | {p['depth']} | {n} | {xs[0]['nodes_D']} / {xs[0]['nodes_H']} | "+
                                 ' / '.join(f"{x['total']/1e6:.2f}" for x in xs)+' |')
    lines += ['', '## 接收配置','',
              '下表代入 d=128；轻量两档在其他 d 下保持注意力宽度 64／128、MLP 隐藏宽度 d/8；加宽档两者都取 d。来源残差的注意力尺寸记为 0，表示未启用。', '',
              '| agg_profile | agg_mode | MLP 隐藏宽度 | 注意力宽度 | 头数 | 每头宽度 |',
              '|---|---|---:|---:|---:|---:|']
    for agg_profile in AGG_PROFILES:
        a = aggregation_config(128,agg_profile)
        lines.append(f"| {agg_profile} | {a['agg_mode']} | {a['agg_hidden_width']} | {a['agg_attention_width']} | {a['agg_attention_heads']} | {a['agg_attention_head_dim']} |")
    lines += ['', '## 三档参数对照','',
              '固定 n=32、d=128；共享适配秩为 8，有 hub 时取 X，否则取 K。单位 M，完整尺寸及秩见 CSV。', '',
              '| 空间图 | 接收配置 | hub | independent | region_shared | layer_shared |',
              '|---|---|---:|---:|---:|---:|']
    for key,h in [('fc-8',0),('lattice-7',0),('exchange-m1-r5',4),('multiscale-compact-9',4),('multiscale-9',4)]:
        for agg_profile, agg in AGG_PROFILES.items():
            values=[count(key,32,h,128,'X' if h else 'K',sharing,8,agg_profile)['total']
                    for sharing in ('independent','region_shared','layer_shared')]
            lines.append(f"| {PROFILES[key]['label']} | {agg['label']} | {h} | "+' | '.join(f'{v/1e6:.3f}' for v in values)+' |')
    lines += ['', 'FC 每层一区，两共享档在相同秩下等价；CSV 保留各档标识供筛选，实验无需重复。区域共享按真实区域及各组实际来源类计数，不由一个固定倍率推算。三档参数量不保证严格有序。',
              '', '## 预算阶梯','', '| n | 合法 k | k/n（%） |', '|---:|---|---|']
    for n in NODE_SIZES:
        ks = budgets(n)
        lines.append(f"| {n} | "+', '.join(map(str,ks))+' | '+
                     ', '.join('100 (pool)' if k=='pool' else f'{100*k/n:g}' for k in ks)+' |')
    lines += ['', 'k=n 的 quad 与 pool 是不同分配函数；相同的活跃代理不表示数值等价。',
              '主评分为 encoded/self_attention：d_z=d/2，每头32维，区域FFN宽度d；区域参数为2d²+2d。三种分配共用参数计数，CSV 按 quad/shannon/pool 分行，页面保留一份参数行。',
              '约束 quad/Shannon 显式使用 received 下界、interval 总量与固定阈值。H 区固定角色 quad；pool 仅替换 D 区。',
              '降低 k 不降低总参数；quad 活跃代理为 base+(k/n)D+H，Shannon/pool 取 total。共享基座去重，假定各区有正接收、H 均参与。准备阶段仍可能访问其他节点参数。',
              '自注意力的投影、成对读取及逐条 MLP 属于准备；加宽配置的准备成本不能按 Full 的 k/n 比例折减。',
              '宽图、更深图与超过 256M 的行全部保留；候选整体覆盖规模及比例，不要求每个小实例达到约 3%。']
    (out/'capacity-summary.md').write_text('\n'.join(lines)+'\n')
