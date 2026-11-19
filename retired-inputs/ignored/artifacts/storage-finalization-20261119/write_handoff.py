"""Atomic current handoff and reviewed storage evidence update."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('j',HERE/'frozen-job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def text(path,value):
    path=ROOT/path
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.handoff-')
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(value);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((HERE/'final-audit.json').read_text())
evidence_path=ROOT/'docs/evidence/storage-finalization-20261119.json'
evidence=json.loads(evidence_path.read_text())
evidence.update(state='completed',final_audit=audit,
    temporary_validation_outputs='Retired after reviewed evidence commit dee2c84d; source hashes, smoke result and deletion inventory remain.',
    scratch_retirement=dict(path='artifacts/storage-finalization-20261119/scratch-retirement.json',
        sha256=sha(HERE/'scratch-retirement.json'),allocated_bytes=audit['temporary_validation_bytes_retired']),
    source_tree_unchanged_from_production_revision=True)
evidence['sources'].update({str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')})
j.atomic(evidence_path,evidence)
gib=2**30
report=f'''# 剩余存储收尾（2026-11-19）

从 `660f3d0f` 接续，归档还原工具固定 `45cb5dad`；生产树仍为 `97c52960`。
本次处理的是已结束工作的载荷与记录，不新增模型矩阵或资格。
数学 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` 的108文件原字节不变。
完整数字、来源、保留清单、校验与终态见[审阅JSON](storage-finalization-20261119.json)。

## 实际释放与剩余

仓库 allocated bytes 从54,809,583,616降至2,430,844,928，约 **51.045→2.264 GiB**，
净减少 **48.782 GiB**。收尾测量在最终文档提交前；少量证据/Git写入会使末位变化。
账户quota约 **42.287/300 GiB**软限，硬限700 GiB；不把部分目录du当完整账户总量。

| 操作 | 文件数 | 释放 allocated bytes | GiB |
|---|---:|---:|---:|
| 明确退役旧整图/中间载荷 | 283个.pt、6669个.f32 | 48,953,257,984 | 45.591274 |
| 大JSON无损归档节省 | 352 | 3,450,470,400 | 3.213501 |
| 原有产物释放合计 | 7304 | 52,403,728,384 | 48.804775 |

新增索引、删除收据及Git证据解释了毛释放与净变化的差别。另清除了本次生成的
314,601,472 B安装/还原测试临时文件，不能将其重复计为原有配额减少。

| 当前存储类别 | GiB | 用途 |
|---|---:|---|
| 保留的78个张量文件 | {audit['categories_allocated_bytes']['retained_tensors']/gib:.6f} | 近期训练前态、局部诊断输入/输出 |
| 可直接阅读的JSON | {audit['categories_allocated_bytes']['plain_json']/gib:.6f} | 小记录、汇总、源码/工作树中受保护的证据及清理身份 |
| 352个压缩JSON | {audit['categories_allocated_bytes']['archived_json']/gib:.6f} | 原始完整记录，可精确还原 |
| artifacts中其他文件 | {audit['categories_allocated_bytes']['other_artifact_files']/gib:.6f} | 源码、配置、日志、原生工具及小诊断材料 |
| artifacts之外的文件 | {audit['categories_allocated_bytes']['outside_artifacts_files']/gib:.6f} | 主源码、math、文档、共享Git及私有环境 |
| 目录块 | {audit['categories_allocated_bytes']['directory_blocks']/gib:.6f} | 全仓目录自身占用 |

注册工作树约0.708 GiB已包含在这些类别内，不能再次相加；其独有源码及各自tracked文件保留。
下游正常安装、训练与推理不需要这些实验载荷或归档。保留主源码/安装环境以及共享Git；
主`.git`同时服务下游worktree，不能将整个仓库当作可删缓存。
本次不修改下游、参考仓库、公共环境或四份用户未跟踪文档。

## 保留范围和明确失去的可重放性

保留六份近期文件：新quad训练anchor与两份checkpoint、真实FFN VJP张量、BF16分配见证、
共享9B分配见证。另保留六个小见证：独立Attention、Attention两种表示、区域settle/timed、
CPU区域舍入和FFN局部诊断；最后一个不构成完整FFN训练轨迹的重放包。
native bias-rounding原始输入与输出两个小包也保留，共70文件448 KiB，
重放仍依赖固定native binary/launcher/SDK；本次没有重新执行该诊断。
74个失败profiler及原失败结果继续保留。

旧CPU anchor、VJP1、update1及旧BF16 anchor等本次已明确退役。
此前update2整树exact重做和删除后摘要exact重做仍是已完成的历史观察，
**现在不再承诺从在线输入直接重放这些整图更新**，也不能重新捕获旧FFN完整反向或原BF16三次更新。
保留局部输入不等于保留全部因果链；标量失败汇总不是完整复现器。
原超差、阈值、source身份和通过范围均不改变。

## 无损归档和还原

352份未跟踪、终态任务的大JSON（≥1 MiB）原占3.323902 GiB，压缩后0.110401 GiB。
排除所有注册工作树tracked文件、source/build副本、近期数值诊断、清理身份、
保留见证及关键可读汇总。使用gzip level6、mtime0，逐份核验完整解压长度和SHA256；
先持久化索引，再删除原件。失败记录的原始字节原样归档，不用摘要替换原文件。

[归档索引](storage-archive-index-20261119.json)记录原路径、两种SHA和字节数。
[退役张量身份清单](storage-retired-payloads-20261119.json.gz)保存全部6952个路径、stat和删除前SHA；
其中只有元数据，没有模型内容。固定计划、过程日志和收据位于
`artifacts/storage-finalization-20261119/`。

在主仓执行以下命令，将原始JSON精确还原到一个尚不存在的路径：

```sh
python scripts/materialize_record.py \\
  --index docs/evidence/storage-archive-index-20261119.json \\
  --record artifacts/ffn-medium-npu-qualification-v1/qualification/result.json \\
  --output /tmp/tide-ffn-medium-npu-result.json
```

工具验证压缩与原始SHA、解压长度及gzip完整性，并原子发布且拒绝覆盖。
`search_report`、`cost_report`等旧工具会读取记录内的原绝对路径或关联输入；
另存JSON只保证可读，不保证旧命令可执行。需要时逐项恢复依赖到原逻辑路径；
已退役张量和trace无法由解压JSON恢复。归档仍在本地artifacts中，Git只保存索引和审阅证据。

## 清理后的验证与资源

还原工具8项检查通过；实测还原141,189,151 B的大报告及一份失败报告，SHA与原件一致，
后者仍为原始`execution_failed`。首次验证脚本误把任务`failed`当作记录状态，
已改为核对真实`execution_failed`；未修改归档或任何数值阈值，初次断言失败另有记录。
352个压缩包再次校验SHA，6952个退役载荷确认不存在，82份受保护文件原字节不变。

清理后从固定 `97c52960` 构建新wheel，23个包文件与源码匹配。在新的私有venv中，
从仓库外用`python -I`运行公开CPU FP32入口（FC n8/layers4/d64，B2/L8）。
训练与checkpoint、下一前向exact，实际配对更新在原阈值内；
另一新解释器恢复后position8→16并真实更新。该检查不依赖退休绝对路径，
只证明安装/入口及小恢复可用，不扩大数值、性能、9B训练或任务质量资格。
结果先提交 `dee2c84d`，再核验终态并删除安装、checkpoint和两次JSON还原临时目录。

两个清理服务与一个CPU冒烟服务都在`background.slice`，共10个记录后代全部终态。
首个服务为补保留80 KiB native原输入，在删除原件前主动停止；336个已压缩文件经完整复核后复用。
第二个清理服务成功，安装检查成功；三者服务时长合计约84.385秒，峰值采样RSS约0.451 GiB。
清理最多8个CPU/16 GiB/2700秒，安装最多2个CPU/4 GiB/300秒；没有NPU任务。
同账户可检查进程未引用候选；六个受权限限制的SSH/PAM进程作为观察限制记录。
服务用时不是模型性能测量。本次存储收尾已完成，不自动重启旧实验。
'''
text(Path('docs/evidence/storage-finalization-20261119.md'),report)
status=(ROOT/'docs/STATUS.md').read_text()
status=status.replace('# 当前交接：清理、数值定位与quad修复已完成','# 当前交接：存储收尾完成，数值资格边界保留',1)
start=status.index('本轮以 `c5bfebaf`')
end=status.index('\n数学固定',start)
status=status[:start]+'''本次从 `660f3d0f` 接续，剩余51 GiB的存储收尾已完成；归档还原工具固定 `45cb5dad`。
当前大小、保留范围、还原入口与清理后安装检查见[存储收尾](evidence/storage-finalization-20261119.md)
和[审阅JSON](evidence/storage-finalization-20261119.json)。本次不改生产树，不新增模型资格。

此前数值定位从 `c5bfebaf` 起步，修复实现与资格固定 **`97c52960`**；
诊断/重放工具固定 `2360728d`、`e1c6aaa0`、`aab9a4d2`、`72c1a9d0`。
原事实见[数值证据](evidence/cleanup-numerics-20261119.md)及其配套JSON，
保留原测量身份和失败，不以文档HEAD追认旧测量。当前有限计划无待运行测点。
'''+status[end:]
start=status.index('## 清理\n');end=status.index('\n## 数值定位与修复',start)
status=status[:start]+'''## 清理

前次约91.150→51.045 GiB；本次再从 **51.045降至约2.264 GiB**，净减少约48.782 GiB。
283个旧.pt和6669个.f32明确退役，释放45.591274 GiB；352份JSON无损归档再省3.213501 GiB。
新增索引/收据/Git证据计入净占用，不将本次临时安装/还原产物的删除重复算为原有配额减少。
账户quota约42.287/300 GiB软限（硬限700 GiB）；账户目录du有不可读Docker子目录。

保留约0.520 GiB的78个张量文件、可读汇总/小记录、压缩原JSON、74个失败profiler及独有源码。
近期六份训练/局部见证、六个旧小见证和native原输入/输出包继续在线；82份保护文件逐字节核验。
旧anchor/update1/VJP1和BF16 anchor已退役，旧整图续更不再有在线输入可直接重放；
此前exact重做是历史观察，原失败及阈值不改。JSON按[索引](evidence/storage-archive-index-20261119.json)
使用 `scripts/materialize_record.py` 精确还原；解压不会重建已退休张量/trace。

清理后新隔离wheel的23包文件匹配 `97c52960`，公开CPU FP32训练、checkpoint、
配对更新及新进程实际恢复更新通过。还原工具8项检查及两份真实记录的精确还原通过。
下游运行不需要实验产物；主math/.git、所有注册工作树源码、私有环境、公共环境和下游保留。
四份用户未跟踪文档原字节保留、不提交。本次无NPU任务，三个服务及10个记录后代全部终态。
'''+status[end:]
status=status.replace('15个服务、110个记录后代全部终态；','此前数值任务15个服务、110个记录后代全部终态；',1)
text(Path('docs/STATUS.md'),status)
roadmap=(ROOT/'docs/ROADMAP.md').read_text()
old='''旧产物40.637 GiB以及本轮可重建源码、测试/恢复/安装临时产物已退役。
其余失败输入、唯一源码及下游依赖保留；不可把所有剩余张量视为未结束作业或无用残留。'''
new='''前次旧产物40.637 GiB和可重建临时产物已退役；随后[剩余存储收尾](evidence/storage-finalization-20261119.md)
明确退役45.591 GiB旧载荷、无损压缩大JSON，仓库约51.045→2.264 GiB。
保留近期局部见证/训练前态、少量旧诊断包、原失败记录、唯一源码及下游依赖；存储收尾已完成。
旧anchor/update1等不再在线，过去exact重做保留为历史事实，不承诺当前整图重放。'''
assert old in roadmap;roadmap=roadmap.replace(old,new,1)
old='''   CPU NUMA计时差异仍未解释。剩余失败大载荷可逐组提取最小见证、验证再生后退休；
   现存旧anchor/update1是已删除update2的复现依赖，不能失去依赖后仍承诺可重建。'''
new='''   CPU NUMA计时差异仍未解释。旧完整载荷已明确退役；若需要更深数值定位，先使用
   保留的局部见证，或另行确定有限重捕获输入与预算，不将原记录路径当可用性保证。'''
assert old in roadmap;roadmap=roadmap.replace(old,new,1)
text(Path('docs/ROADMAP.md'),roadmap)
retention=(ROOT/'docs/artifact-retention.md').read_text()
retention+='''
## Current lossless record archives

The [storage finalization report](evidence/storage-finalization-20261119.md) owns the
current availability decisions. Large raw JSON records may be retained as `.json.gz`;
the [archive index](evidence/storage-archive-index-20261119.json) records original and
compressed SHA256, lengths and logical paths. `scripts/materialize_record.py` restores
exact bytes to an explicit new path, checks both identities, bounds decompression and
refuses overwrite. Originals are unlinked only after complete decompression verification
and fsynced archive/index publication. A failure remains the same original failure.

Git retains the index, reviewed summaries and compressed retired-payload **metadata**;
the archived JSON content remains in local artifacts. Restoring JSON to a new location
does not restore its transitive dependencies or rewrite embedded original absolute paths.
Tools may require related records at their original logical paths. Explicitly retired
tensor/trace payloads cannot be reconstructed by this helper. Historical exact replay is
not an ongoing replay promise after its anchor/update inputs have been retired.

Downstream installation and normal execution require the package/source and runtime,
not experiment artifacts. This does not authorize wholesale removal of `artifacts`:
registered worktrees, unique source and useful witnesses remain protected there.
'''
text(Path('docs/artifact-retention.md'),retention)
index=(ROOT/'docs/evidence/README.md').read_text()
index=index.replace('# Current evidence\n','# Current evidence\n\nThe [storage finalization](storage-finalization-20261119.md) records explicit retirement\nof the remaining old full payloads, verified lossless JSON archives and the post-cleanup\ninstalled training/recovery smoke. Current storage is about 2.264 GiB. Earlier replay\nclaims are historical where their anchor/update inputs were retired; failures remain failures.\n',1)
text(Path('docs/evidence/README.md'),index)
old_report=(ROOT/'docs/evidence/cleanup-numerics-20261119.md').read_text()
old_report=old_report.replace('# 清理、数值定位与quad修复（2026-11-19）\n',
'''# 清理、数值定位与quad修复（2026-11-19）

**后续可用性变更：** [存储收尾](storage-finalization-20261119.md)已退役下文原本保留的
旧CPU anchor/VJP1/update1及BF16 anchor等整图输入；部分大JSON已无损归档。
本报告的数值、阈值及已完成exact重放是原测量事实，不保证现在仍有完整输入可直接重放。
当前保留包和JSON还原入口以该后续报告为准。
''',1)
old_report=old_report.replace('除明确退役的两份update2，其余290个旧',
    '以下是本报告原始收尾时的保留清单，后续退役已覆盖其中大载荷：\n除明确退役的两份update2，当时其余290个旧',1)
old_report=old_report.replace('最小化，不能整组删除。74个旧失败profiler仅约1.43 MiB，也继续保留。\n旧CPU anchor、两份update1仍是已删update2的复现依赖。',
    '最小化；当时没有整组删除。后续已按授权明确退役，并不宣称所有载荷均可再生。\n74个旧失败profiler仅约1.43 MiB，仍保留。旧CPU anchor、两份update1当时是已删update2的复现依赖，现已退役。',1)
old_report=old_report.replace('## 当前可复现入口','## 原测量命令与当前输入可用性',1)
old_report=old_report.replace('# 删除update2后的当前可运行复现：固定72c1a9d0、CPU32线程、原affinity240–271。',
    '# 历史命令：固定72c1a9d0、CPU32线程、原affinity240–271；旧anchor/update1现已退役，不能直接执行。',1)
old_report=old_report.replace('# 原局部FFN诊断固定e1c6aaa0，launcher需 --affinity 240-271；',
    '# 历史FFN重新捕获命令：固定e1c6aaa0，launcher需 --affinity 240-271；旧anchor/VJP1现已退役。',1)
old_report=old_report.replace('仍可直接阅读已保留result/witness及全部原字段比较，并使用上述删除后续更入口。',
    '仍可阅读已保留result/witness和原字段比较；归档JSON须先按索引还原。\n上述整图续更与FFN重新捕获命令已失去在线输入，不能再直接运行；局部见证继续保留。',1)
text(Path('docs/evidence/cleanup-numerics-20261119.md'),old_report)
print('Updated handoffs and evidence with fsynced staging and atomic replacement.')
