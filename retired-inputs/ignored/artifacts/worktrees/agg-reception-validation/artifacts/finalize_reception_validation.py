from pathlib import Path
import json
import os
import tempfile


def atomic(path, text):
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(parent)
        finally:os.close(parent)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


summary=json.loads(Path('artifacts/reception-validation-summary.json').read_text())
assert summary['all_terminal'], 'Wait for actual terminal records; do not publish the partial draft.'
assert all('MainPID=0' in j['terminal_unit_observation']['output'] and not j['recorded_descendants_still_alive'] for j in summary['jobs'])
by_name={v['name']:v for v in summary['jobs']}
static=by_name['agg-paths-static-fp32-v1'];result=static.get('result')
text=Path('artifacts/reception-validation-draft.md').read_text()
text=text.replace('DRAFT：等待 static_view NPU 终态后填写；目前不作为正式资格页。',
    '[精简 JSON](reception-validation-20261118.json)保留全部作业身份、原文件 SHA256、命令、\n'
    '逐项比较、服务退出与记录后代核对。本页没有启动新实验或改写原结果；\n'
    '[三种接收 FP32 数值记录](reception-numerics-20261118.md)保留此前独立范围。')
if result is None:
    static_text='任务外层为 `'+static['status']['state']+'`，exit '+str(static['status'].get('exit_code'))+'；没有完整路径结果。\n该次提交不能授予 static_view NPU 数值资格；失败/日志与命令原样保存在 JSON 指向的原始记录。\n'
else:
    candidates=result['candidates'];ag=result['aggregate']
    base,views=candidates
    assert base['name']=='baseline' and views['name']=='views'
    differing={k:[base['config'][k],views['config'][k]] for k in base['config'] if base['config'][k]!=views['config'][k]}
    static_text=('冻结 `1fbe15c9`，物理 NPU '+'/'.join(str(v) for v in static['queue'].get('physical_ids',[]))+'，进程内 npu:0；\n'
        'compact-9 Settle、n8/h4、d128、X、light64、FP32、B1/L4、chunk2。输入配置选择\n'
        'fused QKV、结构 mask、Reception 重算、SSD prefix/doubling；路径检查器将两候选\n'
        '都解析为 region_batch_size=1。实际配置差异为 `'+json.dumps(differing,ensure_ascii=False)+'`，\n'
        '因此本次单独比较参数索引方式，不是 batch4 的数值资格。\n\n'
        'baseline 为 `'+base['state']+'`，views 为 `'+views['state']+'`。主比较/恢复原阈值均为\n'
        '`atol=1e-6, rtol=2e-3`；'+str(ag['comparison_count'])+' 项比较中 '+str(ag['exact_comparison_count'])+' exact，'
        +str(ag['threshold_failed_comparison_count'])+' 项含超差，共 '+str(ag['threshold_failed_field_occurrences'])+' 次超差字段。\n\n'
        '| 检查 | baseline | views |\n|---|---|---|\n')
    for label,key in [('同owner重复VJP','same_owner_vjp_repeat'),('推理/含图前向','inference_vs_grad_forward')]:
        cells=[]
        for c in candidates:
            v=c['checks'][key]
            cells.append(('exact' if v['exact'] else '非 exact')+'，'+str(v['threshold_failed_fields'])+' 超差字段')
        static_text+='| '+label+' | '+' | '.join(cells)+' |\n'
    static_text+='\nviews 对 baseline 的推理/VJP1/VJP2 超差字段数为 '+ '/'.join(str(views['checks'][n]['threshold_failed_fields']) for n in ('inference_vs_baseline','vjp1_vs_baseline','vjp2_vs_baseline'))+'；\n'
    static_text+='三个连续更新的超差字段数为 '+'→'.join(str(u['comparison']['threshold_failed_fields']) for u in views['updates'])+'。\n'
    static_text+='跨候选初次差异集中于接收 embedding 梯度，且同一 owner 重复 VJP 也非 exact；\n这说明没有恢复/索引切换也能产生有限反向差异，但没有证明后续全部参数超差由此导致。\n'
    top=views['updates'][-1]['comparison']['largest_scaled_row']
    if top:
        static_text+='最后一次最大 scaled 的张量是 `'+top['tensor']+'`，scaled '+format(top['scaled_error'],'.6g')+'，\n该张量 max_abs '+format(top['max_abs'],'.9g')+'；不将其当成全局最大绝对差。\n'
    static_text+='\n结构标志计数：`'+json.dumps(ag['flags'],ensure_ascii=False)+'`；\nNone不一致检查数 '+str(ag['none_mismatch_checks'])+'，模式工作计数全相等 `'+str(ag['all_mode_work_equal'])+'`，\n连续更新工作计数全相等 `'+str(ag['all_update_work_equal'])+'`。\n'
    for c in candidates:
        restore=c['restore'];o=restore['optimizer_update'];checks=restore['checks']
        static_text+='\n`'+c['name']+'` 新进程恢复状态为 `'+str(restore['state'])+'`，保存数据/下一前向 exact 为 `'+str(checks['saved_data']['exact'])+'/'+str(checks['next_forward']['exact'])+'`；\n'
        static_text+='真实更新为 '+str(o['active_parameter_tensors'])+' 个活动 Parameter、step '+str(o['before_steps'])+'→'+str(o['after_steps'])+'，\n'
        static_text+='更新 exact 为 `'+str(checks['actual_update']['exact'])+'`、原阈值超差 '+str(checks['actual_update']['threshold_failed_fields'])+' 字段。\n'
    witnesses=[dict(name=c['name'],**c['first_internal_witness']) for c in candidates if c['first_internal_witness']['state']!='not_collected']
    static_text+='\n内部见证：`'+json.dumps(witnesses,ensure_ascii=False)+'`。返回树的首个字段仍不等于因果起点；\n本页没有额外高精度/算子重放，不据这项小图结果推定任意合批、尺度或后续轨迹等价。\n'
start=text.index('## FP32 static_view 路径\n')
end=text.index('## 终态、身份及复现\n',start)
text=text[:start]+'## FP32 static_view 路径\n\n'+static_text+'\n'+text[end:]
end_text='''## 终态、身份及复现

数学仍固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件核验通过。
路径记录里的 `source_dirty` 原值保留；本页对冻结归档全部 Git 常规文件的字节核对
分别建立实际源码身份，不借文档 HEAD 或外层 exit0 替代内部资格。

| 作业目录（artifacts 下） | 冻结源码 | 外层终态 | 内部结果/范围 |
|---|---|---|---|
'''
for j in summary['jobs']:
    inner='783 passed / 1 warning' if 'tests' in j else (j['result']['state'] if j.get('result') else '无完整结果')
    if j['name']=='agg-placement-light64-fp32-v1' and j.get('result'):
        inner+='；'+', '.join(r['state'] for r in j['result']['observations'])
    end_text+='| '+j['name']+' | '+j['source'][:8]+' | '+j['status']['state']+' / exit'+str(j['status'].get('exit_code'))+' | '+inner+' |\n'
end_text+='''
上述任务均使用 `background.slice`、有限运行时间和采样进程树 RSS 界限。CPU 全套
固定 affinity56–63，安装64–67；BF16 路径184–187，双 NPU 放置188–191，static 路径
220–223。外层均请求4线程，安装机制 harness 实际 ATen/interop 为1；具体资源、
时间/RSS界限、采样峰值和队列身份保存在 JSON。采样 RSS 不是设备显存或可靠 cgroup
硬界限，任务墙钟包含排队/记录开销，不能用于性能对照。

NPU stderr 保留 CANN/SoC 的 32 对齐额外分配提示；放置任务还保留 TypedStorage
弃用提示，均未使这些任务中断。完整原始日志随哈希保存。

NPU 使用协作队列准入，身份为 `exclusive_cooperative_queue; external jobs not controlled`；
主机共享，不声称控制队列之外的使用者。终态核对时，这五个服务均 inactive/dead、
MainPID=0，已记录的 PID/start 身份中没有仍存活的非僵尸后代；这不表示其他项目任务
全部结束。没有清理冻结源、checkpoint、wheel、环境或复现工作树。

复现使用 JSON 的 `jobs[].requested.command` 及各原 `launch.json`/`run.sh`；安装 driver、
构建/安装/恢复命令都随哈希记录。冻结源码和私有环境保持对应身份，所有结果写新目录，
设备重新按[执行契约](../execution-flows.md)准入。CPU 全套入口为：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q --junitxml NEW-tests.xml
```

隔离安装使用 `--execution-variant reception`；公开入口示例/config 属于 checkout 工具，
wheel 安装 `tide` 包。测试、安装和各个硬件点分别给出资格，不能将 CPU 双逻辑组、
恢复下一前向 exact 或 source 菜单可表达性替换为所有设备/配置的训练等价。
公共工具链、参考仓库和下游未修改，本页不声明下游发布/验收。
完整模型仍是 PyTorch；[LibTorch 区域评分切片](native-regional-performance-20261118.md)
沿用自己的原资格。不同数学模型的性能对照及各模式建议由单独证据负责。
'''
text=text[:text.index('## 终态、身份及复现\n')]+end_text
assert 'TODO' not in text and 'DRAFT' not in text
atomic(Path('docs/evidence/reception-validation-20261118.md'),text)
atomic(Path('docs/evidence/reception-validation-20261118.json'),json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
p=Path('docs/evidence/reception-numerics-20261118.md')
old=p.read_text()
addition='''
后续的[最终集成、隔离安装与后端证据](reception-validation-20261118.md)另行记录
`1fbe15c9` 的完整 CPU/新 wheel/static_view，以及 `1553b301` 的 BF16 与双 NPU 路径。
其中活动支持和恢复更新的原超差分别保留，不改变本页三种 FP32 小图的限定结论。
'''
assert 'reception-validation-20261118.md' not in old
atomic(p,old+addition)
print('Wrote final evidence and the numerical evidence cross-link.')
