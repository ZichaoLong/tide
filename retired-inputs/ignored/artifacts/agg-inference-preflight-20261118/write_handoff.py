"""Finalize the handoff only after all submitted jobs have been reviewed."""
import hashlib
import json
from pathlib import Path

from benchmarks.search_report import write_text_atomic
from scripts.job import atomic

base = Path("artifacts/agg-inference-preflight-20261118")
audit = json.loads((base / "release-audit.json").read_text())
assert all(audit["summary"][key] for key in (
    "all_terminal", "all_frozen_regular_files_match", "all_user_files_unchanged", "local_links_valid"))
assert json.loads(audit["upstream"]["stdout"])["state"] == "passed"
assert audit["production_tree"] == "539c5eb15176827ca14a19c7de82f08df5b6c1d1"
assert Path("docs/evidence/reception-inference-20261118.md").exists()
inference = json.loads(Path("docs/evidence/reception-inference-20261118.json").read_text())
assert inference["all_requested_records_complete"]
assert len(inference["runs"]) == 13 and len(inference["cache_diagnostics"]) == 3
checks = json.loads((base / "preset-check.json").read_text())
assert checks["state"] == "passed" and checks["tests"] == 34

path = Path("docs/STATUS.md")
status = path.read_text().replace(
    "# 当前交接：Reception 与 FP32 交付，最后一组 A–D 对照收尾",
    "# 当前交接：Reception、FP32 训练与大参数常驻推理已交付")
status = status.replace("## 连续推理与当前剩余作业", "## 连续推理与缓存")
status = status.replace(
    "当前仅 `agg-inference-hb1b-stream-fp32-v1` 尚在完成 Transformer B–D 部分；\n"
    "Tide与A已完成，Timed长流A–D也已完成。不要重交或修改冻结源。\n"
    "Inference代理整理最终证据，根完成终态审核、导航与最终交接后正常推送。",
    "HB约1B批量流与Timed长流的A–D也已全部完成。Timed Light64 B4的Tide/A/B/C/D\n"
    "稳定decode约6.86/9.58/9.63/197.00/197.19位置/s；该点Tide仍慢于A/B。\n"
    "[推理证据](evidence/reception-inference-20261118.md)给出首输出、prompt就绪、\n"
    "首增量、p95、持续吞吐、实际活跃量、A–D构造器及显存。它们不含任务头，不能称语言模型TTFT。")
status = status.replace(
    "`artifacts/agg-inference-preflight-20261118/jobs.json`是快照，实际状态逐查`status.json`。",
    "`artifacts/agg-inference-preflight-20261118/jobs.json`已刷新终态。52个已提交作业均无\n"
    "活动MainPID或仍存活的已记录后代；外层50 passed、2 failed。两项原失败为首轮CPU集成\n"
    "与BF16接收微测，修复/超差各按证据保留，没有抹去失败。全部冻结归档常规文件与各自\n"
    "提交一致；最终核对在同目录`release-audit.json`，简表并入[验证证据](evidence/reception-validation-20261118.md)。")
status = status.replace(
    "阶段汇报后继续推进，最终完成时把ROADMAP更新为已交付事实与具体剩余机会。",
    "本次有限计划已完成，阶段实现与证据均分开提交并正常推送。接续以实际HEAD/工作区为准，\n"
    "不重交已完成任务；[ROADMAP](ROADMAP.md)只保留交付索引和具体后续机会。")
write_text_atomic(path, status)

roadmap = """# 当前周期交付与后续机会

本轮获准的 Reception、FP32训练与大参数推理七阶段有限计划已完成。
实际事实见[STATUS](STATUS.md)，持续约定见[project scope](project-scope.md)。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`；FP32优先，BF16次选。
实例产品路线继续；通用基座仍是另行明确启动的路线。

## 本轮交付索引

| 阶段 | 已交付内容及证据 |
|---|---|
| 基线与执行边界 | 冻结源码、配额/资源准入、原始输入/参数身份；首主干输出、完整prompt、首增量、稳定decode和有限尾部的独立计时契约。见[推理入口](inference-benchmark.md)。 |
| 接收self-attention | 三种incoming公式/VJP、真实零质量域；QKV融合、结构mask、相容合批和重算；九候选真实多步及新进程恢复。见[执行](reception-execution.md)及[数值证据](evidence/reception-numerics-20261118.md)。 |
| FP32热点 | SSD单位置直接递推、块内doubling；可选静态参数视图；保留block、gather和重算候选。见[成本与设备轨迹](evidence/reception-performance-20261118.md)。 |
| 大参数推理 | FC/HB约0.5B/1B、8.778B单卡、16.707B双卡；256增量、批量流、长prompt；17B合批与并发分开比较。见[推理证据](evidence/reception-inference-20261118.md)。 |
| 四类联合比较 | 三种incoming的推理/含图前向/完整训练，0.5B主力B4训练及推理A–D；实际触及目标、匹配偏差、工作/padding和双边归因。见[训练表](evidence/reception-training-20261118.md)及[比较契约](benchmark-contract.md)。 |
| 连续与完整验证 | 五组FP32百步和新进程第101步；独立缓存淘汰/原时钟检查；783项冻结CPU全套、后续34项入口检查、隔离wheel及双NPU。有限/支持超差仍保留，见[验证](evidence/reception-validation-20261118.md)。 |
| 分模式交付 | 五个实测FP32训练预设；首输出、decode、批量流、BF16及内存候选分别建议。见[执行建议](execution-recommendations.md)、[计算公式与伪代码](computation-and-throughput.md)、[经验](performance-lessons.md)。 |

完成有限计划不表示所有配置逐位等价或全部后端合格。FP32跨候选更新、双卡恢复及
BF16输出/支持/恢复超差按原阈值保留。正常时间、侵入式诊断和非等价Transformer
参考不混为同一结论；大小模型、Settle/Timed的相对表现确实不同。

## 具体后续机会

以下是下一次任务可选择的方向，不把它们伪装成本轮未跑完的测点，也不因列表而启动
全组合搜索或另一条产品路线。

1. **分配成本。** 0.5B的allocation约占一次侵入式前向27.69%；应先区分断点排序、
   解面重构、回退发生率和小算子/同步，再评估局部融合及专用VJP。当前证据没有把全部
   成本归给sort或fallback，也没有证明它们已达下界。
2. **Full补齐与索引。** 0.5B K packing执行/选择约5.25倍；可对有效行数分桶或缩小组，
   同时计入增加的launch。静态视图在9B推理有收益，但训练反向与跨更新差异还需单独
   权衡。保留动态稀疏Full、参数归属、索引梯度、原时刻KV与私有状态。
3. **有限数值归因。** 已有NPU内部捕获到界限；可用更窄的Reception Q/K或索引反向
   见证追踪原FP32更新超差。BF16组合路径的支持变化与placed恢复超差仍需各自处理，
   不用放宽阈值、仅exact下一前向或CPU局部解释代替完整因果证据。
4. **有需求的后端与尺度。** 当前完整模型是PyTorch，LibTorch为区域评分切片。
   真正需要更大训练或更低调度成本时，再选择相应连续训练/恢复测点及编译、C++、
   原生跨卡传输候选；CUDA、HCCL/peer DMA、全模型C++、compiled/resident训练仍无新增资格。

新的执行按[执行契约](execution-flows.md)冻结输入、限定时间/RSS/设备预算并核验终态。
共享NPU授权仅限原对话，不能成为仓库默认。参考、公共环境和下游仍保留各自权限边界；
任务头、数据、训练配方及模型质量归下游。现有复现依赖保留，退役须先核对活跃引用、
工作区与下游安装，遵守[保留规则](artifact-retention.md)。
"""
write_text_atomic(Path("docs/ROADMAP.md"), roadmap)

path = Path("docs/evidence/reception-validation-20261118.json")
data = json.loads(path.read_text())
data["delivery_followup"] = {
    "audit_record": {"path": str(base / "release-audit.json"),
                     "sha256": hashlib.sha256((base / "release-audit.json").read_bytes()).hexdigest()},
    "audit_source": audit["source"], "recorded_at_utc": audit["recorded_at_utc"],
    "production_tree": audit["production_tree"], "summary": audit["summary"],
    "jobs": [{k:v for k,v in job.items() if k not in ("queue", "environment")}
             for job in audit["jobs"]],
    "preserved_user_files": audit["preserved_user_files"],
    "quota": audit["quota"], "upstream": audit["upstream"],
    "preset_targeted_check": checks,
    "scope": "Later metadata/CLI and lifecycle checks; the 783-test frozen result and original numerical failures remain unchanged."}
atomic(path, data)
path = Path("docs/evidence/reception-validation-20261118.md")
marker = "\n## 交付跟进核对\n"
content = path.read_text().split(marker)[0]
content += marker + """
五个实测FP32训练预设及公开入口的后续定向检查为 **34 passed，9.32秒**。
测试在工作区执行，相关文件逐字节核对为提交 `ffeca387`；完整命令、14份被测文件
身份、JUnit和日志SHA保存在JSON的`delivery_followup.preset_targeted_check`。
之后BF16 preset仅删去已经过时的跨精度吞吐描述，未修改其模型、运行或测量记录。
这不是将783全套重标成后来HEAD的另一轮全套。

三项NPU独立缓存诊断的原始时间、逐owner写入/保留集合、实际淘汰和SSD/Conv/pending
检查均通过，具体配置和容量填满/窗口淘汰的区别见[推理证据](reception-inference-20261118.md)。
它们是正常性能运行之外的诊断，不反向认证所有大模型bank均填满。

本轮52个已提交作业均已终态：外层50 passed、2 failed。两项原失败分别为首轮CPU
集成与BF16接收微测；前者有修复后的冻结全套，后者保留21个数值超差，不能改写成
全部通过。所有相关服务MainPID为0、没有仍存活的已记录PID/start后代；失败服务的
failed状态保留。全部冻结归档常规文件与各自Git提交一致，四份用户文件哈希未改变。
原记录中的dirty标记原样保留，另用归档字节核对建立实际冻结源身份。

最终生命周期/配额/固定数学与被测文件身份随`delivery_followup`保存；完整原始审核
位于`artifacts/agg-inference-preflight-20261118/release-audit.json`。源码/文档提交与测量
身份分别记账。没有清理checkpoint、trace、冻结源、wheel或复现环境，也没有下游写入。
"""
write_text_atomic(path, content)
print("Final STATUS, ROADMAP and validation follow-up written atomically")
