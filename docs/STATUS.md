# 当前交接 — KV 查询块与三层复测已交付（2026-11-11）

本轮授权范围已完成：隐藏KV bank/Read/Full块/块伴随默认能力分派、FC4/FC16 CPU/NPU复测，
以及可重复的成本配对、profile/census、数值检查、JUnit/终态审阅。没有活跃项目作业。
本轮停止同候选迭代，向用户汇报后再对齐下一有限边界；没有自动展开LH/GPU/多卡新测试。

## 从这里接续

1. `git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
2. 读[固定实现、资格、三层成本和架构结论](evidence/kv-query-blocks-20261111.md)。
3. 读[ROADMAP](ROADMAP.md)顶部的剩余边界，以及[可重复诊断入口](fc16-cost-attribution.md#repeating-the-investigation)。
   不需要旧运行目录；原raw数据的hash仅作身份，复测按报告命令从固定源码重建。

数学固定 `0770593838a37c790a568226930567ed60f99b83`，19份core-5快照核验通过。
主实现 `5ff3ce2bfe72b4416814b750d5260f5c90038eca`；计时/范围标记 `598b4fb5f487befe008f50afb1fe1e378504c417`；
最后后备预检与诊断设施 `206bc9b33be15b891598997bf12a8cec5e6f337b`。
诊断差异标记补充 `19af25eb9c3f1698813b609eaa48775390cb312b`（3项定向检查通过，无生产计算变动）。
本提交记录证据，实施提交与证据提交分开。用户未跟踪的
`docs/tide-learning-lab-TL-REQ-004-handoff.md`、`docs/tide-learning-lab-TL-REQ-005-handoff.md`原样保留，不纳入本轮提交。

## 已检查的结果

- 主CPU四定向组543通过/17旧xfail/2 SDK路径启动失败；真实SDK补验两dtype通过，原失败保留。
  主NPU流程32通过/1旧xfail，包含设备与CPU辅助。Python/native安装消费者、独立SDK的CPU/NPU三步训练及新进程恢复分列通过。
- 最后206bc9b新native/SDK构建通过，CPU162项、standalone Settle两dtype通过；NPU流程11项（6设备/5CPU辅助）及新native安装消费者训练/恢复通过。
  范围标记另有12项；这些组有重叠，不累加成全库资格。
- FC16 B8/L128/d128：CPU默认/packed-v2/直接FC训练5.835/11.601/1.390秒，NPU10.928/15.124/0.675秒。
  主成本仍是探索观察，不是完整尺度严格等价资格；相同材料FP32投影失败保持，提升FP64后值/末状态/任务梯度通过。
- 同轮三模式、同步阶段、实际块命中、B2/L32算子/私有图、资源和旧失败均已归档在上述报告及两个JSON。
  性能差距没有闭合；下一步判断见ROADMAP，不以本轮局部收益宣称整体完成。

## 作业与复查

持久原始记录 `artifacts/kv-block-20261111/`；生命周期 `artifacts/jobs/kvblock-*/status.json`。
`gef-kvblock-measure-{cpu,npu}`、`gef-kvblock-threads`、`gef-kvblock-triage`、
`gef-kvblock-final-build2`、`gef-kvblock-anchors-cpu`、`gef-kvblock-final-npu`均终态passed；内部严格失败独立保留。
`gef-kvblock-final-build`因启动审阅发现缺显式SDK前缀/ABI，在SDK步骤前取消；更正脚本后接续成功。
早期开发namespace/include及profile宏重名编译失败原日志保留，未改写为通过。

冻结源码分别 `/dev/shm/zlong-gef-block-{5ff3ce2,26ab27d,206bc9b}`；中间26ab27d目录实际已固定598b4fb，
以记录和Git提交为准，不能从目录名推断源码。新源码不使用旧native二进制。
开发激活 `artifacts/kv-block-env.sh`；最终构建 `artifacts/kv-block-20261111/build-final` 与 `build-final-sdk`。
测试/计时/安装命令和资源见 `kv-query-blocks-qualification-20261111.json` 与各job记录。

资源没有阻塞：本轮构建320CPU；主测CPU/NPU不同CPU池，无重型构建重叠。
主NPUphysical11；最终补验和补充直接基线physical1，报告明确区分。只使用空闲NPU，未共享。
账户约153GiB/软300/硬800，文件系统约944GiB可用，shm约398GiB可用；大写入前仍需复查。

Historical source IDs: python scripts/resolve_history_commit.py OLD_SHA; see docs/history-migration.md.
