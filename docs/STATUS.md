# 当前交接 — LH张量分组、REQ-006诊断与resident重写（2026-11-12）

用户已授权继续完成[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
本轮起点a8a4ee028a5d79696d49bc47242a29c4c39aee4d；HEAD 2927ba40；固定数学0770593838a37c790a568226930567ed60f99b83。
19份教材快照再次通过；本分支3个新提交尚未推送。下游及参考仓库只读。

## 已终态作业

- 129bbbdc的gef-rewrite-aligned：native CPU定向99通过/9原数值xfail；gef-rewrite-aligned-device：half 15通过、6个安装训练/新进程恢复通过，FC16 CPU/NPU完整步已测。两unit均MainPID=0、cgroup空。报告待整理。
- c83a2820的gef-rewrite-lh-cpu/npu均passed，三层三模式和CPU32补充已完成；代表176节点/d64/B4/L16、实际61输入位置。NPU训练直接0.6075s、公共native7.1231s、默认native7.4958s。直接dense Full与候选selected Full工作量不同，不能把差额全归包装。
- 同源码gef-rewrite-req006-a/b均failed；四项measure通过，四项profile已导出trace，但CANN字符串时间戳使旧分析器失败。原失败保留，不重复设备profile。所有以上unit已检查无后代。
- 原始结果及矩阵：artifacts/rewrite-20261112/；持久状态/日志：artifacts/jobs/rewrite-*/。

## 当前动作与活跃作业

2927ba40已修正时间戳类型并提供离线CLI，11项短测试通过。
即将启动gef-rewrite-trace-recovery：固定源码/dev/shm/gef-rewrite-20261112/parser-source，
仅离线读取4个既有trace，生成各profile目录下summary-recovered-2927ba40.json。
脚本artifacts/rewrite-20261112/trace-recovery-work.sh；记录artifacts/jobs/rewrite-trace-recovery/；
background.slice，CPU224–231、64GiB、600秒，不用设备、不改原failed状态。

主代理集成linear_blocks及KV prepare_many（开发Python定向通过，native待新构建）。
区域布局正改为按真实候选数量分组，保留每行成员、稳定平票及结构梯度支持；不以padding替换候选集。
resident独立设备块调度原语已完成CPU独立检查，实际数值接入、设备构建/VJP/optimizer/续接仍待完成。
三项工作分边界开发，主代理统一审查、提交与资源调度。正式计时已结束，可安排新构建。

## 下一步与未提交

核验离线分析的实际kernel/形状/同步分布，记录原失败和恢复分析；整理129/c83固定证据。
完成节点组/区域组集成定向测试，固定新提交，再做native/SDK新构建、独立根/恢复和FC/LH有限回归。
resident独立验收，不把旧服务小图通过当新执行器完整交付；普通异步跨NPU反向EE1017保持失败。
用户新增TL-REQ-004/005/006三份handoff保持原样，不纳入提交。
配额284GiB/软300GiB/硬800GiB，主存可用约1.7TiB，shm308GiB；构建源码/build优先有界shm，持久记录留磁盘。
共享NPU未获本轮授权，仅使用通过准入的空闲卡；正式计时避免本任务构建竞争。

```sh
python scripts/status.py
systemctl --user show gef-rewrite-trace-recovery.service -p ActiveState -p MainPID -p ControlGroup
cat artifacts/jobs/rewrite-trace-recovery/status.json
git diff --stat
```

当前状态只在本页，待办/停止与重启条件只在ROADMAP。实现、严格数值、恢复和性能分别报告。
