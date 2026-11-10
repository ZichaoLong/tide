# 当前交接 — 接口、性能与架构统一实施（2026-11-10）

用户已授权持续实施、分阶段提交/推送；数学固定0770593，分支graph-execution-foundation。
启动基线4d777bc，旧核心1f3f62d。用户未跟踪TL-REQ-003/004文档保持原样，不纳入提交。

## 当前增量

三项仓库skills与分主题性能经验已整理；账户skills未改。取消仓库subagent门槛，按工作独立性使用。
固定基线源码清单/审阅工具已完成；本次公共边界实际审阅25文件128函数，其余不冒称完成。
修复Session重复读/decode、插件可声明outputs观察、runtime遗漏子包hash（manifest v2）。
外层/私有autograd census及四模式对照已接入；诊断计时与正式吞吐分开。
KV无trace序列保留实际共享bank视图；query按真实storage区间去重gather/Read K,V投影，
手写一阶VJP保留独立snapshot根。仍有query×history伴随展开，不能称全部重复工作已消除。
对应新契约：docs/weighted-snapshot-bank.md、docs/autograd-census.md；覆盖见相关测试与architecture。

## 已完成的开发检查

Python sequence/新bank首轮37通过，另1旧FP32scalar差异；禁用新bank时该梯度逐位相同。
同材料FP64差3.56e-15，FP32双方距FP64约1.03e-6/2.83e-6；严格失败保留为独立xfail。
修订后新bank定向9通过/1保留xfail；native新文件语法编译通过，实际构建/设备资格尚待运行。
公共边界开发107/67/39/33项组通过（重叠不累加）；图诊断11项通过。
独立审阅补60组重叠/冻结/重复根探针和大int64检查；native存储组改稳定首次出现顺序。
这些是开发检查，不能替代固定源码资格或性能结果。

## 下一动作与作业

先固定当前实现提交，建立artifacts/interface-work/source-bank干净检出及env-bank wheel。
计划background.slice有限服务gef-interface-bank：320可用CPU构建，CPU定向资格和三客户端恢复，
FC16原任务CPU1/16线程及图统计对照；空闲单NPU队列内完成相关资格/成本。
正式启动前记录实际unit/命令；尚无新活跃后台服务。任务日志预定artifacts/jobs/interface-bank/task.log。
共享NPU本阶段无授权；只使用空闲卡，不能复用旧shared adapter。CPU约1.7TiB可用，文件系统约685GiB，
本用户281GiB/soft300/hard800GiB；正式计时与本项目重构建不重叠。

## 基线与剩余工作

旧结果见docs/evidence/fc16-cost-attribution-20261110.md：FC16 B8/L128/d128 FP32完整训练
CPU1线程17.496945s、NPU mixed-c21.064699s；独立FC/Transformer范围不同，原NPU共享/完整数值未验。
本增量尚无新吞吐结论。全面审阅、其余模块/接口治理及代表回归继续按ROADMAP推进；
不得只交付文档或首个KV增量就宣称总体完成。LH脏现场/下游安装锁保持不变。

接续运行git status --short --branch、python scripts/status.py与check_upstream；检查unit及实际日志后继续。
