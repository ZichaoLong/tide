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
修订后新bank定向9通过/1保留xfail；native06b242e构建通过，首资格51通过/1旧xfail/2新增Read梯度失败。
定位为未限定bind调用被C++ ADL选中std::bind，未安装手写VJP；原生shared梯度为零，unshared/Python正常。
编译期static_assert证实错误overload；现改明确attach_bank_vjp名称，待新冻结build验证，不能归为舍入。
公共边界开发107/67/39/33项组通过（重叠不累加）；图诊断11项通过。
独立审阅补60组重叠/冻结/重复根探针和大int64检查；native存储组改稳定首次出现顺序。
这些是开发检查，不能替代固定源码资格或性能结果。

## 下一动作与作业

首服务gef-interface-bank已终态failed/MainPID=0；source06b242e，构建通过，首CPU门禁失败，
相关/SDK/性能/NPU均未执行。原记录artifacts/jobs/interface-bank/status.json及task.log保留。
冻结source-bank/env-bank/build-bank及驱动保留，不修改旧任务输入。
当前main工作树在修bank ADL、compact窗口错误累计live容量，以及区域/循环头批量低垂工作。
三位子代理分属调度、区域、循环模块，主代理统一集成；等待短检查后建立下一固定源码重新构建。
后续仍CPU1/16线程FC16成本、独立census，空闲单NPU资格与成本，SDK三客户端恢复。
共享NPU本阶段无授权；只使用空闲卡，不能复用旧shared adapter。CPU约1.7TiB可用，文件系统约685GiB，
本用户281GiB/soft300/hard800GiB；正式计时与本项目重构建不重叠。

## 基线与剩余工作

旧结果见docs/evidence/fc16-cost-attribution-20261110.md：FC16 B8/L128/d128 FP32完整训练
CPU1线程17.496945s、NPU mixed-c21.064699s；独立FC/Transformer范围不同，原NPU共享/完整数值未验。
本增量尚无新吞吐结论。全面审阅、其余模块/接口治理及代表回归继续按ROADMAP推进；
不得只交付文档或首个KV增量就宣称总体完成。LH脏现场/下游安装锁保持不变。

接续运行git status --short --branch、python scripts/status.py与check_upstream；检查unit及实际日志后继续。
