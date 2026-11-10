# 当前交接 — 接口、性能与架构统一实施（2026-11-10）

用户已授权持续实施、分阶段提交/推送；数学固定0770593，分支graph-execution-foundation。
启动基线4d777bc。用户未跟踪TL-REQ-003/004文档保持原样，不纳入提交；LH现场与下游安装锁不改。

## 已固定与已验证

06b242e交付三项仓库skills、分主题性能经验、源码审阅工具、公共边界修复、autograd census
与实际共享KV snapshot bank；1528df8修复C++ ADL误选std::bind导致原生Read伴随未安装。
固定1528df8的CPU新门禁53通过/1保留xfail，相关248通过/2保留xfail；NPU配置流程72通过/1保留xfail
（含CPU辅助）。Python/native安装消费者与独立SDK CPU/NPU训练、新进程精确恢复通过。
证据：docs/evidence/interface-snapshot-bank-20261110.{md,json}；审阅范围不等于全仓库已审阅。
原FC16 B8/L128/d128 FP32完整训练CPU1线程14.6283秒、CPU16线程16.5985秒、NPU mixed-c20.6338秒；
TTFT CPU3.1947/NPU4.3911秒。原任务完整尺度独立数值/学习not_checked；不宣称NPU普遍加速。
小图B2/L8含私有图13477节点，外层615，SelectBackward3200；仍需削减逐行装配。

## 作业终态

gef-interface-bank：source06b242e，failed；构建通过、2个原生梯度失败，后续未运行。
gef-interface-bankfix：source1528df8，failed；构建/CPU/安装消费者通过，SDK header定位辅助import失败。
gef-interface-bankfix-rest：source1528df8，passed/MainPID0；SDK、有限成本、空闲NPU11检查全部终态。
三者状态/日志均在artifacts/jobs/对应名称/，原失败保留。冻结输入位于artifacts/interface-work/，不修改。
目前无活动长作业。共享NPU本阶段无授权；下次重新走idle-only队列，不沿用旧卡准入。

## 正在集成的第二增量（尚未固定资格）

默认query/region只读张量发布、逐字段unbind、区域无用ID上传/重复回读消冗；
五记忆Describe独立能力选择；compact窗口实际在途容量修复；Python block重复扫描与返回重复传输消冗。
公共数学、独立根、None/零、int64与checkpoint值保持；消费者需修改返回张量时显式clone，见tensor-publication。
开发检查已完成，native/SDK需新构建。现核对非strict KV与Describe分派边界，之后冻结实施提交。
新审阅报告覆盖调度、区域、循环层，文件/函数范围有重叠，不直接累加，也不冒称整个仓库审阅完成。

下一步：完成集成边界检查，提交第二实施增量；新冻结工作树、独立build/venv和background.slice服务
运行相关CPU/NPU/SDK根、训练/恢复，复测FC16成本/census和受影响有限代表；证据单独提交。
使用全部可用CPU/内存与空闲NPU，重构建与正式计时不重叠；当前约681GiB磁盘、1.7TiB可用内存，320CPU。
其余逐模块审阅、接口/性能及代表回归按ROADMAP继续；本阶段成果不等于全部数学/设备/尺度限制解除。

接续：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
