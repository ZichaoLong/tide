# 当前交接 — 继续完成 P1—P6（2026-11-12）

用户已要求继续并完成[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
当前已提交/推送基线2bca34fd；数学固定0770593838a37c790a568226930567ed60f99b83，19份快照起步核验通过。
可整体重写，允许常规构建/测试、分阶段提交推送授权分支；不在阶段提交后等待重复授权。
三个用户TL-REQ-004/005/006 handoff保持只读且不纳入提交；下游与参考仓库不修改。

## 正在实施

- 跨节点KV Read/Full：lh_direct负责Python/C++相容组、独立多输出VJP和默认列调度接入。
  沿真实bank/配置/设备/预算成组；不能以普通stack抹掉未使用根，原始int64位置保持。
- resident_rewrite负责实际Aggregate/Upd、数值阶段与设备choose/commit的完整组合，
  然后逐边路由、跨窗tape、训练owner/optimizer/checkpoint；组件资格不代替整链。
- req006_diagnostics审查P1/P2/P3/P4/P6并补standalone公共实验的新列接入与精确恢复；
  当前standalone实验仍走旧事件Eager是已定位的实际缺口，不能用SDK组件check代替。
- 主代理集成、资源与资格；补LH非空初态及各输出族独立VJP、代表规模多步训练/新进程恢复。
  未提交开发修改分属上述边界，避免冲突编辑；公共构建注册由主代理协调。

## 活动作业

已提交bb27e4b0（非空LH初态/独立输出族）和4e3e4bdb（计划图AdamW/新进程恢复消费者）；
两个冻结开发检查分别21项与3项通过，原终态留在artifacts/jobs/p1p6-{lh-check-dev-01,planned-dev-01}。

两个冻结开发构建已结束（4e3加56文件overlay；清单在artifacts/p1p6-20261112/core-dev-01-overlay.json）：
Python-owned新库构建并31项定向检查通过；SDK构建在新测试的构造参数/类型名处失败，未执行数值资格。
该两处已修复并完成语法检查，旧failed记录保持。两unit已MainPID=0、空cgroup。
随后跨节点查询/Describe/放置/Settle/位置短检查89通过、9项原数值xfail；standalone相关20项Python通过。
这些都是开发检查，不代替固定提交后的两客户端资格。

正在固定跨节点KV Read/Full/Describe合批与standalone列owner/默认实验入口；
下一动作是从新提交建独立Python-owned/SDK库，执行CPU相关完整检查与新进程续接。
resident sequence已稳定，local/整图设备调度继续开发；主代理单独冻结登记，不将组件当整链。
共享NPU准入前重新采样，普通异步跨卡将只做新的PER_STREAM_QUEUE/TASK_QUEUE_ENABLE有限诊断。

源码与开发结果：artifacts/p1p6-20261112/；终态与log：artifacts/jobs/p1p6-*/。
正式资格的新job/unit/source在启动前更新本节。公共工具链不改；构建期不做正式性能计时。

## 已有结果与资源

上轮[节点/区域合批](evidence/node-region-batching-20261112.md)、
[REQ006诊断](evidence/req006-diagnostics-20261112.md)、
[常量表示/初态配对](evidence/req006-floor-20261112.md)、
[resident组件](evidence/resident-blocks-20261112.md)保持各自源码/失败身份。
原FP32超差、旧resident FC16 EE1019/507002及普通异步跨NPU EE1017仍在ROADMAP中，不能被新组件覆盖。
CPU/GPU源码、构建、真机和性能分列，GPU延后不阻塞本机其他工作。

当前对话后续共享NPU已获持续授权，包括正确性、profile和正式性能；不逐作业再询问，
不将该授权延伸到新会话。实际使用须重新记录PID/start/UID、映射与资源余量，结果标共享条件。
遵循已更新优化skill/执行契约。构建可用全部可用CPU和内存；性能计时协调自身争用，
显式CPUAffinity/taskset并核对actual，不凭AllowedCPUs请求推断隔离。
起步quota约287GiB/软300/硬800GiB，无grace；shm约305GiB空闲，主机可用内存约1.7TiB。
大构建继续用有界shm，durable记录在磁盘；没有磁盘阻塞，不清理用户/参考数据。

接续先检查上述新开发检查，审阅各实现后固定提交，重建所需Python-owned/standalone库并执行相关资格。
STATUS是唯一当前交接，剩余范围/停止重启条件仍只在ROADMAP；原记录不改写。
