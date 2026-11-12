# 当前交接 — 区域张量与resident控制图重写（2026-11-12）

数学固定0770593838a37c790a568226930567ed60f99b83，19份快照通过。
分支graph-execution-foundation。用户授权整体重写、阶段提交/非强推；mixed-a/b/c仅为
性能放置预设，不是架构层。当前对话允许共享NPU，GPU资格延后。
三份用户TL-REQ-004/005/006未跟踪handoff只读、不提交。

## 已固定结果

[6011d599资格及44点成本](evidence/tensor-kernels-20261112.md)：Python/C++ KV整组Read/Describe/Full，
Python整组正写入、resident实际bank GEMM。两套core/resident构建通过，SDK三CTests通过。
CPU272通过/9原数值xfail/4旧bucket断言失败；原失败保留，后继测试已改为实际新bucket。
CPU/NPU×Python/native共8个安装消费者三步AdamW/新进程精确恢复通过。
FC16 B8/L128默认native完整训练CPU1/NPU-host2为1.1729/0.3636秒；
LH176 B4/L16为1.5954/3.5953秒，直接公式0.7546/0.6322秒。单点成本不是全尺度数学资格。
resident9组件真机通过；block-kv-messages录入时EE1019（stream task buffer full），停止依赖资格。
厂商摘录已固化到证据JSON，不将新库编译或局部通过冒充完整训练/规模通过。
33文件静态审阅及四条独立profile与计时分列；模块边界和接收节点装配仍有明显成本。

## 当前实施与下一动作

本后继实现包含C++整组正写入、跨节点Read按目标统一传输、Python/C++区域物理来源均值张量/VJP，
以及resident metadata-only optimizer、实际13类存储计费和设备bank循环体复用。
Python46项相关开发检查通过；C++prepare/Read/Full/columns和独立Aggregate开发见证另有记录。
Aggregate以receiver×row×physical-source交接，一次组装和中心化反向；重复边、None/零、独立根保留。
混合放置仍是可简化预设，不要求数值边界固定拆成软件层。

下一动作：提交本实现，以该固定提交建立/dev/shm/gef-region-tensor-20261112/source，
独立构建core-native/core-sdk及resident-native/resident-sdk，然后跑定向CPU与安装消费者。
resident验收顺序projection→block-kv-messages→optimizer→其余组件/完整owner；
组件失败停止依赖路径，主机资格/有限FC与LH回归独立继续。尚无本后继真机结论。
原失败仍保留；P1—P6未全部完成，剩余范围和重开条件只在ROADMAP。
新运行计划与启动器：artifacts/region-tensor-20261112/launch-plan.json及*-build.sh、qualification.py。

## 作业与资源

6011d599所有服务已终态；新资格尚未启动。新构建分别CPU0–159/160–319、各512GiB/2400秒，
通过scripts/job.py与background.slice启动；新记录artifacts/jobs/region-tensor-*。
固定源码/构建：/dev/shm/gef-tensor-rewrite-20261112/{source,core-native,core-sdk,resident-native,resident-sdk,install}。
持久状态和日志：artifacts/jobs/tensor-rewrite-*；原启动cwd缺失的两次失败也保留。
完整矩阵/命令/队列准入：artifacts/tensor-rewrite-20261112/；已审关键结果进入Git证据。
容量：主盘831GiB余量、shm约290GiB余量；用户313GiB/软300/硬800GiB，软限尚有宽限期。
无磁盘阻塞。长作业继续固定源码、background.slice、有限资源/时长并记录真实终态。
重新进入：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
