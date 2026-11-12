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

## 当前未提交实施与下一动作

1. C++正写入整组数值边界：不同R/B、真实共享角色、独立根，27节点私图开发见证；
   新prepare及旧Read/Full、完整columns两步SGD开发通过。
2. Python/C++跨节点Read按目标device/dtype统一映射，保留真正共享参数身份；
   Python25定向/集成和C++9配置/2版本守卫开发通过。C++新测试需注册CMake/build manifest。
3. Python区域均值：一次receiver×row×physical-source装配/中心化VJP，14独立公式测试通过；
   相关集成64通过/5原xfail，测试自身singleton cotangent误断言已修正；C++对应待集成。
4. resident移除全参数layout占位与optimizer/8；实际13类槽/元数据、发布和live workspace计费。
   bank游标在设备复用容量分支，静态B×logR缩为logR，仍按实际最小桶执行。20项CPU/9TU语法通过；
   EE1019修正与实际容量边界尚无新设备资格。raw逐帧快照/重复历史仍未重写。

下一动作：统一审阅这些未提交源码/契约、补新C++目标与短相关开发检查，提交新实现。
以新提交独立工作树/构建，不修改6011d599固定输入；按projection→messages→optimizer/session
逐层验收resident，组件失败即停依赖路径。主机CPU/NPU安装/恢复及LH与FC有限回归独立推进。
相关开发记录在artifacts/resident-bank-dev/及artifacts/tensor-rewrite-20261112/。
P1—P6尚未全部完成，剩余范围与停止/重开条件只在ROADMAP。

## 作业与资源

本轮6011d599所有服务已终态，无活跃构建/测试/测量。
固定源码/构建：/dev/shm/gef-tensor-rewrite-20261112/{source,core-native,core-sdk,resident-native,resident-sdk,install}。
持久状态和日志：artifacts/jobs/tensor-rewrite-*；原启动cwd缺失的两次失败也保留。
完整矩阵/命令/队列准入：artifacts/tensor-rewrite-20261112/；已审关键结果进入Git证据。
容量：主盘831GiB余量、shm约290GiB余量；用户313GiB/软300/硬800GiB，软限尚有宽限期。
无磁盘阻塞。长作业继续固定源码、background.slice、有限资源/时长并记录真实终态。
重新进入：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
