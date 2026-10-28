# 当前交接 — 2026-10-28

授权分支`graph-execution-foundation`；实现`474f9af32d35ca7d8bc7bf6359dbddb3ec91dfb3`已提交推送，
包含`56631eb`局部分配坐标/诊断与全部eager KV低秩求值。固定数学`0770593838a37c790a568226930567ed60f99b83`；
19/5快照已同上游Git对象核对。参考仓库/公共环境只读，不使用子代理，不恢复旧队列。
用户授权持续实现、有限实验、分阶段提交及普通推送，提交后继续。

## 当前资格

固定只读工作树`artifacts/worktrees/core5-private-qualified`检出474f9af。
CPU `core5-factorized-qualify-cpu-v2-20261028`终态passed/exit0：687 passed/2 strict xfail；
另固定源码59项诊断/大共同阈值测试通过，结果`artifacts/diagnostics/core5-factorized-extra-tests`，合计746 passed。
CPU core复用开发产物前已重新核对完整C++内容hash和每个binary hash；独立benchmark在固定源码重新构建。
CUDA `core5-factorized-qualify-cuda-v3-20261028`终态passed/exit0：core/resident sm80/90/100构建、loader和core CPU检查。
本机无NVIDIA设备节点，GPU实机仍待验。
NPU `core5-factorized-qualify-npu-v3-20261028`终态passed/exit0：core/resident构建通过，队列已取得物理1/13，
allocation FP32及flow/config/training/node-training两dtype共9项真机全部通过；SDK和纯Python eager KV各4次更新/恢复通过。
输出统一为`artifacts/jobs/core5-factorized-qualify-{cpu-v2,npu-v3,cuda-v3}`，查看status.json/task.log。
服务均background.slice/Nice10、线程1/build2/MemoryMax128G；CPU30min、设备90min，NPU队列180s/设备1800s。
首批启动因预建输出目录拒绝、设备v2因已有resident目录拒绝均保留原failed；没有覆盖失败记录。

## 数值结果与下一步

开发`core5-factorized-kv-dev`终态failed：198 passed/2 xfail、standalone材料5项通过；
原wide2048可执行完整窗（1.17269s），但keys最大差4.95911e-5仍strict mismatch。原开发日志/快照保留。
D64 Python/native两步均离散结构一致，完整返回/梯度/SGD仍严格不等价。
30组实际分配经独立120位Decimal复核，分支/free集合一致，局部误差最大8.34e-8；
评分差在传入求解器前已出现，不能继续把全部差异归为分配迭代误差。
实际wide操作数的重复行BMM探针仍有2.861e-6差异，只是局部算术证据，不是整图差异的完整解释。
两个原大型残差失败行已存tests/data；本次修复保持dtype/容差/活动集/精确解VJP，未放宽守卫。

CPU `core5-factorized-study-cpu-20261028`已终态failed/exit1：两个medium完整训练通过（56631eb 0.366831s、474f9af 0.363974s；单次约持平），D64及matched2048均完成测量窗后严格核对失败。
D64 keys最大差2.26498e-6；约92.62亿主体参数matched2048 keys最大差4.48227e-5，推理窗130.351s、RSS峰值约50.67GB，不是等价吞吐。
两条显式FP64独立调度两步诊断及medium独立CPU profile通过。原始材料、残差失败和本次完整对照失败均保留。
输出`artifacts/jobs/core5-factorized-study-cpu/{status.json,task.log,runs/study.json}`；systemd同名unit退出1。
原始CPU基线binary与固定56631eb的C++/消费者内容hash一致，身份`.local/core5-factorized-baseline-identity.json`。

NPU `core5-factorized-study-npu-20261028`终态passed/exit0，物理13→npu:0：HB medium纯Python mixed-c，0503593→474f9af单次2.189634s→2.145125s，完整返回/optimizer核对通过。
这是累计修改对照，约−2.0%不足以证明稳定收益；每配置一次流程，外部负载未隔离。
输出`artifacts/jobs/core5-factorized-study-npu/{status.json,task.log,queue-state.json,runs/study.json}`，服务已终止。

阶段记录开发任务`core5-benchmark-progress-dev-20261028`已终态passed/exit0：新standalone构建及6项检查通过（五族FP32完整训练、损坏材料拒绝、FIFO强制中断后阶段可读）。
固定开发树`artifacts/worktrees/core5-benchmark-progress-dev`，源码归档/tree hash在job内；没有改变474f9af已完成的数值测量。
输出`artifacts/jobs/core5-benchmark-progress-dev/{status.json,task.log,development.json,standalone-build/result.json}`。

当前提交设施修正（main.cpp、材料测试与weighted-performance契约）；提交后立即创建
`artifacts/worktrees/core5-benchmark-progress-qualified`，冻结该提交，单独运行CPU定向资格。
准备服务`core5-benchmark-progress-qualify-20261028`：background.slice/Nice10/线程1/build2/MemoryMax32G/15min；
从新固定源码构建消费者、重新校验未改动core内容/binary hash、运行相同6项并保存JUnit。
输出将为`artifacts/jobs/core5-benchmark-progress-qualify/{status.json,task.log,qualification.json,tests.xml}`，当前尚未启动。
下一命令：创建固定工作树→启动`.local/core5-benchmark-progress-qualify.sh`→核验终态→汇总证据/经验/ROADMAP→证据提交、推送。
数值资格仍对应474f9af，阶段记录资格另对应本次设施提交；不得混用。

## 保留边界

[既有完整基座与资格](evidence/README.md)、[性能经验](performance-lessons.md)、[数值/异步隔离](evidence/core5-numerical-isolation-20261028.md)。
eager FP16 GDN AdamW、SDK FP32 affine、默认异步双NPU、FP32反馈PDG及新D64严格失败仍保留。
resident KV仍用其原稠密组合矩阵/tape，本次低秩改变只作用于eager。
CUDA真机、x86_64、大型设备完整训练/跨卡吞吐待验；局部公式或参数匹配不认证全规模性能。
主树有STATUS、standalone阶段记录修正及相关测试/契约未提交；运行任务只读取固定只读工作树。

standalone阶段信息原来仅在窗口结束时持久化，本轮在fixture/构造/warmup/measurement/验证阶段于计时外落盘；不改变474f9af测量或数值算法。
