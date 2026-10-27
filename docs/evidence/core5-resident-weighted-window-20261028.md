# Weighted 完整窗口反向资格

固定实现 `2622e605e77c19360a44b8954cdae5989c3fd724`，固定上游 `0770593`。
[契约与数学对应](../resident-weighted-window-vjp.md)；[机器记录](core5-resident-weighted-window-20261028.json)。

standalone NPU FP32：12个异构图/Read组合和15个正式KV qry配置通过，每组12类独立根，
合计324种根模式；另27次空窗口恒等、预算提前拒绝及旧generation拒绝。
FC/HB、非对齐TimedDAG、PDG反馈和重复边，五记忆、三Read、冻结全部/部分参数均覆盖。
十种发送、固定/上下文控制、match/att/grouped、两key、三正则及来源变换沿固定fixture复现。
候选先行，独立CPU Greedy/autograd核对完整前向及全部参数、外部输入、左raw和pending梯度。
None/连接零、离散支持与int64严格一致；VJP容差保持rtol=3e-5、atol=3e-6。
局部独立公式、KKT与方向导数由阶段/数值组件提供，本报告的调度对照不独立重推全部公式。

NPU另通过三层嵌套服务、18个owner/54提交的静态布局复制，以及21个完整前向案例和15个正式
配置回归。CPU元数据/祖先手算检查17项通过（1.49秒），不称为CPU设备常驻运行。
实际设备物理9→npu:0，LibTorch-NPU2.10/CANN9.0；干净固定源码、六项device gate全部exit0。

CUDA Torch2.10/cu128、Toolkit12.8.1完成相同六个目标及weighted-flow库的构建、loader/help检查。
160个已配置CUDA编译命令均为sm80/90/100；不把未选择目标说成全部已构建。
CUDA资格clean/exit0，无GPU真机，本报告不认证CUDA实际前向、反向或性能。
两后端保留原build receipt，核对未变core源码/二进制，依赖感知构建后记录实际新binary/loader摘要。

开发中，单流任务容量361001由严格LIFO嵌套服务解决，所有阶段串行共享workspace。
完整顺序第9配置曾反复报CANN507011；单独该配置通过。算子跟踪最终定位raw连接位的bool
ScatterUpdate（target[10]/source[2]），更正了此前coordinator元数据操作的错误归因。
NPU静态bool目标改用已知布局连续片段的厂商copy，仍保留bool及精确元数据。
dev8因错误条件宏没有编入修复；dev9修正宏后才首次检验新路径，并通过完整顺序及配置矩阵。
未证明厂商内部根因；原失败、底层日志及两次launcher失败均保留，详见机器记录。

该资格认证实际单窗口完整链式VJP，不认证公共retained训练、optimizer发布/训练checkpoint、
weighted resident FP16/peer或吞吐。后续实现与证据从STATUS/ROADMAP进入。
复现命令见机器记录；重新构建所需源码和fixture均在固定提交，无需依赖旧运行目录。
