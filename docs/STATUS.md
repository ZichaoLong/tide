# 当前交接 — 联合prefill反向装配与代表回归（2026-11-10）

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准持续实施/推送、全部可用CPU及空闲/有余量共享NPU；数值有限诊断后分列推进。
参考/下游只读，无子代理；两份未跟踪TL-REQ-003/004用户handoff原样保留、不提交。

已交付分量窗口、联合KV准备、批量守卫与整序列VJP，见相关evidence索引。
a553bea交付query连续完整根分区合并、按需内部trace、真实选定路径精确恢复及执行身份v4。
固定CPU249通过/2原数值xfail，CPU安装消费者与SDK三项CTest/训练/新进程恢复通过。
原NPU门禁73通过/2原数值xfail/1观测失败，任务failed原样保留；非数值断言失败。
fb406dd仅补反向工作线程的观测上下文，固定CPU24/NPU2通过，
补全NPU两种安装消费者和SDK完整/outputs三步及新进程精确恢复，均已终态passed。
有界同材料诊断：合并前后NPU梯度逐位相等，对FP64最大9.61643e-8。

当前工作区：KV前缀伴随K/V按owner一次拼接，Python全部相关30项通过（FP64/FP32），字面导数和固定concat次数检查包含其中；
C++尚未构建，不冒用旧构建。本提交后固定源码，新建Python/native与独立SDK构建并验证。
任何重型构建须等本项目性能任务终态；活动源码在只读worktree中，不读工作区。

活动任务（原始记录在artifacts/jobs/，详情由scripts/status.py显示）：
- gef-joint-prefill-representatives：固定a553bea/fixed-roots/env-roots，CPU16/NPU18项；
  B4/P16、一次预热/测量、CPU1线程/NPU2host线程，宽FC/HB/LH/sum/qry/SSD/GDN及三客户端。
  独立数值not_checked，精确下一窗恢复逐项验证；512G/320CPU/14400秒。
  日志artifacts/jobs/joint-prefill-representatives/task.log；结果artifacts/joint-prefill/representatives-{cpu,npu}/runs/result.json。
- gef-joint-prefill-resident-small：已准备固定a553bea，NPU6项小型PDG/一般端口/训练/恢复，
  使用已成功构建的resident-roots与build-roots-a；8CPU/64G/1800秒，仅正确性。
  已终态passed：6项实际NPU通过，物理7；不认证整序列resident性能。
  日志artifacts/jobs/joint-prefill-resident-small/task.log；JUnit artifacts/joint-prefill/resident-small-npu.xml。

下一步：审阅每个inner结果；补原FC16的NPU完整/最小返回成本和独立小profile，
再补选定resident/双NPU/大规模及同公式层链基线，不由小图外推性能或全部能力。
原FC16 CPU a553完整29.788/最小22.068秒；NPU对应尚未运行。
不同返回契约、不同设备实际Full/消息不混算等价加速；原任务数值/学习未认证。

起步：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
检查/停止：systemctl --user status/stop 上述准确unit；不能将live任务记通过。
唯一待办与停止条件见ROADMAP，GPU按授权延后。下一提交补证据、经验与导航，保持原失败。
