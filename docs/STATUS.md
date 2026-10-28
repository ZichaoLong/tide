# 当前交接 — 2026-10-28

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`。
参考仓库/公共环境只读；不使用子代理。实现、有限验证、阶段提交和普通推送已获授权。

## 当前结果

实现提交`ad6d1e71d8798a68765e0efa6f8192c00bfc5a94`已完成正式定向资格。
GDN epsilon-L2及GDN/SSD scaled-RMS的FP32解析一阶收缩更稳定；原FP32前向逐值不变，
输入/容差/AdamW默认值不变，不隐式提升到FP64。图v13、checkpoint v5保持。
[本次报告](evidence/core5-fp32-normalization-20261028.md)及同名JSON是固定证据。

- CPU：**453 passed、2原FP16 strict xfail、0 failed**；原两项FP32 GDN/HB/default AdamW
  通过原断言与恢复检查。覆盖独立Jacobian、版本/None/零、相关循环和批量VJP、完整记录与每切面续接。
  standalone另含每dtype 24图×4次更新、实际checkpoint与optimizer恢复。
- NPU SDK2.10/CANN9.0：core构建/CPU两dtype核及FC/HB×GDN/SSD四组FP32通过，
  各4次AdamW、同设备恢复及CPU交接。队列选择物理1→npu:0。
- 当前Python/Torch-NPU2.12/CANN9.1：纯Python mixed-c GDN/HB、SSD非对齐DAG两组通过，
  独立根、四次AdamW、原始状态/optimizer精确解码与CPU交接；物理9→npu:0。
- CUDA2.10/cu128：同源core/绑定构建及CPU两dtype核通过；无GPU真机。resident核和
  benchmark消费者本轮未重建；Python/native NPU适配矩阵不冒充已重验。

## 作业与来源

`core5-fp32-normalization-{qualify,npu,cuda,python-npu}-20261028`均**passed/0**，
无活动作业；transient服务已卸载，终态以持久status/exit code及验收产物为准。
启动时已核对PID/Transient/background.slice。日志/终态位于`artifacts/jobs/core5-fp32-normalization-对应后缀/`。
主资格记录为`qualification.json`、`status.json`、`task.log`；CPU另有`tests.xml`，
Python NPU另有`checks/result.json`；精确命令及hash已入仓库证据，不依赖.local重建。
background.slice/Nice10，build2/算子1；前三者64G/20min，Python NPU64G/10min，队列wait180s。
CPU只读树`artifacts/worktrees/core5-fp32-normalization-qualified`、设备只读树`core5-private-qualified`
均为ad6d1e7。CPU开发缓存`core5-stable-preparation-dev/build/cpu`复用前后完整核验源码hash及23个binary hash；
原开发manifest的1850434基提交保持，不能当成正式资格来源。开发183项通过也另保留。

## 前阶段选型结果与剩余边界

1f1ec57共享准备顺序、真实梯度/optimizer检查和显式FP64已提交推送为0e04da6证据，
见[上一阶段](evidence/core5-stable-preparation-precision-20261028.md)。其CPU原572 passed/2 failed、
旧基线复现、近零梯度/独立Decimal见证均不回写；FP32 GDN的新通过只属于ad6d1e7。
此前一次流程的D64 FP64完整训练5.702/2.896/2.958/2.903s，D2048约92.62亿主体参数匹配
FP64推理151.416s/窗，完整核对通过；后者424候选/199 Full/946消息，测量后RSS峰值约90.35GiB。
外部争用未隔离。KV不调用本次两个函数，因此没有重复大型测量，也没有新增性能收益声明。

尚未解除：D64/D2048 KV FP32完整记录/训练、FP16 GDN默认AdamW、SDK FP32 affine、
FP32反馈PDG、默认异步双NPU VJP；CUDA真机/x86_64、大型设备训练/跨卡吞吐待验。
跨不同输入的私有合批、设备小算子提交、resident组合矩阵/窗口反向仍有空间。
全部剩余事项见唯一[ROADMAP](ROADMAP.md)，历史LH/core-3及本轮经验见[performance-lessons](performance-lessons.md)。

## 收尾与重入

本轮约定的数值定位、修复、有限参数匹配测量和定向资格均已终态；不恢复旧队列。
本次证据与追溯文档作为实现后的独立提交保存；推送对齐由`git status --short --branch`核对。
没有待收尾的数值作业，不需要重复已通过的门禁。
重入先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
后续独立优化按ROADMAP选取能验证的具体问题，保留现有严格失败与精度边界。
