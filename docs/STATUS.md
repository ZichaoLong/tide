# 当前交接 — 直接FC的Full跨样本批量化（2026-11-12）

数学固定 `0770593838a37c790a568226930567ed60f99b83`；授权分支 `graph-execution-foundation`。
用户本轮只授权立即改造直接FC的B循环并给出CPU/NPU吞吐；其余重写/删除/skills治理继续只读对齐。
旧实现、旧测试都可在后续重写中移除，唯一数学锚点是固定上游；本轮不执行该整体清退。

## 本轮范围与当前结果

`scripts/fc_tensor_baseline.py::full` 将选中查询跨B打包成 `[B,H,Qmax,Dh]`，
共享每样本KV；Q、输出与FFN投影只计算实际选中行，Attention补齐到批内最大查询数。
Full中不再有B/L主机循环。独立逐查询指数商公式覆盖不齐/空/单行/全活动、各输入VJP；
开发CPU共31 passed，包括非空状态、跨窗、三步SGD及新进程恢复；这还不是固定设备资格。

待实施提交固定后，用前版本 `bb23a9fa` 与新版本运行同一FC16任务，B8/L128/d128、FP32。
新旧直接公式各三模式，数学不等价Transformer16三模式，另一个新SDPA训练点。
每点独立进程，一次reset warmup/一次完整测量；CPU1线程、NPU host2，正式成本不含profile。
新旧代码独立冻结，资源预算每服务8核/128GiB/1800秒，单点240秒。
脚本 `benchmark_fc_batch_pair.py` 保存逐点命令、原始结果、退出码和源码身份。

## 作业与资源

当前尚未提交后台作业；开发短检查已经结束，无在途任务。
计划任务目录 `artifacts/fc-batch-full-20261112`，作业 `artifacts/jobs/fc-batch-full-{cpu,npu}`，
unit `gef-fc-batch-full-{cpu,npu}`；冻结源码计划位于 `/dev/shm/gef-fc-batch-full-20261112/{before,after}`。
复用项目私有 `artifacts/additive-work/env` 与公共CANN9.2.0模块；本轮无需C++构建。
全CPU/内存及本对话共享NPU授权有效，优先空闲卡，记录实际准入与外部争用。
存储已检查：主盘876GiB、shm273GiB可用，账户269GiB/软300/硬800GiB；不受磁盘阻塞。

## 其他状态与下一动作

上轮主机字段owner与resident KV bank的资格分别见
[evidence/read-region-owner-20261112.md](evidence/read-region-owner-20261112.md)及
[evidence/resident-kv-owner-20261112.md](evidence/resident-kv-owner-20261112.md)。
新resident尚无新性能；LH差距及原数值/异步限制不因本次FC修改解除，待办仍见ROADMAP。
三份用户未跟踪REQ-004/005/006交接文档保持原样，不提交。

下一动作：提交本次实现，冻结前后源码；启动CPU/NPU固定检查及有限成本，审阅终态后
单独提交证据并汇报。其余仅汇总用户对张量贯穿、可删除旧测试和直接样板性能要求的反馈。
