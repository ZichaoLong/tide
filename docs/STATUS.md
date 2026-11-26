# Stage A 的 44 个暂停主体已接续，74 个早期主体已入队复核

北京时间 2026-11-26 19:50，项目旧队列在 14 张物理卡上运行 99 个 NPU worker，CPU 运行 40 个 worker，包含早期复核。270 个主体中已有 206 个选定配方。训练调度记录有 1,722 个完成、138 个运行、23 个待派和 8 个已替换。B 门禁关闭，原截止为北京时间 2026-12-20 00:48:33。[本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

## 44 个主体的训练接续

100 个接续作业的编号为 1946–2045，按 A3 的 18 项、A2 的 48 项、A1 的 34 项依次登记。采样时 4 个完成，83 个运行，13 个等待，86 份训练日志已核实首步等于检查点步数加一。恢复使用原训练程序、后端、父状态和阶段初评。按原逐主体预算，只在取得执行资源时合计追加 19,328 个评估 token。Lattice/S、d128、k2 主体已选出 r2、r1，控制器自动生成两个配方各两种子的 A2 任务（2141–2144）。[本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

本次只切换了实验控制器。原队列派发器保持同一个 systemd InvocationID。切换前 57 个 worker 中，采样时 55 个仍是原 PID 和启动时钟，2 个正常完成。自动使用物理卡 0–11、14、15，CPU 自动目标为 40 个 worker。新作业按驱动物理空闲显存扣除已有任务尚未兑现的峰值预约后准入。[本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

## 74 个早期主体的源码复核

93 个复核作业的编号为 2046–2138，其中 64 个在 NPU 上执行，29 个在 CPU 上执行。不同后端的历史配方按原后端比较。采样时已有 11 份作业完成，1 份运行，81 份等待。每份作业分别启动历史和目标版本，在同一批输入上记录损失、梯度与后继状态。813 个比较点覆盖旧选方及其竞争配方，或尚未选方主体的首个缺口。12 个 Shannon 主体另外比较新的弱质量配方。准备预算预记 6,518,528 个 token。[本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

Settle/K/singleton7 的 CPU 失败批次在旧版再次触发分配梯度溢出。冻结参考基座 `6e14a6cc` 和弱质量基座 `e9bd1483` 均完成第 7,354 次更新。两种子的单步结果不能单独确认完整轨迹或配方排名。36 个主体还有 78 条 A2 比较缺少父状态及本阶段两代检查点，已逐项核对退役记录与私有归档清单。[历史排名与缺口](evidence/pilot-r3-deferred-74-review.json) · [本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

## 定位与接续入口

运行 `python scripts/status.py` 或加 `--json` 查询两组作业。运行 `python -m pilotlab.runq --root ~/.local/state/tide-runq show 2046` 查询单项。启动指针在 `.local/pilot-r3/launch.json`。其 `last_signal_admission` 绑定接续队列编号，`deferred_source_review` 绑定逐主体复核计划、资源和入队记录。每份复核计划目录下的 `observations/comparison.json` 保存实测结果。[队列操作](README.md#自动补位与手动派发)

此前 7 项显存失败的补跑均已完成，健康与因果检查通过。其两项 A2 精确恢复后首步均为 513，另五项 A1 首步均为 1。[原补跑计划](evidence/pilot-r3-oom-seven-20261126.json) · [本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)
