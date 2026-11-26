# Stage A 的 44 个暂停主体已接续，早期主体复核已收齐 89/93 份报告

北京时间 2026-11-26 23:04，项目旧队列运行 79 个 NPU worker 和 40 个 CPU worker。270 个主体中已有 213 个选定配方。训练调度记录有 1,812 个完成、119 个运行、16 个待派和 8 个已替换。B 门禁关闭，批次截止为北京时间 2026-12-20 00:48:33。[队列与复核快照](evidence/pilot-r3-deferred-review-deadline-20261126.json)

## 44 个主体的训练接续

100 个接续作业的编号为 1946–2045，按 A3 的 18 项、A2 的 48 项、A1 的 34 项依次登记。已有 67 个完成，33 个运行。恢复使用原训练程序、后端、父状态和阶段初评，86 份训练日志此前已核实首步等于检查点步数加一。按原逐主体预算，只在取得执行资源时合计追加 19,328 个评估 token。Lattice/S、d128、k2 主体已选出 r2、r1，控制器自动生成两个配方各两种子的 A2 任务（2141–2144）。[队列快照](evidence/pilot-r3-deferred-review-deadline-20261126.json) · [逐项接续证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

44 个主体接续时切换了实验控制器，队列派发器沿用原 systemd InvocationID。当时 57 个 worker 中，55 个以原 PID 和启动时钟继续运行，2 个正常完成。自动使用物理卡 0–11、14、15，CPU 自动目标为 40 个 worker。新作业按驱动物理空闲显存扣除已有任务尚未兑现的峰值预约后准入。[训练接续与资源设置](evidence/pilot-r3-signal-and-deferred-20261126.json)

## 74 个早期主体的源码复核

93 份复核计划（初始队列编号 2046–2138）中有 64 份使用 NPU，29 份使用 CPU。813 个比较点覆盖旧选方及其竞争配方，或尚未选方主体的首个缺口。每个版本在原后端用同一批输入比较损失、梯度与后继状态，12 个 Shannon 主体另测弱质量配方。原计划预记 6,518,528 个准备 token。[复核矩阵与原始入队凭据](evidence/pilot-r3-signal-and-deferred-20261126.json)

89 份报告已经落盘。Timed/K/singleton7 作业 2069 在 900 秒时中断，接续作业 2209 在 Device 7 复用 31 条观测，只补 `weak_log` 种子 2026112002 的 r1–r5，独立结算 2,048 个 token。三个版本各 12 条更新已齐。2065、2070、2071、2072 已在原队列编号上更新执行命令，各等单卡 52 GiB。复核按比较点保存结果，运行时间受批次绝对截止和准备预算约束。[接续、合并与当前预约](evidence/pilot-r3-deferred-review-deadline-20261126.json)

Settle/K/singleton7 的 CPU 失败批次在旧版再次触发分配梯度溢出。冻结参考基座 `6e14a6cc` 和弱质量基座 `e9bd1483` 均完成第 7,354 次更新。两种子的单步结果不能单独确认完整轨迹或配方排名。36 个主体还有 78 条 A2 比较缺少父状态及本阶段两代检查点，已逐项核对退役记录与私有归档清单。[历史排名与缺口](evidence/pilot-r3-deferred-74-review.json) · [本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)

## 定位与接续入口

运行 `python scripts/status.py` 或加 `--json` 查询两组作业。运行 `python -m pilotlab.runq --root ~/.local/state/tide-runq show 2209` 查询单项。启动指针在 `.local/pilot-r3/launch.json`。其中 `last_signal_admission` 绑定训练接续编号，`deferred_source_review.submissions` 指向逐模型的有效复核编号与结果目录，`original_submissions` 指回首次提交。每项报告为 `comparison.json`，有 `observations` 字段时取该目录，否则取原计划旁的 `observations/`。[队列操作](README.md#自动补位与手动派发)

此前 7 项显存失败的补跑均已完成，健康与因果检查通过。其两项 A2 精确恢复后首步均为 513，另五项 A1 首步均为 1。[原补跑计划](evidence/pilot-r3-oom-seven-20261126.json) · [本次队列与逐项证据](evidence/pilot-r3-signal-and-deferred-20261126.json)
