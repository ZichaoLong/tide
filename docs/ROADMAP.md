# Stage A 当前待办

1. 用 `python scripts/status.py --json` 分别查看 270 主体主队列、74 主体判定，以及其中 70 个主体的目标源码补跑。补跑由持久控制器按种子和父检查点自动生成后继。2237 已在 Device 14 启动，2264 已由 2372 在 Device 5 接续，所有后端按原资源设置补位。查看 `deferred_followup.npu_release`、`by_backend`、`blocking_reason` 及训练日志核对进度。[当前交接](STATUS.md) · [运行快照](evidence/pilot-r3-deferred-followup-20261127.json)
2. 逐主体收齐所需训练与确认。49 个比较两个入围配方，2 个确认旧赢家，19 个重做六配方，4 个保留带旧源码标识的配方。重点核验 Settle/K/singleton7 的完整 A3、两种子的精确恢复和活动量。对执行或数值失败保留错误、来源、检查点及剩余预算，其他独立轨迹继续推进。[逐主体处置及依据](pilot-r3-deferred-dispositions.md)
3. 收齐全部 344 个主体与匹配对照的逐项结论后，联合核对 A 门禁、选方证据、匹配质量、来源差异及正式运行预算。根据最终 B worker 的源码和输入形状导出 A 阶段资源表，缺少同形状成功记录的配置按既定方法实测或复核预约。[显存数据与形状匹配](evidence/pilot-r3-shape-hbm-recovery-20261125.json) · [资源表用法](README.md#用阶段-a-的观测申请-npu-显存)
4. 跟进 270 主体内其余 A2、A3 及对照。按各自种子、曝光与来源复核原 44 个暂停主体的最终选方、7 个弱质量配方组、pool 保底统计及低余量 quad 的边界与梯度。收尾迁移结合剩余步数、可用检查点、同卡并发与整批关键路径逐项决定。[原 44 主体接续与队列](evidence/pilot-r3-signal-and-deferred-20261126.json) · [弱质量比较](pilot-r3-weak-mass-comparison.md)
