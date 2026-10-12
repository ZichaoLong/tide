# 无人值守测量的当前入口

当前没有运行或排队的实验。旧选择矩阵已结束，历史 CPU Attention 任务已取消；
不从旧目录、timer 或 launcher 自动恢复。结果边界见 [STATUS](STATUS.md) 与
[历史证据](evidence/README.md)，必要记录保留见 [artifact-retention](artifact-retention.md)。

新测量需要具体问题、新配置与输出目录、固定源码、明确资源/时间上限和停止条件。
默认一次独立进程，不为共享资源争用自动重复；项目重型计时串行。
按 [执行契约](execution-flows.md) 与 [measurement-isolation](measurement-isolation.md)
选取所需运行入口，而不是恢复已关闭矩阵。

长任务使用独立 background.slice user service，和交互会话分离；本机
run-durable-user-task/use-local-ascend 规范负责服务与设备操作。
保存任务身份、退出状态、实际结果与已知争用；提交任务不等于通过。
结束时检查主进程和子进程/控制组均已退出，并更新唯一 STATUS。
新线程日常开发无需读取历史 systemd unit 或旧实验目录。
