# Stage A 已切换至账号队列，自动补位运行中

北京时间 2026-11-24 10:27:11，正式控制器与独立 worker 服务已通过进程身份和驱动核验。38 个 CPU worker、11 个 NPU worker 正在执行。物理卡为 2, 4, 5, 7, 8, 11, 14。自动目标为 8 张 NPU 和 40 个 CPU worker。其余物理卡有外部进程。当前批次共用原 1 TB 主存额度，预计峰值占用会限制追加，即使卡上还有空余显存。用 `scripts/runq status` 查每张卡的实际 worker 数、显存预记余量和队列等待原因。[实机采样](evidence/pilot-r3-runq-live.json) · [调度命令](README.md#自动补位与手动派发)

270 个本批主体中，147 个已选定配方，123 个待选。继承 4,206 个完成请求。本次首波逐一接回 243 个请求，采样时本轮已核验 2 个完成、49 个运行、192 个待派。任务完成数和主体选方数分别统计。另有 74 个早期主体待复核，阶段 B 门禁关闭。绝对截止为北京时间 2026-12-20 00:48:33。[接续审阅](evidence/pilot-r3-runq-readiness.json) · [74 个主体复核](evidence/pilot-r3-deferred-74-review.json)

## 接续入口

本机启动指针为 `.local/pilot-r3/launch.json`，正式实验目录为 `.local/pilot-r3/campaign-runq/`。依赖控制器使用冻结提交 `9b6cc81a`，模型 worker、基座、数据及预算沿用已验收的前序版本。账号队列保存在 `~/.local/state/tide-runq/`，当前派发版本为 `a971fb31`。实机热切换时 28 个 CPU 与 13 个 NPU worker 的 PID 和启动时钟全部延续。手动派发、调整并发和调度器升级均可在活动作业期间进行。[现场与源码身份](evidence/pilot-r3-runq-live.json)

Timed/d128/X/mixed9 的 A3 已在原阶段起点重算全部 16,384 个评估 token，补测期间训练更新数为零。正式 worker 从已发布的第 7,923 步检查点完成第 7,924 步，并越过旧心跳记录的第 7,935 步。成功补测与此前取消的补测分别记回原预算。[恢复凭据与正式首步](evidence/pilot-r3-runq-live.json)

用 `python scripts/status.py` 核实最新科学进度，用 `scripts/runq status --state queued` 查可手动派发的任务 ID。早期冷检查点的恢复清单在账号私有 SFS。[状态与接续](README.md#接续同一批) · [归档恢复](README.md#恢复已归档的检查点)
