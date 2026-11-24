# Stage A 在 Device 7、14 运行，45 个任务保留检查点待恢复

北京时间 2026-11-24 16:43，项目队列在物理卡 7、14 各运行 7 个 worker，CPU 运行 40 个。Device 9 已释放，驱动在 16:43:48 记录到 0 个进程。原卡上 6 个未完成任务已在 Device 7 从各自检查点的下一步继续训练。自动补位限定为物理卡 7、14，CPU 自动目标为 40。控制器切换时，另外 47 个运行中的 worker 有 45 个保留原进程，2 个自然完成。[换卡与续跑记录](evidence/pilot-r3-device9-to7-20261124.json)

此前让出六张卡时暂停的 45 个任务均保留 `stopped_budget` 状态。45 份最终检查点已用冻结的 worker 和基座在 CPU 上解码，核对了文件和训练状态摘要、原 NPU 配置、训练步数、token 数与数据游标。接续清单按任务保存了检查点、原始结果、阶段起点评估来源和本轮已预约的评估预算。收到恢复指示后按该清单开新尝试。[六张卡的交接记录](evidence/pilot-r3-six-card-pause-20261124.json) · [本机逐任务接续清单](../.local/pilot-r3/resource-handoffs/six-card-pause-20261124T153341+0800/handoff.json)

270 个本批主体中，164 个已选定配方，106 个待选。16:43 的任务记录有 426 个完成、54 个运行、45 个暂停、333 个待派。另有 74 个早期主体待复核，阶段 B 门禁关闭。绝对截止为北京时间 2026-12-20 00:48:33。[74 个主体复核](evidence/pilot-r3-deferred-74-review.json)

## 接续入口

本机启动指针为 `.local/pilot-r3/launch.json`，指向控制器 `tide-pilot-r3-dev7-20261124.service` 及其冻结计划。实验输出在 `.local/pilot-r3/campaign-runq/round/`，运行中的实验使用 `~/.local/state/tide-runq`。用 `python scripts/status.py` 核实项目科研进度，用 `python -m pilotlab.runq --root ~/.local/state/tide-runq status` 查看该批队列。账户新任务通过 `~/.local/bin/runq` 提交，使用独立的 `~/.local/state/runq`。下一次整理剩余任务时，先按研究任务身份核对已运行和已完成记录，再决定优先级与放置。[操作入口](README.md#自动补位与手动派发)

项目队列按机器可用主存及未兑现预约准入，原 1 TB 批次额度已解除。[切换与验收记录](evidence/account-runq-rollout-20261124.json)

Timed/d128/X/mixed9 的 A3 已在原阶段起点重算全部 16,384 个评估 token，补测期间训练更新数为零。正式 worker 从已发布的第 7,923 步检查点完成第 7,924 步，并越过旧心跳记录的第 7,935 步。成功补测与此前取消的补测分别记回原预算。[恢复凭据与正式首步](evidence/pilot-r3-runq-live.json)

早期冷检查点的恢复清单在账号私有 SFS。[归档恢复](README.md#恢复已归档的检查点)
