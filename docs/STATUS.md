# Stage A 使用两张 NPU，45 个任务已暂停并保存检查点

北京时间 2026-11-24 15:38，项目队列保留 Device 9、14，有 15 个 NPU worker 和 40 个 CPU worker。Device 2、4、5、7、8、11 的租约已全部释放。45 个 worker 因本次让卡在完整训练步保存检查点后退出，项目状态为 `stopped_budget`，worker 停因是 `signal`。[六张卡的交接记录](evidence/pilot-r3-six-card-pause-20261124.json)

45 份最终检查点已用冻结的 worker 和基座在 CPU 上解码，核对了文件和训练状态摘要、原 NPU 配置、训练步数、token 数与数据游标。接续清单按任务保存了检查点、原始结果、阶段起点评估来源和本轮已预约的评估预算。收到恢复指示后按该清单开新尝试。[本机逐任务接续清单](../.local/pilot-r3/resource-handoffs/six-card-pause-20261124T153341+0800/handoff.json)

270 个本批主体中，162 个已选定配方，108 个待选。15:38 的任务记录有 325 个完成、55 个运行、45 个暂停、189 个待派。另有 74 个早期主体待复核，阶段 B 门禁关闭。绝对截止为北京时间 2026-12-20 00:48:33。[74 个主体复核](evidence/pilot-r3-deferred-74-review.json)

## 接续入口

本机启动指针为 `.local/pilot-r3/launch.json`，指向控制器 `tide-pilot-r3-runq-policy-1a339693.service` 及其冻结计划。实验输出仍在 `.local/pilot-r3/campaign-runq/round/`，运行中的实验使用 `~/.local/state/tide-runq`。用 `python scripts/status.py` 核实项目科研进度，用 `python -m pilotlab.runq --root ~/.local/state/tide-runq status` 查看该批队列。账户新任务通过 `~/.local/bin/runq` 提交，使用独立的 `~/.local/state/runq`。下一次整理剩余任务时，先按研究任务身份核对已运行和已完成记录，再决定优先级与放置。[操作入口](README.md#自动补位与手动派发)

项目队列按机器可用主存及未兑现预约准入，原 1 TB 批次额度已解除。[切换与验收记录](evidence/account-runq-rollout-20261124.json)

Timed/d128/X/mixed9 的 A3 已在原阶段起点重算全部 16,384 个评估 token，补测期间训练更新数为零。正式 worker 从已发布的第 7,923 步检查点完成第 7,924 步，并越过旧心跳记录的第 7,935 步。成功补测与此前取消的补测分别记回原预算。[恢复凭据与正式首步](evidence/pilot-r3-runq-live.json)

早期冷检查点的恢复清单在账号私有 SFS。[归档恢复](README.md#恢复已归档的检查点)
