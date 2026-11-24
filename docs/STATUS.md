# Stage A 自动补位，两项 Timed A2 已越过原失败步

北京时间 2026-11-24 21:59:13，项目旧队列在 8 张物理卡上运行 55 个 NPU worker。CPU 正在运行 42 个 worker。Timed/d256/S、K 的正式补跑 967、968 已越过第 2959、1760 步，采样时各至第 3001、1771 步。两项沿用补值前向，显式对数质量策略已在 CPU 与 Ascend 独立测量。用 `python scripts/status.py` 查看现场读数。[两例正式接续与短窗比较](pilot-r3-weak-mass-comparison.md) · [扩容记录](evidence/pilot-r3-auto-fill-device6-20261124.json)

Stage A 控制器按模型推进。一个模型凑齐本阶段所需的种子结果并通过相应检查后，控制器会登记其下一组作业。派发器按后端约束补 CPU 空位，NPU 先分散到指定的物理卡，再优先试放到 worker 较少的卡。启动前核对主存，以及扣除已有 worker 未兑现预约后的实际空闲显存。卡 10、15 的共享许可绑定已获准的外部进程，每次新增 worker 都复核身份。单卡并发随作业申请和可用资源变化。[派发实现](../pilotlab/runq_place.py) · [显存与共享准入](../pilotlab/runq_devices.py)

卡 10 的任务 654 在首个训练步前触发 AICPU `IndexPut` 错误 `507018`，内核失败的根因仍未确定。正式补跑 909 在卡 7 使用同一冻结任务和初始参数完成了 64 步，通过健康与因果检查。该模型的三个初始化种子已通过门禁，12 个 A1 作业已登记。[补跑验收与原始失败](evidence/pilot-r3-indexput-retry-20261124.json)

物理卡 6 的 16 步并发测试已于北京时间 2026-11-24 19:50:25 完成并释放卡。六档共 32 份作业均通过验收。Settle 同样七份作业，串行、四路、七路的整批用时分别为 349.2、149.7、126.6 秒，首份分别为 49.1、78.3、123.1 秒。四路到七路的满并发训练吞吐增加 6.7%。Timed 同样两份作业，双路使整批用时从 296.7 秒降至 201.3 秒，首份从 146.5 秒推迟至 194.2 秒。每档测量一次，每份 8,192 个训练 token。[逐档结果与原始记录](evidence/pilot-r3-card-sharing-dev6-short-results-20261124.json) · [本机测试报告](../.local/pilot-r3/benchmarks/card-sharing-dev6-16step-20261124/report.md)

Timed 三路与双路的补测于北京时间 2026-11-24 20:24:54 完成并释放物理卡 6。每档运行同样六份 16 步作业，两个种子各占三份，12 份均通过验收。三路比双路把整批用时从 599.4 秒降至 516.4 秒，节省 13.8%，首份结果推迟 56.5 秒。满并发训练吞吐从 90.8 升至 102.9 token/s。三路 worker 合计 HBM 采样峰值为 50.25 GiB，整卡最低可用 10.76 GiB。每档各测一次。[复核数据](evidence/pilot-r3-dev6-timed-three-workers-20261124.json) · [本机对照报告](../.local/pilot-r3/benchmarks/timed-dev6-three-workers-20261124/report.md)

此前 64 步版本在物理卡 1 上完成了一组 Timed 配对。串行整批用时 17 分 28 秒、首份 8 分 38 秒，双路整批 12 分 12 秒、首份 12 分 01 秒。[64 步配对与暂停记录](evidence/pilot-r3-card-sharing-pause-20261124.json)

此前让出六张卡时暂停的 45 个任务均保留 `stopped_budget` 状态。45 份最终检查点已用冻结的 worker 和基座在 CPU 上解码，核对了文件和训练状态摘要、原 NPU 配置、训练步数、token 数与数据游标。接续清单按任务保存了检查点、原始结果、阶段起点评估来源和本轮已预约的评估预算。收到恢复指示后按该清单开新尝试。[六张卡的交接记录](evidence/pilot-r3-six-card-pause-20261124.json) · [本机逐任务接续清单](../.local/pilot-r3/resource-handoffs/six-card-pause-20261124T153341+0800/handoff.json)

270 个本批主体中，165 个已选定配方，105 个待选。采样时 Stage A 有 660 个任务完成、97 个运行、173 个待派、45 个暂停、0 个最新尝试失败。控制器会随模型进展登记任务。另有 74 个早期主体待复核，阶段 B 门禁关闭。绝对截止为北京时间 2026-12-20 00:48:33。[任务快照](evidence/pilot-r3-timed-weak-mass-20261124.json) · [74 个主体复核](evidence/pilot-r3-deferred-74-review.json)

## 接续入口

本机启动指针为 `.local/pilot-r3/launch.json`，指向控制器 `tide-pilot-r3-vjp-20261124.service` 及其冻结计划。实验输出在 `.local/pilot-r3/campaign-runq/round/`，运行中的实验使用 `~/.local/state/tide-runq`。用 `python scripts/status.py` 核实项目科研进度，用 `python -m pilotlab.runq --root ~/.local/state/tide-runq status` 查看该批队列。账户新任务通过 `~/.local/bin/runq` 提交，使用独立的 `~/.local/state/runq`。下一次整理剩余任务时，先按研究任务身份核对已运行和已完成记录，再决定优先级与放置。[操作入口](README.md#自动补位与手动派发)

项目队列按机器可用主存及未兑现预约准入，原 1 TB 批次额度已解除。[切换与验收记录](evidence/account-runq-rollout-20261124.json)

Timed/d128/X/mixed9 的 A3 已在原阶段起点重算全部 16,384 个评估 token，补测期间训练更新数为零。正式 worker 从已发布的第 7,923 步检查点完成第 7,924 步，并越过旧心跳记录的第 7,935 步。成功补测与此前取消的补测分别记回原预算。[恢复凭据与正式首步](evidence/pilot-r3-runq-live.json)

早期冷检查点的恢复清单在账号私有 SFS。[归档恢复](README.md#恢复已归档的检查点)
