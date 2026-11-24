# 当前待办

1. A 阶段已暂停。当前讨论按剩余显存手动派卡的通用队列。接续 270 个主体及匹配对照时，以停机记录中的工作进度、预算与检查点为起点。[已暂停的现场](evidence/pilot-r3-r30-paused.json)

   按运行采样中各任务的接入步核对后续训练与最终评估，跟进 Exchange/d256/S/independent、Timed/K/singleton7 的终态和 pool 保底统计。用正式心跳、NPU 驱动 PID 和检查点记录跟进直接派工后的并发、共享吞吐及硬配额余量。按实际训练样本复核弱余量 quad 的边界与梯度，保留新旧源码的选择依据。收尾时按空卡、剩余步数和可用检查点评估节时。跨设备接续需单独核验裁剪前梯度范数和迁移记录。[r30 接续](evidence/pilot-r3-r30-resource-readiness.json) · [运行采样](evidence/pilot-r3-main-r30-handoff.json) · [quad 验收](evidence/pilot-r3-r24-foundation-quad-total-acceptance.json) · [跨设备验证](evidence/pilot-r3-r23-cpu-npu-handoff.json)
2. 接续 74 个早期主体的迁移审阅。每个模型的有效配置键、旧源码、排名或首个缺口、候选检查点和历次评估预记量已落表。原始 216 个主体中有 74 个早期曝光，剩余 142 个与补充的 128 个组成 r30 的 270 个。[逐模型清单](evidence/pilot-r3-deferred-74-review.json)
   - **42 个有旧选择。** 已用原选择集 NLL 和两个调参种子，在 A1 的 262,144 token 上复算 6 选 2，在 A2 的 1,048,576 token 上复算 2 选 1，并核验两条通往 A3 的接续链。24 个 FC/Lattice 的历史迁移只覆盖 r01 到 r04 的 quad 选方和活动量。18 个 Exchange/Settle 有绑定旧源码的接续凭据。下一步在 `6e14a6cc` 上按配置检查受影响的算子、原批次和竞争配方。若支持集、梯度或排序依据改变，用原两种子和等量曝光重做受影响的比较。[历史迁移范围](evidence/pilot-r3-stage-a-transfer.json) · [选方规则](../pilotlab/gate3.py)
   - **32 个尚无最终选择。** 最远到容量阶段的 14 个中，4 个缺完整容量记录，10 个下一步是初始化诊断。其余 18 个的首个缺口为 A1 比较 5 个、A2 比较 7 个、A3 确认 6 个。已对这些阶段的 95 个已派请求核过候选检查点的完整 SHA-256，另 5 个 A1 请求尚未派出。优先复核 Settle/K/singleton7 的实际分配梯度溢出，依据各 worker 结果区分数值故障与同组中止。用原模型、优化器、私有状态、RNG 和数据游标核对恢复，再对真实后继批次执行首个更新。逐项登记旧选方可复用性、轨迹能否接回和需要重评的范围。[首个缺口与检查点](evidence/pilot-r3-deferred-74-review.json) · [接续核验](../pilotlab/continuation3.py)
   - 此次审阅止于读取证据。后续派工前核对总账和原截止，同一笔继承预记只计一次。4 个 Exchange 主体在 r01 另有合计 1,023,328 个评估 token 预记，逐模型预算须包括这段曝光。[预算来源和逐项数值](evidence/pilot-r3-deferred-74-review.json)
3. 待全部 344 个主体与匹配对照有逐项结论后，核验完整 A 门禁、选参证据、匹配质量与正式运行资源。
4. 按实际训练负载复核多分支路由梯度、独立零先验与弱正权重。继续保留 KV 未写入槽位及 quad 常数下界的原始对照和容差。[路由验收](evidence/pilot-r3-r14-foundation-route-acceptance.json) · [KV 验收](evidence/pilot-r3-r10-foundation-kv-acceptance.json) · [quad 验收](evidence/pilot-r3-r11-foundation-quad-reception-acceptance.json)
