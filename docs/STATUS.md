# 当前交接 — 接口、性能与架构统一实施（2026-11-10）

用户已授权实施与分阶段提交/推送；最新要求：完成当前这一轮后停下汇报、对齐，再授权后继续。
本轮边界：收尾47704c9证据，集成通用Describe/循环用途查询及resident预算诊断，固定源码定向资格和有限代表回归。
single-bank resident投影及进一步新优化留待下一轮授权，不在本轮启动。数学固定0770593；当前分支graph-execution-foundation。
用户未跟踪TL-REQ-003/004文档原样保留且不提交；LH和下游安装/锁不改。共享NPU本阶段无授权。

## 已固定与已验证

06b242e交付仓库skills、分主题经验、审阅与census、公共边界及共享KV bank；1528df8修复native VJP ADL错误。
证据docs/evidence/interface-snapshot-bank-20261110.{md,json}，CPU/NPU三客户端小训练/精确恢复已分列。
47704c9交付只读发布/字段unbind、Describe与Mem能力分离、live-fiber容量及返回搬运去重。
证据docs/evidence/interface-batch-publication-20261110.{md,json}：CPU局部501通过/13xfail；图308通过/1skip/4严格失败。
4失败均FP32 affine dag/pdg Python/native首evidence_weight1.3504e-6；显式诊断完成其余十项比较/优化器/精确恢复，仍记xfail。
同FP32材料提升FP64通过；不改全局容差。NPU配置122通过/3xfail（含CPU辅助），owned-return7通过；
Python/native安装消费者与SDK CPU/NPU完整训练/新进程恢复通过，SDK3CTests与单独return-check通过。
FC16完整训练CPU1/CPU16/NPU12.4468/12.9587/17.2122秒；TTFT CPU3.1589/NPU5.0984秒。
训练下降而NPU推理反升；完整尺度数值/学习未独立验收。小图union11317、SelectBackward640，不冒称全B×L已无碎片。

## 作业与下一步

gef-interface-bank、gef-interface-bankfix、gef-interface-batch保持原failed；各自后续独立作业不改写原失败。
gef-interface-bankfix-rest、gef-interface-batch-rest均passed/MainPID0；没有活动长作业。
最后rest source47704c9于2026-11-10T11:51:32Z完成；状态/日志artifacts/jobs/interface-batch-rest/。
冻结source-batch、batch*.sh、env-batch与build-batch/SDK保持不变，不重跑其已通过检查。
当前main有第三增量源码/测试及resident诊断未提交；代理补SSD完整projection/delta_bias形状验证、Describe整链检查和construction-only入口范围。
完成审阅/短检查后分别提交实现，冻结新source并新建env/build，通过有界background.slice执行native/SDK/空闲NPU资格。
有限代表与无梯度/构图/反向成本使用最终固定源码；构建和正式计时不重叠。每个失败/跳过分别记录。
本轮结束更新STATUS/ROADMAP、证据和经验，提交并授权推送，然后停下与用户对齐。

接续：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
历史提交定位与迁移见docs/history-migration.md；当前限制唯一清单为ROADMAP。
