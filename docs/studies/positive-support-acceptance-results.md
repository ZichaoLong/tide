# TL-REQ-003下游验收：正支持补正

**验收结论：接受用于当前单设备Python eager/KV/FP32的有限稠密比较和训练探索。** CPU真实HB16失败边界、FC16完整干预、同模式精确恢复，以及CPU/NPU小型接入均完成；严格数值等价仍未通过。两项作业全部终态，cgroup和已知后代清空，没有新后继训练。

问题是`positive-floor-v1`能否作为可靠的FP32稠密比较/诊断实现，修复正支持下溢造成的执行中断。它不负责证明Shannon优于稀疏分配，也不把8次更新当作语言质量结论。[预先固定验收表](positive-support-acceptance.md)、[机器证据](../evidence/positive-support-acceptance-v1.json)、[四种方案及选择依据](shannon-numerical-options.md)。

## 已完成的核心检查

| 检查 | 实际结果 | 解释范围 |
| --- | --- | --- |
| 新私有安装 | 固定74e7bbd，249份Python源码与归档匹配，实际安装文件hash与基座交付相同 | 新wheel自行构建，hash另记；旧85d161d环境不变 |
| 基座定向 | CPU30项、NPU流程30项通过，零跳过；NPU其中13项实际设备目标、17项CPU/配置辅助 | 包含实际行/Decimal独立导数、共享根、KV、AdamW、完整下一窗和新进程恢复 |
| HB16/FC16任务接入 | CPU/NPU各两拓扑，B2/L3、原width32；离散支持、因果、共享根、各区域梯度、唯一optimizer、分块和streaming通过 | 保留同一1e-6/1e-5严格阈值及有限失败，不称完整严格等价 |
| HB16真实第87步 | 新安装strict在原86步材料上重现失败；新模式完成87–94步 | 原B8/L32、seed11、cat/ind、tau.25、AdamW和512步日程不变 |
| 同新模式恢复 | 第90步保存后，另一个新进程继续91–94；模型、optimizer槽、RNG、数据进度、所有最终参数梯度及逐步loss/工作/补值逐位一致 | atol=rtol=0；应用按原配方每序列重置图状态，完整图跨窗恢复由定向套件另验 |
| FC16真实干预 | 原128步权重，122块/3878目标全部完成；普通NLL6.864746359，context-reset6.895437009 | 普通相对旧实现仅+4.328e-8；原会中断的干预补8个坐标后完整执行，重置上下文代价+0.030690650 |

HB16八次更新的补值坐标数依次`[1,3,0,0,0,0,0,0]`；观测包含实际分配调用和autograd replay，不冒称全局唯一节点数。每步4096 candidates/4096 Full、8192条内部消息，保持稠密实际计算。loss范围6.46105–7.01366，裁剪前梯度范数1.33306–2.21486，参数和梯度均有限。固定validation8块NLL从6.818646到6.835233（+0.016587）；这不是改善证据，也不能把更新后差别归因于补值。

FC16普通评价没有补值，说明切换整套稳定运算后的已有正常质量基本保持；干预评价发生8次补值并完成。这个单checkpoint结果支持当前使用范围，不能单独证明所有补值都无影响。原失败记录仍是失败，新增结果属于显式策略转换；FC16历史整链未自动改标completed。

验收六轴：执行通过；严格数值有限失败；所测离散/因果/共享梯度机制通过；同模式恢复通过；短程学习稳定观察完成但质量改善未证实；资源/工作量已记录，无吞吐加速主张。至此显式采用74e7bbd及新私有安装，默认`strict`仍保留，未来稠密配置需明确填写`support_policy`。B/Qwen完整模型尚未迁移或复验。

## 严格数值、反证与未检查范围

任务完整状态/记录/VJP比较有有限超差：CPU HB16/FC16为224/1028个元素，NPU为100/1185；最大绝对差分别9.54e-6/4.01e-5/1.00e-5/3.81e-5，整体相对L2约4.33e-7–1.02e-6。离散不匹配和非有限均为0。完整原因未分解，不能把它们一概解释成已证明无害的舍入；此前85d161d的严格失败也不覆盖。独立实际行、普通尺度导数、完整小图高精度参考以及因果/共享/恢复证据支持有限学习探索，但不支持严格实现等价声明。

新策略仍拒绝全行下溢造成的显著相对扰动和不可表示的公开质量伴随；定向反例已实测。多个被补尾部的比例不保留。任意损失根、任意长度、FP16、native/resident、其他记忆及多卡不在这次范围；没有将这些边界扩大成当前学习实验的前置条件。

同一架构的两条随机训练轨迹逐渐不同，不足以认定架构不稳定。若合理数值策略使独立任务质量或主比较结论系统性翻转，才需进一步区分实现、优化与结构敏感性。本次没有这种证据，也没有证明长期完全不敏感。

## 固定身份、查看与复现

- 数学0770593838a37c790a568226930567ed60f99b83；基座74e7bbd62341d2ccd382194a2c52ad4fea033665。
- 实验执行源码fc781722b278a0180f46e5200bef63a34d613a82；只读`.local/source-positive-support-v1`，树hash`c059a058a911ccb17396313d0aee1998575f940842c63f50410b2c5f4b68a31b`。
- 新wheel SHA256 `210f6c1f92d365b38def9718cbca0726abcb5da65bda14a8819a585add7e16f9`；安装文件集合hash `7dbfde4bbbd375e17e320b75a5796164cafdb33c5cfbc07de9fc02d020cd059a`，与上游交付相同。构建/包关系见`artifacts/setup/positive-floor-v1/build-identity.json`。
- 配置[positive-support-v1.json](../../configs/checks/positive-support-v1.json)，候选安装/来源/环境锁在[dependencies/positive-floor-v1](../../dependencies/positive-floor-v1/foundation.lock.json)。旧应用材料从原`.venv`独立解码；迁入参数/optimizer/RNG/进度是显式新数值运行，不冒称跨策略精确恢复。
- CPU1执行真实边界；NPU physical1→logical npu:0执行小型定向与任务检查，未在NPU重跑长HB16。单进程ATen/BLAS1线程，MemoryMax8GiB，background.slice；未更改共享环境和基座工作树。

本次CPU/NPU计算worker分别488.185/91.679秒，合计579.864秒；开发接入检查另59.91秒，共639.774秒（0.1777计算小时），低于3600秒上限。v5计入这些工作后约6.5475/24计算小时；原共同截止22:49:13不延长。已固定原输入/运行源码重新核对，250份安装文件全部匹配。开发检查最初有3项因新环境误选旧锁被身份守卫拒绝；显式选新锁后3项通过，另8项已通过，没有把首次错误标成数值通过。

作业记录为`artifacts/jobs/support-v1-{cpu,npu}/job.json`，内含完整命令/固定输入/生命周期；结果在`runs/support-v1-{cpu,npu}/`。状态命令：

```sh
python scripts/job.py status artifacts/jobs/support-v1-cpu
python scripts/job.py status artifacts/jobs/support-v1-npu
python scripts/job.py logs artifacts/jobs/support-v1-cpu --lines 30
# 子阶段标准输出位于runs/support-v1-cpu/PHASE-wrapper/，指标在PHASE/metrics.jsonl
# 若有新提交仍在运行，停止并验证全部后代：
python scripts/job.py stop artifacts/jobs/support-v1-cpu
```

复现采用新的输出/作业身份，加载相同公共基线并设置`TIDE_LAB_VENV`为`.venv-positive-floor-v1`，`TIDE_LAB_FOUNDATION_LOCK`为冻结源码内`dependencies/positive-floor-v1/foundation.lock.json`；运行固定`accept_positive_support_chain.py`，完整argv在job.json。单独复核已保存的新模式90→94恢复，可调用同一固定`accept_positive_support.py resume`，指定原config、`--material runs/support-v1-cpu/export`、`--prior runs/support-v1-cpu/hb`和全新`--output`，通过新的有界后台作业执行。不能对旧strict checkpoint使用普通exact resume冒充转换。

原链配置有共同绝对截止，过期后不会自动重置预算。未来复现实验须另存新的生命周期预算，保留原科学配置与转换来源；不原地编辑只读快照。新源码的安装身份读取并核对所选锁，避免把新wheel写成旧85d161d；旧`.venv`仅配原冻结源码/锁使用。

Lens项目`tide-learning-lab`：`tl-support-v1-hb`为8步学习/验证，`tl-support-v1-resume`为恢复一致性重放，`tl-support-v1-fc`为同权重普通/上下文重置评价。后两者不是新独立训练重复。私有入口继续在`.local/LENS.md`；技术定向测试只存项目证据，不增加无学习信息的Lens投影。各投影readback在机器证据单列。

三份Lens投影均为verified且readback_matches=true，事件数分别10/4/2。项目JSON/原始状态为权威，查看投影不改变实验判定。

| 源运行ID | Lens实际显示名称 |
| --- | --- |
| tl-support-v1-hb | tl-support-v1-hb-projection-25eff632a569 |
| tl-support-v1-resume | tl-support-v1-resume-projection-f3a0f24adef9 |
| tl-support-v1-fc | tl-support-v1-fc-projection-38a732488d78 |

本次不自动恢复HB40，不添加正背景或bool/log新候选。后续若只为继续语言/稀疏研究，优先使用已验补正模式完成必要稠密对照；只有实际发现效果/梯度域问题才扩大表示工程。
