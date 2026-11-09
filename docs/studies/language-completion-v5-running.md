2026-11-09终态：五项训练/续训、全部评价及必要复评已完成；最后HB40 dense于17:16:18结束，所有本批cgroup/后代清空，无排队或自动后继。当前结论见[终态总报告](language-completion-v5-results.md)及[固定终态证据](../evidence/language-completion-terminal.json)。下面保留实际提交、前提、查看和恢复契约；不要重启已完成服务来查看结果。

# v5补齐：作业、记录与恢复

本页对应[已授权补齐表](language-depth-completion-v5.md)，实时状态以项目记录为准。[终态证据](../evidence/language-completion-terminal.json)保留生命周期、检查、实际指标、预算、Lens身份和固定输入核验；[中途交接快照](../evidence/language-completion-handoff.json)保留原观察时点。更新审阅不会修改运行目录，也不自动生成科学支持。

源码固定为`0e409967641d6963e82942e3d13223900b0afba6`，只读快照`.local/source-language-completion-v5`，树hash`3c591a70cb389f053fdd2f32fde9a0df9dfd823a91336570356302f29cc500e2`。全部配置在`configs/studies/language/completion-v5/`。数学0770593、foundation74e7bbd；环境`.venv-positive-floor-v1`，实际安装锁`dependencies/positive-floor-v1/installation.lock.json`，250个文件已复核。旧安装与旧冻结输入保持。

所有正式训练为CPU单线程、FP32、seed11、独立样本batch8/microbatch8/accumulation1、L32、512步学习率日程。最多6个同时计算worker，各24GiB上限；等待依赖不计已训练。沿用原24累计计算worker小时和北京时间2026-11-09 22:49:13截止。每个候选均有独立短恢复/机制检查、训练、test、推理和原定义的context-reset；未完成阶段不能标通过。

| 作业ID | 配置文件 | 计算与目的 | 本次上限 |
| --- | --- | --- | --- |
| `lm-complete-fc16-sparse-v5` | `fc16-sparse.json` | FC16/width32，4区域串联；quad从零至128步 | 2h |
| `lm-complete-hb16-dense-v5` | `hb16-dense.json` | HB16/width32，4区域/3依赖层级；已验94步材料续至128 | 1.5h |
| `lm-complete-wide-dense-v5` | `wide-dense.json` | FC4/width64；最后完整32步材料续至128 | 1.5h |
| `lm-complete-hb40-dense-v5` | `hb40-dense.json` | 条件HB40/width32，10区域/5依赖层级；从零64步 | 3h |
| `lm-complete-hb40-sparse-v5` | `hb40-sparse.json` | 同一HB40的独立quad64步 | 3h |
| `lm-complete-references-v5` | `references.json` | 旧FC4、FC16、HB16固定权重的必要复评；非新训练 | 0.5h |
| `lm-complete-wide-cal-{cpu,npu}-v5` | `wide-cal-{cpu,npu}.json` | 同B8短校准；均已完成，NPU物理1 | 各10min |
| `lm-completion-budget-v5` | `budget.json` | 统一核算旧v5、补正验收及本次所有尝试；达到上限停止 | 同共同截止 |

CPU/NPU浅宽校准分别5.466/14.379秒每更新，排除进程/模型启动，包含前后向及optimizer；各2个计时更新、1个预热，仅用于当前资源选择，不是通用后端benchmark。CPU1约2.63倍快，因此浅宽也用CPU。已有深层CPU/NPU与batch校准复用，B8维持原采样、曝光和更新口径；不因有空卡追加无收益迁移。

HB40原依赖等待器先等待新HB16 dense整链完成及原HB16 quad完成，再核对两者相对各自从零初始validation NLL下降至少0.1；前提于16:06满足，两候选现已完成。新HB16不是相对94步起点判断“是否能学习”。HB40是新初始化；前驱只是执行前提，不是权重祖先或科学支持。原非有限/离散/机制/恢复失败及预算停止条件保持，不绕过前提。

HB16和浅宽均为显式应用状态转换：核对父文件、manifest、模型/训练配方、数据、采样和日程，保留参数、optimizer、RNG、已消费token及步数，另存`transfer.json`。跨补丁、浅宽跨NPU→CPU不称为旧轨迹精确恢复。新栈B2/L3三步新进程恢复另行通过。HB16原87步失败、浅宽原56步后失败及已消费重算成本完整保留。

## Lens与原始记录

项目`tide-learning-lab`，搜索`lm-complete-`。原生Group未填写，分类用配置`study_id`与下列前缀。实际显示名附带`-projection-…`，精确ID见终态证据的`lens`及每阶段`projection.json`。私有入口和SSH方法只在本机`.local/LENS.md`；服务、认证与访问策略不变。

| 研究组 | 学习曲线名称（投影前） |
| --- | --- |
| TL-A-DEPTH | `tl-lm-complete-fc16-sparse-v5-train-tide-sparse-s11` |
| TL-A-TOPO | `tl-lm-complete-hb16-dense-v5-train-tide-dense-s11` |
| TL-A-CAPACITY | `tl-lm-complete-wide-dense-v5-train-tide-dense-s11` |
| TL-A-NEST | `tl-lm-complete-hb40-{dense,sparse}-v5-train-tide-{dense,sparse}-s11` |

训练逐步记录loss、梯度范数、步时、实际Full/候选/消息、补值行与坐标；区域活动每32步。补值计数包括autograd replay，不当作唯一节点数。`validation/nll`是验证曲线；`test-…`为终评；`inference-…`为训练后推理/严格检查；`context-reset-…`保留旧位置与状态同时重置的定义。技术接入、恢复及校准仅存本地，避免重复技术投影。

27份完成投影均已核对身份、事件数和`verified/readback_matches`。此前[只读前缀读回](../evidence/language-completion-lens-live.json)仅保留运行时的有限观察，不冒称终态exactly-once。已接受的FC16普通/重置评价直接复用`tl-support-v1-fc`，不重复消耗计算。

## 查询、停止与恢复

在仓库根目录查询；将`JOB`替换为上表完整ID：

```bash
python scripts/job.py status artifacts/jobs/JOB
tail -n 40 artifacts/jobs/JOB/job.log
cat runs/JOB/stage.json
cat runs/lm-completion-budget-v5/state.json
python scripts/review_language_completion.py
```

具体阶段日志在`runs/JOB/<stage>.log`，学习原始记录在`runs/JOB/train-…/metrics.jsonl`。HB40等待状态在`runs/JOB-wait/state.json`。外层service active可能只在等待；具体训练步数以完整metrics/checkpoint记录为准。

停止单项：

```bash
python scripts/job.py stop artifacts/jobs/JOB
python scripts/job.py status artifacts/jobs/JOB
```

停止整批时先停止两个HB40等待器，再停止主训练/复评，最后停止`lm-completion-budget-v5`；监督器停止不会自动代替停止其被监督作业。`job.py stop`检查MainPID、实际cgroup和已知后代全部清空；不能只看外层退出码。

恢复必须另建job/output，先读`checkpoint.json`核对文件hash及实际步数，再用**相同固定源码、环境、配置、CPU1和任务参数**调用`language_stage.py train ... --resume /绝对路径/checkpoint.pt`。保持原`--steps 128 --schedule-steps 512 --batch 8 --length 32 --seed 11`，HB40为`--steps 64`，深度用原配置值；有效source/config/dependency身份必须完全匹配。新恢复ID必须加入一个覆盖旧费用的预算监督，仍受原剩余额度与共同截止约束，不能直接重启原service或覆盖原目录。严格同栈恢复会拒绝身份不符；再跨后端/补丁需单独显式转换，不伪装精确恢复。

已执行`python scripts/review_language_completion.py --terminal --output docs/evidence/language-completion-terminal.json`，它先要求全部cgroup/后代清空，再输出固定证据和逐文档配对统计。该命令不填写研究图科学支持；逐组一页结论和六轴审阅现已完成，见[补齐总审阅](language-completion-v5-results.md)。后续选项尚未启动，待用户验收本批后选择。
