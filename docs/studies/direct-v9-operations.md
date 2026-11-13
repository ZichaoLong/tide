# 当前查看：v9训练已全部终态

2026-11-13：58条主轨迹和12条延伸完成，当前无本批活动服务或后代。
只读全终点审阅：`python scripts/review_direct_v9_results.py`；`review_v9_progress.py`仅含主58条。
[总报告](direct-v9-results.md)与[逐run/Lens/父SHA](../evidence/direct-v9-results.json)为当前交接。
延伸精确前缀为`tl-v9-recover-extend-`，分别属于TL-SCALE-v9（HB88/TF256M）和TL-ACTIVITY-v9（R64 64M）。
主轨迹含`tl-v9-main-`及`tl-v9-recover-main-`，都已终态；不能由旧中断projection的recording推断进程存活。
原48h截止没有重置，未来追加训练需要新的有限表；以下保留实际启动/停止/恢复方法。

# v9操作与运行身份

2026-11-12故障恢复补充：原controller保留failed（quota软告警误判），新`v9-recovery-controller`继续同批。
`python scripts/review_v9_progress.py`按明确恢复谱系统计58条主轨迹；新run前缀`tl-v9-recover-main-`，
条件延伸为`tl-v9-recover-extend-`，原study组不变。源码、预算及原失败见[恢复记录](direct-v9-recovery-20261112.md)。
四条旧中断Lens已通过独立收尾收据核对，旧projection文件未改写；见[收尾证据](../evidence/direct-v9-recording-finalization-20261112.json)。

本批表：[语言规模、活动、入口树](language-scale-activity-v9.md)。共同48h起止见
`configs/studies/language/direct-v9/budget.json`；无累计算时上限，最多12个单卡worker。
每个作业的`artifacts/jobs/v9-*/job.json`固定源码commit、输入SHA、环境和内存/时限。
不能修改正在运行的`.local/source-v9-*`或`runs/v9-campaign/*/configs`。

项目`tide-learning-lab`；Lens沿用已有认证实例，私有入口及SSH方式见`.local/LENS.md`。

|组|run前缀|用途|
|---|---|---|
|TL-CALIBRATION-v9|tl-v9-recording-admission|小型记录接入；非训练|
|TL-CALIBRATION-v9|tl-v9-batch-|真实形状的同步完整更新计时；非质量结论|
|TL-CALIBRATION-v9|tl-v9-screen-|36 TF、12 HB配方和24活动终点前缀|
|TL-SCALE-v9|tl-v9-main-hb / fc / tf|数据、容量、依赖深度及浅宽、TF对照|
|TL-ACTIVITY-v9|tl-v9-main-r|原入口、接收/Full比例与语言质量|
|TL-INLET-v9|tl-v9-main-tree-|入口树与原入口的同主体对照|
|相应原组|tl-v9-extend-|条件继承优化器的延伸；非新种子|

精确Lens投影名以各`train/projection.json`为准，运行中recording，终态后verified/readback_matches。
所有run均有对应配置与source/data/install身份；服务退出不产生科学支持边。

只读检查：

```bash
python scripts/review_direct_v9.py
python scripts/job.py status artifacts/jobs/v9-controller
python scripts/job.py logs artifacts/jobs/v9-controller --lines 40
python scripts/job.py logs artifacts/jobs/v9-screen-w0 --follow
```

controller日志报告提交/阶段，训练细节在`runs/v9-<phase>-<id>/train.log`及`train/progress.json`。
共享卡只在进入worker时选择，实际物理卡记在job的queue-state.json；源码使用逻辑npu:0。

停止全批时一次性停止controller和所有v9服务；这些服务有各自的cgroup，不能仅依赖控制器在30秒停止宽限内逐个清理：

```bash
systemctl --user stop 'tide-lab-v9-*.service'
python scripts/review_direct_v9.py
```

停止单个worker用`python scripts/job.py stop artifacts/jobs/v9-screen-w0`等精确目录；其中余下case会成为未启动。停止后检查MainPID=0及后代清空。
每个服务亦受共同截止限制；data独立服务另有2h限时。

恢复必须用新job/output、原固定源码与原配置/data/环境，并传
`campaign_stage.py train --resume <原train/latest.pt> --admission <原check/result.json> --recovery <原recovery.json>`。
继续使用原deadline，不能将重启当作重置预算。提供完整`--config --data --output --run-id --recording-group --deadline-epoch`；
使用同一job.py启动器。微型三步新进程恢复单列；实际长轨迹重启未经实际执行仍未检查。
里程碑`tokens-*.pt`为权重留档，**不含optimizer，不承诺精确续训**；滚动latest才是恢复入口。
校准前缀完成后只留里程碑权重、曲线和失败，退役其临时Adam状态，不做自动延伸。

延伸所需总target在原配置内预留；初始主终点通过`--until-tokens`控制。
LR按初始16M/64M horizon余弦衰减，延伸沿用其10%floor；不会重新warmup或悄悄改变优化器。
中档六条完整64M且有限时才成组延伸256M；活动只选一个原入口形状三方两种子，
低活跃两种子都入目标带才延伸64M。未知效果不得作为普遍scaling结论。

新数据首次使用错误私有解释器缺tokenizers、随后直连超时的两次失败在v9-data/v9-data-r2保留；
v9-data-r3使用原有数据环境和已授权netpath单次重试。训练环境没有安装Lens，
使用已验收的独立JSONL recorder；不修改共享环境或服务策略。

当前活动指标按实际计算的输入位置统计，包括文档尾部padding位置；有效目标另计，不能把global64×L256恒当有效token数。共同数据流与分组保持一致。主checkpoint约39.38GiB、筛选权重约7.94GiB的静态估算不含临时峰值/元数据；控制器每120秒检查真实占用和账号硬余量。
