# direct-v8：查看、停止与恢复

2026-11-12：历史筛选权重已按授权选择性退役，19个代表终点保留，全部结果/日志/失败保留。
恢复或重新评价前先查[检查点可用性与退役清单](checkpoint-retirement-20261112.md)；以下原操作方法不承诺每项旧权重仍在。

2026-11-11终态：本批40个服务全部退出且无后代，没有活动/排队任务。controller实际选19个局部复核，18完成、1失败；单项r2完成。
以下保留固定来源及原作业操作方法；当前查看不会重启训练。结果入口为[终态交付](direct-v8-results.md)。

[研究表](direct-coverage-v8.md) · [实现与数值边界](direct-v8-implementation.md) ·
[Transformer结果](direct-v8-transformer-results.md) · [逐配置/任务/Lens观察](../evidence/direct-v8-observations.json)。
项目文件为权威；报告/研究图是观察时点快照。

## 固定身份与范围

TF筛选/确认：2702d91，`.local/source-v8-tf`；局部公式检查、成本与两anchor：778f9e9，`.local/source-v8-cal`；
其余180格及条件seed22：5db28ae，`.local/source-v8-main`。最后一个快照只新增正支持监视与有限重复调度，
局部数学函数和角色初始化与anchor相同；各阶段仍按实际源码独立记录，不伪称同一source hash。
一个具名quad残差修正复跑使用cd85637、`.local/source-v8-repair`，服务`v8-repair-w0`，
run前缀`tl-v8-repair-`。它等待同一10卡槽位，以原seed/data/1M目标从头运行；不覆盖失败，不修改活动覆盖源码。
原确认controller不自动用新版本结果替换原失败，r2结果需单独审阅。
数学0770593；公开规划74e7bbd；实际安装锁见`dependencies/positive-floor-v1/`及各stage identity。
直接训练只依赖Torch模块，没有调用基座执行器或采用基座工作树的新优化。

182格均通过实际d128、CPU B2/L3前向/梯度/因果检查；另8项独立数学检查通过。
正式每格还需本设备CPU对比、支持/机制检查及新进程三步精确恢复；这些状态与学习分开。
13个NPU成本case完成，CPU1的12个成本case完成；NPU B32各族完整更新约0.13–0.40秒。
HB40 anchor B16/32/64分别约0.261/0.296/0.394秒；CPU1 B16/32约4.99/9.83秒。
B64吞吐更高，本批为保留共同短程更新次数和已校准TF比较采用global/physical B32，10卡独立任务并行。
HB40 B32约5GiB峰值分配，B64约10GiB；后续case的同进程累计峰值不能当各case独立峰值。

总时限截止2026-11-13 17:43:58北京时间（48h），240 NPU卡时、768 CPU核时、128GiB新增磁盘；
单局部case≤8h、TF≤1h。主worker每人18个候选；预计周转以实际进度为准，不等待时限耗尽。
`v8-budget`监督全批；有限重复controller只在182格终态后，从每拓扑10个桶各选至多1项seed22，共≤20。
sending_control并入sending；anchor独立桶。只选择完整有限验证且接入/恢复通过的项；失败仍保留。
新配置归档至`configs/studies/language/direct-v8/confirmation/`，控制器记录选择及拒绝项，
不会自动开启真实稀疏或规模主线，也不会自动生成科学支持。

## 查看

```sh
python scripts/review_direct_v8.py
python scripts/job.py status artifacts/jobs/v8-cover-w0
python scripts/job.py logs artifacts/jobs/v8-cover-w0 --lines 40
```

覆盖worker为`v8-cover-w0`至`v8-cover-w9`；已完成首项为`v8-anchor-w0/1`。
每case目录`runs/v8-cover-<config-id>/`下有check、recovery、train.log；
train/下有progress/result、初始/阶段/最终逐文档validation、inference、metrics和checkpoint。
`runs/v8-cover-submission/w*.json`给每个worker的确切顺序和固定配置；仅外层退出码不足以判断学习完成。
确认控制器为`v8-confirmation-controller`，提交后出现`v8-confirm-w*`；等待状态不等于已经开始复核。

Lens保持原认证实例与访问策略，私有入口/SSH转发写在本机`.local/LENS.md`，不入公开仓库。
在项目`tide-learning-lab`中按组或run前缀查看：

| 组 | run前缀 | 内容 |
| --- | --- | --- |
| TL-DIRECT-TF-v8 | `tl-v8-tf-` | 14格Transformer配方校准 |
| TL-DIRECT-CONFIRM-v8 | `tl-v8-tf-confirm-` | 4个Transformer seed22复核 |
| TL-DIRECT-COVER-v8 | `tl-v8-anchor-`、`tl-v8-cover-`、`tl-v8-repair-` | FC/HB的182格局部候选与单项r2 |
| TL-DIRECT-COVER-v8 | `tl-v8-cal-` | 合成完整步成本，无质量结论 |
| TL-DIRECT-CONFIRM-v8 | `tl-v8-confirm-` | 条件提交的至多20个局部seed22复核 |

显示名带`-projection-...`后缀；确切destination_name/run_id由每项`projection.json`及观察表关联。
`work/full`是逻辑选择，`work/physical_full_rows`才是本实现实际Full行数。
其他观测有train/validation NLL、tokens、梯度范数、发送强度、区域活动、positive floor、RSS/HBM及完整步时间。

## 停止与恢复

统一停止（预算服务会停止本批所有仍活动服务，包含确认控制器）：

```sh
python scripts/job.py stop artifacts/jobs/v8-budget
python scripts/review_direct_v8.py
```

单独停止一个worker用`job.py stop artifacts/jobs/v8-cover-wN`；若不希望失败后controller继续选择剩余候选，
先停止`v8-confirmation-controller`。任何停止都需核对MainPID、cgroup与记录后代清空；不要按python进程名清场。
不修改活动作业读取的源码、配置、依赖或输入。

恢复使用**新的**job、worker plan、case output、run ID，仍使用原冻结源码、原config SHA、data lock、安装、backend与线程数。
复用该case的check/result.json与recovery.json，在train_arguments中加`--resume <原train/latest.pt>`；
checkpoint与两份资格记录都加入job.py的`--input`。复制原job.json的环境、资源策略及启动argv，保持原共享deadline/总预算，
目标仍1M。当前仅保留清单内的latest及相同终点的硬链接别名；其他旧权重已退役。三步恢复通过不认证整条长轨迹独立重放。
预算用完或本批终态后不自动续到更多曝光；再次查看报告无需重跑。
