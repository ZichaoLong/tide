# 历史检查点退役：2026-11-12

用户在只读调查后批准扩大清理：目标约75–80GiB，保护当前v9、关键历史模型和配对对照、失败复现材料；
保留配置、数据、结果、曲线、日志及来源。最新授权允许退役历史筛选权重，覆盖旧文档中笼统的全部产物保留表述。
本页当前是固定执行表，实际执行结果以随后附加的终态凭据为准。

[逐文件固定清单](../evidence/checkpoint-retirement-plan-20261112.json)含文件大小、inode/device、mtime、
终态证据hash、39个保留checkpoint及理由、活动核对和72,901个不改动历史文件的元数据指纹。
只允许清单内路径；不以通配符执行删除，不跟随symlink。

| 动作 | 文件数 | 预计释放GiB |
| --- | ---: | ---: |
| 退役已完成v6–v8的非保留训练检查点 | 1,116 | 66.630 |
| 保留模型的终点重复副本改为硬链接，路径/字节保持 | 76 | 7.531 |
| 退役旧GEF benchmark/恢复状态文件 | 14 | 0.596 |
| 合计 | 1,206 | 74.757 |

选择后的精确数约74.76GiB，接近原约75–80GiB目标；不为凑整数而删除关键对照或失败见证。
不是额外在此前49.84GiB之上释放74.76GiB；本清单已经包括相应重复副本。

## 保留内容与解释边界

- 整个v9批次、运行/条件延伸的父checkpoint、数据、源码、配置和环境保护。
- v6所有非completed轨迹完整保留，包括三项下溢失败、9个恢复probe、failure-boundary/input和HB/FC负面规模轨迹。
  另保留FC/HB的anchor、proportional、pooled-attention终点，5M/10M/19M Transformer终点及10M的共同262K对照。
- v7十条主轨迹的4M终点全部保留：FC16、HB40/88、TF12/28各两seed。
  其余筛选及中间checkpoint退役，包括部分历史续训父checkpoint；原谱系、hash和结果仍在。
- v8保留19个终点：FC/HB anchor、temperature1，FC direct-attention、HB pooled-attention、FC SSD mem-off负面消融，
  调参后TF6/12，以上均两seed；另保留唯一具名r2数值修正。原26个执行失败全部保留。
- 旧A/B、小批接入和原始性能trace、所有历史配置/指标/结果/日志、模型别名及公开权重、源码/环境均不清理。
  GEF14个旧benchmark文件对应10项completed与恢复passed记录；报告保留，原状态字节退役。
- Lens未触及；图的结论、原严格阈值和失败不改写。当前图没有直接指向本清单删除文件的链接。

保留checkpoint仍含原model/optimizer/RNG等内容；本次不转换精度、不抽取权重、不重新运行恢复实验。
同一保留终点的旧路径硬链接共享一个实体，只允许作为后继新作业的只读输入，不就地编辑。
退役权重对应的科学观测继续有效，但不能再立即加载旧模型重评/诊断/续训；重新获得需新身份重训。
过去的恢复通过是历史事实，不表示已退役字节现在仍可用。三步恢复的原资格不升级为长程独立重放。

## 执行方式

执行器`scripts/retire_archived_checkpoints.py`，固定manifest输入；先核对终态、PID/start-tick、
活动v9输入无交集以及可读/proc文件占用，再以2线程计算每个退役文件和保留checkpoint的完整SHA256。
全部hash原子持久化之后才开始修改；相同副本逐个原子替换为硬链接，其余逐文件unlink并fsync目录。
末尾核对删除/链接状态、39个保留checkpoint的SHA256和72,901个原保留文件的元数据。
进程fd检查存在权限限制，记录其数量；不将未读权限声称成完整可见。

后台作业`retire-archives-20261112`：CPU2线程、2GiB RAM、3600秒上限，无NPU，background.slice。
冻结新源码于`.local/source-housekeeping-20261112`，不改活动训练源码。
输入为本清单，输出`docs/evidence/checkpoint-retirement-execution-20261112.json`；
状态/日志位于`artifacts/jobs/retire-archives-20261112/`。

```bash
python scripts/job.py status artifacts/jobs/retire-archives-20261112
python scripts/job.py logs artifacts/jobs/retire-archives-20261112
python scripts/job.py stop artifacts/jobs/retire-archives-20261112
```

执行器拒绝覆盖已有执行凭据，不可直接重跑删除命令。
如中断，以prepared清单中的hash、逐项state及当前路径/inode先只读对账，再为尚未处理项建立新的固定清单和作业；
原凭据不得覆盖。文件退役没有撤销恢复入口，硬链接去重保留原内容。

清理前目录分配量：tide工作目录150,150,791,168字节，GEF2,376,921,088字节；
账户quota353,060,044KiB（336.704GiB），800GiB硬限。活动作业仍可正常写入，因此终态会分别报告
逐文件释放量、目录变化和账户配额，不把其他并行写入混成清理误差。
