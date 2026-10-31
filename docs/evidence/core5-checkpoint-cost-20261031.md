# 大模型 checkpoint 成本与内存文件系统（2026-10-31）

这是独立的 checkpoint 成本观察，不计入合格吞吐。固定 Python 实现
`dba2a5ed45d79f17c93a72f5285c36b60b29bb9d`；CUDA core/resident 沿用 `de999c2`，
standalone 沿用 `eca070f` 的实际二进制身份，没有在新提交重新构建。
[完整小记录](core5-checkpoint-cost-20261031.json.gz)可由公共 `tidegraph results` 查询；
[资格和终态](core5-checkpoint-cost-checks-20261031.json)保留来源、wheel、安装锁和独立检查范围。

## 同材料磁盘与 tmpfs 基线

跟踪配置为 `configs/checkpoint-cost/cuda-disk.json` 和 `cuda-tmpfs.json`。
同一 FC16/8 区域、KV、d4352、4,561,319,928 去重参数、B1/P2/span20、staggered-v1；
native mixed-c、四卡 ordinary、FP32 payload/scoring、TF32 关闭、packed/scalar 完整返回，
ATen/interop/worker 均 1。各自一次独立流程，warmup1+测量1+下一完整窗精确恢复。
`validation=none`，完整返回与精确恢复已实际执行，独立 CPU 数值/离散检查未执行；
原同模型容量流程的独立 FP32 超差仍见[容量证据](core5-platform-capacity-20261031.md)。

| 成本 | 数据盘 | /dev/shm |
| --- | ---: | ---: |
| 保存第0窗 | 57.7345 s | 47.4340 s |
| 保存第1窗 | 62.9729 s | 48.0332 s |
| 完整恢复及下一窗检查 | 289.5869 s | 289.2628 s |
| 整流程 | 453.5596 s | 427.3098 s |
| 进程 RSS 高水位 | 74.880 GB | 74.888 GB |
| 最后 checkpoint 单代 | 20,698,161,792 bytes | 20,698,161,792 bytes |

tmpfs 保存成本减少约18–24%，整流程减少约5.8%。它没有消除 CPU 编码、SHA256 和
恢复比较。磁盘首保存的权重到CPU为10.558 s、序列化14.509 s、fsync9.366 s、hash22.304 s。
tmpfs 文件 fsync 几乎不耗时，但 hash 仍约22 s。记录的 span 是包含子阶段的墙钟时间，不能嵌套相加。

磁盘恢复中，已保存值快照24.506 s、下一窗参数快照17.799 s、重新构造和加载79.593 s；
解码值比较89.854 s、下一窗参数/optimizer比较74.913 s。构造/加载内又包含反序列化11.053 s、
权重校验27.129 s、复制权重1.840 s。CPU完整快照和比较是本次有依据的优化目标，扩盘本身不会降低这些成本。
窗口计算0.784/0.686 s只是两个流程内的成本观察；没有用它们计算存储算法的算子加速比。

## 准入、任务与范围

GPU容器主存上限257,698,037,760 bytes，tmpfs总额128,849,018,880 bytes；
tmpfs未来文件同时计入主存预算和挂载可用空间，不把两者相加为物理容量。
资源快照和逐卡预算保留在结果内。RSS不包含所有文件页；cgroup历史峰值也不是该run峰值。
tmpfs会在重启时丢失，因此不认证持久磁盘的耐久性。

`jobs/platform-checkpoint-cost-dba2a5e`于08:11:48Z退出0，两项内层退出0、exact-next通过；
独立SSH检查无存活监督者或后代。大文件只在审阅小记录、确认终态并检查dry run后退役；
配置、结果、hash、日志和重建入口保留。

固定源码CPU相关67项通过；新私有CUDA安装的小型两卡ordinary `check --scope full`通过
完整独立推理、各独立根、三步训练、optimizer及精确恢复。该检查不是4.561B独立数值资格。
后续显式保存间隔及快照优化以各自固定源码和实际回归另验，不能由本基线推断。

## 干净安装后复现

按[人类手册](../human-workflow.md)安装对应固定提交并填本站实际site/build路径：

```sh
tidegraph plan configs/checkpoint-cost/cuda-disk.json --site SITE.json
tidegraph run configs/checkpoint-cost/cuda-disk.json --site SITE.json --output-dir NEW_DISK_RUN
tidegraph run configs/checkpoint-cost/cuda-tmpfs.json --site SITE.json --output-dir /dev/shm/NEW_TMPFS_RUN
tidegraph results docs/evidence/core5-checkpoint-cost-20261031.json.gz --format table
```

两个大流程串行，先做live准入；输出目录必须新建。本机NPU tmpfs采用独立配置和联合预算，
未完成前不引用GPU比例预测NPU收益。

## 36baeb7快照优化与显式保存间隔

固定实现`36baeb719361d5e5e39c0f05533f9e5cc53c04f7`，同model/workload/execution身份，
`cuda-disk-periodic.json`明确interval2，只发布最后一窗；默认仍每窗保存。
CPU固定110项通过（standalone一项明确未选），CUDA实际返回6项和两卡full检查通过；NPU新包资格另等实际结果。
成本任务08:28:55Z终态exit0、无后代，完整解码及下一窗/参数/optimizer精确检查通过。
仍为validation=none观察，不新增大模型独立数学资格。

| 阶段 | dba2a5e磁盘基线 | 36baeb7 |
| --- | ---: | ---: |
| 保存次数 / 总保存成本 | 2 / 120.707 s | 1 / 58.750 s |
| 已保存值完整快照 | 24.506 s | 9.410 s |
| 下一窗参数快照 | 17.799 s | 9.064 s |
| 解码值比较 | 89.854 s | 62.179 s |
| 下一窗参数/optimizer比较 | 74.913 s | 67.426 s |
| 完整恢复检查 | 289.587 s | 223.215 s |
| 整流程 | 453.560 s | 324.107 s |
| RSS高水位 | 74.880 GB | 59.716 GB |

完整恢复成本降低22.9%，RSS降低20.3%；整流程降低28.5%，但其中包含少保存一次的显式耐久性取舍。
未保存的warmup边界不能用于中断恢复；最终边界和optimizer窗口不变。
消除加速器到CPU后的重复clone与重复parameters快照，CPU完全相等张量正向快捷比较仍先保留有限值检查；
不相等值走原严格断言，原容差、字段、None/零和失败诊断不改。
没有分开消融这些小改动，因此不把整项差值单独归给其中某一行代码。

剩余恢复比较仍129.605 s，构造/加载72.136 s；CPU完整有限值扫描、权重校验和hash仍有成本。
本轮到此停止继续重写codec或比较器；后继需要这些子阶段的具体排他profile和保持全部拒绝语义的方案。
9.123B容量采用同一固定包、显式interval3和单代预算重新准入；原两代磁盘拒绝保留。
