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
