# 当前交接 — GPU 规模推理轮（2026-10-31）

授权分支 `graph-execution-foundation`；本轮起点 `cbb0964124ea7b708e19ee4f388579528cddbce3`，启动干净。
数学固定 `0770593838a37c790a568226930567ed60f99b83`；19份core-5与旧快照校验通过。
用户本轮明确授权大batch、约1B及接近安全显存边界的有限推理；旧“不扩1B”和禁止子代理安排由本轮指令覆盖。
不重跑旧训练矩阵、不开展8B/16B全扫描；旧严格失败和多卡resident硬件边界保留于[ROADMAP](ROADMAP.md)。
阶段实现先提交，再对固定源码资格，证据另提交；普通推送已授权，参考仓库只读。

## 当前工作

启动审计完成，补齐推理资格与不齐输入，随后取得baseline、有限规模测量和至多少数具体packing优化。
三个协作子任务分别负责推理checkpoint入口、真实不齐输入与静态预算、六类packing代码审计；主代理统一集成和资源。
当前未提交源码见 `git status --short`；不编辑远端活跃作业读取的固定源码。
六项问题：相同长度桶、控制元数据同步、重复packing、兼容节点合批、resident填充/容量、反向模板；均须记录证据和处置。

## 已审阅历史

上轮[eager](evidence/core5-cuda-eager-20261031.md)、[resident](evidence/core5-cuda-resident-20261031.md)、
[finite](evidence/core5-cuda-finite-20261031.md)保留原源码、严格失败及适用范围。
本轮推理须独立比较完整返回/状态/控制/消息/history/pending，磁盘新owner精确续接另验；训练失败不替代推理结论。
当前benchmark缺少推理checkpoint许可，默认输入活动/缓存长度相同；正在最小补齐，不另建测量框架。

## 现场和作业

目标 `gpu3090`，SSH入口 `artifacts/cuda3090/ssh.sh`；根 `/root/autodl-tmp/gef-cuda3090`。
重新核对四张RTX3090/24GiB/SM86、驱动595.71.05、Torch2.12.1+cu130/Python3.12.3、Toolkit13.0.88、GCC11.4、CMake3.22.1、Nsight2025.3.1。
私有环境 `venv` 的pytest8.4.2/Ninja1.13.0/JSON3.11.2可复用；实际IEEE FP32矩阵探针通过，12方向P2P查询仍false。
affinity112、配额56核、内存240GiB；启动主存约3.6GiB，数据盘空闲48.2GiB，四卡1MiB/0%，无项目进程。
远端 `dev` 保持cc6b45f的原未提交修改，不作为资格源码；检查命令 `git -C /root/autodl-tmp/gef-cuda3090/dev status --short`。
当前只读源码 `sources/cbb0964`；旧32edb0f库仅在实际源码hash匹配时用于初始profile，并记录其原构建身份。

计划首个作业 `inference-baseline-cbb0964`：在固定cbb0964运行FC/KV width8、B1/B8、P8的Python/native独立profile，各一次，180秒/项，整体900秒。
这是现有实现的同步/批次基线，旧入口没有推理checkpoint，不把profile当完整吞吐资格。
提交命令：`venv/bin/python sources/cbb0964/scripts/container_job.py start --cwd /root/autodl-tmp/gef-cuda3090/sources/cbb0964 --output-dir /root/autodl-tmp/gef-cuda3090/jobs/inference-baseline-cbb0964 --timeout 900 -- /root/autodl-tmp/gef-cuda3090/venv/bin/python /root/autodl-tmp/gef-cuda3090/inference-baseline-cbb0964.py`。
日志/status位于该job目录，输出 `runs/cbb0964-inference-baseline`。当前状态：passed/exit0，2026-10-31T03:13:25Z终止，remaining_live_pids为空。四个profile完整前向核对通过；PyTorch B1/B8 scalar取回515/3623，native657/2981；不授予吞吐/推理磁盘资格。
检查/停止使用同一 `container_job.py status|stop --output-dir JOB`；完成后核对无存活后代。
本机私有 `.local/venv-core5-inference` 继承现行Ascend9.1开发栈，仅补pytest/Ninja；本机尚无长作业。

## 下一命令

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
```

当前没有活动目标作业。推理/输入设施相关短回归已通过；下一步提交该设施，建立新只读源码并构建匹配standalone benchmark。core/resident复用须核对实际源码hash，保留原构建身份。
新显式目录 `benchmarks/core5-cuda-inference-v1.json` 包含staggered-v1输入、14个静态模型/窗组和11个执行选择；不自动全排列。正式逐项选择，近界d8192仅在约1B实测峰值和完整预算允许后运行。
大运行先静态预算参数/状态/KV/workspace/物理副本/返回、主存和磁盘；计时与重型构建不重叠。
规模、精度、算子、packing、线程、执行位置和完整返回口径均需进入运行身份。不得以自身恢复替代独立正确性。
