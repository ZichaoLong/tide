# 当前交接 — GPU规模推理与有限优化（2026-10-31）

授权分支 `graph-execution-foundation`；最后GPU资格来源 `f738692ce4c06ac0530851c0198552eea01aa459`。
数学固定 `0770593838a37c790a568226930567ed60f99b83`，快照校验通过。
最新实现 `909a322f058b7e897e6185334d6915dccca0d06a` 仅修证据查询对内嵌configuration facts的读取与身份校验；固定源码5项及完整索引查询通过。
参考仓库只读；远端dev原有未提交修改保留，从未作为本轮资格源码。

## 本轮实际结果

完整来源、逐配置命令/工作/精度/构建、回归与原失败见[审阅报告](evidence/core5-cuda-inference-20261031.md)及同名JSON。
50项正式测量中47项通过完整独立前向和磁盘新owner精确续接，3项d256 FP32严格失败仍只作成本观察。
另20项profile/工作/诊断/相关检查观察独立记录，包含两项strict_mismatch；外层任务passed不覆盖内部失败。

| 来源/范围 | 已实现、已构建与实际通过 |
| --- | --- |
| eee68c4 | 独立完整推理、无optimizer、staggered-v1真实不齐输入；A21测量18通过/3严格失败；CPU23、resident相关4项通过 |
| 1c903ae / 4960b6c | 桶/MM/BMM/活动观测、批量bool元数据、首差诊断与显式CUDA eager FP64；修复standalone命名编译失败。core/standalone新构建，相关66项与三客户端选定训练9项通过 |
| 9fce486 | 规模静态预算、离线同步API区间并集；GPU literal与两客户端零支持完整训练/恢复通过 |
| 2f17c24 / fd3c72b | 显式完整返回packing，默认scalar。F33目标环境检查（9直接CUDA）通过；H只修Half检查材料，三dtype字节、三客户端小型FP64 SGD训练恢复和10正式项通过 |
| 9acc7aa | FC/KV 1,010,339,302 trainable，显式FP64：三客户端B1和native B8全部通过。B1 PyTorch/native/standalone中位239.231/223.885/330.295ms；native B8 771.993ms |
| f738692 | FC/KV 2,085,424,614 trainable，native mixed-c FP64 B1/P4/span16：384.317ms中位、最高reserved18.541GiB、完整独立记录与第4窗精确恢复通过；另1项轻量工作观察通过 |

本机NPU物理9的A/D/F纯PyTorch完整推理/恢复均通过，中位559.661/527.660/576.500ms。
F另5个NPU字节/根检查及小型HB FP32训练/恢复通过。返回packing同材料成本上升9.26%，保持默认scalar；无standalone NPU新资格。
六项问题均有处置/重启条件，见[performance-lessons](performance-lessons.md#gpu-推理批次与完整返回)和唯一待办[ROADMAP](ROADMAP.md)。

## 终态、环境与资源

没有活动GPU/NPU资格作业。目标机现存35份作业记录均终态、实际监督器/工作进程均不存活；原失败和cancelled探针保持原状态。
最后作业 `inference-near-f738692` 于04:18:50 UTC passed/exit0，remaining_live_pids为空；固定源码sources/f738692，日志jobs/inference-near-f738692/task.log。
本机三个NPU独立background.slice服务均success/exit0、ControlGroup为空。
最终现场04:20:37 UTC：四卡均无计算进程，每卡free24126MiB；数据盘free47,268,839,424字节。

目标入口 `artifacts/cuda3090/ssh.sh`，根 `/root/autodl-tmp/gef-cuda3090`，Python用绝对`venv/bin/python`。
已选栈Torch2.12.1+cu130/CUDA13.0.88、4×RTX3090/24GiB/sm86；affinity112/CPU配额56、内存240GiB、数据盘50GiB。
构建56线程；正式计时GPU0、ATen/BLAS/interop/workers1、TF32关闭。重型构建与计时串行，外部CPU争用未隔离。
core构建4960b6c、最新standalone构建fd3c72b、A的匹配resident库32edb0f分别保留真实身份；不能称所有库在f738692重新构建。
远端dev仍为cc6b45f上的独立脏目录，最终diff哈希在JSON；不要覆盖或用于资格。
当前Ascend站点入口是development列出的dev-workspace-9.1.0 module；旧home指南已退役。

大文件在终态完整审阅、流式SHA256与dry-run后共退役59,095,122,182字节，只含本轮生成checkpoint/standalone fixture。
小结果、配置、命令与哈希保留；无需旧运行目录即可生成新材料，哈希不保证退役字节可恢复。
忽略目录的私有环境、匹配构建和小记录可复用但不是Git复现的隐含依赖，不清理参考仓库或公共工具链。

## 未解除边界与接续

- d256 FP32 B1/8/32首窗keys3/768坐标严格差异、原训练失败/strict xfail均保持；显式FP64通过不解除它们。
- d6144近界预测超过4GiB预留，未运行；standalone d5888事务恢复双bank约33.14GiB拒绝；resident FP64域拒绝，大宽度FP32 Full守卫亦超24GiB。
- 当前12方向P2P仍false，原跨卡原子硬件限制保持。无多卡resident、新通信协议、FP16规模、大尺度训练或8B/16B扫描资格。
- 1B B8的MM平均8.04行、私有BMM全2行；参数规模没有证明充分矩阵利用。六项后置内容只在ROADMAP的具体重启条件满足时继续。

本轮有限范围已完成；实现先提交并固定资格，本页所在后续提交仅记录审阅证据、支持范围与交接。
没有自动续跑队列或未提交实现。后续无需重跑旧矩阵，新的运行须新目录、固定源码和现场准入。
在Python3.10+的匹配私有环境中，从干净检出查询：

```sh
python scripts/status.py
python scripts/check_upstream.py
python -S scripts/index_core5_study.py docs/evidence/core5-cuda-inference-20261031.json --view models --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-index.json --view coverage --topology fc --kind performance --format tsv
```

[报告的干净复现命令](evidence/core5-cuda-inference-20261031.md#从干净检出复现)包含重新构建、三客户端1B、小型resident及失败重建；
只clone/fetch授权新分支，旧身份用`resolve_history_commit.py`定位，不把映射当重新验证。
