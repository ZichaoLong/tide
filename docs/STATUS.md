# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`；授权分支 graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、提交和推送；不使用子代理，参考仓库只读。
总体尚未完成，不在阶段提交后暂停；完整待办以 [ROADMAP](ROADMAP.md) 为准。

## 已固定的阶段资格

均为 CPU FP64/FP32、具体定向范围，不代表总体/设备矩阵完成。

| 实现提交 | 范围 | 通过数 | 证据 |
| --- | --- | --- | --- |
| efb9d62 | 一般 SettleGraph | 247 | [报告](evidence/core5-settle-20261027.md) |
| 714d5f9 | 结构化 Read/int64/SrcClass | 212 | [报告](evidence/core5-structured-20261027.md) |
| 6fb467c | KV 首组合 | 129 | [报告](evidence/core5-weighted-kv-20261027.md) |
| 442e8b2 | affine/delta/GDN/SSD/短卷积 | 544 | [报告](evidence/core5-recurrent-20261027.md) |
| a05d3ff | 熵分配 | 226 | [报告](evidence/core5-entropy-20261027.md) |
| 4a08588 | 十种发送/增益 | 320 | [报告](evidence/core5-sending-20261027.md) |
| cacb645 | 完整评分/evidence/key | 593 | [报告](evidence/core5-scoring-20261027.md) |
| 26dad7d | KV/循环实际 packed 与固定时间卷积 | 378 | [报告](evidence/core5-packed-20261027.md) |
| 7a017bf | State 批量 VJP | 343 | [报告](evidence/core5-state-vjp-20261027.md) |
| d90cfbc | Aggregate 数值/batched VJP、训练恢复 | 704 | [报告](evidence/core5-aggregate-20261027.md) |
| 7b8b469 | Read/Full 数值批量、完整快照 | 377 | [报告](evidence/core5-query-20261027.md) |
| 67854b3 | Read/Full 批量 VJP、训练恢复 | 647 | [报告](evidence/core5-query-vjp-20261027.md) |
| 5391017 | 区域数值/batched VJP、稳定均值反向 | 536 | [报告](evidence/core5-region-20261027.md) |

## 当前任务与精确下一动作

区域数值/VJP 实现 53910174d036e0809be7893527ca1b1c204a460b 的固定源码资格已完成：
unit tide-core5-region-qualify-20261027，inactive/success、exit 0，536 passed / 0 failed / 0 skipped。
干净只读 worktree artifacts/worktrees/core5-bootstrap，tree bfabbb377172fa1c288827923c180717bcae535c；
输出 artifacts/jobs/core5-region-qualify，build manifest 已独立保存。
launcher `.local/core5-region-qualify.sh`，完整命令/环境/源码与日志 hash 见证据。
区域实现/证据已推送到远端 47cde7c；当前活动任务见下文。

下一增量 SourceBinding 已实现四种来源参数绑定（region/node/source/adapted），
低秩 A/B 因子为 (node,class) 私有，参数共享不合并物理原子或状态。
支持统一或按类别的隐藏宽度；同矩阵共享组异宽明确拒绝。默认区域绑定 graph/state
已与 5391017 的冻结实现逐字节比较相同；31 项局部公式/Aggregate VJP 短测通过。
新增测试覆盖四种别名、低秩解析导数、HB/PDG 双样本完整记录、训练和含 pending 恢复。
契约 docs/weighted-source-binding.md；C++ 低秩路径尚待本轮开发构建核验。
所有未提交修改均为本任务所作。

开发 tree bcbaed6b741adca26581b60906f92ffd651b32c6，tide-core5-source-dev-20261027 已终止：
failed/exit 1，201 passed / 2 failed；构建和 standalone 通过。两失败均为 FP32 node/PDG
三步 AdamW 在第二次更新后的 Full query 参数差 2.7641654e-6，前向/梯度比较已通过。
输出 artifacts/jobs/core5-source-dev；launcher .local/core5-source-dev.sh。
诊断 artifacts/diagnostics/core5-source-query-rounding 保存原失败来源、原始 query 梯度与三步权重。
Attention 中心化 VJP 已实现，保留前向、配置和容差；26 项局部/原失败 Python 短测通过。
第二轮开发 tree ab6cda80eb576b4e932f7bd2165603e34ca58749，tide-core5-source2-dev-20261027
已终止 failed/exit 1：400 passed / 2 failed；原 source AdamW 两失败消除，standalone 通过。
新两项是 test_weighted_read_batch 的 FP32 case 3、Python/native 完整 objective 对
nodes.1.read_program.score.encoders.cur.in_weight 的梯度差 3.21865e-6。
已从 git archive 5391017 提取原源码并复现相同 Python 失败；不是本轮来源/Attention 引入。
诊断目录 artifacts/diagnostics/core5-query-encoder-rounding 包含基线和消融脚本；
关闭区域批量或改 Query 投影方向均未消除。归一化 VJP 改为归一化坐标收缩后，
9 项新解析公式/方向差分与原两类 Python 失败短测通过；前向、dtype 和容差保持。
第三轮开发 tide-core5-source3-dev-20261027 已完成：inactive/success、exit 0，
665 passed / 0 failed / 0 skipped，含新两类数值修正、来源/优化器/恢复、query、循环、
评分、区域、State VJP 及 standalone；原失败仍各自保持 failed。
源码 tree fc6ca6b13402e3cc8f76372d600ff02d6b95f836，输出 artifacts/jobs/core5-source3-dev，
launcher .local/core5-source3-dev.sh。当前没有活动任务。
下一步从该 tree 固定本轮实现（排除其中两个未接入的共享草稿文件）并提交，
然后显式提交 SHA 冻结干净源码，启动 tide-core5-source-qualify-20261027。
通过后单独提交证据并推送；新计算复用工作保持未提交、不属于该资格。
主工作区另有下一增量：Aggregate 相同输入/参数的精确计算复用。
python/tidegraph/aggregate_sharing.py、weighted/sharing.py 已接入 blocks.py、weighted/programs.py；
WeightedModel 单位传输保持真实广播 Tensor 别名，layered_inputs 显式复用同一位置载荷。
C++ 新 aggregate_sharing API/实现、weighted_sharing.h 和 block.cpp/stream.cpp/delivery.cpp 亦已接入，尚未构建。
新 tests/test_weighted_compute_sharing.py 的 12 项 Python 检查通过、12 项 native 未运行；
此前 17 项 Python instance/semantics 通过。此增量不属于 fc6ca6b，不可随来源实现提交或认证。
阶段提交必须按 fc6ca6b 的 tree 固定已测试源码，排除 python/tidegraph/aggregate_sharing.py 与
python/tidegraph/weighted/sharing.py；新 shared 改动保留在工作区。STATUS 可以更新到实际终态。
随后精确计算共享、mixed/resident 前向/反向/optimizer/跨卡、standalone 完整训练/graph continuation、
有限 profiling 和最终追溯。总体尚未完成，完整范围见 ROADMAP。
CUDA 源码/构建/真机分开记录，无 GPU 不阻塞其他实施。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64；未改公共环境，磁盘约 3.4 TiB 空闲。
短测设置 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0；
TIDE_BUILD_DIR 指向 artifacts/worktrees/core5-bootstrap/build/cpu。

region4-dev 的 480/2 原失败与诊断在 artifacts/diagnostics/core5-region-pdg-rounding；
独立分支根 46/2 在 artifacts/diagnostics/core5-region-vjp-connectivity。region5-dev 536/0
不覆盖旧失败。前三次启动/编译失败仍留在各 job 状态；证据说明原因。
更早数值/VJP 的失败均留在相应证据/原始记录，不重新排队。status 工具已同时扫描
artifacts/NAME 与 artifacts/jobs/NAME，不递归 worktree/archive 副本。
