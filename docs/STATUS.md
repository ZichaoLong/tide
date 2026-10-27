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
| 503f609 | 来源绑定/类别宽度、稳定 Attention/RMS VJP | 665 | [报告](evidence/core5-source-20261027.md) |
| 52dee1e | Aggregate 精确计算复用 | 543 | [报告](evidence/core5-sharing-20261027.md) |
| 6dc2665 | KV 基座投影复用 | 309 | [报告](evidence/core5-kv-sharing-20261027.md) |
| a7106fc | standalone 训练/完整数值 checkpoint | 20 | [报告](evidence/core5-native-training-20261027.md) |
| 9234c13 | typed weighted placement / packed copy VJP | 508 | [报告](evidence/core5-placement-20261027.md) |

## 当前任务与精确下一动作

placement 实现 9234c13，固定 tree 81d829ed987729e1d697078b10a8fa81615f1ea8 的 CPU 资格已通过：
508 passed / 0 failed / 0 skipped；unit tide-core5-weighted-placement-qualify-20261027，inactive/success、exit0。
输出 artifacts/jobs/core5-weighted-placement-qualify，build manifest 已保存，证据独立提交推送。
第一轮开发缺测试头文件的失败记录仍保留，见该证据。

设备开发 frozen tree 73f75c559f99a2278a8c27de9281fbc959bfde7e，worktree artifacts/worktrees/core5-device，
HEAD9234c13，新增三个检查脚本。**该 worktree 尚被活动 build 使用，禁止修改。**
- tide-core5-npu-python-build-20261027：仍在编译当前 Torch2.12/CANN9.1 的 NPU Python adapter；
  .local/core5-npu-python-build.sh；output artifacts/jobs/core5-npu-python-build；2 workers、1800秒。
- tide-core5-weighted-npu-python-dev-20261027：passed/exit0、19 代表案例全部通过。
  output artifacts/jobs/core5-weighted-npu-python-dev；mixed-c 纯 Python，完整记录/独立根/四步 optimizer/恢复与 CPU handoff；
  physical1→logical npu:0，由 queue 动态选择。开发证据不代替固定提交最终资格。
- 短探针 artifacts/jobs/core5-npu-sorted-probe：exit0；physical3→npu:0，
  标量硬发送设备排序、大 int64 平票、分数 total 和 literal VJP、typed packed copy 已通过。
  厂商报告 int64 argsort 在设备 AiCPU 执行，未转 float、不把它称为 AiCore 核。

主工作区未提交：check_weighted_device.py / weighted_device_cases.py / weighted_device_checks.py /
weighted_device_primitives.py，以及 Python/C++ 标量硬发送的设备 batch 排序入口、standalone 训练检查的
独立 CPU reference 与 placement/case 开关。设备开发第一轮不包含后加的 warm state 根和排序入口。
下一动作：核对上述 build 终态，保存 manifest；冻结最新增量，再跑相关 CPU 与实际 mixed-a/b/c native，
以及独立 libtorch-npu/2.10.0-cann9.0.0 SDK 构建/真机。实现通过后提交，固定资格再独立证据。
继续 weighted resident、低精度、跨卡、有限 profiling 与最终追溯；整体目标仍未完成。

检查当前 build：systemctl --user show tide-core5-npu-python-build-20261027 -p ActiveState -p Result -p ExecMainStatus；
cat artifacts/jobs/core5-npu-python-build/status.json；tail -n 30 artifacts/jobs/core5-npu-python-build/task.log。

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

来源两轮真实失败与旧5391017基线复现保留在来源证据及原始诊断；不因665项新通过覆盖旧失败。

本轮环境探针：当前 Torch-NPU2.12/CANN9.1 栈的实际 FP32 matmul/relu/sum 得到 152，
physical13→logical npu:0（Ascend910_9392），queue core5-npu-operator-20261027 completed/exit0；
仅基础算子探针，不是带权模型资格。queue 记录 artifacts/diagnostics/core5-npu-operator/queue-state.json；
完整 doctor 输出仅在会话工具记录。后续任务重新选卡，不沿用当时物理 ID。
独立公开 C++ 入口 libtorch-npu/2.10.0-cann9.0.0 的 SDK 静态/loader 检查通过，
与 Python2.12 分栈；本轮尚未编译/运行其项目真机检查。未改公共环境。
