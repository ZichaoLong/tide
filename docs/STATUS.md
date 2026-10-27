# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、提交推送；不使用子代理，参考仓库只读。
实现470ce06已推送；本次记录固定设备资格证据后继续，不在阶段提交后暂停。
总体待办以 [ROADMAP](ROADMAP.md) 为准。

## 已固定的阶段资格

此前各阶段为 CPU FP64/FP32；本轮设备资格另列，均为具体范围，不代表总体完成。

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

## 当前设备资格与真实限制

[固定证据](evidence/core5-device-20261027.md)：470ce06112b888ed0b08866c3d791501b3bcd240，
tree6cf3e7a6cd9b2d5c00960b59b6132146751531fa。CPU200项、Python/native mixed-a/b各2、
mixed-c各19、两卡显式ASCEND_LAUNCH_BLOCKING=1各2组合全部passed/exit0。三个worktree
core5-bootstrap/core5-device/core5-sdk均冻结该提交；当前所有相关任务已终止，无活动reader。
完整unit、命令、结果摘要、hash在证据JSON，原始 artifacts/jobs/core5-...-qualify。

默认异步两卡完整Python VJP仍有CANN invalid event；最小复现 artifacts/diagnostics/core5-peer-event。
current0/current1/serial失败保留；普通P2P小probe通过不解除限制。
standalone SDK build/loader已通过，使用匹配Python bin的PATH后开发两图四步通过。
固定24图门禁在fc/affine第三轮forward的evidence字段超容差，max_abs_error3.09944e-6。
原任务 artifacts/jobs/core5-weighted-sdk-qualify保持failed；诊断副本
artifacts/diagnostics/core5-sdk-affine/diagnostic.cpp与build-commands.json，
unit tide-core5-sdk-affine-diagnostic-20261027，output artifacts/jobs/core5-sdk-affine-diagnostic，
同样failed。前两轮forward/gradient/parameter比较通过；后续定位实际参数漂移/数值归约，
不放宽路由或容差，不把其余未执行图推定通过。

native固定profile已passed：31531kernel、1641hostscalar、1372memcpy；Python开发profile
28301/2733/1243，两者32candidate/31Full/44messages。均有20设备AiCPU。非吞吐排名。

## 未提交增量与精确下一动作

FP16草稿仍未提交（git status列出），不在上述资格内：
- binding.py允许half，显式scoring_dtype=float32，基于同一FP32随机参数量化。
- weighted_detail.h及recurrent/gated/query packed C++ Half常量构造草稿尚未编译。
- tests/test_weighted_precision.py检查五记忆、三步FP32 master AdamW与checkpoint后第四步。
- Python normalization.py + gated_delta.py/ssd.py/full_batch.py新增half scaled-RMS稳定一阶VJP，
  保持实际half前向操作数，内部FP32归约；尚未补C++、独立公式与完整契约。
  最新短检查 artifacts/diagnostics/core5-half-python-stable2/test.log：4passed/1failed/6deselected。
  SSD NaN已消除；GDN仍在nodes.4.kernel.conv.weight因小梯度舍入造成AdamW差异.001907，未放宽容差。
  首次stable启动仅缺basetemp父目录5errors，日志core5-half-python-stable.log保留。
- 还要处理half原始int64长时差转换溢出、KV位置/衰减，不能压紧逻辑时间。
  standalone FP16 master/checkpoint、weighted resident尚未实现。

下一动作：提交本轮设备证据并推送；继续定位SDK fc/affine及两卡异步事件问题，
完成FP16稳定VJP/长时间/精度契约与相关验证后独立实现提交。长构建前把所需草稿冻结到
空闲worktree，记录tree与新unit/output，绝不让任务读取主工作区。现有CPU/NPU/SDK build目录
必须分栈使用；更新worktree前先检查相关unit与status.json为终态。

## 环境、命令与保留

现行规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 及 /etc/modulefiles；
旧ASCEND_CODING_AGENT_GUIDE.md不存在。实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation。
Python：source /etc/profile.d/modules.sh，module purge，module load ascend/dev-workspace-9.1.0，
source .local/venv-core5/bin/activate；Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1/GCC10.3.1。
SDK单独module load libtorch-npu/2.10.0-cann9.0.0，ACL_OP_INIT_MODE=0；PATH前置
$LIBTORCH_ROOT/../../../../bin让CANN内嵌Python找到同栈NumPy。无需修改公共包。
ATen/BLAS单线程、2build workers、background.slice/Nice10、有限时长；3.3TiB空闲。
检查：python scripts/status.py；systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读取status.json/task.log。停止：systemctl --user stop UNIT。保留所有失败、最小复现和原证据；
不恢复旧队列、不重跑历史矩阵，CUDA源码/构建/真机与CPU/NPU证据严格分开。
