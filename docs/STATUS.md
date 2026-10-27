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

## 当前任务与下一动作

当前 HEAD 1c314a7 已推送。即将提交设备检查与设备排序增量，然后固定源码资格，单独提交证据。
本增量不包含主工作区的 FP16 草稿。设备 tree2a1af585025625a4710b109ad33e82ca7514e8f4
已冻结于 core5-bootstrap / core5-device；SDK tree01fe819ed2b60345c97701d9a0a8d6882508c930
的 C++ 源码与之相同。新入口增加 dispatch 环境与源码稳定性记录。

### 开发证据（不是固定提交资格）

- 单卡 Python/native mixed-a/b 各2代表组合，mixed-c各19组合通过，包含完整记录、
  独立原始 State/descriptor/output/control 根、三步 AdamW、checkpoint 后第四步与 CPU handoff。
  输出 artifacts/jobs/core5-weighted-{python,native}-mixed-{a,b,c}-dev。
- CPU 首轮198通过/2失败；两处 standalone新增报告字段与旧完整dict断言不符。
  修正后 CPU2 两项通过，且两个 native完整检查通过。原 failed记录保留。
- NPU Python build/build2 passed，manifest已保存；SDK build刚完成 passed/exit0，manifest已保存。
  SDK使用独立Torch2.10/CANN9.0，不混用Python2.12二进制。
- Python完整两卡默认异步在首个NPU all-VJP失败：CANN rtStreamWaitEvent invalid event。
  current-device0/1与串行autograd定位均失败；普通P2P/两个根小探针可通过，尚未定位根因。
  最小复现 artifacts/diagnostics/core5-peer-event，失败job保持failed。
- 显式 ASCEND_LAUNCH_BLOCKING=1 的完整Python两卡检查通过：delta-pdg、heterogeneous-attention，
  四步/恢复/交换owner；output artifacts/jobs/core5-weighted-python-peers-sync-dev。
  不认证默认异步、native或resident。当前开始对应native同步检查。
- Python profile passed：28301设备kernel、其中20设备AiCPU，2733主机scalar、1243memcpy，
  32candidate/31Full/44messages；输出 artifacts/jobs/core5-weighted-python-profile-dev。
  有profiler警告；有限trace未发现命名host fallback，不能泛化。不是吞吐排名。
  int64 Argsort厂商在设备AiCPU执行，保持精确ID，不称为主机回退或AiCore。

### 活动任务与失败后的新检查

| unit（均以 -20261027 结尾） | frozen worktree | launcher / output | 预算 |
| --- | --- | --- | --- |
| tide-core5-weighted-native-peers-sync-dev | artifacts/worktrees/core5-device tree2a1af | .local/core5-weighted-native-peers-sync-dev.sh / artifacts/jobs/core5-weighted-native-peers-sync-dev | 2卡动态队列，同步dispatch，1800秒 |
| tide-core5-weighted-sdk-dev | artifacts/worktrees/core5-sdk tree01fe819 | .local/core5-weighted-sdk-dev.sh / artifacts/jobs/core5-weighted-sdk-dev | 1卡动态队列，两组合，1800秒 |

启动后 systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读取对应 status.json、task.log、checks结果。停止用 systemctl --user stop UNIT。
活动时不得编辑对应worktree；当前SDK build与Python peers-sync均已终止。
SDK首轮实际初始化failed/exit1：CANN内嵌Python找不到numpy（模型尚未运行）。
匹配Torch2.10解释器自带NumPy1.26.4，loader闭包通过；新sdk2-dev仅在launcher的PATH
前置同一配套Python bin，不安装或修改公共包；相同binary/two cases、1800秒。
unit tide-core5-weighted-sdk2-dev-20261027；launcher/output将sdk-dev替换为sdk2-dev。
下一步：完成上述开发检查，隔离FP16草稿提交设备增量；冻结该提交，跑CPU相关200项、
Python/native有限mixed矩阵和两卡同步配置、SDK必要图案例及native profile；固定证据后继续。

## 未提交 FP16 草稿与待解决问题

不混入设备增量：python/tidegraph/weighted/binding.py、cpp/src/weighted_detail.h、
weighted_{gated_packed,query_recurrent,recurrent,recurrent_packed}.cpp、tests/test_weighted_precision.py。
KV首轮2失败（half常量构造、继承payload误差的FP32 descriptor过严容差）；
Python第二轮4通过/2失败：GDN小梯度舍入导致AdamW不同更新、SSD反向NaN。
诊断 artifacts/diagnostics/core5-half-{kv,python}-dev；C++Half常量草稿尚未编译。
后续修复稳定VJP和原始int64长时间差溢出，保持精度契约；standalone FP16 master/checkpoint尚未实现。
weighted resident、CUDA源码/构建/真机分账、有限性能和最终追溯继续，整体目标未完成。

## 环境与保留事项

真实路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前home。
旧 ASCEND_CODING_AGENT_GUIDE.md 已不存在；现行规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md
及 /etc/modulefiles 已核对。Python环境：source /etc/profile.d/modules.sh，module purge，
module load ascend/dev-workspace-9.1.0，source .local/venv-core5/bin/activate。
Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1/GCC10.3.1，aarch64。
SDK单独 module load libtorch-npu/2.10.0-cann9.0.0；ACL_OP_INIT_MODE=0。
未修改公共环境，3.3TiB空闲；2 build workers，ATen/BLAS1线程，background.slice/Nice10。
保留所有历史真实失败、最小复现与原证据；status --all-jobs 可审计，不为恢复上下文重跑旧任务。
