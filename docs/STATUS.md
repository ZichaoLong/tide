# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 graph-execution-foundation。
持续实施、私有环境、有限实验、提交和推送均已授权；不使用子代理。参考仓库只读。
历史等待授权/提交后暂停已被取代。完整未完成范围以 [ROADMAP](ROADMAP.md) 为准。

## 阶段结果

- 一般 SettleGraph `efb9d62`：247 项固定源码 CPU 定向资格，证据 d0acd3c，已推送。
- 结构化 Read/int64/SrcClass `714d5f9`：212 项固定源码 CPU 定向资格，证据 cbdd3fd，已推送。
- KV 首组合 `6fb467c`：129 项固定源码 CPU 定向资格，证据 572d4ff，均已推送。
- affine/delta、完整 GDN/SSD、固定时间短卷积与异构绑定
  `442e8b2701164eb7e4241da56c8a395a832ebeb0`：**544 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[契约](weighted-recurrent.md)、[证据](evidence/core5-recurrent-20261027.md)。
  证据 955912c 与实现已一起推送。首组合和循环阶段均不代表完整总体任务完成。

- Bernoulli/Shannon/固定 alpha ent 分配 `a05d3ffc946985cfeb124a5e3b3074ceeece306d`：
  **226 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-entropy-20261027.md)。

- 十种发送/节点增益 `4a0858880831cb46ad96191ab9dd58fca5260223`：**320 passed / 0 failed / 0 skipped**，
  CPU FP64/FP32；[证据](evidence/core5-sending-20261027.md)。实现与证据 76c698e 均已推送。

- 完整评分/证据/key 与共同阈值 VJP `cacb645adc64ed06e454f7ac8d8572bf09b98ee2`：
  **593 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-scoring-20261027.md)。

## 当前任务与下一动作

评分资格 `tide-core5-scoring-qualify-20261027` 已终态 inactive/success、exit 0；
实现 cacb645 与证据 8a6208e 已推送。详情及该阶段原失败见评分证据。

实际 packed 增量已完成开发：KV ragged 投影、四种循环仿射前缀、GDN/SSD 原始时间卷积。
[契约](weighted-packed.md)明确输入域、数学对应、解码和实际统计；不调用 scalar Upd 补写数值。
零接收保留 raw tuple，时间差先 int64，完整正前缀可解码。语义 VJP 重放仍保留，
批量反向/Read/Full/区域/设备推进尚未宣称完成。无用户修改。

精确联合记忆实现 `26dad7d22d61468823a986fb67608fc8df3b74e0` 已完成不可变资格：
**378 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-packed-20261027.md)。
unit `tide-core5-packed-qualify-20261027` 已终态 inactive/success、exit 0；
干净只读 worktree artifacts/worktrees/core5-bootstrap，源码 26dad7d。
原始 job/status/log/source/development 在 artifacts/jobs/core5-packed-qualify；
launcher `.local/core5-packed-qualify.sh`。实现与证据 727a7b5 已推送。
当前活动开发任务如下。

State 批量 VJP 实现 `7a017bf4eea93787ca2d8e72baf436b128266afd` 固定源码资格：
**343 passed / 0 failed / 0 skipped**，CPU FP64/FP32；[证据](evidence/core5-state-vjp-20261027.md)。
unit tide-core5-state-vjp-qualify-20261027 已终态 inactive/success、exit 0；
worktree artifacts/worktrees/core5-bootstrap 干净固定 7a017bf。
输出 artifacts/jobs/core5-state-vjp-qualify；launcher `.local/core5-state-vjp-qualify.sh`。
实现与证据 4468439 已推送。原开发 337/6 与修正 81/0 分别保留，原因/修复见证据，未覆盖原失败。

WeightedAggregate 增量完成开发：class 投影批、完整 fiber 归约与独立批量 VJP。
数值开发 unit tide-core5-aggregate-dev-20261027 已终态 inactive/success、exit 0，**288 passed**；
开发 tree d9758c37765f4bda35f4a91bd32ba42402d49ddd，原记录在 artifacts/jobs/core5-aggregate-dev。
批量 VJP 开发 unit tide-core5-aggregate-vjp-dev-20261027 已终态 inactive/success、exit 0，
**696 passed / 0 failed / 0 skipped**，tree 4ebfbaeb9848217a423a922c101c2615913190f1；
记录在 artifacts/jobs/core5-aggregate-vjp-dev。额外 HB/PDG batched AdamW/checkpoint **8 项通过**。
最初局部短测的 4 项错误是固定 scale 不需要梯度却列为根，已修正夹具为显式独立根；
修正局部 **9 项通过**，未改变数学公式或容差。契约 docs/weighted-aggregate-batch.md。

准备提交该实现并启动 `tide-core5-aggregate-qualify-20261027`，launcher `.local/core5-aggregate-qualify.sh`。
资格只读 artifacts/worktrees/core5-bootstrap，冻结本次实现提交；独立于主工作区后续 Read/Full 草稿。
输出 artifacts/jobs/core5-aggregate-qualify/{status.json,task.log,development.json,source.tar.gz}。
background.slice、Nice=10、两个 build workers、ATen/BLAS 单线程、15 分钟上限。
精确校验已有 build 的 C++ 源码与全部二进制 hashes，运行上述相关测试及新增训练恢复。
检查 `systemctl --user show tide-core5-aggregate-qualify-20261027 -p ActiveState -p Result -p ExecMainStatus`；
日志 `tail -40 artifacts/jobs/core5-aggregate-qualify/task.log`；停止同名服务。结果尚未知。
终态成功后单独提交证据并推送，继续实施，不在阶段提交后暂停。

主工作区下一增量尚未提交：weighted/query_batch.py、read_batch.py、full_batch.py，
cpp/src/weighted_query_batch.h、weighted_query_kv.cpp、weighted_query_recurrent.cpp 和
 tests/test_weighted_read_batch.py。Python 局部数值 **60 项通过**；C++ 尚未接入/构建，
Read/Full 仍未切换能力，不能声明完成。所有未提交修改均本任务所作，无用户修改。
后续为 Read/Full/区域的数值与批量 VJP、计算共享（包括正式参数绑定方式），mixed/resident、
standalone 完整训练/graph continuation 与有限 profiling。总体任务尚未完成。

## 环境与保留事项

实际路径 /mi/data2T/zlong/home/llm/graph-execution-foundation；/home/zlong 指向当前 home。
规范 /opt/software/ascend-locks/dev-workspace-9.1.0/README.md 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境。
CUDA 源码/构建/真机分别记录；无 GPU 不阻塞其他工作。

旧 bootstrap 缺 Ninja、缺 TIDE_BUILD_DIR 的 24 项前置失败、structured-dev 的 210/2、
weighted-kv-dev 的 C++ Options 构建失败，以及 recurrent-dev launcher 失败保持原记录。
recurrent-build 的 212/1 是比较目标字段归约顺序问题，修复后新开发和不可变资格分别通过；
原因与修复详见循环证据。原失败日志/源码留在相应 artifacts/jobs 目录。
