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

主工作区下一增量未提交：嵌套参数捕获/别名保留、零/正接收分组的 State VJP，
及空卷积缓存的无连接分支。Python 新连接检查 13 项和 packed/legacy 53 项先通过。
相关 C++ 已构建。StateKernel 能力增加结构 key，依赖对象全部重建，未复用旧 ABI。
standalone C++ packed 输出根的手算反向与能力统计已接入并通过。

开发 `tide-core5-state-vjp-dev-20261027` 已终态 failed/exit 1：**337 passed / 6 failed**。
冻结树 `96d7b2d8c253ffbeab9b6896e42c233013ce2c5c`，原日志/源码/status 保留 artifacts/jobs/core5-state-vjp-dev。
FP32 affine Read0 score.bias 差 7.63e-6；SSD Read−/+ input.2 差 2.13e-6 / 3.87e-6；
Python/native 各三项。完整前向/新增连接测试已通过。独立诊断
`.local/diagnose-state-vjp.py` / 该 job 的 diagnosis.txt 定位：共享归一化先合并 cotangent 再
做一次 RMS VJP，改变 scalar 各分支先反传再相加的浮点顺序。分开 norm 的变体全通过。
正式 Python/C++ 保留写入、GDN 控制、p_v 摘要的各批量归一化分支；
未改数学/数值前向、容差、dtype 或裁剪梯度。修正后 Python 定向 **56 项通过**。

修正开发 `tide-core5-state-vjp-norm-20261027` 已终态 inactive/success、exit 0：
**81 passed / 0 failed / 0 skipped**，冻结树 d3282cab7f596a296434d1d046c79cabffa23ac3。
launcher `.local/core5-state-vjp-norm.sh`，记录 artifacts/jobs/core5-state-vjp-norm；
新 weighted_state_vjp/weighted_packed、六个原失败、standalone 全部通过。
当前无 live job；实现/测试/契约同增量提交后，以固定提交运行不可变相关资格。
下一动作：冻结到 task-owned worktree 并核验干净只读，通过 develop.py --reuse-build 校验构建，
新 state-vjp-qualify job 沿 state-vjp-dev 的完整八入口运行，终态后独立记录证据并推送。
契约 weighted-state-vjp.md；整段/每切点/optimizer 与旧 State VJP 回归需要在资格中保持。
再推进 Aggregate/Read/Full、mixed/resident、standalone 完整训练与有限 profiling。
总体任务尚未完成；无用户修改，持续推进。

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
