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
launcher `.local/core5-packed-qualify.sh`。实现与本证据提交后推送；原失败保留如下。
当前无 live job。

主工作区已开始下一增量：嵌套参数捕获/别名保留、零/正接收分组的 State VJP，
以及空卷积缓存的无连接分支。Python 新定向 13 项与 packed/legacy 回归 53 项通过；
相关 C++ 能力/结构 key 已写但尚未构建。不属于 26dad7d 的资格；无用户修改。
下一命令：资格终态并记录证据后，冻结新 VJP 树，构建并运行
tests/test_weighted_state_vjp.py tests/test_weighted_packed.py tests/test_state_batched_vjp.py
与相关带权执行/训练回归；新参数树/反向契约见 weighted-state-vjp.md。
再推进 Read/Full、mixed/resident、standalone 完整训练与有限 profiling。
总体任务尚未完成，持续推进。

## 当前增量保留的失败

- packed-dev，树 9897132ce8ff886bb885790f04401d3ef242a9d4：构建 failed/exit 1。
  核心编译通过，两个 standalone 测试遗漏 NoGradGuard 头文件；已补显式 include，pytest 未运行。
- packed-build，树 9b40c97226d71b0d8033342cd19c10c60d48af83：**363 passed / 5 failed**，exit 1。
  FP32 SSD Read−/+ 的 Python/native input.2 根差 2.13e-6 / 4.83e-6；PDG Read− key.out_bias 差 1.91e-6。
  `.local/diagnose-packed-order.py` 与该 job 的 order-diagnosis.txt 定位为行批投影的浮点顺序；
  列批和单行列向量变体通过 SSD 两快照及 PDG 全 cut，行批/bmm 保留失败。
  正式 Python/C++ 统一权重矩阵乘输入列批；未改公式、输入、容差、dtype 或裁剪梯度。

原 status/source/log 保留在 `artifacts/jobs/core5-<job>/`，新通过不覆盖旧失败。

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
