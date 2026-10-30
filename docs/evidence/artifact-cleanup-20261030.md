# 第一轮磁盘清理 — 2026-10-30

用户授权先清理旧构建、可退役工作树、已有报告的原始trace及已通过的LH大checkpoint；
34.5GiB参数材料、失败checkpoint和未提交源码留待说明后另行决定。
基线为`cc89ca89e10c149fd6b658b2c5bd314a1c7e5f6b`，数学固定0770593，未修改运行代码或数学定义。
[机器凭据](artifact-cleanup-20261030.json)记录完整清单、Git归属、保留身份、实际终态和检查结果。

## 实际完成

清单84项全部移除，按删除前分配块合计 **58.876 GiB**。
仓库由只读调查的约147.24GiB降至约 **88.36GiB**；没有清理同盘其他项目、工具链或只读参考仓库。
临时runner另在任务结束后移除。Git登记、新记录及临时目录使净变化与候选合计略有差别；共享盘df变化不充作本项目回收量。

| 删除范围 | 数量 | 删除前占用 |
| --- | ---: | ---: |
| 干净且HEAD已在本分支历史中的worktree，含其旧构建 | 51 | 37.279 GiB |
| 保留源码副本中的旧build目录 | 18 | 6.320 GiB |
| SHA256已出现在审阅证据中的原始trace.json | 14 | 4.160 GiB |
| LH约1B、已完整通过的resume.pt | 1 | 11.117 GiB |

只移除上述精确路径，没有整棵删除artifacts/jobs。原515份作业终态记录逐一hash未变；
配置、结果、错误日志、源码归档、诊断材料及其他checkpoint保持。
删除前另保存并逐字节核对747份元数据/未跟踪源码和56份补丁，压缩档约1.7MiB，
位于`artifacts/cleanup-round1-20261030-v2/preserved-records.tar.gz`，其SHA256见机器凭据。
该任务档属于受保留的本地材料，未作为新的原始归档提交Git；现存28份脏树源码也仍在原位。

## worktree是否已合并

81个历史worktree的HEAD **全部是当前授权分支基线cc89ca8的祖先**。
这证明这些已提交基线已在`graph-execution-foundation`的历史中，没有独立遗留的worktree提交。
它不证明未提交实验差异已被采用，也不代表ROADMAP中的所有功能/数值问题均已解决。

其中53个worktree干净，28个含未提交修改/新文件。本轮移除51个干净副本；
另外2个承载保留的开发构建。最终有主工作区加30个历史worktree，其中28个仍保留原未提交源码。
“干净”本身还不足以删除：预检另外核对提交归属、忽略文件、使用中的文件/进程和构建依赖。
清理使用git worktree remove同步移除登记，没有删除Git历史或强推分支。

保留的开发入口：

- `artifacts/worktrees/core5-checkpoint-qualified/build/`：b255的CPU core/client和standalone NPU SDK链。
- `artifacts/worktrees/core5-compact-fiber-qualified-v2/build/`：04b5215的Python/SDK常驻基线。
- `artifacts/worktrees/core5-independent-qualified/build/npu-python/`：既有Torch2.12 NPU core基线。
- 主工作区及`.local/venv-core5`、公共环境保持。

87个保留的构建文件/manifest在删除前后逐项hash相同。不同基线不自动获得当前源码资格：
CPU和最新SDK C++源对应b255；较早Python NPU二进制仍须按自身source/hash及[构建规范](../development.md)使用，不能混用版本。
清理后原生C++ smoke实际CPU/FP32通过，另6项CPU接入检查通过：native Settle独立端口/别名/梯度FP32及FP64，
以及4项下游锁/原子记录/checkpoint检查。没有重跑历史性能矩阵或新增NPU/GPU资格。

## 仍保留的大文件是什么

| 材料 | 内容与用途 | 对新实验/开发的意义 |
| --- | --- | --- |
| `matched2048.tensors.bin`，34.50GiB | 2da3f2c的约9.262B参数性能用例；固定种子生成的FP32初始参数和合成输入，配套JSON保存图/模块/共享绑定。导出时没有执行前向 | 不是训练出的模型，也不是公共库运行依赖。可按固定源码、配置、种子和匹配环境重新生成；需要时间和较大内存 |
| FC/HB约1B CPU checkpoint，合计20.91GiB | 短测试的参数、图状态/history/pending、时钟及SGD槽。两者自身逐值恢复通过，随后与独立参考比较时有少量梯度/槽差异 | 文件本身未被证明损坏。主要方便调查旧数值差异；新实验无需加载 |
| FC约1B NPU checkpoint，4.89GiB | 原>4GiB文件预算复验中保存并成功加载的实际状态；随后原owner下一窗出现非有限梯度 | 方便从当时状态继续诊断；不是可用训练成果，也不是基座启动依赖 |

这些大型材料共约60.31GiB，另有配套FC静态fixture约3.61GiB。小型workers2、FP16、异步等反例也继续保留。
普通后续实验依赖公共实现和自己的配置/数据，不依赖这些旧文件。
它们适合在保存必要源码身份、配置、种子、环境、结果/错误和小型反例后退役；
删除checkpoint的代价是失去当时已保存的精确状态，调查旧失败时需重新运行，跨环境不保证逐位重建。
本轮按授权边界保留它们，没有将“可再生”自动解释成当场删除。

未提交worktree也不必永久整份常驻：若未来退役，应先确保基线、完整工作区/暂存区补丁、
未跟踪源码及必要输入有受管理的一份归档；不能仅因HEAD在历史中便丢掉未提交差异。
这与把所有旧实验补丁合并到公共实现是两件事。

## 执行与保护检查

固定cc89ca8 runner和只读任务程序，独立background.slice、Nice10、CPUQuota200%、MemoryMax2GiB；
清单任务限600秒，删除任务限1800秒。清单SHA、任务程序SHA和归档SHA在删除前绑定，拒绝路径越界、
错误清单hash、目标漂移、活跃任务/文件及源码修改；删除前后核对保留项。
工作树/构建/trace/checkpoint精确列表可从机器凭据查看，不应把该历史清单原样用于新一轮删除。

- `cleanup-round1-plan-20261030`：passed/0，清单及归档完成。
- `cleanup-round1-apply-20261030`：failed/1。路径拼接TypeError发生在预检，未开始删除；当时84项均仍存在，没有删除result文件。
- `cleanup-round1-apply-v2-20261030`：仅修正该表达式，清单目标和保护范围保持；passed/0，84项全部完成，约104.57秒。

原失败未改为通过。源码差异/未跟踪文件内容、保留构建hash及原作业记录一致；
大型保护文件另核对inode/大小/mtime，未为此次清理重读约60GiB原材料做内容hash。
单位回收后依据持久status、逐项result和实际路径验收；本轮临时runner也已删除，没有活跃任务。
原绝对路径继续作为历史来源标签，后续开发按[开发入口](../development.md)重建，不恢复旧队列。
