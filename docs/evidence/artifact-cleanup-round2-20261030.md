# 第二轮磁盘清理 — 2026-10-30

用户在[第一轮说明](artifact-cleanup-20261030.md)后，明确授权清理旧大材料、失败checkpoint及历史临时worktree，
包括已归档的未提交实验差异。本轮执行基线`12d3e16f1513d0a6f8be7de2720f0096f68b44d7`，数学固定0770593。
[机器凭据](artifact-cleanup-round2-20261030.json)列出精确路径、提交、内容hash、保留检查和实际终态。
没有修改数学/运行实现、参考仓库、公共工具链或历史验证结论。

## 实际结果

35个清单目标全部退役，删除前分配块合计 **65.852535GiB**。
承接第一轮的约88.36GiB，本轮临时runner结束并移除后，仓库约 **22.50GiB**。
两轮清单累计退役约 **124.73GiB**，相对最初约147.24GiB。Git登记、日志和新凭据使净变化略有差别；
不以共享盘df变化冒充本项目回收量。

| 退役范围 | 数量 | 删除前占用 |
| --- | ---: | ---: |
| 历史源码worktree，不含单独保留的构建 | 30 | 1.928013GiB |
| 静态参数/合成输入二进制 | 2 | 38.117100GiB |
| 旧测试checkpoint | 3 | 25.807423GiB |

之前说明的60.31GiB不含worktree，也不含配套FC静态材料3.61GiB；本轮将它们一起处理。
Git目前只登记主工作区，原81个历史worktree和两轮临时runner均已移除。
三组仍可用的构建共约3.28GiB，已移至`artifacts/retained-builds/`；路径和身份见
[保留构建](../artifact-retention.md#本机保留构建)。旧worktrees路径只保留三个build兼容链接，父目录没有历史源码或`.git`。

## 退役材料的含义

- 34.50GiB `matched2048.tensors.bin`：2da3f2c的约9.262B参数用例，固定种子7生成的FP32初始参数与合成输入；不是训练成果。
- 3.61GiB FC `fixture.tensors.bin`：e50b6d8的FC约1B同类静态材料；配套图、模块、共享绑定、配置和hash的JSON仍保留。
- FC/HB约1B CPU `resume.pt`：e50b6d8的参数、图状态、history/pending/时钟和SGD槽。各自精确续接通过，但独立核对分别有2个SGD槽坐标、59个梯度坐标失败；不称文件损坏。
- FC约1B NPU `resume.pt`：b255的5.25GB预算复验文件，实际保存和加载通过，原owner下一窗非有限梯度；完整续接与独立核对未完成。

所有大文件删除前均计算SHA256，并在删除预检再次核对；静态材料还与旧记录hash一致。
精确文件清单/大小/hash和配套manifest身份在机器凭据中。配置、结果、错误日志仍原位保留，
已有[参数匹配说明](core5-performance-transfer-20261028.md)、[自然规模说明](core5-natural-study-20261029.md)和
[文件预算复验](core5-checkpoint-budget-20261029.md)保留固定源码、生成/运行入口与原结论。

普通新实验不依赖这些旧二进制。重新调查旧问题需从相应源码、配置、种子和匹配环境重新生成材料；
删除checkpoint后无法直接取回当时精确状态，也不保证跨环境逐位重建。ROADMAP已同步这项保留边界，严格标准和停止/重启条件保持。

## worktree和失败经验

30个已提交HEAD都是本轮基线的祖先；28个worktree包含未提交差异，另外2个干净。
HEAD已在历史中不证明未提交差异已被采用，脏树也不自动等于失败。
按本次明确授权，无需为了退役而把每份实验补丁合并到公共实现。

删除前再次验证第一轮源码归档：

- `artifacts/cleanup-round1-20261030-v2/preserved-records.tar.gz`，1,692,495字节，约1.61MiB。
- SHA256：`1ebb2d173889de3a7d3399620f745e9badc91df4c54deefeee4c5c83deaf5dcd`。
- 全档803个成员；本轮对28份脏树的56份完整工作区/暂存区补丁与182份未跟踪源码逐项核验，均与现有副本相同。
- 各基线、git状态、补丁hash和新文件hash见机器凭据的`retired_source_records`。

归档成员使用`artifacts/worktrees/<名称>/diff-HEAD.patch`、`diff-index.patch`及原新文件路径。
如确需恢复某一尝试，先校验归档，在新目录检出所记HEAD，审阅后应用工作区补丁、用暂存区补丁恢复index，再恢复未跟踪文件。
不要直接运行归档里的历史启动器。此归档作为受保留的本地材料存在，未把新的原始归档加入Git。

失败经验可保留为结论、配置/源码身份、错误证据和必要的小型复现；不需要长期保留完整失败目录。
本轮只清理并连接已有记录，详细失败经验治理另行安排。旧workers2、FP16、异步及实际操作数等小型材料不在删除范围。

## 保护检查与开发链复核

518份原作业状态文件hash保持，包括第一轮预检失败；66份相邻配置/结果文件按预检身份保持，
其中不超过16MiB的文件另核对内容hash。没有整目录删除jobs，也没有改写旧失败为通过。
三组构建移动前后全部166,531个目录/文件条目的相对路径、inode、大小、mtime和mode一致；
另对87份运行库/manifest/loader等文件逐项核对SHA256，旧路径兼容链接解析到新位置。
这是保留检查，不宣称为全部构建中每个文件都重新计算内容hash。

搬迁后，在固定12d3e16源码上使用新CPU构建目录完成：

- 原生C++ CPU FP32 smoke通过。
- 公共`load_native`通过manifest/ABI/实际二进制hash校验并加载新路径。
- native Settle端口/平行边/来源/别名/梯度两dtype与4项下游接入检查，共6项通过，2.15秒。

日志、smoke JSON与JUnit随任务记录保留；未运行历史性能矩阵或新增NPU/GPU资格。
构建保持原版本：b255 CPU/SDK、04b常驻、bdd7f6c旧Python NPU core；不同版本不能混用或冒称当前完整资格。
旧CMake源码路径已退役，后续修改需从固定源码在新目录重建，不在搬迁缓存中继续增量构建。

## 执行与终态

固定只读runner及任务程序，独立background.slice、Nice10、CPUQuota200%、MemoryMax2GiB；计划限900秒、删除限1800秒。
清单SHA256为`add883804a6dcd4779992d9ee7f4c1cd0c84c0c1ad624d54c473d023b4900e90`。
先核对归档、祖先关系、忽略数据、保留构建、活跃作业/文件及目标内容；随后逐项持久记录迁移与删除。

- `cleanup-round2-plan-20261030`：passed/0，04:58:17—04:59:32 UTC。
- `cleanup-round2-apply-20261030`：passed/0，04:59:56—05:01:38 UTC；包含完整预检，实际迁移/删除/后检约28.10秒。

状态与日志在`artifacts/jobs/<任务名>/`；精确plan/result、程序副本、launch、smoke、finalization在
`artifacts/cleanup-round2-20261030/`。本轮无失败任务，第一轮原失败保持。
终态后移除临时runner并检查实际路径和Git登记；无活跃作业，不恢复旧队列。
