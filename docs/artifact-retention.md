# 历史产物与可重建边界

2026-10-12 用户确认：旧实验默认不必复核，保留必要记录即可。
本规则替代过去“所有被报告引用的外部文件长期在线保留”的要求。
不改变已经发生的通过、失败、取消、编译或设备验证结论。

## 必要记录

Git 保留源码、测试、构建/验证/测量入口、固定教材快照，以及已审阅的报告。
退役记录另外保存旧运行的配置/结果、源码身份、共享资源观察、任务终态和
可用的失败诊断。历史绝对路径作为来源标签保留，不是新线程的依赖路径。
必要记录应可由仓库取得；大型原始 trace、缓存、二进制和重复源码副本无需常驻。

2026-10-12的退役归档直接随 Git 保存，约 38.5 MiB；索引、SHA256 与数量见
[退役凭据](evidence/artifact-retirement-20261012.json)。635 个旧工作副本的基线
提交都在当前分支历史中，431 份相对 HEAD 的补丁和 5,049 个未跟踪源码文件
按原字节保留（包括构建模板），不必保留重复 checkout。
配置/结果小记录保留原文，大结果保存明确标记的摘要及原文件哈希，日志只保留尾部。
原始张量、完整 profiler 时间线和构建二进制不在归档保证范围；旧报告中的
逐项观察量不一定可从摘要展开，需要时应重新生成并验证。

旧 CPU/NPU 性能只是当时源码与环境下的观测。共享资源争用不可原样重放；
即使同源码重新编译，也不保证二进制哈希、内存峰值或浮点轨迹逐位相同。
新结果使用新配置/环境/源码身份，不改写旧通过数或严格失败。

## 按需读取旧记录

正常接续无需读取归档。需要定位某条历史来源时，在仓库根目录执行：

```sh
python scripts/read_retired_record.py --list-prefix /mi/data2T/zlong/tide-accelerator/sources/native-dev01
python scripts/read_retired_record.py \
  --path /mi/data2T/zlong/tide-accelerator/sources/native-dev01:git-diff-HEAD \
  --output /tmp/gef-retired-native-dev01.patch
```

工具先校验整个归档；`--path` 单独使用只显示元数据，`--output` 校验并提取
一个记录，拒绝覆盖已有文件。不执行旧脚本，也不向归档中的绝对路径写回。
源码复现需要相应 Git 基线、补丁、未跟踪文件和匹配依赖；先审阅并在新目录组装。
历史绝对路径是查询标签，不能直接当作本机现在存在的输入。

## 重建

- 普通开发从 [development](development.md) 和固定 Git 提交开始，重建本机库。
- 构建 manifest 由新构建生成；删除旧库后，loader 应拒绝缺失或不匹配的显式路径。
- CPU/NPU/GPU 的工具链、驱动和 environment modules 是外部软件依赖，未纳入产物清理。
- 工作负载由 `prepare_execution_flow.py` 等仓库入口及声明的配置生成；
  新实验使用新目录和有限预算，不依赖旧 launcher 的绝对路径。
- 原始 LH 对照是额外门禁，需按 [LH selector](lh-selector.md) 从原仓库建立固定快照；
  普通 Tide 开发不需要原始 LH。只读参考仓库不在本次清理范围。

## 后续清理

先检查任务状态、实际目录、待保留记录与 dry run，再删除明确的项目产物。
保留活动任务输入、未保存的用户修改、不可再生输入和新发现问题的最小复现。
清理失败时如实保留部分完成状态，不把计划当成已回收空间。
清理产生的记录不重新开启历史队列，也不要求重跑历史全矩阵。

工作树干净只表示无未提交的非忽略修改；另外检查HEAD是否在授权分支历史中、忽略目录中的实际数据及使用依赖。
已提交基线在历史中不表示未提交实验差异已合并，也不表示所有问题已解决。需要保留差异时可归档完整补丁与未跟踪源码，
不要求永久保留整个checkout。删除工作树使用Git管理登记；历史报告引用不是永久在线依赖。

运行任务的外层passed可能只表示调查流程完成，内部用例仍可失败；不能仅按外层状态批量丢弃失败材料。
`scripts/clean_artifacts.py`只覆盖旧平铺作业布局，不扫描当前jobs/worktrees，不能作为完整退役清单。

2026-10-30第一轮[清理凭据](evidence/artifact-cleanup-20261030.md)记录58.876GiB退役及当时的保留项。
用户随后明确授权退役未提交的历史临时副本及大材料；[第二轮凭据](evidence/artifact-cleanup-round2-20261030.md)
记录30份源码副本和5个大文件的移除，约65.853GiB。全部历史worktree的已提交HEAD在授权分支历史中；
28份未提交差异与完整补丁/新文件归档再次核对后退役，不宣称这些差异均已合并，也不把脏树一律解释为失败。

本地源码归档仍为`artifacts/cleanup-round1-20261030-v2/preserved-records.tar.gz`，约1.61MiB；
身份、成员核对及恢复边界见第二轮证据。它属于受保留的本地材料，未新增原始归档到Git。
配套配置、种子、结果、错误日志、小型反例和作业终态保持。失败现场无需永久整目录保存；
删除大checkpoint意味着旧精确状态不再在线，重启诊断须重新生成，跨环境不承诺逐位恢复。
失败经验的进一步整理另行安排，不因清理改写已有结论或重新开启旧任务。

## 本机保留构建

当前只有主工作区被Git登记。仍可用的构建已与历史源码副本分离，共约3.28GiB：

| 相对仓库根目录的路径 | 原身份与用途 |
| --- | --- |
| `artifacts/retained-builds/core5-checkpoint-qualified/build/` | b255e27；CPU core/client与standalone NPU SDK链，实际有效resident为resident-sdk-v3 |
| `artifacts/retained-builds/core5-compact-fiber-qualified-v2/build/` | 04b5215；Python-owned与SDK常驻基线 |
| `artifacts/retained-builds/core5-independent-qualified/build/npu-python/` | bdd7f6c；Torch2.12/Torch-NPU2.12原core基线 |

原`artifacts/worktrees/<上述名称>/build`只剩指向这些构建的兼容链接，用于旧RPATH及路径引用；
父目录没有旧源码或`.git`登记。新任务使用retained-builds路径并核对manifest/二进制hash与环境。
迁移保留文件身份与运行库hash，CPU接入另复核；不新增NPU/GPU资格，也不把旧NPU库冒称当前全源码构建。
这些是预编译产物，CMake缓存仍记录历史源码路径，不能直接在搬迁目录增量重建。
需要修改或重建时按[development](development.md)从固定源码在新的build目录构建；旧工作树无需恢复。
`.local/venv-core5`、主源码和公共环境保持。

2026-10-12退役的具体目录、必要记录位置、校验值和实际回收量记录在
[evidence/artifact-retirement-20261012.md](evidence/artifact-retirement-20261012.md)。
