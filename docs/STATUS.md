# 当前交接 — 2026-10-30

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
第二轮磁盘清理完成，执行基线12d3e16f1513d0a6f8be7de2720f0096f68b44d7；无子代理、无活跃作业。
仅修改交接、保留规则、开发导航与证据；未改数学/运行代码、参考仓库、公共环境或旧失败结论。

## 本轮交付

35项全部退役65.852535GiB：30份历史源码副本约1.928GiB，2份静态材料38.117GiB，3份旧checkpoint25.807GiB。
仓库约88.36→22.50GiB；两轮累计清单退役124.73GiB，初始约147.24GiB。
原81个历史worktree和本轮临时runner均已移除；Git目前只登记主工作区。
未提交差异不宣称已全部合并；28份脏树已与完整归档逐项核对后按用户明确授权退役。

28份源码的归档仍在artifacts/cleanup-round1-20261030-v2/preserved-records.tar.gz，约1.61MiB。
518份旧作业记录、66份相邻配置/结果、失败日志、小型反例及原失败结论保持。
大checkpoint的旧精确状态已不在线，诊断重启须重新生成；ROADMAP已同步这一保留边界。
证据入口：[第二轮清理](evidence/artifact-cleanup-round2-20261030.md)及[机器凭据](evidence/artifact-cleanup-round2-20261030.json)。
第一轮历史结果仍见[原报告](evidence/artifact-cleanup-20261030.md)。

## 当前开发入口

预编译库约3.28GiB，已分离到artifacts/retained-builds/，详情见[保留构建](artifact-retention.md#本机保留构建)：

- core5-checkpoint-qualified/build/：b255 CPU core/client及standalone NPU SDK链。
- core5-compact-fiber-qualified-v2/build/：04b Python/SDK常驻基线。
- core5-independent-qualified/build/npu-python/：bdd7f6c旧Torch2.12 NPU core基线。

旧artifacts/worktrees/<名称>/build仅为兼容链接，没有对应源码或Git登记；不在旧CMake缓存中增量重建。
新构建按[development](development.md)使用固定源码和新build目录，核对manifest及环境；不混用版本。
.local/venv-core5、主源码、公共工具链保持。搬迁后CPU原生smoke、显式公共加载及6项接入检查通过。
这只是开发链复核；没有新增NPU/GPU或全矩阵资格，也没有重跑历史实验。

## 作业终态与接续

- cleanup-round2-plan-20261030：passed/0，04:58:17—04:59:32 UTC。
- cleanup-round2-apply-20261030：passed/0，04:59:56—05:01:38 UTC；35项全部完成。

独立background.slice，Nice10/CPUQuota200%/MemoryMax2GiB，900/1800秒上限；只读runner已在终态后移除。
状态/日志在artifacts/jobs/<任务名>/；plan/result、程序副本、launch、smoke及finalization在artifacts/cleanup-round2-20261030/。
归档、构建文件身份/87项内容hash、历史作业记录与相邻记录核对通过。第一轮预检failed/1记录仍保留。
本轮提交内容仅上述文档与凭据；提交/远端是否同步以git状态核对，不把文档当作推送完成证明。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
git worktree list
```

无待恢复历史队列；按用户后续任务接续。详细失败经验治理另行安排；现有约束和停止/重启条件仅由[ROADMAP](ROADMAP.md)维护。
多实验仓库协作从[协作规范](downstream-contract.md)和[安装示例](../examples/consumer_core5/README.md)进入；
既有安装资格、配置覆盖、性能和硬件边界不因清理改变。
