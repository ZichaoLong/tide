# 当前交接 — 2026-10-30

历史清理、远端发布、本地切换和临时材料退役已完成。后续仍在`graph-execution-foundation`开发。
原历史在远端`graph-execution-foundation-old`冻结于`83d384a4a5efb9df38ce77c395c2d9c5222b4e0e`。
本地仅跟踪新分支；目录约39MiB，只保留`.git`和跟踪文件，没有活动作业或待恢复的旧实验队列。
旧本地检出及本轮临时源码、私有环境、构建、原始日志已删除，没有另存一份大仓库。
数学仍固定`0770593838a37c790a568226930567ed60f99b83`；公共数学/API/容差和原数值、异步、硬件限制不变。

## 交付与证据

- [迁移规则与映射](history-migration.md)：716个原提交全部保留，181个改号；只从新历史移除40,407,191字节归档。
- [独立资格](evidence/history-migration-20261030.md)：固定`b0969fc`，逐提交核对全部其他路径/模式/blob、父链、元数据和消息。
  新CPU构建160线程，13 CTest、FP32/FP64 smoke、177定向检查、4消费者单元/8安装训练恢复/4拒绝通过。
- [原子发布、干净新检出与最终清理](evidence/history-publication-20261030.md)：精确lease原子保存旧分支、发布新分支，独立远端核验；
  大归档blob在新本地对象库中实际缺失，替换前保护原文件/refs。两次网络失败原样保留，后续成功未覆盖它们。
- 历史报告/JSON/二进制及依赖锁身份保持。当前导航使用新源码位置；映射不是重新运行历史实验。

## 接续

```sh
git status --short --branch
PYTHONDONTWRITEBYTECODE=1 python scripts/status.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_upstream.py
PYTHONDONTWRITEBYTECODE=1 python scripts/resolve_history_commit.py 35d6358 --json
```

新clone按[development](development.md)使用`--single-branch --branch graph-execution-foundation`；
不要把旧分支fetch/merge回日常检出。旧工作区先保存私有修改再另建新clone，不能直接pull两个历史。
环境与库按需重建；旧归档按[迁移规则](history-migration.md)单独取回，正常开发无需下载。
本次SSH认证HTTP CONNECT已实际读取和克隆成功，现行本机netpath要求认证；相关入口与无凭据helper见发布证据。
没有修改全局Git/SSH/代理配置；其凭据属于用户私有配置，不是仓库依赖。
问题→定义/源码/输入→测试/证据从[reproduction](reproduction.md)、[semantic-map](semantic-map.md)、[architecture](architecture.md)进入。
原性能任务不恢复；当前限制与停止/重启条件只维护在[ROADMAP](ROADMAP.md)，下游升级按[协作规范](downstream-contract.md)另行验收。
