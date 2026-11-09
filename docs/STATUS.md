# 当前交接 — 默认prefill增量已验证（2026-11-10）

分支graph-execution-foundation；数学0770593。核心实现7fe97f2，SDK查询策略checkpoint守卫5f563f6。
本轮固定资格、四项成本及独立profile均终态；证据/工具随本交接提交归档，未启动后继优化。
实现、证据均在授权分支；远端同步状态以git status --short --branch核对。
两个用户未跟踪handoff保持原样；参考/下游只读，不使用子代理。

## 交付与事实

- 公共默认auto按能力选择packed-v2；跨时间区域批、原始KV槽批量gather、独立VJP与策略恢复守卫已实现。
- 固定安装CPU397 passed、2既有dense-v1 xfail；NPU26实际设备+7CPU辅助通过。
  Python/native及standalone的小型独立训练、optimizer、新进程精确恢复分别通过。
- 原FC16/B8/L128完整步：CPU旧62.177→新38.953秒，NPU旧237.119→新101.756秒。
  每设备内候选/Full/消息总数相同；原任务逐张量/路由/梯度及学习验收未运行，不能称严格等价吞吐。
- profile显示区域批512→4已兑现，私有KV写投影仍12,288次/最大批4；NPU仍慢于CPU。
  含预热设备峰值从6059.1升至6536.0MiB；原FP32失败和其他平台/尺度边界保留。
- 证据及重建命令：[固定报告](evidence/prefill-default-20261110.md)、[机器记录](evidence/prefill-default-20261110.json)。
  默认契约weighted-dense-query；代表回归prefill-regression；通用经验performance-lessons-prefill。

## 作业终态

无活动任务。gef-prefill-default-cost通过，MainPID=0、ControlGroup为空；其余本轮服务亦无运行后代。
固定cwd为artifacts/prefill-default/fixed-7fe97f2，SDK fixed-sdk；日志/status在artifacts/jobs/prefill-default-*/。
成本及profile在artifacts/prefill-default/cost-*/result.json和profile-npu-auto/；
安装资格在qualified/{cpu,npu,sdk-cpu,sdk-npu}/。旧构建失败、pip取消、Ninja失败均保留，未改公共环境。
原始产物只是本机来源提示，干净checkout从报告生成输入、重建，不要求恢复旧运行目录。

## 下一步

1. 接续先读固定报告及ROADMAP；本轮已核对源码/构建/驱动hash、13份作业终态、文档链接与目录可列出性，无需重复已通过资格。
2. 后继只按ROADMAP的TL-REQ-004行：先检验现有序列私有投影复用和必要元数据批量校验，再决定新的有限成本对照；不重跑旧全量矩阵。
3. 下游在新私有安装使用默认auto，独立验原任务及应用恢复，再决定升级依赖锁；本轮没有跨仓库写入。

接续命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
报告中的具体构建/检查/成本命令均使用新输出目录和匹配固定源码；当前不需恢复任何旧服务。
