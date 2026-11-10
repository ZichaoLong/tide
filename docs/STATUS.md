# 当前交接 — 默认分量窗口与块内批量继续实施（2026-11-10）

分支graph-execution-foundation；数学0770593。用户已批准完整计划、全部320 CPU与
空闲/有余量共享NPU；资源以尽快得到结果为准，舍入差异有限诊断后分列推进。
参考与下游只读；不使用子代理。两份未跟踪用户handoff保持原样。

## 已交付与验证

- aab8cb7联合KV准备已推送；证据1d5d598已推送：CPU80、NPU相关33，
  Python/native安装消费者三步独立检查/新进程精确恢复通过。
  FC16/B8/L128完整步CPU35.486秒、NPU96.776秒；原任务独立数值仍未验。
  正式导航[evidence/joint-kv-preparation-20261110.md](evidence/joint-kv-preparation-20261110.md)。
- 默认区域SCC窗口调度：Python/native/standalone接线、实际分量计数、执行v3与旧greedy身份复现。
  五记忆/变长B×L/一般端口/反馈/历史/独立根/SGD/精确恢复已有开发检查。
  window-b新构建完成，494通过；唯一失败为旧测试仍要求PDG auto拒绝prefill，已按新契约修正。
  定向重验与正式固定资格分列，不以开发通过代替设备/SDK资格。

## 当前作业与资源

无活动长作业。gef-joint-prefill-window-a失败于缺少声明头，未运行测试；原记录保留。
gef-joint-prefill-window-b已终态（原测试断言1失败/494通过），冻结源码
artifacts/joint-prefill/source-window-b，原始tree身份window-b-snapshot.json；
新构建artifacts/joint-prefill/build-window-b。全320核、512G/3600秒，未与重型计时重叠。
固定提交资格将复用内容hash一致的build-window-b，安装新wheel，CPU新相关/NPU/SDK分别验收。
现有CANN9.2私有环境env-block-a；新SDK必须独立于Python运行库构建。

## 正在推进与下一步

1. 提交默认窗口调度后冻结资格，NPU用有界单卡准入；小型代表先独立核对，有限舍入保留原阈值。
2. 工作区另在开发KV原始时间切片和输入有限检查批量化，不能混入调度资格来源。
   随后批量State结构分支元数据；全部保留中间状态独立根与None/零。
3. 扩展五记忆、标准层链及大宽度/多节点HB/LH/sum/反馈代表，CPU/NPU/mixed/resident分列。
   标准层链需要同公式直接基线，不能把外部LM路径冒称Tide调度已优化。
4. 最终原FC16成本及小profile回答实际剩余碎片；新经验进入performance-lessons。
   前述目录/历史尺度不等于新默认真机通过，GPU按用户授权延后。

下一命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
长作业源码/脚本/日志均在artifacts/joint-prefill与artifacts/jobs，完整待办只见ROADMAP。
