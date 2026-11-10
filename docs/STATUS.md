# 当前交接 — KV整序列反向固定资格（2026-11-10）

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准持续实施和推送；320 CPU、空闲及有余量共享NPU可使用。浮点超差有限诊断后分列推进。
参考仓库/下游只读；不使用子代理。两份未跟踪用户handoff原样保留，不纳入提交。

## 已交付与当前开发

- 默认分量窗口、联合KV准备、批量守卫及原始时间切片已交付至0915781；证据见
  [分量窗口](evidence/component-window-20261110.md)。
  guards固定终态尚待汇编证据；已测成本CPU33.478秒/NPU86.324秒，一次完整返回观察，不等于原任务数值资格。
- 当前改动：未公开中间State时，KV整序列VJP批量回收前缀伴随；trace=True保留中间状态因果根。
  Python/C++开发143项通过（seq-b.xml），覆盖三个Read、FC/HB/反馈、独立根、暖态、共享、SGD与精确恢复。
  公共eager_runner(trace=False)额外6项FP32检查通过。旧seq-a多余头文件构建失败原样保留。
- 代表配置生成器与有界GDN/HB数值诊断准备中；目录不等于运行通过。

## 活动作业和下一步

当前无运行的本项目服务。所有已有服务已终态；seq-b源为0915781加冻结overlay，
build-seq-b，日志artifacts/jobs/joint-prefill-seq-b/task.log，143通过不是固定提交资格。
即将提交本实现，在干净worktree分别安装验收CPU/NPU及独立SDK，记录真实采用的joint_state_vjp_events。
随后有限比较原FC16完整/最小返回；补大宽度FC、多节点HB/LH、sum、循环及标准层链、混合/多卡/resident。
长任务固定源码、background.slice、资源/时长有界；构建与重型计时不重叠。

下一命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
唯一待办/停止条件见[ROADMAP](ROADMAP.md)。GPU按用户授权延后；NPU已有严格超差保留，不无限阻塞其他工作。
