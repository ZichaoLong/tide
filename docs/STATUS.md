# 当前交接 — TL-REQ-004 首个优化增量交付（2026-11-10）

显式 `query_execution="dense-v1"` 的实现、固定资格和有限性能流程已结束。
[交付报告与下游命令](evidence/dense-query-tl-req-004-20261110.md)、
[机器证据](evidence/dense-query-tl-req-004-20261110.json)、
[公共域/数学对应](weighted-dense-query.md)是本次成果入口。
数学固定0770593；实现/包/native固定`75a502790985044b12125bee83d7f0dddfa39079`，
诊断驱动`26ca1b6b32c43adfb78434b6402498dc4c53e40b`，两者Python/C++源码相同。
默认legacy、严格阈值、图v13/checkpoint v5保持；新路径有独立执行身份。

## 实际结果

- CPU固定资格25 passed/2数值xfail；NPU15 passed包含13实际设备和2辅助。
  CPU FP32独立evidence根约1.044e-6严格失败保留；相同材料FP64对照支持该局部舍入解释。
- FC/HB/反馈PDG三步AdamW、完整记录/独立根、共享/None/零、pending/cut和换owner精确恢复通过。
  四个CPU/NPU×Python/native安装消费者的独立三步及新进程3步vs2+1恢复通过。
- 原FC16/B8/L128、FP32/native同安装单次完整步：CPU62.839→45.162秒（16.30→22.67 token/s），
  NPU mixed-c252.922→183.549秒（4.05→5.58 token/s）。未达约32.4 token/s，NPU仍慢于CPU。
  Full/消息计数及裁剪前梯度范数不同；这些是有限成本观察，不是大图严格等价或学习资格。
- B2/L8 NPU profile确认FFN由1–2行变为2–15行，但device Index/IndexPut仍各1170次。
  下一布局/数值处理的停止与重启条件只见[ROADMAP](ROADMAP.md)，经验见[performance-lessons](performance-lessons.md)。
- 下游0439efd只读；未更新其依赖、安装、运行或验收。两份用户handoff未跟踪、未修改、不纳入提交。

## 作业终态与产物

`gef-tl-req-004-{build-a,check-b,qualify-c,cost-d}`均passed/exit0，
MainPID=0、ControlGroup为空且cgroup不存在；两个NPU队列completed/return_code0，无活跃任务。
外层passed不改变内部2项严格数值失败。成本六子流程均completed，最后服务于
2026-11-09 16:28:08 UTC终止；全部日志/状态hash见机器证据。

本机原始材料在`artifacts/tl-req-004/`（约2.1GiB），日志/状态在
`artifacts/jobs/tl-req-004-*/`。source-c/source-d为冻结源码；build-c、wheels、installed-env
为本轮私有安装。公共环境ascend/dev-workspace-9.2.0保持只读。
资格构建用全部320 affinity CPU、512GiB；性能CPU1/NPU2 host线程、scheduler1、96GiB，
服务≤3600秒，项目重型工作串行，外部主机争用已记录。不得恢复旧队列或把产物路径当新安装默认。

## 下一次接续

1. `git status --short --branch`、`python scripts/status.py`；先审阅上述交付，不重跑本轮已完成矩阵。
2. 下游在新私有环境按报告构建固定wheel/native，`tidegraph lock create/verify`；
   调用`eager_runner(..., query_execution="dense-v1")`或配置`execution.query_execution`。
   `python examples/consumer_core5/dense_query.py --device cpu --client native --native-library BUILD --output-dir NEW`。
3. 下游自己的原任务check/恢复、有限学习验收后再决定升级锁；基座交付不自动代表实验已验收。
   需要新优化时从实际Index/VJP热点或数值首差提出具体方案，按ROADMAP有界重启。

正式导航和证据随本交付提交；当前实现无未提交变更。两个用户未跟踪文件
`docs/tide-learning-lab-TL-REQ-003-handoff.md`与`docs/tide-learning-lab-TL-REQ-004-handoff.md`继续保留原字节。
