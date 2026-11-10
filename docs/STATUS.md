# 当前交接 — 默认分量窗口与块内反向（2026-11-10）

分支graph-execution-foundation；数学0770593。用户批准持续实施及推送，320 CPU、空闲与有余量共享NPU；数值舍入有限诊断后分列推进。参考/下游只读，不使用子代理。两份未跟踪用户handoff保留原样。

## 已交付

- aab8cb7联合KV准备、1d5d598证据已推送；原FC16/B8/L128 CPU35.486秒、NPU96.776秒。原任务完整独立数值未验。
- da2f41b默认区域SCC窗口；b148ba1修正跨设备测试器。固定CPU391通过；NPU五记忆28通过/2 GDN-HB evidence根超差，另33相关通过。两安装消费者及SDK CPU64/32、NPU32均独立训练/新进程精确恢复通过。见evidence/component-window-20261110.md。
- 3f4ade5原始KV时间切片、输入finite及结构分支合批。开发1313全部通过；固定安装CPU146通过，NPU/消费者及一次成本、profile正在审阅。

## 当前作业

gef-joint-prefill-fixed-guards：固定3f4ade5，artifacts/joint-prefill/fixed-guards；build-guards-a核对C++内容hash复用，新env-guards安装。MemoryMax192G/3600秒；日志artifacts/jobs/joint-prefill-fixed-guards/task.log。CPU完整步33.478秒；NPU完整步86.324秒，单次观察，独立数值/争用分列。小profile尚待终态。无本项目重型构建重叠。
所有window/SDK旧服务已终态；缺头文件、旧拒绝断言、启动器及同设备误拒绝记录不改写。

## 当前开发及下一步

工作区开发未公开中间State时的KV整序列VJP；trace=True保留逐状态独立梯度链，trace=False在相同公开输出/边界状态/输入/共享参数根上批量回收前缀伴随，避免逐时私有构图。Python10个原型检查通过，新native/完整训练恢复检查待构建。不能据原型认证默认发布。

1. 固定guards结束后冻结新开发，320核构建并运行新序列VJP/既有独立根回归。确认每个返回/None/版本/精确续接。
2. 提交实现再冻结安装/设备/SDK资格及有限成本；补大宽度FC、多节点HB/LH/sum、标准层链及多设备代表；resident独立范围不能继承host通过。
3. 有界分析GDN/HB evidence梯度超差，不无限阻塞后续。当前约3.889e-6，原因未判定，原阈值保留。
4. 更新performance-lessons与唯一ROADMAP，新增证据从仓库直接导航。GPU按授权延后。

下一命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
当前源码/脚本/日志在artifacts/joint-prefill与artifacts/jobs；待办只见ROADMAP。所有活跃源冻结。
