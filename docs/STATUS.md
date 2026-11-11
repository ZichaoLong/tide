# 当前交接 — 实施轮次完成，成本对齐已补充（2026-11-11）

用户已授权的“加法基线与默认接口接入”有限轮次完成。按用户要求本轮汇报后停下，
后继先对齐再实施；不要因旧的持续任务安排自动重启性能队列。
分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`，
离线快照核验通过。起点`a61e028`；计算实现`e6f7bd5`、成本/默认调用探针`0e6916d`、
half拒绝见证修正`57439d7`。正式资格与成本证据已在独立文档提交`868ec05`中。

2026-11-11仅审阅既有记录，补充[FC16/LH吞吐与阶段差额](evidence/performance-accounting-20261111.md)及仓库优化skill的模式覆盖/成本归因方法。
没有新构建、测试、性能运行或计算实现修改；仍保持实施暂停，后继先对齐。

## 交付与结果

- 仓库develop/optimize/maintain skills加入有限加法/减法、公共接口增量对照及采用/拒绝/延期方法；账户skills未修改。
- FC/KV/SSD紧凑基线支持非零raw左态、共享、真实三步训练、跨窗与新进程投影checkpoint。
- Python/C++默认packed Full采用适用的共享bank矩阵，保留独立快照VJP与通用后备。
- resident首次借用只计真正owner新增存储；实际小额度、真实不足和Describe数值见证通过，未放宽总预算。
- [固定资格](evidence/additive-interface-qualification-20261110.md)：CPU主组493通过/7xfail/4失败；其中两项SDK启动路径另以实际程序通过，两项既有Read-sum FP32严格超差保留。NPU配置37通过/1xfail/相同两项CPU辅助失败；默认实际调用CPU4/NPU2通过。三客户端小型训练/精确恢复、resident组件/小图分别列出，未认证新resident完整训练或吞吐。
- [有限成本](evidence/additive-interface-performance-20261110.md)：局部/层链/FC共42个成本点、12个代表训练/精确下一窗、NPU约1B推理/精确下一窗、图和阶段独立诊断。FC16训练CPU1/NPU12.283/15.511秒，首token2.737/4.055秒；局部前向更快，NPU完整训练未改善。完整尺度独立数值/离散/学习仍未验。
- 原严格失败、half原见证失败和公式census启动失败均保持；修复/补验不改写原作业状态。当前限制和重启条件仅由[ROADMAP](ROADMAP.md)维护。

## 作业与资源终态

全部在固定清洁源码、`background.slice`下结束，无本轮live作业或持有的NPU锁。

| 作业 | 终态 | 原因/范围 |
| --- | --- | --- |
| `gef-additive-build` | passed | 320 CPU，216.5秒；Python native、独立SDK及resident构建 |
| `gef-additive-qualify` | failed | 保留上述数值、SDK路径及half原断言失败；各内层结果分别审阅 |
| `gef-additive-budget` | passed | 固定57439d7，两dtype recurrent真实预算拒绝及独立数值/VJP |
| `gef-additive-measure` | failed | 计时/代表/1B恢复均完成；四个公式census误传native参数，在计算前拒绝 |
| `gef-additive-supplement` | passed | 仅补四个未执行census，另存CPU默认分派4项JUnit |

日志/终态：`artifacts/jobs/additive-*/{status.json,task.log}`。
冻结现场`artifacts/additive-work/source-{e6f7bd5,probes,budget}`；安装/构建分别为
`env`、`build-native`、`build-sdk`、`sdk`、`device`、`device-budget`，运行/补验目录同根。
每次只用空闲physical11→npu:0，已释放；未使用共享卡、未新增GPU或多卡任务。
本机全部可用CPU/内存与空闲NPU可用，共享NPU需当前对话明确授权。
最后磁盘检查：文件系统约670GiB可用、账户约297GiB/300GiB软配额（800GiB硬限）；
没有自动清理参考仓库、公共工具链或本轮失败材料。新大写入前重新核验配额/容量。

## 接续入口

先运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
按[资格](evidence/additive-interface-qualification-20261110.md)、[成本](evidence/additive-interface-performance-20261110.md)、
[经验](performance-lessons.md)、[架构](architecture.md)和[修改维护表](core5-study.md#修改时同步维护什么)
定位配置、实现、检查及复现；新clone无需旧运行目录。不要重跑已终态的整组实验恢复上下文。

用户未跟踪文档`docs/tide-learning-lab-TL-REQ-003-handoff.md`和
`docs/tide-learning-lab-TL-REQ-004-handoff.md`原样保留，未纳入本轮提交。
