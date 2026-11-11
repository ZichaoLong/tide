# 当前交接 — FC默认路径、直接公式与Transformer比较（2026-11-11）

用户已授权本轮完整对比、诊断与架构调查；先收拢FC4/FC16，LH和广泛后继暂不扩大。
起点`b504f34`，数学固定`0770593838a37c790a568226930567ed60f99b83`，离线核验通过。
本轮可修改诊断设施并执行有限CPU/NPU实验，不把省略独立根的诊断变体当成生产实现。

## 当前动作

- 核对三层实际计算与接口职责；FC4为2区域×2节点，FC16为原4×4，均d128/H2/FFN512。
- 补齐匹配的无梯度请求、含grad前向、完整训练；自然完整步计时与阶段同步诊断分列。
- 比较实际算子/形状、Read/Full/区域/快照/私有VJP及必要状态职责；诊断变体有明确投影域。
- 默认复用已固定e6f7bd5的Python/LibTorch构建（当前C++内容无差异），诊断源码另冻结。
- CPU与NPU任务可在不同CPU亲和范围并行，构建不与重型计时重叠。只使用空闲NPU。

当前只有诊断源码修改，没有live作业；启动前将补充固定提交、unit和精确命令。
相关数值/模式检查后提交设施，再运行固定源码，最后独立提交证据与架构结论。

## 资源与停止边界

本次检查：账户292GiB/300GiB软配额、800GiB硬限；文件系统955GiB可用；
/dev/shm约400GiB可用，主机可用内存约1.8TiB。现有环境/构建可复用，无需复制工具链。
临时源码/大型profile可放项目独占shm目录，小型配置、日志、终态和审阅结果持久保存。
若实际存储不足会限制必要构建或并行，应停止相关大写入并汇报，用户可清理非本仓库空间；
不得静默缩减范围。共享NPU仍需当前授权，本轮未授权；不等待共享卡阻塞CPU调查。
每一有限矩阵/诊断有时限；严格超差保留，有界分析，不无依据扩大或重复全部组合。

## 交付与结果

- 仓库develop/optimize/maintain skills加入有限加法/减法、公共接口增量对照及采用/拒绝/延期方法；账户skills未修改。
- FC/KV/SSD紧凑基线支持非零raw左态、共享、真实三步训练、跨窗与新进程投影checkpoint。
- Python/C++默认packed Full采用适用的共享bank矩阵，保留独立快照VJP与通用后备。
- resident首次借用只计真正owner新增存储；实际小额度、真实不足和Describe数值见证通过，未放宽总预算。
- [固定资格](evidence/additive-interface-qualification-20261110.md)：CPU主组493通过/7xfail/4失败；其中两项SDK启动路径另以实际程序通过，两项既有Read-sum FP32严格超差保留。NPU配置37通过/1xfail/相同两项CPU辅助失败；默认实际调用CPU4/NPU2通过。三客户端小型训练/精确恢复、resident组件/小图分别列出，未认证新resident完整训练或吞吐。
- [有限成本](evidence/additive-interface-performance-20261110.md)：局部/层链/FC共42个成本点、12个代表训练/精确下一窗、NPU约1B推理/精确下一窗、图和阶段独立诊断。FC16训练CPU1/NPU12.283/15.511秒，首token2.737/4.055秒；局部前向更快，NPU完整训练未改善。完整尺度独立数值/离散/学习仍未验。
- 原严格失败、half原见证失败和公式census启动失败均保持；修复/补验不改写原作业状态。当前限制和重启条件仅由[ROADMAP](ROADMAP.md)维护。

## 接续入口

运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
先看本页当前作业，再读[成本汇总](evidence/performance-accounting-20261111.md)、
[成本契约](fc16-cost-attribution.md)、[架构](architecture.md)与[ROADMAP](ROADMAP.md)。
上轮作业均已终态，日志保留在`artifacts/jobs/additive-*/`；不恢复旧队列。
用户未跟踪的两份TL-REQ-003/004交接文档原样保留，不纳入提交。
