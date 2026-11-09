# 当前交接 — TL-REQ-003 正支持数值模式开发（2026-11-09）

用户已授权实施下游 [TL-REQ-003](../../tide-learning-lab/docs/requests/TL-REQ-003.md)。
启动HEAD `85d161d5c9148eac2b164f89a0ba4204b4331898`，授权分支graph-execution-foundation。
已有用户未跟踪交接 `docs/tide-learning-lab-TL-REQ-003-handoff.md` 保留；其最新优先级为
保持cat/ind与训练配方，评估显式最小正值数值模式。数学固定0770593，快照检查通过。

## 当前工作与边界

原严格支持拒绝与数值阈值不变。首先处理实际HB16失败行的Python公共scalar/packed路径，
以及加权聚合、Shannon一阶VJP、KV读写、完整状态/共享根/optimizer/新进程恢复。
先CPU，再通过本机队列检查NPU次正规数和实际依赖路径；不先重跑原训练或全后端矩阵。
新模式必须显式配置/身份和适用域，native/resident未实现时明确拒绝，不能静默按旧路径执行。
小型前向成功不代表完整梯度或训练通过。需求原文与下游验收仍由实验分支拥有。

## 环境、作业与接续

本基座当前无活动作业；下游/他人活动环境、源码和权重不修改。
复用公共 `ascend/dev-workspace-9.2.0` 和调用者选择的私有venv。
当前开发私有环境在 `.local/venv-cann9.2.0`；路径不是公共接口。
账号约169GiB/硬800GiB；本次新增小型测试/固定源码/wheel预算≤2GiB，无需tmpfs。
相关长检查固定提交、background.slice，CPU1/8GiB，单项≤30分钟；NPU申请1张真正空闲卡，
等待/服务均有时限，不停止其他项目进程。正式资格先提交实现，再独立提交证据。

开发结果：显式positive-floor-v1及公共配置/身份/拒绝、稳定Aggregate/KV与未补值Shannon VJP已实现。
65项定向开发检查通过（2026-11-09），包含CPU实际行、FP64反例、两区域完整记录/独立根、
AdamW/跨窗/新进程精确恢复。原严格拒绝未变。空闲单NPU次正规算术探针通过，不代替整图资格。
较宽开发回归一次误选native而报缺库（87通过/3加载失败）；随后Python范围在60秒交互上限终止，
不计完整通过。下一步在固定源码/新wheel安装中完成相关回归和NPU实际依赖路径。

计划固定工作树 artifacts/tl-req-003/source-a，独立安装 artifacts/tl-req-003/qualification-venv。
持久服务 gef-tl-req-003-cpu-a / gef-tl-req-003-npu-a，分别最多1800秒、CPUQuota=100%、MemoryMax=8GiB，
NPU队列最多等待600秒/1卡。固定源码2c11695ca66a5dcd028f7d777ec26b177add0378；新wheel已构建/安装且pip check通过。
CPU/NPU实际进入background.slice并记录clean固定源码。NPU-a已终态exit0，29项定向通过，
真实卡physical1→npu:0，公共行消费者通过；服务inactive/MainPID0/控制组已消失。
CPU-a的29项新策略通过，相关Python回归仍运行，不能提前称整个CPU流程通过。
入口：bash scripts/qualify_positive_support.sh NEW_OUTPUT cpu|npu；需匹配私有固定wheel，NPU在站点队列内。
状态/日志：artifacts/jobs/tl-req-003-{cpu,npu}-a/{status.json,task.log}；
检查 systemctl --user status gef-tl-req-003-{cpu,npu}-a；停止用 systemctl --user stop 对应服务并核对控制组为空。
实现提交、固定源码资格与证据提交分开；不覆盖下游安装，不自动重启HB16训练。

既有CANN9.2有限迁移、三客户端与严格失败见
[原证据](evidence/cann92-project-migration-20261106.md)，并未解除或重新认证。
其他历史边界和后继条件只在[ROADMAP](ROADMAP.md)维护。
