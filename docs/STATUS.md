# 当前交接 — 联合prefill本轮终态（2026-11-10）

授权分支graph-execution-foundation；固定数学0770593838a37c790a568226930567ed60f99b83。
用户授权持续实施/提交/推送、充分使用空闲CPU/NPU及经身份核实的有余量共享卡；GPU延后。
参考/下游只读，无子代理。两份未跟踪TL-REQ-003/004用户handoff保持原样，不纳入提交。

## 本轮已完成

默认分量窗口、B×L联合KV准备、整序列VJP、query完整根汇合和按需trace已交付。
f15c4a5/3d88b0b的owner一次装配、描述投影/有限检查/分配metadata合批完成固定资格。
3d CPU281通过/4原数值xfail、NPU目标流程68通过（含CPU辅助）；安装消费者和独立SDK
训练/新进程恢复分列通过。原FC16同minimal完整步CPU18.640405/NPU22.902820秒。
证据入口：docs/evidence/prefill-local-batches-20261110.md。

34项CPU/NPU代表矩阵见prefill-representatives-20261110.md；后继16项部署尝试（含失败）见
prefill-deployment-20261110.md/json。宽FC、FP16、五记忆异构LH、CPU线程、1.051B/9.314B、
双卡、B32均有实际结果；尺度独立数值/离散仍not_checked，不把恢复或计时当完整数值资格。
1.051B/176节点/B32/L16：CPU32线程12.262388秒，NPU host2线程33.965264秒；活动约22.2%。
9.314607738B/B2/L8双NPU普通推理13.613896秒，公共精确下一窗通过。

同公式KV/SSD层链：CPU13/显式IEEE NPU7项小型检查通过，12项成本完成；
B4/L32/d128四层KV native CPU/NPU0.512/0.934秒，张量投影0.029/0.022秒；
SSD native0.894/2.853秒、张量投影0.044/0.032秒。两者返回职责不同，默认入口仍明显较慢。
原vendor卷积HF32数值差异及同材料FP64定位保留，见layer-chain-baseline-20261110.md/json。
不认证标准Transformer/Mamba模型或已达到原生性能。

485a3e2修正实际选定推理的事件预留和恢复诊断：只对已审核ranked-local、无梯度无插件、
host分量窗口PyTorch/Python-native缩减全图trace预留，保留其他分区和全部保守后备。
46项新私有固定安装检查通过；复用3d C++且相关源码diff为空，不称重新构建。
原NPU B32的115.34GB静态估算拒绝保留；新24.92GB预留准入，实际完整计时及精确恢复通过，
含恢复peak allocated/reserved约6.07/6.40GB。数学、算法、精度、容差未替换。

## 作业终态与保留失败

当前本轮没有活动作业。gef-joint-prefill-record-capacity已passed/0，MainPID=0、cgroup为空。
固定485a3e2 worktree fixed-record-capacity、env-record-capacity、3d库；日志
artifacts/jobs/joint-prefill-record-capacity/task.log，驱动hash在record-capacity-driver.json。
同目录各任务原始终态和输入继续保留；source记录不是全部内部通过的替代。

closure为failed/1：六CPU层链成功；旧B32 NPU准入拒绝；修正预算resident两项构造拒绝；
CPU B32不计时重建后公共exact_next逐值通过，但临时汇总脚本随后对dict调用record()报错。
原CPU计时在checkpoint-save取消、未发布staging和后继恢复通过分别记录，不重跑计算或改写旧状态。
日志artifacts/jobs/joint-prefill-closure/task.log，结果已审阅。旧throughput-sample/resident-accounted/
layer-final-cpu取消事实保留；后两者原先等待期间未执行，不能写为当时拒绝。

默认ordinary双NPU FC16曾在warmup反向挂起，gdb停于NPUStream::stream/_index_put_impl_；
对应unit已停止且后代消失。显式blocking完成/恢复是另一配置，默认异步限制未解除。
resident三轮目录/分区/实际数值预算结果分列；最后FC16 qry和四层SSD分别在KV qry/
local-row工作区构造拒绝，无性能/恢复。停止统一加预算重试，下一步按具体工作区分解。

## 下一步与导航

1. git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
2. 从docs/prefill-regression.md、architecture.md及evidence/README.md查配置→实现→独立检查→结果；
   原始目录不是复现依赖，报告附干净检出命令。只保留ROADMAP当前待办/重启条件。
3. 优先按层链/原FC16 profile推进块内状态快照表示、索引/私有根装配、循环模块VJP和小提交，
   保持B×L接口及PDG合法后备。每次改动选相关小门禁和少量成本；大型9B降低频率，不无依据重跑。
4. resident先导出逐节点remaining、query/tape及row表达式实际保留量，再考虑workspace/服务复用；
   默认异步双卡须有事件/流所有权的新方案，原失败不由blocking或推理通过解除。
5. 有限数值超差做有界诊断并记录后可继续独立探索；非有限、离散、通信、容量、精确恢复另列。
   不降低原阈值，不把GPU缺席、旧失败或NPU小图资格外推；下游尚未升级/验收。

按既有授权使用当前分支提交/推送，不强推；同步状态以git status为准。最近quota约281/300/800GiB，
磁盘可用约800GiB；大写入前重新核对。后继原始产物清理须按artifact-retention先审阅dry run，
保留必要复现/失败输入，不清理参考仓库或公共工具链。
