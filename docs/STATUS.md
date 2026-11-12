# 当前交接 — P1—P6阶段汇报点（2026-11-12）

固定数学 `0770593838a37c790a568226930567ed60f99b83`，19份快照核验通过。
工作分支graph-execution-foundation；最近实现 `55f8c631`，resident资格固定 `a3c4f62d`。
本次文档提交固定下列证据；实际HEAD与推送状态以 `git status --short --branch` 为准。
用户授权重写、提交/非强推及当前对话共享NPU；GPU后继延后。
用户TL-REQ-004/005/006三份未跟踪handoff只读，不提交。参考仓库未改。

## 下一动作

已集成[22点成本/生命周期](evidence/source-columns-performance-20261112.md)、
[43文件静态审阅](evidence/lh-block-static-20261112.md)和[resident公共资格/规模拒绝](evidence/resident-block-public-20261112.md)。
语义、能力、复现导航与性能经验同步；当前在用户允许的阶段汇报点，P1—P6未全部完成。
继续时先按下文命令核对状态，随后从ROADMAP P5选择具体表示边界：owner/角色分组投影和共享历史bank，
同时审视columns的Q/K尺寸分桶与逐节点小操作。先定义forward/VJP/观察物化及字节模型，再做有限实现和相关检查。
用户要求静态算法/循环/张量形状/复制/伴随审阅与运行观测并重；LH主要差距未闭合时可在下一停点汇报。
不以小图通过或公共接口接入宣称P1—P6完成，不继续无依据增加预算或重复同一规模失败。

## 已固定结果

- `00bb5ac2`来源MLP按输入及完整参数槽合批/精确复用；CPU/NPU 22成本点、四安装LH176生命周期已完成。
  当前mixed-c默认LH176训练CPU1 2.3187s/NPU-host2 5.1350s，直接公式0.7568/0.6238s；主要差额在公共列层。
  同FP32材料首Aggregate差1.92e-9，提升FP64全根最大1.71e-13，原严格失败和None/零边界保留。
- `4248bde1`实现KV sum typed tape/独立Mem-total根，Model及experiment execution-v7默认接入，实际完整解码及schema2 archive；
  `a3c4f62d`加入standalone真实训练/新进程资格入口。冻结源码 `/dev/shm/gef-p1p6-20261112/resident-v3-source`。
- 固定a3c：Python-owned/SDK resident均构建成功，复用hash相同的00bb C++ core。CPU70通过，SDK Model检查通过。
  原SDK外层因benchmark缺JSON头failed；显式已有JSON头后新benchmark构建及4 CTest通过，原失败不改写。
  native设备6进程、SDK9进程、standalone sum/qry两项通过，含三步optimizer、实际raw、新进程精确恢复。
  d00f异宽source反向device index_select已有公共消费者真机覆盖。本轮真机范围是新block单NPU FP32、KV sum/qry相容Settle域。
- `55f8c631`独立LH基线改用完整有序参数角色，修复去重root列表误合并低秩槽；CPU开发22通过。
  不改变无低秩LH176旧材料，不当作新性能测量或a3c设备资格。

## 规模与架构边界

固定a3c各一次，原FC16 B4/L16/d128预算及LH176 B4/L16/d64边界试验均failed，无有效训练/吞吐/恢复。
FC16构造成功，warmup qry保守预留13,670,416,384字节大于可用6,120,144,499；该值不是实测峰值。
LH176在optimizer子分区的canonical layout守卫拒绝：optimizer_bytes/8=50,331,648，
FP32 layout+owner位需59,625,113字节；这是子预算拒绝，不是设备OOM。旧FC16 EE1019未由更早拒绝解除。
静态审阅确认columns主要矩阵已分组，仍按实际Q/K尺寸分裂及逐节点小操作；resident多bank按行复制矩阵，
逐帧复制raw、重复投影历史，单工作block按B/N/L执行及反向重复结构根扫描。
后继具体边界为owner/角色分组投影、共享initial/write bank与快照索引/块伴随，以及对应精确容量模型。
不能靠仅换外层调用或放松预算声称实现这些替代。P5及未迁移多卡/half/其他记忆仍未完成，详见ROADMAP。

## 作业与资源

当前无活跃构建/测试/实验。所有 `gef-p1p6-resident-v3-*` 已终态；保留的failed服务MainPID=0且ControlGroup为空。
作业记录 `artifacts/jobs/p1p6-resident-v3-*/`：native-build/benchmark-build/native-qualification/
sdk-qualification/standalone-qualification passed；sdk-build及scale failed。
输入/launch/逐项日志在 `artifacts/p1p6-20261112/resident-v3-*`；native设备5、SDK10、standalone/scale5。
两套队列设置均unset launch-blocking/TASK_QUEUE_ENABLE/PER_STREAM_QUEUE；不继承双卡普通异步资格。
构建各CPU160核/512GiB/2400秒，资格各独立CPU池/256GiB/1800秒，规模每点600秒；未盲目重试。
冻结core、native/sdk resident及安装在 `/dev/shm/gef-p1p6-20261112/`；正式复现须按development重建，不依赖这些目录。
磁盘约841GiB、shm约292GiB可用，没有存储阻塞、没有清理用户或参考数据。

重新接续：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`；
历史失败查 `python scripts/status.py --all-jobs`。三份用户handoff保持未跟踪，不属于本次交付。
后继源码导航：`rg -n "banks|index_select|bmm" tools/device_online/weighted_projection.cpp`、
`rg -n "for|old|proposal" tools/device_online/ascendc/block_kv_sequence_math.h`；
新源码的独立公式、跨窗及客户端检查从上面的资格报告进入，不重跑原失败队列。
