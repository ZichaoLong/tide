# 当前交接 — 2026-10-29

授权分支`graph-execution-foundation`；数学固定`0770593838a37c790a568226930567ed60f99b83`，离线19份+旧5份校验通过。
用户要求逐项推进非GPU限制，给出实际改进、严格验收、有限预算、停止/重启条件；GPU延后，不用子代理。
已获修改、有限实验、提交和普通推送授权；参考仓库与公共工具链只读。

## 当前成果与未提交工作

起点121f5b1，生产基线f03e9be。诊断bdd7f6c、证据fe5e6e7均已提交推送；
[固定隔离证据](evidence/core5-boundary-isolation-20261029.md)记录D2048实际操作数局部连续FP64改进、
原half归一化FP64 VJP仍失败、默认async完整双NPU CPU桥接仍EE1017。未解除原限制，没有重跑9B原全图。

eager单行Query/Aggregate VJP消冗、异形字段copy和标量/native分配观察完成开发检查，准备提交。
未提交的resident因子KV/失败快照/反向阶段计量及benchmark实际槽、跨卡、精确恢复设施正在验证。
没有更改原容差、optimizer、默认dtype、正支持或None/零语义。

## 终态开发记录

所有记录在`artifacts/jobs/NAME/{status.json,task.log}`，完整dirty source archive/hash另存；旧失败不改。
- core5-boundary-dev：passed/0，单行VJP CPU完整构建、177 passed/2原FP16 strict xfail。
- core5-boundary-python-dev：passed/0，122+17项；core5-boundary-python-npu-dev因用例名错误failed/2，v2五图mixed-c完整训练恢复passed/0。
- core5-boundary-native-dev、sdk-dev、pynative-dev：failed/1，新增诊断ArrayRef→vector编译错误；已显式复制修正。
- core5-boundary-native-dev-v2：passed/0，329项（分配/observer/传输/设施），包含最小subnormal未算残差不得填零。
- core5-boundary-pynative-dev-v2：failed/1，core构建完成，CANN9.1旧合并对象增量复用unknown file type；未执行模型。
  v3改为新resident目录、正式builder，不修改公共工具链。
- 固定诊断CPU job failed/1（两设施+首写入passed，half strict_mismatch/2）；bridge job failed/1（完整VJP EE1017）。

## 活跃任务与冻结输入

两个service在background.slice，build各2线程、ATen/BLAS各1线程、MemoryMax=16G、RuntimeMaxSec=3600；
资源属性不保证cgroup硬隔离。设备排队180秒、执行1200秒；没有GPU排队，没有性能计时并行。
两者只读同一任务专用树`artifacts/worktrees/core5-independent-qualified`：bdd7f6c+完整dirty快照，
其旧f03e9be任务已终态，复用前保存原manifest。任何一个未终态都不得更新该树。

| unit（均后缀-20261029.service） | 入口 / 输出 | 目的 |
| --- | --- | --- |
| core5-boundary-sdk-dev-v2 | `.local/core5-boundary-sdk-dev-v2.sh` / `artifacts/jobs/core5-boundary-sdk-dev-v2` | LibTorch2.10/CANN9.0核+KV/allocation/flow/training/node-training两dtype、双卡相关检查；build/npu-core-v2及npu-resident-v2 |
| core5-boundary-pynative-dev-v3 | `.local/core5-boundary-pynative-dev-v3.sh` / `artifacts/jobs/core5-boundary-pynative-dev-v3` | Torch2.12/CANN9.1核+新build/npu-python-resident-boundary；eager六组和公共resident/实际SGD解码恢复 |

## 精确接续

先`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
查看上述两个status/task.log、development.json；不可把活跃任务写为通过。原CPU可复用缓存
在`artifacts/worktrees/core5-boundary-dev/build/cpu`，与main当前cpp hash一致；正式资格前逐一验证binary hash。

接着固定eager提交并在干净树跑原D64 FP32两步/精确恢复、反馈FP32四窗/恢复及必要相关资格。
NPU开发通过后提交resident/设施，干净资格包括新公共失败codec测试、实际SGD/恢复。
全部构建/资格终态后串行执行一次/配置有限性能：CPU中型和D64显式FP64、NPU中型mixed/resident；
D64九百万参数单/双卡（显式blocking独立配置）有界训练/恢复，容量/数值/运行时失败即停。
补窗口反向构造/执行成本、保留容量/表示下界和每项停止/重启判定；结果更新ROADMAP及performance-lessons。
已冻结或失败的源码/记录不覆盖，最终源码与证据分开提交并普通推送，持续完成不在阶段提交后暂停。
