# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
已推送cb0dd30（6f709f8来源低秩实现及固定证据）。总体未完成，不在阶段提交后暂停。
当前完整backlog见[ROADMAP](ROADMAP.md)，执行契约见[execution-flows](execution-flows.md)。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、实际packed/VJP/计算共享、
CPU完整训练与恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
470ce06 FP32设备、01689c0 FP16设备：Python/native有限mixed、显式同步双卡及master恢复通过。
670ead3 standalone masters/schema2：CPU24、NPU FP16显式SGD24图/AdamW2图和CPU handoff通过。
同提交CUDA新栈core构建/CPU24/显式设备不可用拒绝通过，GPU真机未验。
2814b87常驻Alloc：CPU29、NPU六正则配置完整VJP/重放/容量、CUDA sm80/90/100构建通过。
8e7298d常驻发送：CPU34、NPU八配置完整五输入VJP/96物理行80非空帧、CUDA三架构构建通过。
后两项仅局部原语，无主机逐次决定，但不代表完整weighted resident图或吞吐优化。

## 当前阶段与下一动作

6f709f8来源低秩链和固定证据cb0dd30均已推送。此前完整Region ce63aef、Aggregate92751c0、
source6f709f8的CPU/NPU及CUDA三架构构建资格通过；[证据索引](evidence/README.md)保留各项精确范围。
它们是局部常驻组件，不代表完整图。CUDA无真机证据。

canonical参数归约尚未提交；符号与契约见resident-weighted-parameters.md。
开发任务tide-core5-resident-parameters-dev-20261027.service已failed/exit1，
冻结tree18b4665df189605cfb6c62baa38097a0f18edc9f；CPU3项通过、NPU构建通过，
设备检查在canonical None/zero optimizer counter断言失败。
最小原因：检查fixture传入显式空参数group（语义是选择零参数），未将canonical owners加入group；
主工作区已修正fixture，数值实现和容差未改。原source.tar/source.tree/status.json及检查日志保留在
artifacts/jobs/core5-resident-parameters-dev；原失败不改passed。

新开发任务tide-core5-resident-parameters-dev2-20261027.service已inactive/exit0，
冻结tree3f166d4cb17373dba759c64bccef6c2d8af208c4。CPU3项通过；NPU完整组合通过：
跨bank/角色10个canonical owners、SGD/Nesterov与AdamW/AMSGrad各4步、全部参数/slots、
None/零计数、新program恢复后两步精确状态、非有限事务与513维尾块溢出/容量拒绝。
输出artifacts/jobs/core5-resident-parameters-dev2保存source.tar/source.tree/status.json/task.log/checks/result.json。

2e335e7固定资格CPU3、NPU完整局部训练组合及CUDA三架构构建通过。
core5-resident-parameters-{npu,cuda}-qualify均clean/exit0/inactive，证据
[evidence/core5-resident-parameters-20261027.json](evidence/core5-resident-parameters-20261027.json)。
NPU保留原dev2 manifest并核验全部摘要；CUDA无GPU实测。worktree固定2e335e7，无活动reader。

主工作区正在实现typed raw state：weighted_raw_state.h/.cpp、weighted_raw_update.h/.cpp、
ascendc/weighted_raw_update_math.h及四个登记入口；尚未集成构建或运行，不声称通过。
全部五记忆的原始字段与int64元数据布局、严格CPU编码/解码、正/零接收与原始时间缓存更新、
独立字段VJP和事务adopt。下一步补CPU独立公式及standalone检查，冻结开发/固定资格后继续数值记忆。

完整weighted resident仍缺typed raw state/五记忆/卷积、Describe、完整图在线就绪与推进、
retained训练/optimizer/精确checkpoint/跨卡；继续局部组件与完整生命周期，不借旧ContentFlow
或CPU捕获轨迹宣称完成。随后有限profiling/性能选型与最终追溯，不做全开关笛卡尔积。

## 未解除的严格限制

- FP16 GDN默认AdamW：Python/native nodes.5.kernel.conv.weight严格更新差约.001984，专用strict xfail；
  只接受该字段，未改容差或输入，SGD通过不解除。原175pass/2fail与95pass/2fail任务保留failed。
- 默认异步双NPU完整Python VJP：CANN invalid event；显式ASCEND_LAUNCH_BLOCKING=1通过不解除。
  core5-peer-event及current0/1/serial复现保留，不恢复旧队列。
- 独立SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a相同；原失败/诊断保留。
- CUDA实际GPU单卡/跨卡正确性、完整训练/恢复、常驻与性能待硬件，不能用CPU或构建冒充。

## 环境与操作

实际路径/mi/data2T/zlong/home/llm/graph-execution-foundation，zlong，约3.3TiB空闲。
现行Ascend规范/opt/software/ascend-locks/dev-workspace-9.1.0/README.md及/etc/modulefiles。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate（Python3.12.13/Torch2.12/CANN9.1）。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；sdk_python=$LIBTORCH_ROOT/../../../../bin/python，
PATH前置其bin，ACL_OP_INIT_MODE=0，PYTHONPATH追加项目python但保留module提供的CANN路径。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh，Torch2.10+cu128/Python3.11/Toolkit12.8.1；
服务PATH追加项目私有Ninja但保持CUDA Python优先。公共包与工具链未修改。
长任务background.slice/Nice10、2build workers、ATen/BLAS单线程、新输出/有限时长；
先更新STATUS、固定源码，不编辑活动reader。源码/构建/实际设备/性能分别记录，旧失败不改成功。
检查python scripts/status.py及systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读status.json/development.json/task.log/checks/result.json。停止systemctl --user stop UNIT。
STATUS原子更新用scripts/durable_records.py并回读。活动输入/失败复现与未提交代码保留。
