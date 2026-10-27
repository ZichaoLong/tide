# 当前交接 — 2026-10-28

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、packed/VJP/计算共享、
CPU完整训练及恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
Python/native有限mixed、显式同步双NPU、half/master及standalone恢复见ROADMAP；严格失败保留。
5efce32完整weighted resident FP32推理固定资格CPU50、standalone NPU21图/15配置/安装consumer、
Python-owned23项通过；8eebcec显式CUDA80/90/100完整图构建与安装loader通过；无GPU真机。
ebe5471完整数值阶段VJP固定NPU24配置/342根模式和完整前向回归、CUDA三架构/安装loader通过；
证据core5-resident-weighted-stage-20261028.{md,json}。阶段实现与证据已推送ad03d13。

最新实现ad8e7b82a20b1e41403ff5274fa034c1c43a8a10：实际消息祖先反向路由。
重复物理边各贡献一次、精确int64发送时间/样本、独立left-pending/external根；None/连接零、错误37/41。
固定资格通过，证据core5-resident-weighted-route-20261028.{md,json}已随a3472a1正常推送：
- unit tide-core5-weighted-route-npu-qualify-20261028.service inactive/success/exit0；clean ad8e7b8；
  CPU12项/.76秒、NPU12设备案例/dtype拒绝；物理9→npu:0。
- unit tide-core5-weighted-route-cuda-qualify2-20261028.service inactive/success/exit0；clean ad8e7b8；
  路由kernel/wrapper构建/loader通过，160个已配置命令仅80/90/100，无GPU真机。
输出artifacts/jobs/core5-weighted-route-npu-qualify/及core5-weighted-route-cuda-qualify2/，status/task.log、
control-build/original-control-build、NPU checks/queue-state；component摘要b58cc9508d0bafa2e1e2775068d80ff96a25792ac6321a67e8b006a6a612f1e5。
开发dev两后端缺头失败、CUDA首次资格launcher错误要求重编摘要不变的失败保留，详见机器报告。

## 当前开发与下一动作

未提交：完整window VJP、实际fiber/边界journal、nested TaskServices、NPU静态bool布局copy、
独立窗口/正式配置/copy检查和完整持有资源预算。设备逆序循环重算局部tape，沿真实消息祖先及
raw前驱回传并累加canonical参数；参数可全部/部分冻结。公共retained token、optimizer发布及
训练checkpoint仍待实施。数学/表示见[窗口VJP契约](resident-weighted-window-vjp.md)。

开发dev9已inactive/success/exit0，status与checks/result均passed：
- unit tide-core5-weighted-window-dev9-20261028.service；物理9→npu:0；tree ce76fb54886950b314dd5c4d804aa0f3f725b4de。
- 三层services、18 owner/54提交静态copy、12异构窗口、15正式配置通过。
  每个窗口配置12类独立根，共324模式；另空窗口、预算拒绝/可重试、旧generation拒绝。
- 输出artifacts/jobs/core5-weighted-window-dev9/：source.tar/tree、status、task.log、control-build、checks/result及各log。
- launcher .local/core5-weighted-window-dev9{,-workload}.sh；helper .local/build-weighted-window9.py。
  SDK冻结checkout artifacts/worktrees/core5-sdk、build/weighted-flow-dev；已无活动reader。
  下一次干净资格前将其旧kernel_meta生成物移入ignored artifacts，不能删除源码。

CUDA开发构建已inactive/success/exit0，status passed、binary/loader摘要已复核：
- unit tide-core5-weighted-window-cuda-dev-20261028.service；tree 282d21d5b27fc3413f2994173626547495e0d634。
- checkout artifacts/worktrees/core5-cuda；build/weighted-flow-cuda-dev；目前无活动reader。
- 输出artifacts/jobs/core5-weighted-window-cuda-dev/；launcher .local/core5-weighted-window-cuda-dev{,-workload}.sh；
  helper .local/build-weighted-window-cuda.py。2workers、1200秒、background.slice/Nice10。
  四个检查程序及weighted-flow库构建/loader/help通过，160个已配置CUDA命令仅80/90/100。
- 依赖感知Ninja构建、明确sm80/90/100、loader/help；无GPU真机。
  检查systemctl --user show tide-core5-weighted-window-cuda-dev-20261028.service -p ActiveState -p Result -p ExecMainStatus。
  停止systemctl --user stop tide-core5-weighted-window-cuda-dev-20261028.service。

保留的开发失败（均终态，不能改为passed）：artifacts/jobs/core5-weighted-window-dev{,2,3,4,5,6,7,8}/。
dev1测试辅助set与ATen ADL冲突，dev2单流任务容量361001；后者通过严格LIFO嵌套服务拆分。
dev3/6/7在完整顺序第9 nonaligned/proposal窗口报CANN507011/0x91；dev5独立同用例通过。
dev7实际PID2991769 stream411/task29定位raw连接位bool ScatterUpdate，target[10]/source[2]；
不是coordinator stage/node元数据scatter。底层日志及更正存dev3/6/7的diagnostics。
dev8仍同样失败，54 copy提交通过但新分支被错误#ifndef排除；后端宏为0，故不能检验修复。
dev9修正为#if !TIDE_RESIDENT_CUDA后首次真正启用静态bool copy，完整顺序通过；未推断厂商内部根因。
dev4在设备/构建前因预建输出目录被job.py拒绝；core5-cann-lifecycle探针因缺compile_commands在编译前失败，
helper已修正但未重跑。原始status/log/reproducer保留，定向修复通过后不重复历史诊断。

定向CPU开发检查17项/1.93秒通过（flow packing、实际消息反向路由），git diff --check通过。
下一步：实现与审查通过，提交实现；冻结确切提交，NPU窗口/正式配置和
必要forward回归、CUDA构建/loader分别资格，证据另提交并正常推送。然后继续公共retained VJP、
optimizer及精确训练恢复，再FP16/peer、有限profiling/代表性完整成本选型与最终追溯。
初态常量不伪装成checkpoint叶子；禁止CPU预跑或主机逐事件推进。总体未完成，提交后继续。

## 未解除的严格限制

- FP16 GDN默认AdamW：Python/native nodes.5.kernel.conv.weight严格更新差约.001984，专用strict xfail；
  只接受该字段，未改容差或输入，SGD通过不解除。原175pass/2fail与95pass/2fail任务保留failed。
- 默认异步双NPU完整Python VJP：CANN invalid event；显式ASCEND_LAUNCH_BLOCKING=1通过不解除。
  core5-peer-event及current0/1/serial复现保留，不恢复旧队列。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a相同；原失败/诊断保留。
- CUDA实际GPU单卡/跨卡、完整训练/恢复、常驻与性能待硬件，CPU或构建不冒充实机。

## 环境与操作

实际路径/mi/data2T/zlong/home/llm/graph-execution-foundation，zlong，约3.3TiB空闲。
现行Ascend规范/opt/software/ascend-locks/dev-workspace-9.1.0/README.md及/etc/modulefiles。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate（Python3.12.13/Torch2.12/CANN9.1）。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；sdk_python=$LIBTORCH_ROOT/../../../../bin/python，
PATH前置其bin，ACL_OP_INIT_MODE=0，PYTHONPATH追加项目python但保留module提供的CANN路径。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh，Torch2.10+cu128/Python3.11/Toolkit12.8.1；
服务PATH追加项目私有Ninja但保持CUDA Python优先。公共包与工具链未修改。
长任务background.slice/Nice10、2build workers/BLAS单线程、新输出/有限时长；
先更新STATUS、固定源码，不编辑活动reader。源码/构建/实际设备/性能分别记录。
检查systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus及status.json/task.log/checks/result.json；
停止systemctl --user stop UNIT。STATUS用scripts/durable_records.py原子更新并回读。
