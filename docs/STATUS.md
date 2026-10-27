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
固定资格通过，证据core5-resident-weighted-route-20261028.{md,json}待单独提交推送：
- unit tide-core5-weighted-route-npu-qualify-20261028.service inactive/success/exit0；clean ad8e7b8；
  CPU12项/.76秒、NPU12设备案例/dtype拒绝；物理9→npu:0。
- unit tide-core5-weighted-route-cuda-qualify2-20261028.service inactive/success/exit0；clean ad8e7b8；
  路由kernel/wrapper构建/loader通过，160个已配置命令仅80/90/100，无GPU真机。
输出artifacts/jobs/core5-weighted-route-npu-qualify/及core5-weighted-route-cuda-qualify2/，status/task.log、
control-build/original-control-build、NPU checks/queue-state；component摘要b58cc9508d0bafa2e1e2775068d80ff96a25792ac6321a67e8b006a6a612f1e5。
开发dev两后端缺头失败、CUDA首次资格launcher错误要求重编摘要不变的失败保留，详见机器报告。

## 当前开发与下一动作

主工作区未提交完整window VJP：weighted_window_{vjp,roots,owner}.cpp、header及独立检查；
forward journal新增实际fibers/offsets、left pending/原始metadata/存在位及begin，预算同步；
复用参数累加拆成公共静态target。内部设备逆序重算局部tape、串消息和raw状态、归约canonical参数。
尚未编译/验收；公共retained训练owner、optimizer发布及训练checkpoint还未完成。
开发服务已提交，结果未知：unit tide-core5-weighted-window-dev-20261028.service。
冻结tree见artifacts/jobs/core5-weighted-window-dev/source.tree，checkout artifacts/worktrees/core5-sdk；禁止编辑活动reader。
输出同job目录source.tar/tree/status/task.log/control-build/checks；launcher .local/core5-weighted-window-dev{,-workload}.sh，
build helper .local/build-weighted-window.py。内部layout已变化，清理完整tide-weighted-flow target后依赖构建；
不变组件逐源核验ebe5471。CPU metadata/route17项；NPU12配置×12根模式、空窗、初态常量与generation拒绝。
1200秒/queue180秒/2workers/BLAS1/background.slice/Nice10。
查看systemctl --user show tide-core5-weighted-window-dev-20261028.service -p ActiveState -p Result -p ExecMainStatus；
查看task.log/checks/result.json；停止systemctl --user stop该unit。
先提交推送路由证据（不混入新window代码），继续处理窗口资格，随后接公共retained VJP、optimizer及精确训练恢复。
初始绑定常量不能伪装成checkpoint叶子；不能CPU预跑或主机逐事件推进。
随后FP16/peer、有限profiling和代表性完整成本选型、最终追溯；不做全开关笛卡尔积。

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

