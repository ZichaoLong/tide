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

最新实现2622e605e77c19360a44b8954cdae5989c3fd724完整window VJP已固定资格通过：
- CPU17项/1.49秒；NPU三层services、18owner/54copy提交、12异构/15正式配置共324根模式，
  27空窗/容量/旧generation拒绝，以及21图+15配置完整前向回归通过；物理9→npu:0。
- CUDA相同六目标及weighted-flow库构建/loader/help通过，160已配置CUDA命令仅80/90/100；无GPU。
- unit tide-core5-weighted-window-{npu,cuda}-qualify-20261028.service均inactive/success/exit0；clean2622e60。
  输出artifacts/jobs/core5-weighted-window-{npu,cuda}-qualify/，status/task.log/control-build/original-control-build；
  NPU另cpu.log/checks/result/queue-state。源码/二进制/loader/结果摘要已复核。
- 证据[evidence/core5-resident-weighted-window-20261028.md](evidence/core5-resident-weighted-window-20261028.md)及同名JSON，
  待本次证据提交推送；窗口契约resident-weighted-window-vjp.md。公共训练仍未完成。

主工作区未提交保留层：weighted_window_snapshot/bridge及对应检查，WindowAdjoint新增保存操作数入口。
journal/raw/pending/参数快照脱离前向生命周期；三窗反向经实际物理pending与raw边界连接，canonical参数汇总。
文档resident-weighted-retained.md。尚未编译/运行，公共token/optimizer/checkpoint尚未实施。
下一开发验证需新冻结快照/新job，在固定资格终态后再复用SDK checkout，不能改活动输入。
保留层首次CUDA开发任务failed/exit1：weighted_window_snapshot.cpp缺tide/parameters.h完整声明，
未进入设备运行；另将版本计数类型改为int64，补requires_grad修改检查。当前主工作区已修正，未再编译。
unit tide-core5-weighted-retained-cuda-dev-20261028.service；tree404ed95b7e8eab68a54002adf2ff9c360e219dea。
输出artifacts/jobs/core5-weighted-retained-cuda-dev/：source.tar/tree、status/log/original-control-build、partial-build。
partial-build仅记录失败后实际cache字节，不伪称全部二进制来自新源码；原通过build receipt未重写。
主工作区另改生成脚本仅写变化内容，三后端重复生成均保持内容/mtime；减少无意义全kernel重编。
NPU脚本.local/core5-weighted-retained-npu-dev{,-workload}.sh已准备尚未启动；helper .local/build-weighted-retained.py。
下一步新冻结快照运行NPU开发与CUDA dev2（需明确读取旧失败partial bytes后依赖感知重建），
900秒/queue180秒/2workers/BLAS1/background.slice/Nice10；两个旧checkout目前均无活动reader。

窗口开发dev1–dev9、诊断日志与launcher失败已集中记录于上述证据和artifacts/jobs/原status。
不得将历史失败重标passed，亦无需为接续重复诊断。当前只继续保留层、公共token/optimizer/恢复，
随后FP16/peer、有限profiling/代表性完整成本和最终追溯。总体未完成，提交后持续实施。

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
