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
实现与这些证据已正常推送至2ae8496。

最新实现ebe547115e8a70347c42ae420d669f1c6ea65fba：完整数值阶段VJP，尚未推送。
Full→Describe→Mem→Upd→Aggregate，三种Read快照、独立raw/描述/Content/control/Full根，
物理输入和来源低秩链、角色切片与canonical共享参数。阶段不等于完整retained图训练。
固定资格已通过，证据evidence/core5-resident-weighted-stage-20261028.{md,json}待单独提交：
- NPU unit tide-core5-weighted-stage-npu-qualify-20261028.service inactive/success/exit0；
  clean ebe5471，metadata5项/.78秒、24配置/342根模式及非有限拒绝、完整forward21图/边界；物理3→npu:0。
  输出artifacts/jobs/core5-weighted-stage-npu-qualify/：status/task.log、control-build、checks/result.json、queue-state。
  复用dev5全部源码/component/core/binary摘要一致构建，原manifest保留，不声称重编。
- CUDA unit tide-core5-weighted-stage-cuda-qualify-20261028.service inactive/success/exit0；
  clean ebe5471，stage/flow/lib与安装consumer构建/loader通过；159个已配置CUDA命令仅80/90/100；无真机。
  输出artifacts/jobs/core5-weighted-stage-cuda-qualify/：status/task.log、control-build、installed/result.json。
共同component摘要17fa110fa6a8724e9631f5e9411028deba4de23efd403aa0e42b47b6e0199bd6。
NPU600秒/queue180、CUDA1200秒，2workers/BLAS1/background.slice/Nice10。
两checkout artifacts/worktrees/core5-{sdk,cuda}当前clean ebe5471、无活动reader。
阶段dev1/2编译失败、dev4参考漏control独立根的失败保持；dev3/5通过，精确tree见机器报告。

## 当前工作与下一动作

跨阶段消息反向路由未提交：weighted_reverse_route.{h,cpp}、AscendC kernel/math、C++设备检查、
tests/test_resident_weighted_reverse_route.py以及CMake/组件检查注册。
按实际int64发送time/source/sample寻找早期Full；重复物理边各贡献一次，left-pending/external按完整六元坐标独立根。
连接位独立于数值零；身份/ancestry拒绝41、非有限/溢出拒绝37，sticky error使整个导数失效。
CPU语义10项/1.22秒通过；C++wrapper/NPU/CUDA尚未构建运行，尚未接完整window reverse owner。
先单独提交推送已完成阶段证据（不混入新路由实现），再冻结开发源码做路由设备检查。
短检查：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate；TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_resident_weighted_reverse_route.py。
继续完整retained VJP/optimizer和精确训练恢复：扩展实际device journal的fibers/offsets，
设备逆序读取局部tape/旧态并重算，链接消息生产者与状态前驱，汇总canonical参数；
不CPU参考重放、不主机逐事件推进。初始绑定常量不能伪装为checkpoint叶子。
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
