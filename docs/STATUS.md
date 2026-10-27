# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、实际packed/VJP/计算共享、
CPU完整训练与恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
470ce06/01689c0 Python/native有限mixed、显式同步双卡及master恢复；670ead3 standalone masters/schema2
CPU24、NPU FP16显式SGD24图/AdamW2图和CPU handoff通过。CUDA新栈core构建/CPU24通过，无GPU实机。
常驻Alloc/发送/head/Region/Aggregate/来源低秩/canonical参数/raw state/固定时间卷积已逐项固定资格；
均为局部组件，不代表完整weighted resident图。旧失败及精确范围见各项证据，不自动重跑。

最新实现2af401dc0b2912fcf09faac9ab1f31a43aec489a：共享投影/可选RMS与bias。
固定资格CPU4（1.03秒）、NPU四配置48行/三bank/完整VJP/None零/保存版本与容量；CUDA sm80/90/100构建通过。
证据[evidence/core5-resident-projection-20261027.json](evidence/core5-resident-projection-20261027.json)。
core5-resident-projection-{npu,cuda}-qualify均clean/exit0/inactive；无活动reader。
NPU保留开发treee741c50847de858517264e9546eb503e20d36cb4原manifest并核验全部摘要。

## 当前工作与下一动作

主工作区未提交KV Upd草稿：tools/device_online/weighted_kv_update{.h,.cpp,_internal.h,_vjp.cpp}，
ascendc/weighted_kv_update_math.h及tide_weighted_kv_{prepare,roots,merge}.cpp。
它们组合base+U@V、独立summary投影与原始时间raw更新；尚未编译验证，未宣称通过。
接着补CMake/registry、独立完整公式与字段根/参数VJP检查、文档对应；冻结开发测试后提交实现，
再固定提交资格和独立证据。随后继续数值Mem/Describe及四循环记忆、完整在线图生命周期。

准确入口：python scripts/status.py；rg -n weighted-projection tools/device_online/CMakeLists.txt scripts/device_component_checks.py。
开发冻结使用.local/freeze-resident.py，所有reader终止后才重写artifacts/worktrees/core5-resident。
当前该worktree固定2af401d。不得把主工作区草稿混入已完成projection的证据身份。

完整weighted resident仍缺五记忆数值/Mem/Describe/Full、图在线就绪与推进、retained训练/optimizer、
精确checkpoint/跨卡；继续局部组件与完整生命周期，不借旧ContentFlow或CPU捕获轨迹宣称完成。
随后有限profiling/性能选型与最终追溯，不做全开关笛卡尔积。

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
