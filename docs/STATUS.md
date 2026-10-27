# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
已推送e875dc1；8e7298d发送实现及本次证据提交后继续推送。总体未完成，不在阶段提交后暂停。
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

resident worktree artifacts/worktrees/core5-resident 固定8e7298d，clean/read-only，无活动reader。
发送npu-qualify/cuda-qualify均exit0/inactive；输出artifacts/jobs/core5-resident-sending-{npu,cuda}-qualify，
具体命令/source/build/hash见evidence/core5-resident-sending-20261027.json。
NPU复用build2，完整component/core/binary摘要相符，原manifest source字段保留。
开发sending-dev的bool→float AscendC失败保持failed；build2-dev通过不改旧状态。

当前未提交：resident weighted head（ScalarHead/Query共同双仿射+SiLU），实际vendor bmm/SiLU及显式VJP，
动态int64 owner聚集、共享bank汇总、None/零、实际input/parameter版本保存；CPU元数据/归约8项通过。
新文件tools/device_online/weighted_head*、ascendc/tide_weighted_head_*、tests/test_resident_weighted_head.py，
文档resident-weighted-head及注册变更。尚未构建或实际设备通过。
下一步冻结main dirty snapshot，独立head开发任务构建SDK并执行
scripts/verify_device_control.py --build-dir NEW --output-dir NEW --device npu:0 --checks weighted-head。
通过后提交实现/固定资格；随后组合区域context/query与Alloc/发送形成完整SelStep，再推进图生命周期。

完整weighted resident仍缺Agg、typed raw state/五记忆/卷积、Describe、完整图在线就绪与推进、
retained训练/optimizer/精确checkpoint/跨卡；不能借旧ContentFlow或CPU捕获轨迹宣称完成。
后续有限profiling/性能选型与最终追溯不做全开关笛卡尔积；CUDA源码/构建/实机分账。

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
