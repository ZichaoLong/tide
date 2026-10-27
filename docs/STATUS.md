# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
已推送d8b4646（92751c0完整Aggregate及固定证据）。总体未完成，不在阶段提交后暂停。
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

完整区域开发tree4a161a96505deac564ec9ccbf3c5bd4e18dbd551已冻结并通过，
任务tide-core5-resident-region-dev-20261027.service inactive/exit0。
artifacts/jobs/core5-resident-region-dev保存source.tar/source.tree/status.json/task.log/checks/result.json。
CPU33项（8.54秒）；NPU14配置/140物理行112非空帧，全部字段/共享参数一阶VJP、
独立原始版本反向、重放、int64支持/history、容量拒绝与失败不提交通过。
实现为weighted_region{,_vjp}.cpp/.h及五个登记入口；契约docs/resident-weighted-region.md。
参考RegionKernel使用完整静态成员并调用validate_weights，独立CPU公式另作数学锚点。
resident worktree原HEAD161ccea、无活动reader，NPU build/resident-region-dev保留原开发manifest。

ce63aef已推送；固定源码Region资格CPU33（8.69秒）、NPU完整14配置、CUDA sm80/90/100构建均通过。
core5-resident-region-{npu,cuda}-qualify任务均clean/exit0/inactive，详见evidence/core5-resident-region-20261027.json。
NPU复用dev构建且组件/core/binary全摘要相符，原manifest保留；CUDA新构建，无GPU真机。
resident worktree固定ce63aef、clean/read-only，无活动reader。

Aggregate开发tree8f9bc2bfbdcecb586bcad6d9343cabf8d5a113bd通过；
tide-core5-resident-aggregate-dev-20261027.service inactive/exit0。CPU15（1.13秒），
NPU两配置/24物理行20非空纤维，vendor RMS/源变换、全部载荷与源参数VJP、
同类重复、零/负质量、四种补零隐藏宽度、None/零、原来源改写后的保存版本反向、容量/失败不提交通过。
实际记录artifacts/jobs/core5-resident-aggregate-dev/{status.json,task.log,source.tree,checks/result.json}。

92751c0已推送；Aggregate固定资格CPU15、NPU完整组件、CUDA sm80/90/100构建均通过。
core5-resident-aggregate-{npu,cuda}-qualify均clean/exit0/inactive，详见evidence/core5-resident-aggregate-20261027.json。
NPU复用dev构建且完整component/core/binary摘要相符；原manifest保留。CUDA无真机证据。
resident worktree固定92751c0、clean/read-only，无活动reader。

来源低秩链开发treea3ee2b018d751ef6374b50227cf242a3b9f0ba26通过，
tide-core5-resident-source-dev-20261027.service inactive/exit0。CPU guard+Agg数学18项；
NPU独立逐原子公式与autograd比较完整Agg、七种参数角色/四因子VJP、共享使用者、
None/零、动态因子/长度重放、原来源改写后的保存版本反向、容量拒绝/失败不提交通过。
原首次setup遗漏registry的3 errors（无数值检查）保留treebf59ee14及
artifacts/reproducers/core5-source-guard-registration/{source.tar,result.json}，不改passed。

当前提交source实现，随后冻结commit运行core5-resident-source-{npu,cuda}-qualify。
NPU重验CPU18与完整组件，仅在component/core/binary摘要一致时复用dev构建；
CUDA新建sm80/90/100组件/loader，无真机证据。background.slice/Nice10，2build workers/BLAS单线程，
NPU600秒/队列180秒、CUDA1200秒。输出artifacts/jobs同名目录；正式结果尚未知。
命令.local/core5-resident-source-{npu,cuda}-qualify{,-workload}.sh；检查/停止同名tide-UNIT-20261027.service。

主工作区另有weighted_parameter_vjp.h/.cpp、weighted_parameter_reduce.h及CPU检查初稿，
参数角色按ParameterRegistry真实TensorImpl身份汇总为已有ParameterVjp；CPU3项通过（1.26秒）。
尚未登记设备入口、构建或接设备optimizer验证，不包含在source实现/资格中。
下一动作：source固定资格与独立证据提交推送，继续canonical参数owner+optimizer/恢复检查，
再接五记忆/Describe、在线图调度与完整生命周期。

完整weighted resident仍缺参数owner集成、typed raw state/五记忆/卷积、Describe、完整图在线就绪与推进、
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
