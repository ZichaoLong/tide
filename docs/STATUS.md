# 当前交接 — 2026-10-28

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 `graph-execution-foundation`。
HEAD/origin 为 `31e017f8416ede8bf714b4c3f175aa7accc9916f`。用户授权持续实施、私有环境、有限实验、分阶段提交与正常推送；不用子代理，参考仓库只读。
总体未完成，阶段提交后继续。唯一待办 [ROADMAP](ROADMAP.md)，持续契约 [execution-flows](execution-flows.md)。
重入已核对实际路径/git/jobs；固定上游19文件与旧版5文件校验通过。

## 已审阅资格

- FP32公共训练实现343429d，证据56d751d：[报告](evidence/core5-resident-weighted-training-20261028.md)。
  CPU28、Python-owned NPU28、standalone5配置20更新、saved-window12、retained8×5、推理21图/15配置及安装消费者通过。
  CUDA三架构build/loader通过；C++内存checkpoint与Python磁盘/新进程恢复分别核验。
- FP16原始状态/共享投影/固定时间卷积实现2e4d445，证据a585376：[报告](evidence/core5-resident-half-primitives-20261028.md)。
  CPU32、NPU三个组件各两dtype、CUDA三架构构建及三dtype CPU codec通过。
- FP16 KV读取db85db6与更新69bf4f0，证据5ff8679：
  [读取](evidence/core5-resident-half-kv-read-20261028.md) CPU27/NPU四组合；
  [更新](evidence/core5-resident-half-kv-update-20261028.md) CPU26/NPU四组合；各自CUDA三架构build/loader/help通过。
  全部独立raw/输入/参数VJP；次正规、正质量下溢仍追加、溢出拒绝且owner不变。
  两批没有重新跑codec；明确沿用前期独立codec证据。

FP16组件：实际half存储，FP32局部算术/保存值/伴随，存储cast声明identity VJP，int64元数据不变。
局部资格不代表完整half图、训练、peer或性能；所有CUDA真机范围待硬件。

## 当前任务与固定源码

1. **bcef2c4四循环/KV Full资格已完整审阅：[报告](evidence/core5-resident-half-memory-20261028.md)。**
   checkout `artifacts/worktrees/core5-half-memory` 固定bcef2c49dfdbde277d4115f450e4fec07d7f8dc9，跟踪文件只读。
   units `tide-core5-half-memory-{npu,cuda}-qualify-20261028.service`；build/npu、build/cuda。
   outputs `artifacts/jobs/core5-half-memory-{npu,cuda}-qualify/`：status.json/task.log/control-build.json，NPU cpu.xml/checks/result.json/queue-state.json。
   两unit复核inactive/success/0，持久passed/exit0；NPU物理9→npu:0，四组合通过。
   范围四循环24配置×两dtype、KV Full4配置×两dtype、CPU24；CUDA111步构建完成。
   launchers `.local/core5-half-memory-{npu,cuda}-qualify{,-workload}.sh`；1200秒、2workers/BLAS1、Nice10/background.slice、queue180秒/device360秒。
2. **31e017f公共图FP16资格仍running。**
   checkout `artifacts/worktrees/core5-sdk` 与 `artifacts/worktrees/core5-cuda` 固定31e017f，跟踪文件只读，不能修改。
   units `tide-core5-half-flow-{npu,cuda}-qualify2-20261028.service`；各自build/weighted-half-flow。
   outputs `artifacts/jobs/core5-half-flow-{npu,cuda}-qualify2/`，launchers `.local/core5-half-flow-{npu,cuda}-qualify2{,-workload}.sh`。
   范围完整推理21图/15配置各两dtype、CPU27；CUDA三架构build/loader/help。1200秒、2workers/BLAS1、queue180秒/device600秒。
   首次无2后缀任务failed/exit1：官方build入口拒绝已存在build目录，未编译/运行数值；保留原记录，新任务改新build目录。

公共图31e017f已实现：half队列/raw/实际发送payload，FP32局部算术与量化参数bank；实际payload journal用于解码。
正c存储下溢拒绝，G下溢允许，溢出拒绝；CPU guard/packing/raw27通过。最初误指不存在测试路径未收集，纠正后通过。
**参考问题已发现，尚未修复/资格：**原half eager默认scoring被拒绝，且逐算子half舍入不等于resident声明的FP32算术。
必须使用独立CPU原数学函数与调度、显式raw/payload存储舍入参考；不能仅放宽dtype/路由比较或把投影等价冒充完整记录等价。
活动资格仍检验原31e017f；若失败保留它，不修改活动源码。

## 未提交实现与确切下一步

主区正在扩展half公共训练：adjoint/rolling/raw/window根FP32；half journal反向route；half owner/FP32 masters；发布先half舍入再拓宽bank；cotangent独立要求FP32。
涉及python/tidegraph/weighted/resident_training.py、scripts/device_component_checks.py及tools/device_online/weighted_{flow_adjoint,reverse_route,training_*,window_roots,window_vjp}与CMakeLists。
新增未跟踪 `tools/device_online/weighted_precision_reference.{h,cpp}`：独立FP32原数学函数、raw/发送half存储及identity-cast VJP；保留alias与int64。
训练轨迹扩为两dtype，half reference使用独立FP32 master与量化发布；**尚未编译/测试，不算通过**。

下一步：
- 修正flow/config/boundary及Python half参考；完整记录字段dtype必须明确，输入/checkpoint参考拓宽，candidate仍实际half。
- 完成训练实现、文档、CPU测试后提交；新固定源码资格含C++与Python-owned公共调用、完整独立根、多步优化器、磁盘/新进程恢复。
- 之后继续peer、有限profiling、最终追溯；不要在局部组件阶段结束总体任务。

短检查：`python scripts/status.py`；`python scripts/check_upstream.py`。
任务检查：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止：`systemctl --user stop UNIT`。
不要把live job写为passed；旧失败不因新通过而改写。

## 保留限制与环境

- eager FP16 GDN默认AdamW约.001984严格差，strict xfail；SGD不解除。
- 默认异步双NPU完整Python VJP的CANN invalid event；显式launch blocking通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a同见证；原输入失败保留。
- CUDA全部真机待硬件；不阻塞本机其余实现。

实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，最近空间约3.3TiB。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与 `/etc/modulefiles`；旧guide失效且已获授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加本项目.local/venv-core5/bin提供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0；保留模块CANN PYTHONPATH。
不预建job.py输出目录，service stdout写.local；官方build入口要求新build目录。
STATUS用scripts/durable_records.py原子写并回读；公共环境与参考仓库未改。
