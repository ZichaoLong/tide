# 当前交接 — 2026-10-28

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 `graph-execution-foundation`。
最近证据提交 bf71bae 已推送；本提交包含公共FP16训练及独立精度参考。用户授权持续实施、私有环境、有限实验、分阶段提交与正常推送；不用子代理，参考仓库只读。
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
2. **31e017f公共图FP16：CUDA passed；NPU终态failed/exit1。**
   checkout `artifacts/worktrees/core5-sdk` 与 `artifacts/worktrees/core5-cuda` 固定31e017f，跟踪文件只读，不能修改。
   units `tide-core5-half-flow-{npu,cuda}-qualify2-20261028.service`；各自build/weighted-half-flow。
   outputs `artifacts/jobs/core5-half-flow-{npu,cuda}-qualify2/`，launchers `.local/core5-half-flow-{npu,cuda}-qualify2{,-workload}.sh`。
   范围完整推理21图/15配置各两dtype、CPU27；CUDA三架构build/loader/help。1200秒、2workers/BLAS1、queue180秒/device600秒。
   首次无2后缀任务failed/exit1：官方build入口拒绝已存在build目录，未编译/运行数值；保留原记录，新任务改新build目录。

公共图31e017f已实现：half队列/raw/实际发送payload，FP32局部算术与量化参数bank；实际payload journal用于解码。
正c存储下溢拒绝，G下溢允许，溢出拒绝；CPU guard/packing/raw27通过。最初误指不存在测试路径未收集，纠正后通过。
**已修正主区参考边界：**完整图/配置/边界与训练检查使用独立CPU数学及调度，在raw和发送处显式half存储。
每类字段dtype分别检查后，无损拓宽half以比较全部记录。公开训练Full根另用舍入前G，不能用已存储payload替代。
原31e017f NPU FP32完整图已通过，随后FP16失败；原日志保留，配置检查尚未运行。新21ed86d资格另记。

## 本提交的实现与下一步

公共half训练已实现：adjoint/rolling/raw/window根FP32；half journal反向route；half owner/FP32 masters；
发布先half舍入再拓宽bank；cotangent独立要求FP32；恢复校验named等于master舍入值。
两dtype standalone五轨迹20更新/完整独立根、生命周期拒绝、累计检查；Python八轨迹三更新、磁盘/新进程恢复已扩两dtype。
独立精度参考 `tools/device_online/weighted_precision_reference.{h,cpp}` 与 `tests/weighted_precision_reference.py` 只参与检查。
新增KV手算key/summary/weight根与RMS Jacobian/外积检查，不以两个调度器互比代替局部公式锚点。

CPU开发命令已通过32项/3.90秒：
`python -m pytest -q tests/test_weighted_storage_reference.py tests/test_resident_weighted_payload_precision.py tests/test_resident_weighted_flow_packing.py tests/test_resident_weighted_reverse_route.py`。
C++训练扩展尚未构建/真机资格，Python-owned尚未运行；不要把源码已实现当作资格通过。

实现21ed86d019d5dcbe3bffa363be02506433e23a7f已冻结到 `artifacts/worktrees/core5-half-training`，三个独立新目录 build/npu、build/cuda、build/python：
- `core5-half-training-npu-qualify`：standalone两dtype完整图/配置/训练，CPU32；NPU queue180秒，gate600秒。
- `core5-half-training-python-qualify`：Python-owned完整图/训练/手算根与磁盘/新进程恢复；queue180秒，pytest600秒。
- `core5-half-training-cuda-qualify`：三架构选定完整图/配置/训练build/loader/help；无GPU数值结论。
新任务均1800秒、2workers/BLAS1、Nice10/background.slice；已启动，均核对active/running、Transient=yes、Slice=background.slice；暂无新资格结果。
units均为`tide-JOB-20261028.service`，输出`artifacts/jobs/JOB/`、launcher `.local/JOB{,-workload}.sh`。
固定checkout跟踪文件只读，三个任务运行期间不可改动。

短检查：`python scripts/status.py`；`python scripts/check_upstream.py`。
任务检查：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止：`systemctl --user stop UNIT`。
下一步核对固定source与干净工作区后启动上述资格，跟进原NPU终态；根据真实失败修复再固定新提交。
其后继续peer、有限profiling与最终追溯；总体未完成。

## 本提交跨卡增量（待固定资格）

活动21ed86d checkout保持只读；主区新增样本副本step_replicas：设备梯度求和、精确optimizer状态校验、
全部proposal→共同拒绝决定→commit/publish，独立raw状态不切分。接口/契约在resident-weighted-peers.md。
新文件tools/device_online/weighted_training_peer.cpp、ascendc/tide_weighted_peer_equal.cpp及CPU guard测试；
公共header/binding/Python/CMake及standalone/Python两dtype多步、完整根、恢复与共同拒绝检查已接入。
尚无多卡构建/数值资格，不能宣称跨卡已通过。
CPU guard首跑3通过/1失败：测试传入temporary.clone().data_ptr()后clone提前释放；改为持有clone后4项通过/0.76秒；原失败保留此处。
短C++语法检查已通过：独立参考、half轨迹、peer函数及peer检查；不是设备构建资格。
Python接口/测试及检查注册模块AST通过。下一步等当前三任务终态并审阅结果，再冻结新提交资格；不自动重复已通过的完整矩阵。

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
