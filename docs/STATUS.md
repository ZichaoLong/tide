# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
用户授权持续实施、私有环境、有限实验、分阶段提交及正常推送；无子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为唯一待办，[execution-flows](execution-flows.md)为持续契约。
已推送343429d完整FP32公共训练，以及2097dc9/9d547ea原始状态FP16及合法次正规数见证修正。
661a92e新增FP16共享投影/固定时间卷积：统一FP16存储，明确FP32算术、保存值与伴随，最终一次舍入，
存储溢出拒绝、underflow连接保留；契约见resident-weighted-projection/convolution。尚未编译/设备资格。
CPU相关32项通过/2.92秒（projection_norm、convolution、raw_update）；diff --check通过。

## 完整训练固定资格：343429d

固定源码343429db76c091541ae24ef7b3792ce027e9bea2，三个checkout跟踪文件干净，component
`d3e07a3fc3bc378243515dde065f6d766c61c506c0423da0b05b06efe24c2ec3`。
公共WeightedTrainingSession/WeightedResidentTraining已实现实际journal保留、完整字段根、
跨窗raw/pending VJP、canonical共享参数、SGD/AdamW/累计、四分区预算和完整checkpoint。
参数为CPU FP32边界镜像，设备训练FP32；元数据int64。不是half/peer/性能资格。

| 后端 | unit | checkout / build | 已读状态 |
| --- | --- | --- | --- |
| NPU standalone | tide-core5-weighted-training-npu-qualify-20261028.service | artifacts/worktrees/core5-sdk / build/weighted-flow-dev | passed，inactive/success/0 |
| CUDA | tide-core5-weighted-training-cuda-qualify-20261028.service | artifacts/worktrees/core5-cuda / build/weighted-flow-cuda-dev | passed，inactive/success/0；无GPU |
| Python-owned NPU | tide-core5-weighted-training-python-qualify-20261028.service | artifacts/worktrees/core5-device / build/weighted-training-python-qualify | passed，inactive/success/0；物理9→npu:0 |

output均为`artifacts/jobs/core5-weighted-training-{npu,cuda,python}-qualify/`，task.log/status.json。
launchers `.local/core5-weighted-training-{npu,cuda,python}-qualify{,-workload}.sh`；NPU另有-device.sh。
全部1800秒/2worker/BLAS1/background.slice/Nice10；NPU queue180秒/NPU1。不得改活动checkout。
NPU范围copy/saved-window/retained/training/flow/config、两独立安装消费者、CPU28项；
Python训练+flow实际28项（10+18），无skip；原预估33错误，源码完整收集28；CUDA同目标构建/loader/help，不含真机。
已运行`.local/record-weighted-training-evidence.py`，复核源码/core/binary/loader/消费者/systemd与CPU/Python28项。
证据见`docs/evidence/core5-resident-weighted-training-20261028.{md,json}`，本次仅提交证据及相关文档。

## 原始状态FP16固定资格：9d547ea

源码`9d547ea354a54ce73f368c339539d4fbb4a24ed6`，checkout `artifacts/worktrees/core5-raw-half`，跟踪文件只读。
units `tide-core5-raw-half-npu-qualify2-20261028.service`、`tide-core5-raw-half-cuda-qualify3-20261028.service`。
outputs `artifacts/jobs/core5-raw-half-{npu-qualify2,cuda-qualify3}/`；build/npu2、build/cuda3。
launchers `.local/core5-raw-half-{npu-qualify2,cuda-qualify3}{,-workload}.sh`；官方build/verify脚本。
两任务已终态：CUDA passed；NPU failed，FP32完整通过后，FP16在入口被默认RuntimeOptions门禁拒绝（NPU requires float32）。
修正需显式args.allow_npu_float16=true；尚未验证该FP16数值。预算900秒/2worker/BLAS1/Nice10/background.slice，queue180秒。
NPU范围weighted-raw两dtype/每dtype五schema40行、独立FP32字段VJP、位提交/拒绝、16项CPU及三dtype codec；
CUDA三架构build/loader/help及CPU codec通过，尚待汇总摘要。core使用已核验匹配源码的SDK/CUDA已有build。

保留2097dc9失败：npu-qualify构建通过，CPU codec因新增测试将非零次正规数放入必须为零的value字段而拒绝；
9d547ea将见证改到合法keys/H字段。cuda-qualify漏私有ninja PATH，cuda-qualify2误改core路径，均未构建。
新launcher显式补PATH并固定实际cuda-tools2，旧失败不改写。NPU也未在旧失败任务执行数值检查。

## 接下来

投影/卷积实现661a92e已冻结到新`artifacts/worktrees/core5-half-math`，跟踪文件只读，不修改活动raw或training源码。
units `tide-core5-half-math-{npu,cuda}-qualify-20261028.service`；output `artifacts/jobs/core5-half-math-{npu,cuda}-qualify/`。
launchers `.local/core5-half-math-{npu,cuda}-qualify{,-workload}.sh`；task.log/status.json为日志状态。
NPU已主动停止并保留cancelled/143：同一入口未启用half，继续会在数值前拒绝；CUDA仍构建中。
主区未提交：三个check入口显式启用NPU FP16。待CUDA终态后固定这些修正，重新资格raw/projection/convolution。
用新build/npu、build/cuda和独立900秒background服务，官方`build_device_control.py --checks weighted-projection weighted-convolution`
及`verify_device_control.py`检查两dtype；CPU固定32项。CUDA仅构建/loader/help，不冒充真机。
两组件VJP使用实际量化输入上的独立FP32公式；原始时间和int64保持，反向拒绝FP16根。
其后扩展五记忆/Full与完整图FP16、peer、有限profiling及最终追溯；组件通过不代表总体完成。

任务检查：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止：`systemctl --user stop UNIT`。
不预建job.py输出目录，service stdout放.local；STATUS用scripts/durable_records.py原子写并回读。

## 保留问题与环境

完整训练dev4/dev5失败原记录保留：nonaligned/delta/partial update1 restored backward的CANN507011，
plog定位Bool target[2,4]/axis1/source[2,1]的aclnnInplaceIndexCopy；实际调用来自recurrent_detail::insert。
NPU改静态逐行copy后dev6完整原顺序通过；未证明厂商内部根因。诊断在
`artifacts/jobs/core5-weighted-training-npu-dev5/diagnostics/`，固定资格随后验证。其它开发失败从脚本/证据归档追溯。
- FP16 GDN默认AdamW约.001984严格差，strict xfail；SGD不解除。
- 默认异步双NPU完整Python VJP的CANN invalid event；显式launch blocking通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a同见证；原输入失败保留。
- CUDA全部GPU/跨卡/完整训练恢复/性能真机待验。
- C++公共checkpoint目前完整CPU内存记录；Python另验磁盘及新进程恢复。

实际路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，约3.3TiB可用。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`及`/etc/modulefiles`；旧guide失效且已获授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：独立shell module load libtorch-npu/2.10.0-cann9.0.0，Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH加本项目.local/venv-core5/bin提供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0，保留模块CANN PYTHONPATH。
公共环境和参考仓库未修改；上游19文件已重新校验通过。
