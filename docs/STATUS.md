# 当前交接 — 2026-10-27

固定上游0770593838a37c790a568226930567ed60f99b83；授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
已推送e875dc1（含2814b87常驻Alloc及固定CPU/NPU/CUDA构建资格）。总体未完成，下一阶段见[ROADMAP](ROADMAP.md)，不在阶段提交后暂停。

## 当前实现与证据

一般SettleGraph、结构化SrcClass/int64、五记忆/短卷积、正式发送与评分、
实际packed/VJP/精确计算共享、standalone完整训练/checkpoint的分阶段CPU证据
从[证据索引](evidence/README.md)进入；各定向资格不等于全部矩阵。

[470ce06 FP32设备](evidence/core5-device-20261027.md)：CPU200，Python/native mixed-a/b各2、
mixed-c各19、显式同步两卡各2组合。默认异步及SDK严格见证未解除。
[8c8824f half CPU](evidence/core5-half-20261027.md)：99passed/2strict xfailed，后两项不计通过。
归一化/Attention内部FP32、一阶VJP、int64先相减的长时差与KV位置、独立公式及训练恢复。
[01689c0 half设备](evidence/core5-half-device-20261027.md)：CPU12；Python/native各mixed-c
AdamW4+SGD2、mixed-a/b各1、显式同步两卡各1，均完整反向/四步/raw与master精确恢复/CPU handoff。
native half profile35945kernels/1222hostscalar/1368memcpy，32candidate/31Full/44messages，
20设备AiCPU；是有限算子归属调查，不是吞吐排名或完整FP16配置认证。

## 当前阶段与活动任务

[670ead3 standalone masters/schema2](evidence/core5-native-master-20261027.md)：
CPU固定资格24项通过；独立SDK2.10/CANN9.0构建、loader与实际NPU FP16训练通过。
显式SGD24图、AdamW的fc/kv和feedback/delta两图；均包括精确raw/master/slots恢复和设备→CPU第四步。
开发门禁native-master-dev20项、final-dev9项、handoff-dev9项及原manifest/snapshot保留。
固定CPU、SDK build/qualify均已inactive/success，具体job/commands/hashes见证据JSON。
loader检查的shell rg缺失，已单独Python验证同一loader文件（通过），未重跑workload。
bootstrap/sdk/device/resident/CUDA均无活动reader；修改冻结worktree前仍核验对应unit终态。

CUDA私有栈已重新定位并读其现行README：/mi/data2T/zlong/gpu-toolchains，激活activate-cuda.sh；
Torch2.10.0+cu128/Python3.11.15/Toolkit12.8.1，available=false。
准备tide-core5-cuda-eager-qualify-20261027：干净670ead3 worktree artifacts/worktrees/core5-cuda，
output artifacts/jobs/core5-cuda-eager-qualify，.local/core5-cuda-eager-qualify.sh；
首次配置因服务PATH无Ninja失败；原failed保留。第二次launcher准备未执行而服务引用缺失脚本，exit127，
无workload，postmortem已保存。准备独立tide-core5-cuda-eager-tools2-qualify-20261027，
对应同名job/.local launcher，追加项目私有Ninja入口，使用新build/cuda-tools2。
新CUDA-linked核心构建/2workers/1800秒，以上四个CPU测试入口验证新栈，GPU真机仍待验。
当前未提交：weighted resident Alloc组件与显式精确解VJP（quad/bern/ent），文档/29项独立CPU源码检查通过。
首次resident-allocation-dev（tree b05f7226）CPU29通过，但AscendC编译因min/max/abs名称与内建冲突失败，
未运行设备；原failed及源码包保留。已修正限定名，并让紧凑行由单一scalar worker拥有，避免cache line共享写。
第二次build2-dev（tree9b925b05）CPU29通过，kernel/host编译通过；check可执行文件缺-fPIC导致链接失败。
已按现有standalone检查规范补-fPIC，未运行设备，两次失败记录均保留。
build3-dev（tree1ecb80b3）完整SDK/AscendC构建、链接、loader通过；运行因launcher覆盖CANN PYTHONPATH，
初始化找不到tbe，未运行数值检查。原failed保留，构建manifest已存job目录。
runtime-dev已终止failed：tbe路径已修复；SDK不支持at::tensor(bool列表)，fixture构造在求解前失败。
改用int32→bool建立完全相同的独立根，增加rule/row/field数值诊断，不改输入、算法或容差。
build4-dev（tree748c3735）完整SDK/AscendC构建与真实NPU FP32开发检查通过，exit0，已inactive。
六种regularizer、同program两组长度/输入重放、72物理行/60非空帧，完整六输入VJP/None与零、
失败不提交/sticky error/容量拒绝通过。构建、checks/result、source.tar/tree保留在对应job。
组件CPU独立公式29项已通过；修复只涉及SDK检查构造与诊断。提交实现后以干净固定源码
资格CPU29/NPU同检查，并独立构建CUDA组件（无GPU）；不将开发snapshot计正式资格。
2814b87固定源码CPU29（2.67秒）、NPU六种分配配置/完整VJP/重放与拒绝通过；
独立CUDA sm80/90/100组件/loader构建通过，无GPU。npu-qualify和cuda-qualify均clean/exit0/inactive。
证据core5-resident-allocation-20261027，原manifest/build4 source字段和所有开发failed记录保留。
resident worktree无活动reader，仍固定2814b87。当前未提交：weighted sending常驻原语及其独立CPU测试34项通过，连同Alloc共63项通过；
Direct/Unit/Pooled attention、proportional、sorted fractional fill与hard，完整五输入VJP、int64 stable IDs。
sending-dev（treeeb57f3c5）CPU63通过，AscendC拒绝bool直接转float；未运行数值检查，原failed保留。
用明确1.f/0.f写相同饱和掩码；build2-dev（tree6d94073d）CPU63、SDK构建/loader、
真实NPU FP32八配置完整VJP/96物理行80非空帧/两组输入长度重放/拒绝与容量全部通过，已inactive/exit0。
原failed保留。提交sending实现后固定源码进行CPU34+NPU发送资格，CUDA三架构组件构建分开。
下一步区域控制/Query的真实批量vendor head及完整SelStep组合，再接图生命周期。
完整weighted resident仍需Agg、typed state/五记忆、Describe/全部发送、图调度、训练/恢复/跨卡。
CUDA tools2干净670ead3已构建、CPU24项（143.99秒），显式不可用拒绝通过，证据已推送；GPU真机未验。

## 未解除的真实失败

- half GDN默认AdamW：Python/native nodes.5.kernel.conv.weight更新严格差异约.001984；
  专用GDNAdamWParityError strict xfail，只接受该字段；显式replay/标量舍入顺序诊断仍失败。
  未改输入/阈值，SGD通过不解除。half-normalization-dev175pass/2fail、half-clocks-dev95pass/2fail保持failed。
- 默认异步两卡完整Python VJP：CANN invalid event；core5-peer-event及current0/1/serial复现保留。
  显式ASCEND_LAUNCH_BLOCKING=1的FP32/half通过不解除默认异步限制。
- standalone SDK2.10原FP32门禁：fc/affine第三轮evidence误差3.09944e-6；mixed-a相同。
  前两轮forward/grad/update通过，小梯度经AdamW累积；未执行图不推定通过。
  core5-sdk-affine/core5-sdk-affine-drift诊断及原failed状态保留。

## 环境、资源与操作

实际路径/mi/data2T/zlong/home/llm/graph-execution-foundation；zlong，约3.3TiB空闲。
现行规范/opt/software/ascend-locks/dev-workspace-9.1.0/README.md与/etc/modulefiles；旧指南已不存在。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate。Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1/GCC10.3.1。
SDK单独shell：module load libtorch-npu/2.10.0-cann9.0.0；ACL_OP_INIT_MODE=0；
PATH前置$LIBTORCH_ROOT/../../../../bin提供匹配NumPy，显式sdk_python使用同栈Torch。
公共包未修改。长任务background.slice/Nice10、ATen/BLAS单线程、2build workers、新输出/有限时长。
活动reader源码不可编辑；复用前核验C++及全部binary摘要，保留原manifest的build source字段。
检查python scripts/status.py；systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus；
读取status.json/development.json/task.log。停止systemctl --user stop UNIT。
不恢复旧队列，不重跑旧矩阵；失败保留，源码/构建/实机与性能证据分别记账。
