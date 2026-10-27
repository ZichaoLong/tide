# 当前交接 — 2026-10-28

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施、私有环境、有限实验、阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、packed/VJP/计算共享、
CPU完整训练及恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
Python/native有限mixed、显式同步双NPU、half/master及standalone恢复范围见ROADMAP；严格失败保持。
常驻各数值组件与显式局部VJP已有固定资格，完整图训练仍需组合。

最新实现5efce32471f3b407b25fa669706e56e88436a3e9：完整weighted resident单设备FP32推理。
设备决定就绪、选择、队列、路由与推进；NPU局部阶段用TaskServices通知握手和串行共享workspace。
五记忆、三Read快照、十种发送/正式配置、异宽来源、一般端口、完整记录与raw checkpoint；
独立C++/Python入口、TideWeightedResident安装包及跨DSO配置识别已实现。
固定资格evidence/core5-resident-weighted-flow-20261028.{md,json}：
- CPU50项/22.91秒、kernel FP64/FP32、DSO双向检查通过。
- standalone NPU metadata5项、services、21图/边界、15配置及安装消费者通过；物理3→npu:0。
- Python-owned NPU23项/24.14秒（18公共图+5metadata）通过；物理1→npu:0。
三个unit tide-core5-weighted-flow-{cpu,npu,python}-qualify-20261028.service均inactive/success/exit0。
输出artifacts/jobs/core5-weighted-flow-{cpu,npu,python}-qualify/；status/task.log、build manifest、checks和consumer日志。
源码均clean 5efce32；checkout分别artifacts/worktrees/core5-{bootstrap,sdk,device}，目前无活动reader。
原开发manifest保留；复用编译源码/core/binary摘要核验，安装规则变化只重配置，不声称重编。
开发dev1–8、Python dev1–3、安装末尾rg127及CPU fixture缺字段均保留原失败/取消终态，精确tree见本次机器报告。

## 当前工作与下一动作

前向证据3d1bd20与实现5efce32已正常推送。
CUDA开发core5-weighted-flow-cuda-dev2已inactive/success/exit0；冻结tree9f02feaa11168e4473270f30d8efbc2c2032ec2f。
三架构所有CUDA编译命令集合精确80/90/100，完整图/services/config及安装consumer构建/loader通过；无真机。
输出artifacts/jobs/core5-weighted-flow-cuda-dev2/；checkout artifacts/worktrees/core5-cuda目前无活动reader。
旧cuda-dev tree78b86b672ce185f0341ae4cbbb2baaab1c00a5cf failed：Torch隐式加sm50，peer __nanosleep编译失败。
CUDA修复8eebceccc8e209a6f5f08caf4d262a4e388d280b固定资格通过：架构8项/.12秒、DSO、完整图/services/config三架构构建与安装loader。
证据evidence/core5-weighted-flow-cuda-20261028.{md,json}；unit tide-core5-weighted-flow-cuda-qualify-20261028.service inactive/success/exit0。
源码clean 8eebcec，检查159个配置CUDA编译命令仅80/90/100；实际所选目标构建，无GPU真机。
输出artifacts/jobs/core5-weighted-flow-cuda-qualify/；checkout artifacts/worktrees/core5-cuda无活动reader。
实现和本次证据待正常推送；不重跑无关NPU数值。
完整数值阶段VJP开发已写入主工作区：weighted_flow_{adjoint,node_vjp,region_vjp}与阶段检查，
复用forward局部绑定、独立raw/Describe/Content/control/Full根、角色切片/canonical共享参数；尚未构建/实机验收。
任务core5-weighted-stage-dev正在构建，active/running：
- unit tide-core5-weighted-stage-dev-20261028.service；冻结artifacts/worktrees/core5-sdk（tree1e3813f8d1d9ca6fae998ee6ea565d55a6022b9d，已归档source.tree/source.tar），不得编辑。
- 输出artifacts/jobs/core5-weighted-stage-dev/；launcher .local/core5-weighted-stage-dev{,-workload}.sh。
- 1800秒、2workers、BLAS1、background.slice/Nice10；queue180秒、阶段检查600秒；重编整个flow及新check，
  新参数归约kernel按需构建，其余逐源码/二进制比对已有dev9和manifest。
- 15个五记忆×三Read数值阶段配置/各独立根与完整forward回归；结果未知。
检查`systemctl --user show tide-core5-weighted-stage-dev-20261028.service -p ActiveState -p Result -p ExecMainStatus`及status/task.log/checks。
停止`systemctl --user stop tide-core5-weighted-stage-dev-20261028.service`。
继续完整retained VJP/optimizer和精确训练恢复：反向读取候选实际device journal，链接消息生产者/状态前驱，
复用已验局部VJP并汇总canonical参数；不得CPU参考重放或主机逐事件推进冒充resident。
随后FP16/peer、有限profiling与代表性完整成本选型、最终追溯；不做全部开关笛卡尔积。

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
