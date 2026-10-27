# 当前交接 — 2026-10-28

固定上游0770593838a37c790a568226930567ed60f99b83，授权分支graph-execution-foundation。
用户授权持续实施/私有环境/有限实验/阶段提交推送；不使用子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为完整待办，[execution-flows](execution-flows.md)为持续契约。

## 已实现与固定证据

一般SettleGraph、SrcClass/int64、五记忆/逻辑时间短卷积、正式配置、packed/VJP/计算共享、
CPU完整训练与恢复从[证据索引](evidence/README.md)进入；定向资格不认证未测矩阵。
Python/native有限mixed、显式同步双NPU、half/master及standalone恢复范围见ROADMAP；严格失败保持。
常驻Alloc/发送/head/Region/Aggregate/来源低秩/canonical参数/raw state/卷积/共享投影/实际KV Upd、
pool、sum/qry Mem及KV Full已逐项固定资格；均为局部组件，不代表完整weighted resident图。
CUDA源码/三架构构建通过范围见各报告，无GPU真机；不自动重跑历史矩阵。

本段最新固定资格：
- 1eb4b72：旋转/周期归约与真实巨大age数值转换。CPU13、NPU rotary及KV sum完整回归、CUDA三架构。
  evidence/core5-resident-rotary-20261027.json；开发dev/dev2设备失败均保留，dev3 tree7207e6dd66b334935329d3ce896863b1f4f38702通过。
- 8b1be80：完整qry Mem。CPU38、NPU120行四根及全部参数/输入/rawVJP、CUDA三架构。
  evidence/core5-resident-kv-query-20261027.json；dev启动脚本路径错误exit4保留，dev2 treeef1c888bdb4a9d39f085deaeda7e7e25fb59f062通过。
- 386970a9dcd0b826e390be020c2a86ea0c7a3437：公共Full残差与多头KV Full。CPU50、NPU residual48行/Full96行，
  全参数/input/raw VJP、保存版本、None/零及容量；CUDA三架构构建/loader。
  evidence/core5-resident-kv-full-20261027.json；开发tree5c4d2c92c6335c6c330d7254376e5187b017dbaf通过。
  core5-resident-kv-full-{npu,cuda}-qualify均inactive/success/exit0；无活动reader。
所有NPU固定资格保留原开发manifest并核验全部源码/core/binary摘要。

## 当前工作与下一动作

HEAD/已推送344276c；配置接口ce21a8359146854696af786379b48367756fd456固定资格通过：
Python/native45项/21.78秒，C++kernel FP64/FP32均通过。
证据evidence/core5-weighted-binding-20261028.{md,json}；unit tide-core5-weighted-binding-qualify-20261028.service inactive/success/exit0。
Describe cd576d2、四循环Upd/Mem/Full 6e2ecb7固定资格见各自20261027/28报告；CUDA三架构仅构建/loader。

未提交增量：weighted_flow*、三个metadata kernel、公共C++/Python入口、契约与完整图检查。
完整weighted resident图已通过下述开发设备验证；当前FP32 no-grad单设备入口不代表训练闭环。
完整图各阶段现拆为TaskServices的设备通知程序：source、Aggregate、Upd、Mem、Describe、Region、Full；
coordinator仍在设备上决定就绪/选择/路由/队列/状态提交；没有主机逐事件推进。
CUDA同源码仍追加到conditional graph。总tensor跨程序唯一计账，串行阶段共享最大需求workspace。
dev6 tree e92e0d50554d8b9839e681f3c8c07470260aa713已failed/exit1：构建/loader通过，
首个图在逐服务累加workspace预算时拒绝，未完成前向。严格通知顺序使数值阶段互斥，
已改为共享按最大需求分配的workspace，不提高原预算；此改动尚未验证。
dev7 treece2603ddb84acd8d4d2d656c9a7f788239d28e88仅动态库测试fixture漏selection_width而failed/exit8；
core构建成功，设备未启动。已补正式match配置字段。新SDK任务core5-weighted-flow-dev8，
增加设备通知多轮/空窗口/复用与共享workspace独立检查；动态库双向检查已通过。
dev8 tree53fbe96589efd27e6bd736d6f34322190c7bb65a发现/fast依赖扫描未更新导致flow旧对象布局后主动停止，
未运行设备检查。新dev9使用新版增量helper2，受影响目标整体clean/rebuild，不保留旧flow对象。
unit tide-core5-weighted-flow-dev9-20261028.service；launcher .local/core5-weighted-flow-dev9{,-workload}.sh；
输出artifacts/jobs/core5-weighted-flow-dev9/，1800秒/2workers/BLAS1/queue180秒，各检查360秒。
dev9 tree7059bdd745bd71d6db90560c682b8b82abfac6cb全部通过：services独立循环/空窗口（workspace16777472字节），
21图案例/边界、15配置；物理9→npu:0。unit inactive/success/exit0；SDK checkout已无活动reader。

保留的完整图失败（原始记录artifacts/jobs/对应名称，精确tree在source.tree/source.tar）：
- core5-weighted-flow-dev：tree5bf94a7eccaed71608d1eb04df23721bd905df5b，C++局部配置变量重名；未运行设备。
- dev2：tree97f3f6f3dede48f84da2ba286b3c8b77c6ef1755，共享库已构建，检查程序缺runtime链接；未运行设备。
- dev3：tree36345adf1e4a2f6d987da563373eb3a1ae4136d0，仅复用预检拒绝未参与standalone的binding变动。
- dev4：同tree36345ad，构建/loader成功，CANN507002 stream task buffer full（ACLNN361001）。
- dev5：tree918aeaa37daf8e4a497e356f052437fe77bdd3a9，显式HUGE stream仍同样失败；该方案已撤回。
以上unit均failed/exit1，不由后续成功覆盖；相关最小重现/日志保留。
SDK冻结checkout artifacts/worktrees/core5-sdk供dev9读取；复用build/weighted-flow-dev。
新增TaskServices须显式重编device-control/device-peer/device-services/weighted-flow及两个检查程序，
其余组件逐源码对比dev2归档及core/binary摘要后才复用。

Python任务tide-core5-weighted-flow-python-dev2-20261028.service现failed/exit1：
冻结artifacts/worktrees/core5-device，tree918aeaa37daf8e4a497e356f052437fe77bdd3a9；
输出artifacts/jobs/core5-weighted-flow-python-dev2/，build/weighted-flow-python-dev2。
独立weighted Python模块构建/loader成功，第一项设备测试在构造阶段拒绝installed weighted绑定（1failed/16.38秒），未运行完整图。
原因是core匿名命名空间类型跨独立动态库不可识别；已在主工作区修为具名类型，并新增两份独立静态core的双向检查。
冻结checkout现无活动reader；后续重编core及独立resident模块后再验。
Python首轮dev tree7896685d56b2809d35a3ec9eda8bd83c21f8012c因旧CMake目标守卫失败，原记录保留failed。
资源1800秒/2workers/BLAS1/background.slice/Nice10；queue180秒、pytest600秒。
检查：systemctl --user show tide-core5-weighted-flow-python-dev2-20261028.service -p ActiveState -p Result -p ExecMainStatus
并读对应status.json/task.log/checks或queue-state.json；停止：systemctl --user stop同一unit。

新CPU检查core5-weighted-binding-dso-dev：冻结artifacts/worktrees/core5-bootstrap，
unit tide-core5-weighted-binding-dso-dev-20261028.service；输出artifacts/jobs/core5-weighted-binding-dso-dev/；
launcher .local/core5-weighted-binding-dso-dev{,-workload}.sh；1800秒/2workers/BLAS1/background.slice/Nice10。
运行安装配置/native45项、独立动态库双向检查、C++kernel FP64/FP32；tree7757a47807c9df81dd4d0a8df45cf6466880e516，结果未知。
Python dev3 treece2603dd构建成功、跨库配置识别已修复；第一项执行因旧flow对象读错Impl布局而failed/exit1（1failed/15.73秒）。
Python dev4 tree7059bdd745bd71d6db90560c682b8b82abfac6cb已passed：18项/23.54秒，unit inactive/success/exit0，
物理9→npu:0；五记忆/两执行模式/异构/一般端口/精确checkpoint及版本拒绝通过。
完整重编控制/peer/services/flow/Python入口，复用源码一致的局部数值/内核。
unit tide-core5-weighted-flow-python-dev4-20261028.service；checkout artifacts/worktrees/core5-device；
输出artifacts/jobs/core5-weighted-flow-python-dev4/；launcher .local/core5-weighted-flow-python-dev4{,-workload}.sh。
1800秒/2workers/BLAS1/queue180秒/pytest600秒；原manifest与全部摘要保留。
新CUDA源码/构建开发任务core5-weighted-flow-cuda-dev：unit tide-core5-weighted-flow-cuda-dev-20261028.service；
冻结artifacts/worktrees/core5-cuda，tree78b86b672ce185f0341ae4cbbb2baaab1c00a5cf；
重编build/cuda-tools2后独立build/weighted-flow-cuda-dev，sm80/90/100。
输出artifacts/jobs/core5-weighted-flow-cuda-dev/；launcher .local/core5-weighted-flow-cuda-dev{,-workload}.sh；
1800秒/2workers/BLAS1/background.slice/Nice10，动态库CPU检查与services/full-flow/config构建loader；无CUDA真机。
独立TideWeightedResident安装包/组件及public-header-only手算消费者已通过构建/loader与NPU运行。
任务core5-weighted-flow-installed-dev tree4eb35420624e7526545479ea4c84665b0558d599；
unit tide-core5-weighted-flow-installed-dev-20261028.service failed/exit127，仅末尾rg不在服务PATH；consumer本身exit0/marker通过，物理1→npu:0。
冻结artifacts/worktrees/core5-sdk；输出artifacts/jobs/core5-weighted-flow-installed-dev/；
仅安装CMake改动，不重跑完整图；先严格比对dev9全部编译源码，再重新生成安装规则和manifest。
launcher .local/core5-weighted-flow-installed-dev{,-workload}.sh，600秒/2workers/BLAS1/queue180秒，单次consumer120秒。
当前将前向实现提交；相关开发通过为SDK21图/边界+15配置+services、Python18项、独立动态库与安装消费者。
下一动作：固定实现提交后，NPU/Python以原二进制和完整源码摘要做新输出资格（安装改动只重生成CMake规则）；
CPU以新固定源码重编正确fixture并跑45项及kernel/DSO，CUDA完成当前构建后固定源码核验与安装检查。
记录新unit/source/日志后启动；不把当前活跃CUDA/CPU开发任务写为通过。检查CUDA dev与CPU binding-dso-dev；旧CPU dev冻结fixture缺selection_width，预期新DSO单项会失败，
相关45项与kernel回归仍有独立结果；不将任务整体写passed。修正版已由SDK独立双向通过，后续CPU定向重编验证。
CPU短metadata/loader6项/0.93秒通过；git diff --check/py_compile通过。
停止：systemctl --user stop tide-core5-weighted-flow-dev9-20261028.service。
前向和公共接口通过后提交实现，固定提交资格，证据单独提交推送；接续retained VJP/optimizer、
精确训练checkpoint/跨卡/half，随后有限profiling与选型/最终追溯，不做全开关笛卡尔积。

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
