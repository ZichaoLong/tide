# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
最近推送1289bc9；d072c17 SDK/Python任务仍在运行，CUDA构建已passed。修正检查器与运行库源码身份分列。
用户授权持续实施、有限实验、提交与正常推送；不用子代理。总体goal active，**未完成**。
唯一待办[ROADMAP](ROADMAP.md)，持续规则[execution-flows](execution-flows.md)；参考仓库/公共工具链只读。

## 当前完成范围

公共core-5、一般Settle与端口、五记忆及正式发送/评分/证据/key、独立调度、packed/VJP、
mixed、常驻FP32/FP16完整图/训练与恢复从[semantic-map](semantic-map.md)进入。
[完整half图/训练](evidence/core5-resident-half-training-20261028.md)及
[样本副本/master复验](evidence/core5-resident-weighted-peers-20261028.md)保持各自固定源码资格。
样本副本只处理独立私有状态，不能替代单样本节点切分。

[ceae7bf有限性能](evidence/core5-weighted-performance-20261028.md)：24配置22通过，CPU反馈PDG两项严格失败。
完整调用/训练含调度、packing、返回、VJP、optimizer和同步；本组小中规模CPU成本最低，
medium HB standalone prefill约350ms/streaming456ms，NPU mixed-c约1.41–1.61s/resident2.60–2.71s。
仅限声明尺寸；无稀疏发送、多卡或CUDA吞吐结论。旧profile失败与修复经过原报告导航。

## f72a6cc启动/内存补充已终态

[报告](evidence/core5-weighted-observation-20261028.md)和同名JSON，审阅脚本`.local/record-weighted-observation.py`。
四个新进程均small KV/HB FP32/SGD，1warmup+1测量窗，完整独立记录/loss/最终参数通过；
CPU记录器5项通过，CPU/NPU SDK/CUDA三个consumer build/loader/help通过。CUDA没有真机。
启动至ready约CPU Python1.448s/standalone0.313s，NPU Python15.530s/SDK7.965s；
边界含首个设备算子/同步与库验证，未清OS缓存。两NPU都物理9→0，项目计时串行，未排除其他用户争用。
测量期allocator allocated峰值约83.10/75.10MiB；不含所有vendor/driver内存，不作大模型容量推断。

源码`f72a6cc2f78befdcda1f93a9ed00d774a54030f7`，冻结checkout
`artifacts/worktrees/core5-weighted-observation`，跟踪文件只读，build/{cpu,npu,cuda}独立。
原core/组件身份与binary、loader、源文件、输入、日志和结果hash都审阅，未混用Torch2.12与SDK2.10。

jobs：`core5-weighted-observation-{build-cpu,build-npu,build-cuda,cpu,npu-python,npu-standalone}`。
units：`tide-JOB-20261028.service`；launcher`.local/JOB.sh`及可选`JOB-workload.sh`；
记录`artifacts/jobs/JOB/{status.json,task.log}`，测量项另有`*-process/`和`*-run/`。
六个job均持久passed/exit0；提交时确认background.slice/独立cgroup，成功瞬态unit已回收，
最终show为inactive/not-found；不能把默认Result/ExecMainStatus当终态依据，实际以持久status及输出校验为准。
两worker/ATen及BLAS1/Nice10，构建600秒、每候选180秒、NPU queue180秒/整测量job420秒。
**d072c17节点放置资格即将提交。**

job `core5-weighted-node-placement-{npu,python,cuda}-qualify`，unit `tide-JOB-20261028.service`；
冻结checkout `artifacts/worktrees/core5-weighted-node-placement`，跟踪文件只读；build/{npu,python,cuda}。
launcher `.local/JOB.sh`、`JOB-workload.sh`；输出`artifacts/jobs/JOB/{status.json,task.log}`。
NPU standalone另有checks/和cpu.xml；Python有tests.xml；CUDA只有三个help和build manifest。
构建1500秒/两个worker，设备资格900秒/queue180秒，整任务2640秒；background.slice/Nice10/ATen及BLAS1。
三项结果均待审；不编辑活动checkout。CUDA无真机；不重跑原24项性能矩阵。
启动后复核发现d072c17通用资格脚本漏传weighted-node-peer的必需output-dir；主工作树已补传
独立case目录，并把两个weighted双卡检查从默认单设备列表移除。原NPU任务预计会在新检查入口退出，
尚未实际失败；保留其真实终态，不修改活动脚本。构建成功后可在新冻结源码用同component hash
复用已建SDK二进制做新资格，明确区分组件源码d072c17与修正后的qualifier源码。
修正入口357477b已冻结到`artifacts/worktrees/core5-node-qualification-entry`。
后继job `core5-weighted-node-entry-npu-qualify`在等待依赖时已主动取消，持久cancelled/exit143，
unit MainPID0/failed；没有排队申请设备或执行资格。原因是反例复核发现检查器原先只给首节点输入，
队列=2会在后续阶段拒绝，不能断言所有节点仍为初态。本提交改为首区域两个跨卡节点同时输入，
用独立CPU参考确认第一阶段2个作用/4条消息/4条pending，再检查共同拒绝未采用任何state/history。
另补实际raw/Full/参数bank设备断言和等长错误后端列表，防止仅返回请求配置而未真实放置。

为了避免重复编译完全未改动的vendor内核，待原SDK构建完成后，可单独编译修正后的C++检查器，
链接d072c17已建运行库。先验证两提交`tools/device_online`除`weighted_flow_peer_check.cpp`外逐字节相同，
并核对原manifest全部binary hash；分别记录运行库源d072c17与检查器新源码。常规干净重建亦可复现。
原d072c17脚本预期入口失败仍须等真实终态；单设备flow/training回归用其未改动检查器独立验收。

修正检查器1289bc9冻结到`artifacts/worktrees/core5-weighted-node-check`。
`core5-weighted-node-check-cuda-qualify`保留failed/exit1：新检查器6步编译/链接及loader/help成功，
辅助脚本错误要求安装前后运行库字节摘要相同。审计确认ELF仅.dynstr/.dynamic改变，符合CMake去RPATH。
`core5-weighted-node-check-npu-qualify`在等待依赖时主动取消，持久cancelled/exit143，无设备运行。
不修改两旧任务结果或其输入。新的`.local/build-weighted-node-check-v2.py`在经manifest校验的私有
库副本上执行CMake RPATH_REMOVE，再与installed库逐字节核对，并记录安装前/后摘要及实际loader路径。

`core5-weighted-node-install-{npu,cuda}-qualify`在job.py启动前因输出目录已存在而失败，
捕获Traceback与postmortem记录保留；未构建/未申请设备。不要预建job.py要求独占创建的输出目录。
新jobs `core5-weighted-node-install-{npu,cuda}-v2-qualify`即将提交，units `tide-JOB-20261028.service`。
冻结源码仍1289bc9，只编检查器；链接d072c17原运行库。每项依赖等待900秒/构建360秒(2worker)，
NPU双卡queue180秒/设备900秒，整job2340秒、unit2400秒，background.slice/Nice10/ATen及BLAS1。
launcher `.local/JOB.sh`、`JOB-workload.sh`、NPU另有`JOB-device.sh`，均只读。
记录`artifacts/jobs/JOB/{status.json,task.log,build/result.json}`；NPU另有checks/和single-device-regression/。
结果待审；CUDA仅build/loader/help，没有真机。

## 原工作树在实施的下一步

1289bc9/d072c17前向资格见上文；Python-owned50项、修正SDK两dtype各9配置已通过，
CUDA修正检查器build/安装RPATH逐字节复现/loader/help passed。SDK单设备flow/training回归仍运行。
尚未形成前向证据提交；等待任务真实终态后审阅、记录并单独提交。

原工作树未提交的窗口反向实现：`weighted_flow_peers`增加journal旧态请求和局部根/VJP响应，
`weighted_flow_views`/region_vjp读取coordinator实际字段；`weighted_window_snapshot`保存放置，
边界gather真实raw并计预算；`weighted_window_vjp/owner`保持逆序/祖先路由/参数归约在coordinator，
局部tape/VJP在原owner。相关模块契约已更新；6个实现TU SDK syntax-only通过（非设备证据）。
新`weighted_window_peer_check.cpp`定义4组合×12独立根与2个三窗桥；复用独立CPU调度/autograd。
`weighted_retained_case.cpp`从旧checker提取，旧入口保持8组合。4个检查器TU SDK syntax-only通过；CPU静态owner定义域8项通过，git diff --check通过。
公共训练放置、optimizer发布与FP16 placed反向仍未接通/资格，不能据此前向或内部源码宣称完成。

接着执行：
1. 审阅`core5-weighted-node-install-npu-v2-qualify`全部终态，原失败/取消保持独立记录；提交前向证据。
2. 本次窗口反向实现提交后冻结到`artifacts/worktrees/core5-weighted-node-window`，跟踪源码只读。
   jobs `core5-weighted-node-window-{npu,cuda}-qualify`，units `tide-JOB-20261028.service`，
   launcher `.local/JOB.sh`及`JOB-workload.sh`，记录`artifacts/jobs/JOB/{status.json,task.log}`。
   SDK构建weighted-node-window/window/training；双卡新4组合×12根与2个三窗桥，单设备
   HB/proposal与feedback/old窗口、FP32/FP16训练必要回归；CUDA三架构只build/loader/help。
   构建1500秒/2worker，NPUqueue180秒/设备2400秒，整job4140秒、unit4200秒；
   background.slice/Nice10/ATen及BLAS1。当前尚未提交服务，不称资格通过。
3. 完成公共training的node_devices构造/恢复、共同optimizer与参数bank发布，FP32/FP16训练/恢复分别验收。
4. 最终公共接口/能力矩阵/追溯审计。CUDA真机继续独立待验，不阻塞本机工作。

## 环境与必要保留

实际cwd`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，可用磁盘约3.3TiB。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`和`/etc/modulefiles`；旧guide已失效。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加本项目.local/venv-core5/bin供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0、ATen/BLAS1，保留CANN PYTHONPATH。
长任务固定源码、独立background.slice/有限时长，不编辑活动checkout；STATUS用durable_records原子写并回读。

严格失败保留：eager FP16 GDN默认AdamW约.001984 strict xfail；默认异步两NPU Python VJP CANN invalid event
（launch blocking通过不解除）；SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6/mixed-a同见证；
ceae7bf FP32 PDG性能两项maxabs4.11272e-6/2.71201e-6。独立FP64提升完整记录通过，不解除原失败。
PDG最小复现保留`artifacts/core5-weighted-measurement-fixtures/kv-pdg-medium.json`、原CPU性能job两result/log、
CPU consumer、`core5-weighted-pdg-rounding-diagnose/diagnosis.json`及`.local/diagnose-weighted-pdg-rounding.py`。
原672e948 standalone bad_alloc检查器失败不改写；7db0b08修复后master/peer复验通过。
原诊断9个kernel_meta文件带hash移存旧诊断job/cann-kernel-meta，未删除，不恢复旧队列。

重入：`git status --short --branch`；`python scripts/status.py`；`python scripts/check_upstream.py`。
任务：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`并核对持久status；停止`systemctl --user stop UNIT`。
