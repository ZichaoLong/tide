# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
最近推送d072c17（节点放置前向）；三项资格仍在构建。当前提交修复通用资格脚本的新入口参数，组件源码不变。
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

## 原工作树在实施的下一步

单样本节点放置的前向源码已写入本提交、尚未做设备资格；不是f72a6cc观测证据的一部分。
涉及`tools/device_online/weighted_flow*`、公共header/CMake/check注册、Python resident及placement adapter、
新模块契约`docs/resident-weighted-placement.md`与semantic-map链接。
每个远端设备拥有完整节点局部计算/私有raw状态；coordinator保留全区域选择/队列/journal，
完整纤维请求→Read返回→全局选择→Full返回→队列预检→共同采用。空窗/错误stop协议已写。
原先单设备区域计算拆为实际字段view，远端view只复制实际结果，不执行参考。
CPU静态放置定义域8项开发检查通过；C++编译及双卡资格尚未进行。
放置后的retained VJP、公共训练/optimizer发布尚未接入；暂时显式拒绝从placed flow创建adjoint，继续实施，不能据此前向宣称目标完成。

接着执行：
1. 本提交后建立只读checkout `artifacts/worktrees/core5-weighted-node-placement`；CPU静态8项已通过。
2. 固定源码NPU SDK构建`weighted-node-peer`和单设备`weighted-flow`/`weighted-training`必要回归；
   双卡FP32/FP16九配置、完整记录/跨放置磁盘恢复/空窗/共同拒绝。随后Python-owned适配及CUDA三架构构建。
   `scripts/device_component_checks.py`已注册新检查；`weighted_flow_peer_check.cpp`是真实双卡独立调度对照。
3. 扩展`weighted_window_snapshot`/`weighted_window_vjp`、canonical贡献归约及公共training/checkpoint，
   不把step_replicas包装成单样本切分；反向、共享参数、optimizer与恢复分别验收。
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
