# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
用户授权持续实施、私有环境、有限实验、阶段提交及正常推送；无子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为唯一待办，[execution-flows](execution-flows.md)为持续契约。
本阶段实现提交前基线HEAD/origin为`0f5e0323df7c13b5ab3abe0df54d03a8a1e62d28`。
当前全部源码修改均为本任务，已准备作为一个公共保留训练增量提交。

## 已实现与开发检查

已有固定资格及数学对应从[证据索引](evidence/README.md)、[semantic-map](semantic-map.md)进入。
最新固定实现2622e60的完整窗口VJP：NPU27配置/324根模式、36前向回归、CPU17项及CUDA三架构构建，
[evidence/core5-resident-weighted-window-20261028.md](evidence/core5-resident-weighted-window-20261028.md)随0f5e032推送。
上游19文件本段重新核验通过；没有CUDA真机证据。

本阶段新增实际journal保留snapshot、跨窗口raw/pending VJP、canonical共享参数归约、
公共WeightedTrainingSession/WeightedResidentTraining、SGD/AdamW/数值累计、所有角色发布、
四分区容量准入、精确raw checkpoint、Python磁盘/新进程恢复和独立安装训练消费者。
参数为CPU FP32边界镜像，设备训练FP32；时间/计数等元数据始终int64。
契约：[保留窗口](resident-weighted-retained.md)、[公共训练](resident-weighted-training.md)。
开发通过不代表固定提交资格，不认证weighted resident FP16/peer/性能。

已复核的dirty源码开发结果：
- NPU dev6：tree`74c86493f6d396b2b383e310ba8d336c6a951548`，inactive/success/exit0；
  18owner/54提交布局复制；5配置20更新，三窗链/空窗/重复端口/部分及全部冻结、两优化器、恢复、
  两组数值累计与拒绝检查；推理/训练两个独立安装消费者实际运行通过。
  unit `tide-core5-weighted-training-npu-dev6-20261028.service`；output `artifacts/jobs/core5-weighted-training-npu-dev6/`；
  checkout `artifacts/worktrees/core5-sdk`，build `build/weighted-flow-dev`；物理9→npu:0。
- CUDA dev5：tree`a4e43c152c1d74381b1ea2c9bd8253dcb733e7fc`，inactive/success/exit0；
  copy/retained/training检查和两库、两安装消费者构建/loader/help通过；160个已配置命令仅80/90/100。
  unit `tide-core5-weighted-training-cuda-dev5-20261028.service`；output `artifacts/jobs/core5-weighted-training-cuda-dev5/`；
  checkout `artifacts/worktrees/core5-cuda`，build `build/weighted-flow-cuda-dev`；无GPU真机。
  两后端最新component均`d3e07a3fc3bc378243515dde065f6d766c61c506c0423da0b05b06efe24c2ec3`，binary/loader已复核。
- Python dev4：tree`b4bb3263fdcfc9c3132bf1ab26e0c3bfaec58a81`，inactive/success/exit0；
  10通过/75.338秒/无skip，八配置三更新及FC新进程恢复；物理9→npu:0。
  unit `tide-core5-weighted-training-python-dev4-20261028.service`；output `artifacts/jobs/core5-weighted-training-python-dev4/`；
  checkout `artifacts/worktrees/core5-device`，build `build/weighted-training-python-dev`。
  此后新增NPU循环bool插入修复与检查定位；Python新源码将用全新构建固定资格。
- retained旧dev3：tree08e251d16aad2d99582f20d21ac6832e0be6a836，live12/saved12/config15/retained8×5通过；保留原记录。
- CPU最新28通过/8显式deselect/2.82秒；XML `artifacts/checks/core5-weighted-training-cpu-dev6.xml`。
  describe/row、reverse-route、flow packing与公共训练CPU准入；不冒充设备运行。git diff --check通过。

## 当前任务与精确下一步

当前无活动任务，以上单位均已终态。先提交本阶段实现并正常推送，再冻结该提交分别资格：
1. `git add -A`；`git commit`；`git push origin graph-execution-foundation`。
2. `.local/freeze-weighted-{sdk,cuda,python}.py PRIOR_UNIT SOURCE_COMMIT` 写入冻结工作区；
   helper不移动HEAD，须另核对detached身份并将HEAD更新到SOURCE_COMMIT，使git status干净。
3. 独立background.slice服务运行 `.local/core5-weighted-training-{npu,cuda,python}-qualify.sh`；
   output `artifacts/jobs/core5-weighted-training-{npu,cuda,python}-qualify/`，对应unit加`tide-`前缀及`-20261028.service`后缀。
   NPU/CUDA原build复用，先验证同源码component、core和kernel二进制；NPU只重建host目标避免CANN重复preprocess。
   helper `.local/qualify-weighted-training.py`；NPU workload/device脚本同前缀。
   Python用全新 `build/weighted-training-python-qualify`，官方build脚本，实际训练+flow测试33项预期。
   资格包括copy、saved-window、retained、training、flow/config及两安装消费者；CUDA仅构建/loader/help。
   NPU资格附28项CPU门禁。1800秒/2worker/BLAS1/queue180秒/NPU1/background.slice/Nice10。
4. 核对systemd/持久status及binary/loader、实际测试；证据单独提交并推送，继续FP16、peer与性能/最终追溯。

启动前更新本节的SOURCE_COMMIT、unit和实际状态。不要预建job.py输出目录，service stdout放.local。
检查：`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止：`systemctl --user stop UNIT`。
不可编辑活动任务读取的checkout；不要将running当passed。

## 本阶段诊断与保留失败

原失败记录全部保留，`python scripts/status.py --all-jobs`可审计，不能因dev6成功而改旧状态。
新定位：NPU dev4/dev5完整顺序在nonaligned/delta/partial update1 restored backward失败，
CANN507011。dev5 tree322fccde9713e34a12a0dc41330f9c4440366a3d，物理9，
stream1153/task152/model287对应aclnnInplaceIndexCopy Bool target[2,4] axis1/source[2,1]。
plog和结构化定位在`artifacts/jobs/core5-weighted-training-npu-dev5/diagnostics/`。
来自recurrent_detail::insert；NPU改为静态逐行copy，保留bool、不读取主机值；dev6完整原顺序通过。
旧weighted_flow_copy静态bool修复保留。可选TIDE_CANN_TRACE_OPS记录提交，用法见device-control。
未声称证明厂商内部根因；诊断跟踪关闭后的原顺序通过才作为开发通过。

其它已修复开发失败：未重建重复端口layout、optimizer组误用所有alias、pybind枚举type::of、
恢复比较未使用padding、row accumulate读取无连接NaN；均保留独立回归或严格原条件。
恢复比较只忽略未定义journal/padding；实际逻辑raw字段、完整解码和checkpoint仍严格检查。
CANN9.1曾重复preprocess导致ld.lld拒绝已合并object，记录保留；未改公共工具链。

## 未解除的严格限制与环境

- FP16 GDN默认AdamW：Python/native nodes.5.kernel.conv.weight严格更新差约.001984，strict xfail；SGD不解除。
- 默认异步双NPU完整Python VJP：CANN invalid event；显式ASCEND_LAUNCH_BLOCKING=1通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a相同；旧原输入失败保留。
- CUDA所有实际GPU、跨卡、完整训练/恢复与性能待硬件。

实际路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，约3.3TiB可用。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`及`/etc/modulefiles`；旧guide已失效。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；
source .local/venv-core5/bin/activate（Python3.12.13/Torch2.12/CANN9.1）。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；配套Python $LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh（Torch2.10/cu128/Toolkit12.8.1，无GPU）。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、ACL_OP_INIT_MODE=0、保留模块CANN PYTHONPATH；2build workers/BLAS1。
STATUS使用scripts/durable_records.py原子写并回读。公共环境、参考仓库均未修改。
