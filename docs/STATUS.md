# 当前交接 — 2026-10-28

固定上游`0770593838a37c790a568226930567ed60f99b83`，授权分支`graph-execution-foundation`。
用户授权持续实施、私有环境、有限实验、阶段提交及正常推送；无子代理，参考仓库只读。
总体未完成，阶段提交后继续。[ROADMAP](ROADMAP.md)为唯一待办，[execution-flows](execution-flows.md)为持续契约。
实现2e4d445及此前完整训练证据56d751d已正常推送。本提交只记录基础FP16组件资格，不混入KV读取开发。

## 最新固定资格

- 完整FP32公共训练：实现343429d，[固定证据](evidence/core5-resident-weighted-training-20261028.md)。
  CPU28、Python-owned NPU28（训练10+推理18，无skip）、standalone5配置20更新，
  saved-window12、retained8×5、完整推理21图/15配置及两独立安装消费者通过；CUDA三架构build/loader通过。
  SGD/AdamW/梯度累计/全部或部分冻结/完整字段根/跨窗raw和pending/精确恢复分别受检。
  C++ checkpoint是完整CPU内存记录；Python另核验磁盘及新进程恢复。不是half/peer/性能资格。
- FP16原始状态、共享投影、固定时间卷积：实现2e4d445，[固定证据](evidence/core5-resident-half-primitives-20261028.md)。
  CPU32、NPU三个组件×两dtype、CUDA三架构build/loader/help与三dtype CPU codec通过。
  实际FP16存储，FP32算术/保存值/一阶伴随；int64不变。次正规数与下溢连接、存储溢出拒绝单列。
  全部systemd inactive/success/0、持久passed/exit0、干净source/binary/loader摘要已复核。
  CUDA所选依赖12条编译命令；已配置160条均80/90/100；没有GPU真机。

最新资格units `tide-core5-half-primitives-{npu,cuda}-qualify-20261028.service`，均已终态。
checkout `artifacts/worktrees/core5-half-math`固定2e4d445，build/npu2、build/cuda2；跟踪文件只读。
output `artifacts/jobs/core5-half-primitives-{npu,cuda}-qualify/`，task.log/status.json；实际NPU物理9→npu:0。
launchers `.local/core5-half-primitives-{npu,cuda}-qualify{,-workload}.sh`；900秒/2worker/BLAS1/Nice10/background.slice。
记录脚本 `.local/record-half-primitives-evidence.py` 已验证并生成报告。当前没有活动项目任务。
完整训练三任务与原失败的精确路径/摘要从上述机器记录进入，不在STATUS重复成旧队列。

## 当前开发与下一步

主区未提交：KV window接受FP16 raw，设备拓宽后在FP32读取；sum/qry的Read输入、参数、
memory/total和全部伴随明确保持FP32。raw、描述与反向精度分开，时间和窗口成员始终int64。
sum分配改用实际window输出dtype，完整预算计入拓宽缓冲；query原有FP32输入/参数域不变。
两个standalone检查新增half raw域，在实际量化raw上独立FP32公式/四根VJP，保存版本仍独立。
额外sum见证检查半精度最小次正规K/V/权重、正的短期FP32证据低于半精度下限仍保留、四根连接及int64 age。
尚未编译/设备运行。CPU相关27项通过/2.79秒：
`python -m pytest -q tests/test_resident_weighted_kv_window.py tests/test_resident_weighted_kv_query.py tests/test_resident_weighted_pool.py`。
下一步同步resident-weighted-kv-sum/query契约，审查dtype/预算/独立期望后提交本组件。
所有旧资格均终态后可将core5-half-math冻结到该新提交，在新build/kv-read-npu、build/kv-read-cuda构建：
`python scripts/build_device_control.py --core-build CORE --build-dir NEW --runtime standalone --jobs 2 --checks weighted-kv-sum weighted-kv-query`，
NPU加`--ascendc-soc Ascend910_9392`，CUDA加`--cuda-architectures '80;90;100'`；实际NPU用verify_device_control.py两dtype资格。
先写STATUS再启动新的900秒background.slice服务，queue180秒，不编辑活动源码。
其后继续KV Upd/Full、四循环与完整图FP16、peer、有限profiling及最终追溯；组件通过不代表总体完成。

## 保留失败与环境

原始状态FP16初期失败/cancelled均已记入机器报告：codec测试放错零适配字段、两个launcher错误、
NPU检查入口遗漏half准入；全部原状态保留，不能因新通过改写。完整训练dev4/dev5 Bool ScatterUpdate
诊断在`artifacts/jobs/core5-weighted-training-npu-dev5/diagnostics/`；NPU静态逐行copy原顺序资格通过，未证明厂商内部根因。
- FP16 GDN默认AdamW约.001984严格差，strict xfail；SGD不解除。
- 默认异步双NPU完整Python VJP的CANN invalid event；显式launch blocking通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a同见证；原输入失败保留。
- CUDA全部GPU/跨卡/完整训练恢复/性能真机待验。

实际路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，约3.3TiB可用。
现行规范`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`及`/etc/modulefiles`；旧guide失效且已获授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH加本项目.local/venv-core5/bin提供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0，保留模块CANN PYTHONPATH。
任务检查`systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`；停止`systemctl --user stop UNIT`。
不预建job.py输出目录，service stdout放.local；STATUS用scripts/durable_records.py原子写并回读。
公共环境和参考仓库未修改；上游19文件已重新校验通过。
