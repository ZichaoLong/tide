# 当前交接 — 2026-10-28

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支 `graph-execution-foundation`。
用户授权持续实施、私有环境、有限实验、分阶段提交与正常推送；不用子代理，参考仓库只读。
总体未完成，阶段提交后继续。唯一待办 [ROADMAP](ROADMAP.md)，持续契约 [execution-flows](execution-flows.md)。

## 最新固定资格

- FP32公共训练实现343429d，证据56d751d：[报告](evidence/core5-resident-weighted-training-20261028.md)。
  CPU28、Python-owned NPU28、standalone5配置20更新、saved-window12、retained8×5、推理21图/15配置及安装消费者通过。
  CUDA三架构build/loader通过；C++内存checkpoint与Python磁盘/新进程恢复分别核验。
- 原始状态/共享投影/固定时间卷积FP16实现2e4d445，证据a585376：[报告](evidence/core5-resident-half-primitives-20261028.md)。
  CPU32、NPU三个组件各两dtype、CUDA三架构构建及三dtype CPU codec通过。
- KV FP16 raw读取实现db85db6：[报告](evidence/core5-resident-half-kv-read-20261028.md)。
  CPU27、NPU sum/qry各两dtype通过，四Mem根/完整输入参数raw VJP；CUDA build/loader/help通过。
- KV FP16 raw更新实现69bf4f0：[报告](evidence/core5-resident-half-kv-update-20261028.md)。
  CPU26、NPU KV更新/raw各两dtype通过；正质量下溢仍追加条目/时间/计数及质量根，存储溢出拒绝且owner不变。
  CUDA build/loader/help通过。新证据均复核source/binary/loader和systemd inactive/success/0、持久passed/exit0。

FP16组件使用实际half存储、FP32算术/保存值/一阶伴随，Cast声明identity VJP，int64元数据不变。
上述局部资格不能代表完整half图、训练、peer或性能；所有CUDA真机范围待硬件。

## 当前源码与下一步

KV两批证据5ff8679已提交推送。当前实现扩展四循环raw FP16及KV Full读取：
FP32快照和完整算术，Update最终一次half存储及溢出guard；Mem/Full输入参数结果仍为FP32。
独立24配置检查扩到两dtype，增加次正规快照、零接收按位保留、弱正接收、完整字段VJP与失败不提交。
契约已同步，定向CPU24项通过/2.90秒；C++尚未构建/设备验证，不属于上述证据。
本提交后冻结到core5-half-memory，启动两后端局部资格。

四个KV资格任务已全部终态，不再有活动源码读者：
- `tide-core5-half-kv-read-{npu,cuda}-qualify-20261028.service`；
  checkout `artifacts/worktrees/core5-half-math` 固定db85db6，build/kv-read-npu、build/kv-read-cuda。
- `tide-core5-half-kv-update-{npu,cuda}-qualify-20261028.service`；
  checkout `artifacts/worktrees/core5-half-update` 固定69bf4f0，build/npu、build/cuda。
- 输出 `artifacts/jobs/core5-half-kv-{read,update}-{npu,cuda}-qualify/`，task.log/status.json，NPU checks/result.json及queue-state.json。
- launchers `.local/core5-half-kv-{read,update}-{npu,cuda}-qualify{,-workload}.sh`；900秒、2workers、BLAS1、Nice10/background.slice。
  NPU queue180秒/device gate240秒，实际物理9→npu:0。记录脚本 `.local/record-half-kv-evidence.py read|update`。

已通过的定向CPU：`python -m pytest -q tests/test_resident_weighted_recurrent.py tests/test_resident_weighted_kv_full.py tests/test_resident_weighted_raw_update.py`。
再冻结新实现，用`build_device_control.py --checks weighted-recurrent weighted-kv-full`和匹配`verify_device_control.py`资格。
其后完成完整图FP16、训练/恢复、peer、有限profiling与最终追溯；不要在局部组件阶段结束总体任务。

## 保留限制与环境

- FP16 GDN默认AdamW约.001984严格差，strict xfail；SGD不解除。
- 默认异步双NPU完整Python VJP的CANN invalid event；显式launch blocking通过不解除。
- SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6，mixed-a同见证；原输入失败保留。
- 原始half初期入口/launcher/codec见证错误及cancelled均保留在基础组件报告；旧失败未改写。

实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`，zlong，约3.3TiB可用。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 和 `/etc/modulefiles`；旧guide失效且已获授权采用现行规范。
Python：source /etc/profile.d/modules.sh；module purge；module load ascend/dev-workspace-9.1.0；source .local/venv-core5/bin/activate。
SDK独立shell：module load libtorch-npu/2.10.0-cann9.0.0；Python为$LIBTORCH_ROOT/../../../../bin/python。
CUDA：source /mi/data2T/zlong/gpu-toolchains/activate-cuda.sh；PATH追加本项目.local/venv-core5/bin提供ninja。
TORCH_DEVICE_BACKEND_AUTOLOAD=0、PYTHONDONTWRITEBYTECODE=1、ACL_OP_INIT_MODE=0；保留模块CANN PYTHONPATH。
任务检查 `systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`，停止 `systemctl --user stop UNIT`。
不预建job.py输出目录，service stdout写.local；STATUS用scripts/durable_records.py原子写并回读。
公共环境与参考仓库未改；上游19文件重入校验通过。
