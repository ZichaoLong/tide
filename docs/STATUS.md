# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支
`graph-execution-foundation`。持续实施、私有环境、有限实验、提交及推送均已授权；不使用子代理。
历史等待授权/提交后暂停要求已被取代。完整未完成范围以 [ROADMAP](ROADMAP.md) 为准。

## 已完成的阶段

- 一般 SettleGraph `efb9d629438fb77582842bb133158be7f9f70afb`，固定源码 CPU 247 项通过；
  [证据](evidence/core5-settle-20261027.md)，实现与证据 d0acd3c 已推送。
- 具名 Read、int64 状态、typed packing、SrcClass `714d5f963b676d95a0ae763171a7d93806f183d3`，
  固定源码 CPU 212 项通过；[证据](evidence/core5-structured-20261027.md)，实现与证据 cbdd3fd 已推送。
- 带权 KV 首组合 `6fb467c25755a2b3f8938952f1b2ec1011cbcd11`，Python/native、FC/HB 正式绑定，
  独立公式、完整记录/根、多步优化器和恢复，固定源码 CPU FP64/FP32 **129 项通过**。
  [证据](evidence/core5-weighted-kv-20261027.md)单独提交；首组合不代表完整任务已完成。

## 当前任务和下一动作

KV 不可变资格 `tide-core5-weighted-kv-qualify-20261027` 已终止：inactive/success、退出 0。
源码 6fb467c，冻结 checkout `artifacts/worktrees/core5-bootstrap` 干净；
记录 `artifacts/jobs/core5-weighted-kv-qualify/{status.json,task.log,development.json}`。
命令 `.local/core5-weighted-kv-qualify.sh`，background.slice、Nice=10、单线程、10 分钟上限。
无 live job；该 checkout/build 可在确认终态后用于下一冻结增量。

下一步继续 affine/delta、固定逻辑时间短卷积、完整 GDN/SSD，随后正式分配/评分/证据/key、
异构、精确联合/packed/批量 VJP、mixed/resident 与有限选型。不能在 KV 首组合处暂停。
未提交 `python/tidegraph/weighted/convolution.py`、`tests/test_weighted_convolution.py`
是后续短卷积原型，8 项短测通过，尚未接入 GDN/SSD 或 C++，不属于 KV 资格。
先补 int64 最大时间 projection 边界，再实现基础循环。
短测环境下执行 `python -m pytest -q tests/test_weighted_convolution.py --dtype both`。

## 环境与保留事项

实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`；/home/zlong 指向当前 home。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与对应 /etc/modulefiles。
`source /etc/profile.d/modules.sh; module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境，无用户未保存修改。
ObsidianVault/lh/fractal-latcarf 只读。CUDA 源码/构建/真机分别记录，GPU 真机待验不阻塞其余工作。

旧 bootstrap 缺 Ninja、缺 TIDE_BUILD_DIR 的 24 项前置失败、structured-dev 的 210/2、
weighted-kv-dev 的 C++ Options 构建失败均保留原记录；后续成功不改写旧失败。
