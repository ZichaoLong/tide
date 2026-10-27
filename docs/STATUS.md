# 当前交接 — 2026-10-27

固定上游 `0770593838a37c790a568226930567ed60f99b83`，授权分支
`graph-execution-foundation`。用户授权持续实施、私有环境、有限实验、提交及推送；
不使用子代理。历史等待授权/提交后暂停要求已被取代。

## 已实现与证据

- `efb9d629438fb77582842bb133158be7f9f70afb` 采用 core-5 的 19 份固定原文，
  实施一般端口 SettleGraph：同秩区域、稀疏日程、逐边/端口、原样嵌入和任意切点。
  不可变 CPU FP64/FP32 定向资格 247 项通过，证据提交 `d0acd3c`，两者已推送。
- 结构化表示已提交 `714d5f963b676d95a0ae763171a7d93806f183d3`：具名 Read、独立字段根、typed packing、int64 状态槽、
  SourceClasses。`core5-structured-dev` 已构建成功，但测试 **210 passed / 2 failed**，
  原因是 Python 通用校验读取 BoundaryWeights 不存在的 kernel；主工作区已修复；新冻结树 `f29dfb13d2aae33786dd39cd8b5d8d447ad0afa1`
  的 `core5-structured-fix` 已通过 **212 项**、退出 0。
  原失败保持，冻结开发树 `464b7a002a77027f94157bce9961955b8d547115`。
- 带权首组合的 Python 工作正在主工作区独立开发，尚未提交/不可变资格：
  `weighted/` 提供 quad 精确解/VJP、KV sum/qry、显式初态/旧/新 Read、cat 评分/证据、
  regional_total、(c,G) 广播、FC/HB 物理拓扑。50 项定向检查通过；
  包含独立枚举/手算/方向差分、非对齐 DAG/PDG 反馈、各切点、共享参数、AdamW 与恢复。
  不认证 native、packed、mixed/resident 或其他正式候选。
- 旧 `core5-bootstrap` 缺 Ninja 的失败及缺 TIDE_BUILD_DIR 的 24 项前置条件失败仍保留；
  后续成功记录不改写旧失败。详细完整待办见 ROADMAP。

## 当前任务与下一动作

结构化表示开发检查已终止：`tide-core5-structured-fix-20261027`，212 项通过、退出 0。
日志/结果：`artifacts/jobs/core5-structured-fix/{status.json,task.log,development.json}`。
`core5-structured-dev` 的原 2 项失败保持，不改写为成功。

不可变 `714d5f963b676d95a0ae763171a7d93806f183d3` 的资格已通过：
`tide-core5-structured-qualify-20261027` 终态成功、退出 0、212 项通过，
冻结 checkout 干净。证据 [core5-structured](evidence/core5-structured-20261027.md)。
实现 `714d5f9` 与证据 `cbdd3fd` 均已推送至授权分支。

带权首组合（kv/sum/+、cat/ind、regional_total/context/cat/quad）及
KV qry、Read 0/-/+、位置/窗口、FC/HB 正式来源/位置绑定已实现 Python 与 C++。
开发树 `010ee393e329a420b7a107280f915475b6e9b0c3` 的任务
`tide-core5-weighted-kv-fix-20261027` 已通过 **112 项**，CPU FP64/FP32、退出 0。
含 Python/native 完整记录、独立根、AdamW、切点/恢复、standalone C++ 手算与共享根。
首轮 `core5-weighted-kv-dev` C++ 测试构造缺 Options 的构建失败保持失败。
本组合不代表完整候选、packed/mixed/resident 或 GPU 资格。

首组合最终开发树 `aeac08dc415016f4a1e121b6f5ff12dfe4cc8002` 已通过 125 项，
`tide-core5-weighted-kv-final-dev-20261027` 终态成功、退出 0，完整日志/清单在
`artifacts/jobs/core5-weighted-kv-final-dev`。另补缓存初态独立根和多头/部分 RoPE 各 2 项通过；
当前实现/测试合并范围为 129 项，尚未作该合并源码的不可变资格。
新增保护包括冻结配置身份、C++ 单位传输尺度、native pending 在新 owner 中的 checkpoint 恢复。

下一步提交当前已暂存 KV 组合，在该固定提交上复用匹配 C++ source/binary hash 的构建，
用相同定向集合运行 129 项资格；拟 unit `tide-core5-weighted-kv-qualify-20261027`，
日志 `artifacts/jobs/core5-weighted-kv-qualify`。实际提交及启动状态将在启动前更新。
通过后证据另提交并推送，继续全部正式候选、批量/设备/性能，不在首组合处暂停。

后续独立工作：未暂存 `weighted/convolution.py` 与 `tests/test_weighted_convolution.py`，
固定逻辑时间短卷积 8 项手算/弱事件/快照/普通 grouped conv1d/VJP 检查通过；
尚未集成 GDN/SSD 或 C++，不在本次 KV 提交/资格内。

## 环境与未提交内容

实际路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`，`/home/zlong` 是当前 home 的软链。
现行规范 `/opt/software/ascend-locks/dev-workspace-9.1.0/README.md` 与 `/etc/modulefiles`。
环境：`module purge; module load ascend/dev-workspace-9.1.0; source .local/venv-core5/bin/activate`。
Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1、aarch64。未修改公共环境。
当前无用户未保存修改或数学歧义；参考仓库只读。GPU 真机待验单列，不能替代为编译证据。
已暂存 KV 增量即将提交；短卷积两个未跟踪文件为下一增量，不属于 KV 资格；没有用户修改。
