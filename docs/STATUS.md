# 当前交接 — KV bank／查询块／块伴随实施中（2026-11-11）

用户已授权在默认能力分派中实现状态bank—Read/Full块—块级伴随，并复测默认Tide／直接FC／Transformer；
交付可重复用于后续拓扑/执行路径的诊断能力。本轮聚焦FC4/FC16，不扩展LH/GPU/多卡。
数学固定0770593838a37c790a568226930567ed60f99b83，19份core-5快照启动核验通过。

## 当前工作

起点`e3452962b928d8cb82d73b1b08fd1ef69e41a119`；Python/native块接口与默认分派已实现。
开发构建dev3与42项查询/selected-boundary测试通过；补充别名/冻结参数/执行身份定向26项通过。
这些组有重叠，不累计为独立资格。可重复诊断入口新增blocks有限套件和显式配对成本收集器。
现在提交实现，然后以该固定源码构建Python native NPU与standalone SDK并做正式有限资格/复测。
末状态、显式trace、过期/位置或不匹配布局保留原有合法后备；不会合并公共独立根。
然后固定实现提交，进行相关CPU/NPU正确性、训练/续接及三层成本比较，证据另行提交。
首轮开发构建`gef-kvblock-build-dev`因两个新增文件的namespace/include遗漏失败，原日志保留。
已修复，接续构建`gef-kvblock-build-dev2`通过；构建/相关测试已通过：`gef-kvblock-dev3`，冻结含工作修改的源码`/dev/shm/zlong-gef-block-dev-20261111`，
快照身份`artifacts/kv-block-20261111/development-snapshot.json`。CPU构建（含native与standalone CPU），
全部320可用CPU、512GiB/1800秒；日志/终态`artifacts/jobs/kvblock-dev3/`（首轮失败与dev2构建另保留）。
三个开发unit均已终止，无活跃作业。这是开发构建，不是固定实现提交资格。旧fccompare四作业均为终态，原严格数值失败保持。
Python局部及已有序列定向组48项通过；新增默认路径CPU FP64三种Read模式3项通过。

## 接续与资源

运行`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
实现入口：`python/tidegraph/weighted/kv_block_{state,math,query}.py`、`block_attention.py`；
集成入口：`packing.py`、`weighted/query_vjp.py`、`weighted/query_execution.py`及对应C++。
独立根/续接检查沿用`test_kv_sequence_vjp`、`test_kv_prefix_roots`及新增块检查。
前轮固定诊断见[evidence/fc-three-layer-comparison-20261111.md](evidence/fc-three-layer-comparison-20261111.md)。

开发环境`artifacts/additive-work/env`，CANN9.2；旧native构建`artifacts/additive-work/build-native`
只匹配旧生产源码；C++变更后须新固定源码构建。空闲CPU/内存/NPU授权保持；共享NPU需当前授权。
长任务使用独立background.slice服务、固定输入、有限资源/时长、独立输出与持久记录。
存储无已知阻塞；大写入前重新检查配额/磁盘/shm，若实质阻塞及时汇报。

用户未跟踪文件`docs/tide-learning-lab-TL-REQ-004-handoff.md`、`docs/tide-learning-lab-TL-REQ-005-handoff.md`
原样保留，不纳入本轮提交。TL-REQ-005未在本轮另行接纳，不混作已修复事项。
