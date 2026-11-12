# 当前交接 — 批量数值边界重写进行中（2026-11-12）

数学固定 `0770593838a37c790a568226930567ed60f99b83`，19份快照通过。
授权分支 `graph-execution-foundation`，起点 `deda0e791a11e8a47c7289e82b9988d6c32b6eb4`。
用户已再次授权整体重写；旧源码、接口及近期columns/block实现均非架构约束。
继续实施、阶段提交与非强推已授权。当前对话允许共享NPU；GPU资格延后。
三份用户TL-REQ-004/005/006未跟踪handoff保持只读、不提交。

## 当前工作与下一动作

首轮新数值边界已实施，准备固定实现提交并构建资格；尚无本轮真机/性能结论。
Python KV正写入、Read/Describe/Full采用角色张量贯穿，C++独立Read/Full数值边界同步；
允许不同Q/K，真实相同参数保留单矩阵，共享QK，独立字段根/None零/版本保持。
resident投影/head/Upd改为设备实际owner稳定分组和bank GEMM/VJP，删除逐行矩阵及对应预算假设。
raw逐帧快照、重复历史投影及optimizer固定子预算尚未由这些修改解除。

开发检查：Python92通过/5原FP32严格数值xfail；同FP32材料提升FP64与零根通过。
本轮proportional out-weight最大差2.8611e-6，原始strict失败保留在
`artifacts/tensor-rewrite-20261112/development-numeric/`，没有放宽比较通过阈值。
resident39项CPU核/公式检查通过、18TU语法通过；C++新数值边界33配置及私图粒度检查通过，
这些是开发检查，不代替固定源码或设备资格。参考数学快照仍通过。

下一动作：以本实现提交建立 `/dev/shm/gef-tensor-rewrite-20261112/source`，
独立background.slice服务分别构建Python-owned和SDK core/resident；各分配互不重叠CPU池、
内存512GiB和2400秒。构建完成后跑相关CPU/单NPU根、三步optimizer/新进程恢复，
再做FC16 B8/L128与LH176 B4/L16的direct/public/default有限三模式成本对照。
源码运行期间只在主工作区继续开发；固定输入不改。后继证据另提交。

## 已固定证据与保留限制

- [22点成本/生命周期](evidence/source-columns-performance-20261112.md)：00bb5ac2；
  LH176默认native训练CPU1/NPU-host2为2.3187/5.1350秒，直接公式0.7568/0.6238秒。
- [静态数据流](evidence/lh-block-static-20261112.md)：43文件；columns按精确Q/K分组并重复节点小操作，
  resident按行复制参数、按帧复制raw及重复历史投影。此次修改须真实改变这些数据流。
- [resident公共资格及规模拒绝](evidence/resident-block-public-20261112.md)：a3c4f62d小图、三步训练及新进程恢复通过；
  FC16被qry保守预算拒绝，LH176被optimizer子预算拒绝，两者无有效规模吞吐。原失败不改写。
- 55f8c631修正独立直接基线完整有序参数角色；CPU22通过，不改变无低秩旧LH材料。

## 作业、资源与接续

启动时无活跃作业，所有p1p6-resident-v3任务终态；旧failed服务无存活cgroup。
旧源码/构建位于 `/dev/shm/gef-p1p6-20261112/`，仅匹配component hash时复用，正式复现不依赖旧目录。
长任务使用固定源码、独立background.slice、明确CPU/内存/设备/时长上限及持久记录。
重新进入：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
保留旧失败查 `python scripts/status.py --all-jobs`。本轮源码编辑及实际测试状态以git和新增作业记录为准。
