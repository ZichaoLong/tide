# 当前交接 — 2026-10-28

分支 `graph-execution-foundation`；路径 `/mi/data2T/zlong/home/llm/graph-execution-foundation`。
固定数学 `0770593838a37c790a568226930567ed60f99b83`；19份当前/5份旧快照校验通过。
用户批准的 A—E 有界工作已完成：限制复现定位、LH/fractal/core-3经验、五族参数/匹配用例设施、
具体性能改进及有限独立验证。已实现、通过范围、性能负结果和保留限制分别记录；不是全部数值域/硬件/优化资格。
没有活动或排队作业，不恢复历史队列。参考仓库和公共环境保持只读，全程不使用子代理。

## 交付与固定来源

- `b0321b9`/`62073dd`/`12ad204`：GDN首次梯度分歧、纯Torch双NPU异步反例和历史经验调查。
- `2da3f2c76c13cff0c41af45fddb5f24f2e99421f`：五族真实参数/别名清单，core-3主体预算匹配，
  FC/HB/ranked/非对齐DAG/反馈PDG及v2二进制材料；共享低秩计算、初始化消冗、ready复制、冻结bank及发布去重。
- `0503593d71a64d26f24e0a4168e16af8c1633baa`：有界无损材料解码、独立FP32/FP64训练诊断、设备scratch隔离和失败metrics保留。
- `07e56ca03b72e177f2e7ee5cbcb47994710e04ac`：quad实际居中响应/残差/VJP修复；合法共享base/输入组内
  私有矩阵真实BMM及独立行一阶VJP，保留None/零、别名贡献、版本、私有状态和精确恢复。
- 实现已推送；本交接所在文档提交单独保存证据，不修改已测试源码。数学、图v13/checkpoint v5、原容差不变。

[最新固定证据](evidence/core5-private-projection-20261028.md)及同名JSON记录源码/二进制/材料/日志身份、
独立手算、资格与严格失败；[前轮设施/迁移证据](evidence/core5-performance-transfer-20261028.md)、
[数值/异步定位](evidence/core5-numerical-isolation-20261028.md)保留原结果。
[经验文档](performance-lessons.md)包含本仓库core-3、LH、fractal及本轮正/负经验。

## 最新终态

job路径均为 `artifacts/jobs/NAME`，unit为 `NAME-20261028`，状态/日志为 `status.json`/`task.log`。
固定只读源码 `artifacts/worktrees/core5-private-qualified` 在07e56ca；CPU复用core前核对全部源码/二进制hash。

| NAME | 终态及范围 |
| --- | --- |
| core5-private-qualify-cpu | passed/exit0；574 passed、2 strict xfail；standalone两dtype各24图×4更新和精确checkpoint |
| core5-private-qualify-npu | passed/exit0；物理1/11，9个组件dtype项，完整单/双卡训练恢复；SDK eager KV及纯Python mixed-c KV通过 |
| core5-private-qualify-cuda | passed/exit0；新core/resident sm80/90/100、loader/help和CPU两dtype核验，无GPU真机 |
| core5-private-study-cpu | failed/exit1；同材料medium前后均通过；ranked64/其profile严格失败，D2048残差拒绝；Python/native FP64两步通过，FP32仍失败 |
| core5-private-study-npu | failed/exit1；物理1，同材料mixed-c前后均通过；SDK affine step2原数值见证复现 |

CPU研究2026-10-28 09:00:39 UTC终态，随后NPU串行测量于09:03:10 UTC终态。
开发quad332项、private244项通过；首次private新测试括号/include编译失败仍在原job/source.tar.gz，不重标。
所有既有严格失败保留，最新通过不替换历史失败；当前无须重跑完整旧矩阵。

## 实测结论与保留限制

同材料CPU完整训练388.921→387.704ms（单次约−0.3%），pure Python mixed-c为1874.480→1988.877ms
（单次约+6.1%）；均完整对照通过，没有稳定提速结论。NPU本轮测量allocator峰值略增，旧resident存储小收益另列。
ranked64真实私有矩阵批最大32，但只覆盖34/479次State；State回调最大批仍1。完整FP32记录、梯度和SGD参数严格失败，
显式FP64两步通过不解除FP32限制。profile保留22501次mm/12次bmm/100523次copy和自身采集分析成本。
92.62亿参数匹配材料完成实际构造/warmup，随后仍残差拒绝，metrics为空；没有大型吞吐或训练峰值结论。

尚存：eager FP16 GDN/AdamW、默认异步双NPU、SDK FP32 affine、FP32反馈PDG及D64训练、D2048残差边界；
CUDA真机/x86_64、大规模设备/跨卡吞吐待验。不同base/输入的私有模块批、物理缺席行和窗口反向构造仍有优化空间。
正支持/容量等公开拒绝保持，不通过改dtype、容差、活动集或无声截断制造通过。

## 下一步与复现

唯一待办为[ROADMAP](ROADMAP.md)，持续约束为[execution-flows](execution-flows.md)。
后续优先取得D2048失败分配的实际输入并定位数值机制，再扩大私有矩阵批或复用反向程序；新硬件单独资格。
这些是保留边界和后续候选，不是待恢复的后台队列。

重新进入：`git status --short --branch`、`python scripts/status.py`、`python scripts/check_upstream.py`。
`python scripts/status.py --all-jobs`审查历史原失败；不从已回收unit默认状态推断通过。
新checkout按[开发](development.md)、[workload](weighted-workloads.md)和报告公共命令生成材料/构建/验证；
正式成果无需旧运行目录。长任务继续固定源码、独立background.slice、有限资源/时长及新输出目录。
本次收尾仅提交文档/证据；原始日志、构建、材料和本机launcher保持忽略，不进入源码。
