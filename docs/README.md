# 文档导航

本仓库采用 20-tide 的 SettleGraph 语义，并维护自己的实验实例与实现证据。根据当前问题选择入口；下列分类是职责分工，不是必须依次完成的研究阶梯。

| 范围 | 文档 | 回答的问题 |
| --- | --- | --- |
| 上游语义 | [上游语义引用与本地对应](upstream-semantics.md)；[固定来源锁文件](upstream/tide-core.lock.json) | 采用哪个上游版本，哪些定义被继承，哪些是本地限制或扩展？ |
| 本地实验实例 | [SettleGraph 实验实例、局部公式与命名](experiment-semantics-and-naming.md) | Base 接入、局部算子、拓扑、状态、发送、loss 与实验名称怎样定义？ |
| 实现 | [SettleGraph 实现与等价性验证计划](settlegraph-implementation-plan.md) | 当前软件支持什么，执行路径和数据边界是什么？ |
| 测试与资格 | [等价性测试契约](equivalence-test-contract.md)；[core-v1 资格计划](core-v1-qualification-plan.md) | 比较哪些对象，怎样构造 fixture、判定结果并形成限定范围的资格声明？ |
| 证据 | [执行器等价性开发验证状态](executor-equivalence-development-status.md) | 哪个提交实际运行过什么，结果与未覆盖范围是什么？ |
| 研究路线 | [checkpoint 到 SettleGraph + BO 实验路线](checkpoint-to-bo-experiment-roadmap.md) | 怎样组织模型接入、学习实验以及实验需要的性能工作？ |
| 历史 | [采用 tide-core-2 前的语义备份](archive/experiment-semantics-and-naming-pre-tide-core-2.md) | 旧记录原先依据怎样的文档？此备份非现行规范。 |

上游定义共同语义与一般定理，本地实验文档选择具体实例。实现计划、测试契约、资格工作单和运行报告分别说明软件、判定规则、待完成目标和已取得证据，均不能通过同名术语扩大上游定义。具体配置超出采用范围时，应在[本地对应表](upstream-semantics.md)登记，不能由执行器自行解释。

`tide-core-2` 与本地 `core-v1`、`extension-v2`、Plan schema 和 parameter schema 是不同版本坐标。文档采用新上游版本，不会改写旧 run 的语义身份，也不会自动增加已验证能力。返回[仓库入口](../README.md)。
