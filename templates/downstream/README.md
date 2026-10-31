# 下游协作模板

流程为固定安装/依赖锁→基座检查→下游接入→配置实验/结果→需求→基座交付→下游复验→显式升级。
完整说明见所采用提交的[协作规范](../../docs/downstream-contract.md)与[人类操作手册](../../docs/human-workflow.md)。

- `foundation.lock.example.json`记录源码/数学/协议和安装材料的来源；null字段是待填写项，不能作为实际运行锁。
- 可选`installation_lock`引用`tidegraph lock create`生成的`tide-experiment-lock-v1`及文件SHA256；旧来源锁无需补此字段。
- 可选`consumer`连接下游提交、任务包wheel及源码hash；没有插件的旧消费者无需此字段。
- CLI安装锁验证实际包/所选库与环境；它不证明Git来源、数学正确性或设备资格，不能替代构建记录与独立检查。
- request/delivery/acceptance分别留在其权威仓库，互链同一编号；复制模板不会创建真实需求或通过记录。
- AGENTS.fragment可并入下游规则；人类执行步骤写进README，不能只给AI接续提示。

这些是`tide-downstream-contract-v1`的兼容说明/可选字段，采用方仍固定文档提交；数学、图和checkpoint版本未因模板改变。
