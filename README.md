# Graph execution foundation

本分支是 20-tide 训练、推理实验的公共图执行与等价性验证基座。
实验项目拥有数据、模型头、loss 和训练配方；本库提供图、局部程序、执行策略、
状态续接、声明的一阶反向与可独立比较的实现。当前数学依据固定为 **tide-core-5**（`0770593`）。
已实现一般图与端口、五种带权记忆、正式发送与评分、在线批量执行和训练续接；
各后端的固定源码资格及剩余限制见 [能力范围](docs/execution-capabilities.md)，当前任务见 STATUS/ROADMAP。

## 开始阅读

| 目的 | 入口 |
| --- | --- |
| 新线程接续、当前完成范围和剩余事项 | [当前状态](docs/STATUS.md)、[唯一待办表](docs/ROADMAP.md) |
| 开发规则和执行契约 | [AGENTS.md](AGENTS.md)、[执行契约](docs/execution-flows.md) |
| 数学定义、输入输出、依赖与实现对应 | [语义索引](docs/semantic-map.md)、[本地契约](docs/semantics.md) |
| 当前采用的教材原文 | [版本锁定](docs/upstream.json)、[固定快照](docs/upstream/README.md) |
| 源码分层和模块导航 | [架构](docs/architecture.md) |
| 从干净源码开始构建、检查和开发 | [开发入口](docs/development.md) |
| 作为 Python/LibTorch 依赖调用 | [公共库](docs/library.md)、[安装客户端](examples/consumer/README.md) |
| core-5 配置/拓扑/规模、运行和维护导航 | [候选与测试入口](docs/core5-study.md) |
| 按配置查询正确性/性能覆盖 | [统一查询入口](docs/core5-study.md)、[逐配置索引](docs/evidence/core5-index.json)、[覆盖修正资格](docs/evidence/core5-coverage-correction-20261030.md) |
| 当前core-5性能设置、合格结果与严格失败 | [自然规模对照](docs/evidence/core5-natural-study-20261029.md)、[大文件与下一窗边界](docs/evidence/core5-checkpoint-budget-20261029.md) |
| core-5 带权模块、训练与有限性能 | [公共绑定](docs/weighted-message.md)、[五记忆与短卷积](docs/weighted-recurrent.md)、[常驻训练](docs/resident-weighted-training.md)、[节点放置](docs/resident-weighted-placement.md)、[样本副本](docs/resident-weighted-peers.md)、[性能消费者](docs/weighted-performance.md) |
| CPU/CUDA/NPU 与常驻后端 | [加速器](docs/accelerators.md)、[设备控制与 GPU 验证](docs/device-control.md)、[常驻库](docs/resident-library.md) |
| 旧实验结论及产物保留规则 | [历史证据](docs/evidence/README.md)、[产物退役](docs/artifact-retention.md) |

## 支持边界

语义层级为 `SettleGraph → 编码后的 TimedDAG → PositiveDelayGraph`。
Python 参考、LibTorch 核心和特化调度保留独立执行顺序；常驻后端拥有独立的
CANN/CUDA 控制流实现。支持范围以配置能力、验证门禁及对应证据为准。

CPU/NPU 已完成证据所列固定源码与声明范围的本机验证。CUDA eager/常驻源码已交付；
历史构建仅认证其记录的提交，当前新增源码的CUDA构建仍待核验。
**真实 GPU 正确性、常驻性、多卡、内存和性能仍待目标机验证**。
自定义 Python 程序不会自动转换为 C++ 或常驻内核；不支持的请求明确失败。
完整矩阵、精度和兼容性边界见 [执行契约](docs/execution-flows.md)。

```sh
python scripts/status.py
python scripts/check_upstream.py
```

以上不需要 Torch、设备或历史实验文件。实际构建与测试需要匹配的工具链，
按 [开发入口](docs/development.md) 创建新构建和输出目录。
源码不依赖本机 skill、旧实验队列或某个已有编译目录。

## 新语义和新特例

先固定用户确认的教材提交，比较定义域、值域、依赖和观察量，再确定实现变化。
新的特例可以实例化现有接口，也可能暴露需要扩展的表示或设备能力；不能只改版本号，
也不能用通用执行器包装冒充独立特化。先建立小型独立参考与相关回归，再考虑性能。
当前已采用固定 core-5；后续定义变更仍需明确任务授权。旧性能任务不自动恢复。
