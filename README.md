# Graph execution foundation

本分支是 20-tide 训练、推理实验的公共图执行与等价性验证基座。
实验项目拥有数据、模型头、loss 和训练配方；本库提供图、局部程序、执行策略、
状态续接、声明的一阶反向与可独立比较的实现。当前数学依据是 **tide-core-3**。

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
| CPU/CUDA/NPU 与常驻后端 | [加速器](docs/accelerators.md)、[设备控制与 GPU 验证](docs/device-control.md)、[常驻库](docs/resident-library.md) |
| 旧实验结论及产物保留规则 | [历史证据](docs/evidence/README.md)、[产物退役](docs/artifact-retention.md) |

## 支持边界

语义层级为 `SettleGraph → 编码后的 TimedDAG → PositiveDelayGraph`。
Python 参考、LibTorch 核心和特化调度保留独立执行顺序；常驻后端拥有独立的
CANN/CUDA 控制流实现。支持范围以配置能力、验证门禁及对应证据为准。

CPU/NPU 已完成声明范围的本机验证。CUDA eager/常驻源码和本机构建已交付，
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
当前新版教材尚未采用，不自动恢复旧性能任务。
