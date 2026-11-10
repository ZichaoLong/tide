# Autograd 图诊断

这组工具用于回答“构建了多少求导节点、反向实际执行了多少次、哪些边界保存了张量”。
节点数不能直接换算成耗时、kernel 数或峰值内存；正确性仍由独立数学、梯度与恢复测试验收。
计量实现见 [Python Census](../python/tidegraph/autograd_census.py)、
[native census](../cpp/include/tide/autograd_census.h) 和
[Python/native bindings](../cpp/src/bind_metrics.cpp)。

## 两个复现入口

先按 [development](development.md) 配好当前解释器、依赖与需要的 native 构建。
从仓库根目录运行；每种模式使用新进程和新的输出目录。以下是小型 CPU 诊断，
扩大规模后的长作业遵循 [execution-flows](execution-flows.md)。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python scripts/diagnose_autograd_graph.py \
  --output-dir artifacts/census-public-backward \
  --baseline graph --client python --device cpu --dtype float32 \
  --width 4 --depth 2 --region-width 2 --batch 2 --length 3 \
  --mode backward --threads 1
```

[公共诊断入口](../scripts/diagnose_autograd_graph.py) 使用固定种子的输入和 hidden MSE，
记录源码、工作区状态、解析后的模型/放置配置、参数与输入哈希、实际工作量，以及
`before_backward`、`after_step`、`native` 计数。所有阶段时间均包含诊断开销。
可用 `--baseline fc-tensor` 比较独立 FC 数值投影，或 `--baseline transformer` 比较
不同数学函数的常规 Transformer；二者使用 Python ATen，参数量和比较范围会单独记录。
native 图调用改用 `--client native --native-library /absolute/path/to/build`，要求匹配的
`build-manifest.json` 和 census bindings，缺失时明确失败。该入口没有 resident 调度资格声明。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python scripts/benchmark_dense_query.py \
  --downstream-repo ../tide-learning-lab \
  --output-dir artifacts/census-downstream-grad-forward \
  --device cpu --client python --baseline graph --batch 1 --length 2 \
  --mode grad-forward --threads 2 --warmup-state reset --autograd-census
```

[固定下游任务入口](../scripts/benchmark_dense_query.py) 只读提取下游提交
`0439efd2af31cf10b07ce9e6423b0195f7449968` 中的任务与配置，不采用下游当前工作树。
它先执行普通 warmup/计时，再执行独立 census pass，写入 `autograd-census.json`；
`result.json` 指向该文件。默认 `--warmup-state reset` 恢复初始参数/优化器，
记录诊断所用参数哈希。选择 `continue` 时不能假定诊断与计时使用相同参数状态。

| 模式 | 两个入口共有的行为 |
|---|---|
| `nograd` | 完整序列前向和 loss，关闭梯度记录 |
| `grad-forward` | 同一完整序列任务，记录求导图，不执行反向或更新 |
| `backward` | 同一任务并执行反向，不执行优化器 |
| `training` | 同一任务、反向、优化器；公共诊断用 Adam，下游入口沿用任务配方与裁剪 |

下游入口额外的 `inference` 只计算末位置 head 与首 token 返回，任务范围不同，
不能直接当作上述完整序列 `nograd` 对照。诊断使用的线程可配置；两个入口记录生效线程数。
正式性能结论使用关闭 census 的测量及对应资源记录，不从 instrumented phase 计算吞吐。

## 如何解释计数

- `unique_nodes` 是已登记根可达图的节点并集；共享节点只计一次。
  `connected_edges` 保留重复边，`null_edges` 单独记录未连接输入。
- `node_executions` 是已登记节点的反向执行次数。保留图后重复反向、局部 VJP 多次调用，
  都可能让执行次数超过唯一节点数；未用于 loss 的返回根也可能从未执行。
- 两个入口登记完整返回的 tensor 字段和任务根；查询、区域、聚合、状态的嵌套私有图
  在构建位置显式登记。只遍历外层 `next_functions` 无法发现 custom Function 内隐藏的图。
  各 `groups` 可重叠，不能相加得到总数。
- Python 能遍历经 Python 接口暴露的 native 外层节点；native observer 又会登记同一外层图，
  并补充其显式私有图。**Python/native 是重叠的观察集合，两个 total 不能相加。**
  比较实现时分别说明覆盖范围，不能把未观察到的私有图当作零成本。
- 手写 VJP，如 [KV sequence](../cpp/src/weighted_kv_sequence_vjp.cpp) 和
  [snapshot bank](../cpp/src/weighted_snapshot_bank.cpp)，没有嵌套私有 autograd 图。
  它们作为外层节点被计数，执行反向公式时由节点执行次数体现；不得虚构 `private` 图，
  也不能据此推断其内部 ATen 算子数量或成本。

Python 在首次登记节点时检查可见的 saved fields，保留无法访问字段的计数，
不安装 `saved_tensors_hooks`，不替换原有 SavedVariable 版本检查。
native 的 `saved(...)` **只覆盖已插入登记的显式保存边界**，不是全部 ATen 内部保存张量。
保存次数与逻辑字节包含重复引用；unique TensorImpl、unique StorageImpl 及 storage capacity
另行去重。storage capacity 可能大于视图有效数据量；still-alive 表示采样时仍存活的已观察存储。
这些都是累计观察量或瞬时存活量，均不是 allocator 峰值、真实设备占用或进程 RSS。

## 生命周期与维护

observer 保存计数、名称和弱身份，不持有 tensor、storage payload 或求导图所有权。
Python 离开 `Census` 上下文会移除执行 hooks；native 停止时丢弃 observer state，
旧 hooks 只有失效的弱引用。业务自身仍持有的结果、图和优化器不会被诊断器释放。
native 必须在 `reset_autograd_census(False)` 前读取 metrics；重置应在工作线程静止时进行。
Python scope 不自动传播到新的工作线程，调用者需要显式传递 observer。

新增隐藏图或显式保存边界时同步维护登记位置、本文范围与
[Python 测试](../tests/test_autograd_census.py)、
[native 测试](../tests/test_autograd_census_native.py)。测试覆盖共享根去重、重复执行、
别名存储、真实私有查询梯度、无梯度模式、版本检查及不延长 tensor 生命周期。
测试通过仅证明这些诊断契约；完整模型/设备资格见对应固定源码证据。
