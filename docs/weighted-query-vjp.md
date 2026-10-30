# 带权 Read / Full 的批量一阶 VJP

固定 `0770593` 的[共同架构 §3–4](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)、
[KV §2–4](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)与
[循环模块对应](weighted-recurrent.md)。这是同一 Read/Full 函数的实际批量图反向；
它不采用 HST，也不改变区域精确分配映射的 VJP 身份。
数值布局、完整可见窗口、int64 时钟与有限描述头运算次序见 [query batch](weighted-query-batch.md)。

## 公共接口与计算

| 层 | Python | 独立 C++ |
| --- | --- | --- |
| Read 能力 | `read_vjp.supported/evaluate` 的确切 `Describe` 类型 | `ReadKernel::fields_batch_grad`；weighted `Describe::batched_autograd` |
| Full 能力 | `FullProgram.batched_autograd/batch_grad`；`WeightedFull` | `FullKernel::batched_autograd/batch_grad`；weighted `Full` |
| 私有图与根 | `weighted/query_vjp.py::_Query,evaluate` | `weighted_query_vjp.cpp::QueryVjp,read_vjp,full_vjp` |
| 实际局部计算 | `read_batch.evaluate` / `full_batch.evaluate` | `query::describe` / `query::full` |

packed Read 在 grad mode 自动选择已经声明的结构化批量能力。
Full 使用显式 `full_autograd="batched"`，默认 `replay` 仍可调用；unpacked + batched
或缺少能力的 Full 提前拒绝。Python 的旧 ProjectionEmit 通过相同能力接口保留原路径。
通用 Read/Full 不因本地带权绑定而获得未经实现的能力。

每组捕获记忆与对应 Read/Full 程序的注册参数，建立私有叶和真实批量计算图。
参数树中的 Tensor/模块别名保持；相同参数不同输入仍独立计算，所有使用者的梯度累加。
每行保留自己的快照浮点槽、int64 非求导槽、content，以及 Full 的 control。
Read⁰ 使用真实绑定初态；Full 始终读本请求 comparison，不借用 Read 的快照。

分组依据状态有无、槽名/形状、dtype/device/requires_grad 元数据，以及 KV 正权重
Attention 的零/非零分支。全零权重或空可见集不连接本来未参与的 Q/K/V 查询；
正权重集合内的数值零导数保持连接。没有按梯度数值、epsilon 阈值或实际 loss 裁剪图。
实际 Mem/Full 数值来自候选批量图；不调用 scalar Read、Mem 或 Full 重放生成反向。
有限 Describe 头仍保留逐行运算，不能将这一能力表述为所有算子的融合。
标量 MatchScore 的 residual gate/up 与 Encoder/GroupedEvidence 一样采用教材列向量次序；
该数值约定避免 Python/native 方向不同的微小投影舍入被复合独立根放大，公式与容差不变。

KV §3/4 的 Attention 读取保持原 softmax 概率加权和前向；一阶反向先以首个值为基准
形成 `delta_i=v_i-v_0`，logit 导数为 `p_i*dot(bar,delta_i-sum_j p_j*delta_j)`，
value 导数为 `p_i*bar`。这是精确映射的等价导数表达，不微分数值求解或作阈值裁剪。
相同值的 logit 根是连接零；先有正权重筛选和原总量增益，零分支仍不连接 Q/K/V。
实现：[Python](../python/tidegraph/weighted/attention_mean.py)、
[C++](../cpp/src/weighted_attention_mean.cpp)，分别接入 scalar/batch 的 Mem 与 Full。
[独立测试](../tests/test_weighted_attention_mean.py)用二项解析 Jacobian、平移不变性、
相同值不变性及独立标量方向差分；[standalone](../cpp/test/weighted_attention_mean.cpp)另算解析值。
此改进由来源 node 绑定的 FP32 PDG 第二次 AdamW 更新发现的微小 query 梯度消减误差驱动；
不改变前向函数、输入、dtype 或既定容差，具体资格与原失败见来源绑定证据。

架构 §1 的共同 RMS 归一化另采用 `(bar-n*mean(bar*n))/denominator` 一阶 VJP，
其中 `n=x/denominator`，前向表达式不变。先在归一化坐标内收缩，再除一次 denominator，
避免独立形成逆幂项。实现 [Python](../python/tidegraph/weighted/normalization.py) /
[C++](../cpp/src/weighted_normalization.cpp)；[解析 Jacobian/方向差分](../tests/test_weighted_normalization.py)
及 [standalone](../cpp/test/weighted_normalization.cpp) 单独验证。
扩展回归在此前 169bf67 复现的 FP32 混合记忆评分梯度差异由这一等价表达修正；
原失败与后续资格分别记录，不修改比较容差。

## 梯度与所有权

公开每行的每个 Describe 字段是独立根；Full 的 value 与广播槽复用该行实际 payload。
反向按**已定义 cotangent 的字段模式**分组：只求这些字段与所用行的输入梯度，
共享参数得到所有使用者贡献。其他行的未使用输入返回 None；已连接的零 cotangent
产生连接零。重复使用同一公共输入、跨程序参数别名都由外层 autograd 累加。

原始输入/参数的版本计数被保存，反向前修改会报错；正常 retain_graph 可重复使用。
只声明一阶 VJP，高阶请求明确拒绝。私有图不保存新的 checkpoint 格式；持久化仍是
声明的 detach 边界，精确原始状态/时间戳/pending 由现有 checkpoint v5 编解码。
设备就绪/选择/推进与此次 eager 局部合批不同，不能由 CPU 测试推断 resident 支持。

## 独立验证与证据边界

- [局部根](../tests/test_weighted_query_vjp.py)：五种记忆与全部 Mem 读取，空/过期/零权重/
  Read⁰、每字段/状态/输入/控制/输出槽的独立根及 None/零；跨程序别名与重复公共输入。
- 同文件的 FP64 方向差分同时扰动参数、实际浮点快照、content/control，对照独立标量公式；
  该检查验证光滑分支的一阶导数，不能替代离散支持和结构连接断言。
- [完整执行](../tests/test_weighted_read_batch.py)、[训练与续接](../tests/test_weighted_query_training.py)：
  独立标量调度、HB 与反馈 PDG、所有局部批量能力同时启用、保留图切点、三步 AdamW、
  含 pending 的 checkpoint/换 owner 恢复、共享模块身份和实际零 replay 计数。
- [standalone](../cpp/test/weighted_query_vjp.cpp)：literal SiLU 导数、单项 KV 加权读出导数、
  未使用行与状态槽的 None、连接零、广播别名、原始版本与一阶限制；不依赖 Python 调度。

[数值批量资格](evidence/core5-query-20261027.md)只认证其固定提交的数值和 replay 路径。
本批量 VJP 的开发/固定源码资格见 STATUS 与后续独立证据，不冒用既有通过数。
