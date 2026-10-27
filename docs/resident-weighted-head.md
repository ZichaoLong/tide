# 常驻批量 ScalarHead / Query

固定 `0770593` [区域分配 §3/S.1](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
共享的两层仿射形状为 `y=W2*SiLU(W1*x+b1)+b2`；ScalarHead输出维1，
Query的输入/隐藏/输出维均为selection_width。具体上下文与query的特征构造由区域组合层拥有。
本组件没有把通用MLP当成完整SelStep，调用与验收边界见[ROADMAP](ROADMAP.md)。

`weighted_head.h::append_weighted_head` 接受FP32 `input[rows,I]`与
int64 `owners[rows]`。`-1`标记不存在行，其输入可为NaN；其余ID在设备上选择参数bank。
四个bank分别是 `[owners,H,I]`、`[owners,H]`、`[owners,O,H]`、`[owners,O]`。
同owner共享参数，每行输入和激活保持私有，不按数值相同去合并计算。
不同语义角色的参数别名由外层参数owner绑定；本组件返回四个独立角色的偏导，
外层须再汇总被显式绑定到同一叶的角色贡献。

设备先检查owner、构造安全索引与有效输入，vendor index_select真正聚集参数行，
batch_matmul/SiLU/SiLUBackward完成矩阵与激活计算。没有用自定义标量循环实现矩阵乘。
DeviceProgram需要连续storage，权重转置用设备permute显式实现，两份临时布局计入scratch预算。
CUDA采用对应cuBLAS/设备算子入口。KEEP_DTYPE不允许隐式HF32/FP16替换FP32。
缺席行在算子前清空，真实零输入仍保留；无主机逐行分支或读取实际owner。

tape拥有实际输入、owner、聚集参数、preactivation和hidden；后向不再读取可变参数bank。
`append_weighted_head_vjp` 接收每行独立cotangent/存在标志，以批量矩阵乘求偏导，
再按设备owner稳定汇总所有使用者，返回input梯度、四个bank梯度及row/owner bool连接。
零cotangent仍连接；未被任何根使用的bank不连接。数值零不能作为root存在性替代。
只声明一阶；尚未接宿主autograd、optimizer或retained图owner。
tape下一次调用覆盖前须由训练owner保留；存原始来源的引用不等于已保存权重版本。

独立前向/反向scratch准入，设备错误2为owner元数据、20为非有限值、23为反向溢出。
失效索引先改为安全索引用于内部算子，sticky error阻止提交；这不是把非法调用当成合法回退。
无效/缺席行不能影响活跃输出；参数或梯度的溢出不能发布部分结果。

| 对应 | 实现与独立验证 |
| --- | --- |
| ScalarHead / Query | `weighted_head.cpp`，vendor batched affine/SiLU |
| owner与存在性 | `tide_weighted_head_plan/guard.cpp`；CPU源码检查大int64/非法ID/poison缺席行/None与零 |
| 共享参数完整VJP | `tide_weighted_head_reduce.cpp`；手工使用者求和、未用bank和溢出拒绝 |
| 实际数值和版本 | `weighted_head_check.cpp`；逐行独立ATen公式/autograd，ScalarHead/Query两种形状、同program owner重放、原bank改写后从实际tape反向、失败不提交/容量 |

CPU元数据适配不能证明厂商算子或GPU执行。目标检查为
`verify_device_control.py --checks weighted-head --device npu:0`；固定源码、构建和实际设备证据分列。

固定161ccea的[CPU/NPU/CUDA构建资格](evidence/core5-resident-head-20261027.md)已记录；完整区域及GPU真机分开验收。
