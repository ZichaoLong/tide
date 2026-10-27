# 常驻来源函数的低秩参数链

固定 `0770593` [共同局部架构 §2.2](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md#22-来源函数与参数共享)。
本组件为[常驻Aggregate](resident-weighted-aggregate.md)构造
`A=baseA+LA@RA`、`B=baseB+LB@RB`，并将有效A/B/e的VJP传回真实基座和四个因子。
同名配置的参数共享规则仍由[来源绑定](weighted-source-binding.md)规定。

`weighted_source_adaptation.h::append_weighted_source_adaptation`接受FP32 bank：
基座 `[banks,H,d]`、`[banks,d,H]` 和embedding `[banks,H]`；因子依次为
`[banks,H,r]`、`[banks,r,d]`、`[banks,d,r]`、`[banks,r,H]`。
每数值组有共同正整数d/H/r；异构配置按静态形状分组。
所有配置参数须有限，包括当前没有消息使用的bank。
厂商batch_matmul构造有效矩阵，未使用自定义标量循环代替矩阵乘。

实际前向tape复制四个因子及有效参数。原始基座/因子随后改变时，反向仍使用该次保存版本。
`append_weighted_source_adaptation_vjp`接受有效A/B/e梯度及逐bank独立root标志，
返回三种基座角色和四种因子角色的完整偏导与连接。
基座梯度是有效矩阵梯度；各因子通过批量矩阵乘和显式vendor permute求导。
没有根的bank在矩阵乘之前清空梯度与操作数，不能用数值零猜测None，
也不能让无连接的poison值经`0*NaN`污染其他bank。

组件返回各bank/角色偏导；跨bank、不同角色的真实同一叶仍由canonical参数owner汇总。
有效参数梯度与低秩因子梯度是不同返回，不能互相替代。
只声明一阶；retained图、optimizer、精确checkpoint和完整模型绑定仍由上层负责。
前向/反向分别进行scratch准入，vendor workspace另设；非有限参数/结果与反向溢出
形成sticky error，外层必须在统一决定后发布，不能提交部分更新。

| 对应 | 实现 | 验证 |
| --- | --- | --- |
| §2.2 有效A/B | `weighted_source_adaptation.cpp::append_weighted_source_adaptation` | standalone逐bank独立ATen矩阵公式，接实际Aggregate输出 |
| 四因子一阶VJP | `append_weighted_source_adaptation_vjp` | `weighted_source_adaptation_check.cpp`独立逐原子公式/autograd，七种参数角色、共享使用者及载荷根 |
| None/零与非法值 | `tide_weighted_source_guard.cpp` | `test_resident_weighted_source_guard.py`，poison缺席、真实非有限值和sticky error |
| 参数版本/事务 | 保存因子与guard/commit组合 | standalone重放不同因子/长度，改写所有原始来源后的独立反向、失败不提交、scratch拒绝 |

检查入口 `verify_device_control.py --checks weighted-source --device npu:0`。
CPU guard、NPU完整组合、CUDA构建和真机证据分列；本项不单独认证canonical别名owner或图生命周期。
