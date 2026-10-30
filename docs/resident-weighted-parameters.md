# 常驻带权模块的 canonical 参数归约

对应固定 `0770593` 共同局部架构 §2.2 的参数共享、实现核对 §1.3 的完整参数/VJP追溯。
[来源低秩链](resident-weighted-source.md)、[Aggregate](resident-weighted-aggregate.md)
和[区域/head](resident-weighted-region.md)返回各语义角色偏导；本层按实际Parameter身份归约。
参数共享不合并输入、私有状态或独立根，数值相等也不构成参数别名。

`weighted_parameter_vjp.h::append_weighted_parameter_vjp` 接受公共 `ParameterRegistry`
及有序 `WeightedParameterPartial{name, values, connected}` 列表。
每项是一次真实角色/使用贡献，FP32 values与该名称的原参数形状相同，
connected是同设备一个bool；缺席partial的内容可为poison，连接零仍参与。
多个名称指向同一个TensorImpl时，所有贡献进入同一canonical owner。
不同TensorImpl即使数值相同或storage重叠，也不在本归约中自动合并。

宿主只读取注册名、形状、别名与buffer地址，构造静态owner/segment/tiles表。
窗口内按实际设备bool决定贡献，稳定按输入列表顺序归约每个owner的每个坐标，
没有数值CPU预跑、动态主机分支或伪造None。向量tile保留尾部边界，
每个tile的scalar错误使用独立64字节行，避免多worker cache-line写冲突。
设备原生变长别名与独立连接规则需要这一小型元数据/归约kernel；局部矩阵乘仍使用vendor算子。

结果是既有 `ParameterVjp{owners,offsets,values,connected}`。
没有任何潜在角色的owner用offset=-1；有潜在角色而本次无根则保留slice与false。
一次失败不能发布partial总和或触发optimizer更新。未知名称、形状/类型、容量非法在提交前拒绝，
非有限实际梯度或求和溢出设置sticky error。只声明一阶，真实partial每次贡献一次由调用者保持。
源buffer被program保留，但跨窗口retained图仍须由训练owner保存实际tape。

归约直接接已有 `DeviceOptimizer` 和 `append_owner_parameter_publish`：
canonical master/slot只更新一次，随后发布到每个使用该参数的设备bank/角色。
这些旧组件的来源和资格仍按原证据保留；新组合必须单独验收。
Optimizer自身拒绝不同owner的重叠storage更新，归约保留身份不代表这种更新顺序已被支持。
FP32梯度可归属于FP32/FP16注册参数；本项新完整组合检查首先限定FP32。

| 对应 | 符号 | 独立检查 |
| --- | --- | --- |
| 共享参数的所有使用贡献 | `weighted_parameter_vjp.cpp`、`weighted_parameter_reduce.h`及登记入口 | `test_resident_weighted_parameter_reduce.py`手工分段求和、513维尾块、1/32 workers、None/零、poison、重放/溢出 |
| 真实TensorImpl身份 | `ParameterRegistry::owners/canonical_name` | `weighted_parameter_check.cpp`跨bank/角色的真实别名，同值独立叶、无角色owner |
| 梯度/更新/发布 | head VJP → canonical reducer → DeviceOptimizer → publisher | 同check逐行独立ATen/autograd、SGD/AdamW四步，全参数/slots及None计数 |
| optimizer恢复和事务 | 原optimizer snapshot/restore，新的program/owner | 同check换owner恢复后两步完整状态按字节一致，非有限输入不改master/slot/已发布bank，容量拒绝 |

入口 `verify_device_control.py --checks weighted-parameters --device npu:0`。
这项局部训练组合不包含图continuation、retained跨窗口反向或磁盘checkpoint；完整图生命周期另验收。
CPU helper、NPU实际执行、CUDA构建/真机分别记录。

固定源码资格见[f000c3e运行证据](evidence/core5-resident-parameters-20261027.md)。

## 冻结参数池补充

`ParameterBankPool`只在一个私有冻结owner/反向重放内存活，按实际TensorImpl、切片和设备
共享FP32数值bank；同值独立参数不合并，跨设备仍有各自副本。源句柄保留，源版本变化拒绝，
首次分配前检查剩余预算；复用不重复扣实际分配。通用`copy_parameter_bank`仍提供独立拷贝。
各角色仍保存自己的参数名和VJP贡献；训练发布按物理目标去重并校验canonical source一致，
避免对共享bank重复并发写。checkpoint的参数别名、状态私有性和optimizer边界不变。
CPU独立切片/别名/版本/预算反例在`cpp/test/parameter_bank.cpp`；设备完整根、训练、恢复
与实际节省须由当前固定源码资格和测量认证，不能仅凭少扣费推断内存收益。
