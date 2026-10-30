# 常驻共享投影与可选输入 RMS

本层实现固定 `0770593` 共同归一化 N_d 及 KV §1.2、循环 §2、GDN/SSD §2 使用的线性投影因子。
完整记忆函数的门、矩阵更新、快照、Mem/Describe/Full 仍由组合层实现；公共投影不替代这些验收。

`weighted_projection.h::append_weighted_projection` 接 input[R,d]、int64 owners[R]、
matrix[B,o,d] 与可选 bias[B,o]。可选 epsilon 显式控制是否先用 N_d 归一化；
启用时必须有限且为正，关闭时保持输入原值。每个 numerical group 具有固定尺寸；
不同输出宽度分组，不把 Read/payload/聚合/状态强制成同一宽度。
owners=-1 为缺席；其他 owner 必须在 bank 范围内。相同 bank 的全部实际使用者贡献归约；
不同输入行始终是私有根。公共参数别名由[canonical归约](resident-weighted-parameters.md)进一步处理。

input、matrix、可选bias及output使用同一显式FP32/FP16存储dtype。FP16输入/参数通过设备Cast
拓宽，RMS、BMM及bias相加在FP32执行，最后一次舍入为FP16；不声称使用FP16 GEMM。
tape保存FP32实际运算值，反向根及所有返回伴随为FP32，Cast声明一阶identity VJP。
该契约独立于CPU FP16叶梯度的累加舍入，也不是舍入阶梯的数学导数。
有限FP32结果若在写入FP16时溢出，则通过设备共享错误决定拒绝；次正规数保留和舍入下溢
分别核验，数值零不删除行/参数的连接。新增拓宽与结果检查缓冲计入scratch budget。

输入计划和根守卫复用已声明的 head 元数据组件。数值使用 vendor RMSNorm、index_select、
permute 和 batch_matmul；可选 bias 用 vendor add。没有输入值的主机预跑或逐行主机决策。
无 bias 时没有该参数角色，不能用固定零 bias 伪造可训练参数。

`append_weighted_projection_vjp` 使用保存的实际归一化输入/rstd/参数行，以 vendor BMM
计算输入与逐使用者矩阵梯度，再归约共享 bank。输入 RMS 使用
`rstd*(bar−normal*mean(bar*normal))` 的普通一阶收缩；关闭 RMS 时为线性输入 VJP。
每行 bool root 独立于数值零；没有参与目标的 bank 保持 None，连接零保留连接。
保存版本不依赖原输入或参数后续内容。不声明高阶；tape 由 owner 保持不变。

显式 FP32/FP16 常驻域、scratch budget、参数/输入/output有限性检查沿用 head 错误决定。
有限性/owner/容量失败不提供有效结果；完整模型须在共享错误决定后更新 live owner。
本组件没有图调度、参数更新或 checkpoint 的独立副本。

| 定义/接口 | 独立验证 |
| --- | --- |
| N_d 的一阶导数 | `test_resident_weighted_projection_norm.py` 独立 FP64 autograd 与联合方向差分；零输入、127维尾宽、None poison、非有限拒绝 |
| 线性投影与共享使用 | `weighted_projection_check.cpp` 在实际量化输入上用独立FP32 ATen/autograd；有/无bias、有/无RMS，动态owner/输入，所有输入/参数VJP；FP16前向仅在最终存储舍入 |
| 保存版本/连接 | 同check 48重放行、三个bank、None/零、修改原输入/参数后的独立反向、容量拒绝 |

入口 `verify_device_control.py --checks weighted-projection --device npu:0`。
CPU helper、实际NPU、CUDA构建与GPU真机分别记录；局部投影资格不自动扩展为五记忆或完整常驻图资格。

FP32固定源码资格见[4a88a54运行证据](evidence/core5-resident-projection-20261027.md)。FP32/FP16组件扩展见[122202f固定资格](evidence/core5-resident-half-primitives-20261028.md)。
