# 带权常驻发送原语

固定 `0770593` [区域分配 §4.2–4.4/§6.3/S.3](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
[公共发送契约](weighted-sending.md)定义函数适用域、正支持、平票及各自反向身份；
正温受约束分配使用独立[常驻Alloc](resident-weighted-allocation.md)。
本页是发送原语的设备表示，不将局部组件等同于完整图常驻。

`weighted_sending.h::append_weighted_sending` 将完整候选行表示为FP32
`scores/caps/received[rows,width]`、`parameters[rows,2]=(B,tau)`，以及int64
`ids[rows,width]`和`lengths[rows]`。length=0表示不存在区域，padding完全忽略；
真实零cap候选仍存在。Hard输入已调整score，SortedFill输入已计算总量B。
上下游控制头和区域证据另行组合，不能由主机依据真实轨迹预先生成发送结果。

| kind / base | 对应函数与输入依赖 | 反向身份 |
| --- | --- | --- |
| Direct / cap、received | direct_attention；`c=mu*p`，p依score/base/tau | attention精确map加外乘 |
| Unit / cap、uniform | unit_attention；uniform只依实际候选存在性 | attention精确map，含零base右导数 |
| Pooled / cap | pooled_attention；`c=min(1,sum(mu))*p` | attention精确map加B |
| Proportional / 不适用 | `mu*min(k,sum(mu))/sum(mu)`；原点返回mu | 比例map，原点恒等导数 |
| SortedFill / 不适用 | 按score降序、int64 ID升序分数填满B | 固定排序分支VJP，score=None |
| Hard / 不适用 | 前k个正调整score；全非正取最高正cap一项 | 所选cap分支VJP，score=None |

接口以CapBase编码无base配置；k仅Proportional/Hard使用，其余必须0。
Attention不读取B，比例不读取score/received/B/tau/ID，sorted/hard不读取received/tau。
输入的未使用槽不是额外数学依赖；无效适用配置明确拒绝。
不同反向没有HST或有限迭代求导。正支持在FP32下溢明确报错，不裁剪成零消息。

前向tape是实际controls、4字段detail、3字段统计、int64 branch/length。
attention保存p、中心化score、log比率、cap；sorted保存满项/部分项之前的满项掩码。
`append_weighted_sending_vjp` 接受独立行root存在标志，返回
`vectors[rows,3,width]=(score,cap,received)`、`scalars[rows,2]=(B,tau)`及五项bool连接。
调用者再将连接与实际requires-grad相交。全零Attention是声明的连接零map；
零base右导数、uniform的cap不连接、分数填充的B连接零与hard的B=None均分别保存。
收到的g只检查实际存在root的有效元素，padding NaN不参与。

ragged稳定排序、正支持与结构连接分支由单个设备scalar worker完成，Exp/Ln使用厂商原语。
当前DeviceProgram没有能覆盖这些分支/None规则的统一ATen归一化或排序节点；
连续buffer不构成吞吐优化声明。窗口内没有主机选择/迭代，完整图仍须自行接入就绪与推进。
前向及反向scratch分别预算。沿用错误2/20/21/23和sticky汇总；只有error=0时可提交proposal。
tape在下一次program调用覆盖，retained训练owner须先journal；本组件不代替checkpoint。

| 实现与独立证据入口 | 能验证的内容 |
| --- | --- |
| `ascendc/tide_weighted_send.cpp`；`test_resident_weighted_sending.py` | plain exp商公式、LP顶点、稳定大int64 ID、分数填充反例、原点与正支持 |
| `tide_weighted_sending_vjp.cpp`；同测试 | 公式autograd/手算、方向差分、全部独立字段、None/零/缺席根、padding与拒绝 |
| `weighted_sending_check.cpp` | 独立C++发送公式；同device program不同输入/长度重放、全部梯度/连接、失败不提交与容量 |

CPU适配直接编译登记的kernel源，不能认证NPU/CUDA；实际设备检查入口
`verify_device_control.py --checks weighted-sending --device npu:0`。
源码、构建、实际设备与完整图范围分别记录到证据和ROADMAP。

固定ce5fa53的[CPU/NPU/CUDA构建资格](evidence/core5-resident-sending-20261027.md)；完整SelStep及GPU实机未据此认证。
