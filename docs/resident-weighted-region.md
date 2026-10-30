# 常驻完整带权区域组合

固定 `0770593` [区域分配 §2–4/S.1](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)。
`weighted_region.h::append_weighted_region` 把完整候选描述归约、ScalarHead/Query、
评分、发送映射、活动集合和singleton history接入同一DeviceProgram。
这是SelStep组合；Agg、Describe、图就绪/推进、训练owner和checkpoint仍各有独立验收。

每行一个完整区域帧，配置在批内一致。FP32 `received/caps/score[R,N]` 对应
`w/mu/s_loc`，`evidence[R,N,E]`、`current/key[R,N,D]` 是已算出的描述字段。
它们不是原始消息或状态；局部发送增益已反映在mu中，不能重复应用。
数学输入满足 `0<=w<=mu<=1`；组件允许独立字段根以检查局部偏导，
Describe的增益绑定和可达描述由调用者保持。
不适用向量字段不构成输入依赖，内部dummy不产生梯度连接。

`lengths[R]`、`ids[R,N]`、`times[R]` 为int64。长度0表示没有作用，忽略该行
所有输入和owner；padding不是候选，允许poison值。非空帧ID非负、互异，
时间非负；所有动态值在设备检查。设备输出controls、严格`c>0`的active和
实际time/history_present；空行不要求图owner覆盖已有history。
无参考递归参与解码，未把原始int64坐标转成浮点。

context特征是 `[sum(w*e)/max(1,sum(w)), log1p(sum(w)), log1p(sum(mu))]`。
Query按配置的received或mu归约current；regional函数必须用received。
ScalarHead和Query使用[已声明的vendor批量算子](resident-weighted-head.md)，
各自的int64 owner选择独立参数bank。参数共享不合并行状态或输入。
总量B、阈值与评分经device finish送入[精确分配](resident-weighted-allocation.md)
或[发送原语](resident-weighted-sending.md)，十种正式发送保留各自函数身份。

`tape`保存实际字段、坐标、head输入/参数版本、映射分支及结果。
`append_weighted_region_vjp` 从这些实际值连接精确解/分支VJP、head显式VJP、
加权归约和log/clamp导数；不对分配迭代求导，不使用HST。
每行独立cotangent/root标志，返回六字段梯度与bool连接、两个head共享bank梯度。
零根值仍连接；不适用字段/硬分支的score和head保持None。
跨head或角色的实际参数别名由外层owner汇总；该组件不拥有optimizer。
下一次前向重用tape前，保留训练owner必须保存它；组件只声明一阶。

前向/反向各有scratch准入，vendor算子workspace另设。sticky error阻止外层提交，
非法元数据、非有限值、正支持下溢及反向溢出保持明确错误。
此组合使用FP32；FP16整图与CUDA真机不能从组件源码或CPU结果推断。

| 数学/接口 | 实现 | 独立检查 |
| --- | --- | --- |
| 完整候选摘要、Query、B/threshold | `weighted_region_prepare.h` 与prepare/finish入口 | `test_resident_weighted_region_math.py`，独立Torch公式+已验发送映射，25组合 |
| 完整一阶链式VJP | `weighted_region_reverse.h` 与reverse入口、`weighted_region_vjp.cpp` | 同上六字段独立根；`weighted_region_check.cpp` 标量RegionKernel/autograd与共享参数根 |
| 坐标、缺席与支持 | prepare/result入口 | `test_resident_weighted_region_kernels.py` 实际登记入口CPU适配，大int64/poison/重复ID/重放 |
| vendor与设备完整组合 | `weighted_region.cpp` | standalone check的14配置、不同长度重放、全部字段/参数、原来源改写后独立反向、失败不提交/容量 |

运行入口 `scripts/verify_device_control.py --checks weighted-region --device npu:0`。
独立公式验证局部数学；标量RegionKernel使用独立执行与autograd，但复用已单独验收的
公共发送数学接口。目标设备证据才证明组合算子实际执行；固定源码资格另行记录。

固定bc935fe的[CPU/NPU/CUDA构建资格](evidence/core5-resident-region-20261027.md)已记录；完整图生命周期和GPU真机分开验收。
