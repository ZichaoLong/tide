# 常驻四循环记忆数值

固定教材：[affine/delta §1–3](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/affine-delta-learning-note.md)、
[GDN §1–3](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/gated-delta-learning-note.md)、
[SSD §1–3](upstream/tide-core-5/local-semantics/weighted-message/recurrent-memory/ssd-learning-note.md)及
[原始时间卷积](resident-weighted-convolution.md)。公共入口 `weighted_recurrent.h`。

`append_weighted_recurrent` 接 `RecurrentConfig`、Update/Memory/Full action、实际 raw state、
`time[R] int64`、`mass[R]`、`content[R,d]`、`owners[R] int64`、`present[R] bool`及参数bank。
raw存储显式FP32/FP16；mass/content/参数、有效快照、算术、Mem/Full返回及全部伴随为FP32。
int64时间/计数/缓存时间戳分离；raw布局及精确编解码见[状态契约](resident-weighted-state.md)。
参数bank与私有状态行分离，owner指定实际共享参数；缺席由present决定。
Update的mass<=0直接保留整个raw元组，正接收更新全部候选；Full由调用者绑定实际active与proposal。
Memory明确绑定所需初态/旧态/候选态，不用有效状态替代checkpoint原始元组。

| 定义 | 数值和返回 | 实现 |
| --- | --- | --- |
| 共同有效状态 | Δ=max(0,θ-s)，每头H、两通道Z/W按固定率衰减；缓存原样读取 | `weighted_recurrent_snapshot.cpp`，int64到FP32最近偶数转换，vendor乘法 |
| affine/delta Upd | 有界key、softplus保留、外积/预测误差；Z/W共同写入 | `weighted_recurrent_algebra.cpp::update` |
| GDN Upd | 固定时间卷积/SiLU、epsilon L2、每头门与Delta更新 | 同上；`WeightedRowGraph::convolution` |
| SSD Upd | 分组B/C、dt外积、独立每头保留 | 同上；共享组重复读取并汇总全部反向贡献 |
| Mem sum/state | 四个独立根；basic两矩阵查询，GDN/SSD短摘要+长矩阵查询 | `weighted_recurrent_algebra.cpp::memory` |
| Full | basic两头输出之和；GDN逐头RMS后门；SSD先直接项/门后全局RMS；共同SwiGLU残差，G未乘c | `weighted_recurrent_algebra.cpp::full` |
| 完整raw Upd | 精确时钟/计数/缓存保留及未加权新卷积项×w | `weighted_recurrent.cpp` + `weighted_raw_update.cpp` |
| 保存一阶反向 | 独立raw字段、mass、content及每个参数角色；身份路径与有效快照路径相加 | `weighted_recurrent_vjp.cpp` |

`weighted_recurrent_parameter_shapes(B,config)` 是严格物理布局；未声明/不适用角色拒绝。
线性参数 `.matrix[B,out,in]`、可选 `.bias[B,out]`；retain/write向量表示为单输出投影，
`log_rate`、`delta_bias`、`full.rho/direct`等向量表示为 `[B,width,1]` 乘常值1。
Basic `read.out.0/1`、`full.out.0/1`分别对应两个数学矩阵；参数别名通过canonical角色归约绑定。
`conv.weight[B,taps,features]`、`conv.bias[B,features]`保留所有物理tap，不能压紧稀疏事件。
公开shape包含一个配置的完整Upd/Mem/Full参数；某action未使用的参数返回None连接，不读取数值。

RowGraph的矩阵乘法使用vendor BMM，反向为保存输入的两次转置BMM；除法、exp、sqrt、
softplus和归一化是设备批量算子。bounded key用sqrt(max(1,||k||²))避免零向量奇异导数，
等号取clamp的声明导数1；softplus与现行Torch阈值20数值约定一致。
配置的正衰减率若在FP32中下溢成零则提前拒绝；数值衰减下溢零仍连接原H/Z/W字段，不删cache、不改逻辑活动集合。
容量准入覆盖局部张量和独立vendor workspace，非法值或超容量显式拒绝。

FP16旧态精确拓宽到FP32，完整Update完成后只在最终raw存储处舍入一次，所有实际
固定/变长字段再次校验；padding不参与校验。有限FP32结果溢出half时拒绝，不提交live owner。
零接收保留原始位模式；正FP32接收即使最终W或缓存值下溢成half零，仍保留更新、
时间/计数和原始时间缓存条目。Cast声明identity一阶VJP，不能把FP16叶梯度累加或
舍入阶梯导数当作此组件反向。raw伴随始终FP32，FP16 cotangent拒绝。
快照与完整组合分别预算拓宽/舍入/guard缓冲，完整组合末尾核验实际保留张量量。

`test_resident_weighted_recurrent.py`独立核验巨大原始age、负差钳制、下溢连接、
精确metadata和身份路径；`test_resident_weighted_describe.py`核验扩展标量VJP。
`weighted_recurrent_formula.cpp`逐行直接写教材公式，不调用候选局部代数、快照、更新或卷积；
`weighted_recurrent_check.cpp`覆盖四族×两种Mem×三action，完整字段及共享参数根、
零接收、空缓存、两组四头SSD、非对齐时间、保存版本与动态出现。原有Python手算公式是另一独立锚点。
入口 `scripts/{build,verify}_device_control.py --checks weighted-recurrent`。
两dtype检查直接在实际量化raw的独立FP32公式上求导；仅Update前向期望最终舍入。
额外四族见证精确检查次正规快照、零接收按位保留、2^-25正接收后零W/缓存条目及质量根、
生成值溢出不提交。两dtype源码扩展的设备资格另记，不追溯改变旧报告范围。

这部分仍是局部数值组件；全图在线就绪/选择/推进、retained训练、精确图checkpoint、peer与性能另行验收。

固定源码资格见[6e2ecb7运行证据](evidence/core5-resident-recurrent-20261028.md)。
