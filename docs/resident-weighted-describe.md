# 常驻完整 Describe

固定 [区域教材 §3、5.1](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)
与[共同架构 §3](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)。
`weighted_describe.h` 将实际 Mem 四返回、接收强度和内容组合为适用的完整节点描述。
旧/新/绑定初态由调用者明确提供，Describe 不重放记忆递归或替换快照。

输入独立字段 `mass[R]`、`content[R,d]`、`short_memory/long_memory[R,ds]`、
`short_total/long_total[R]`；整数 `owners[R]` 选择共享参数 bank，-1 表示缺席。
只有真实行读取输入/参数；强度与统计量须非负，数值字段须有限。
输出 fields 的标量统一编码为 `[R,1]`，向量为 `[R,width]`；这些轴不合并独立根。

| 配置与定义 | 实际返回/参数角色 |
| --- | --- |
| cat §3.1 | `score` ScalarHead 读 f=[N(u),log1p(r),Ms,Ml,log1p(Ws/Wl),p*Ms,p*Ml] |
| match §3.2 | `score.encoders.{cur,mem,stat}`；`score.gate/up/vector/bias`，内积和门控残差的两尺度独立 |
| context/cat evidence §3.4 | `evidence` 线性投影后 RMS |
| context/grouped evidence §3.4.1 | 三独立 encoder 与 cur/mem/stat/interaction 四投影、偏置后 RMS |
| fixed regional control | evidence 为定义的常值零；没有该分支的参数 |
| att/current §3.6.1 | `current` 线性映射 N(u)，返回 x |
| att/key §3.6.2 | content_stats 用线性 key；content 用独立两层归一化 key encoder |
| regional upper bound §2 | w=min(1,r)，mu=min(1,gamma_max*w) |
| node gain §5 | fixed_gain、可训练 gain.logit 或 context ScalarHead；mu=min(1,gamma*r) |

`weighted_describe_parameter_shapes(B,d,ds,config)` 是物理bank的完整维度接口，严格拒绝
缺项/多项和不适用配置。projection 使用 `.matrix[B,out,in]`、可选 `.bias[B,out]`；
head 使用 `.matrix[B,hidden,in]`、`.hidden_bias[B,hidden]`、
`.projection[B,out,hidden]`、`.bias[B,out]`。这对应 ScalarHead/Encoder 的四个教材参数。
`summary.matrix[B,ds,d]` 对应记忆的 W^S；节点记忆仍私有。
match 的向量用 `score.vector.matrix[B,1,hidden]`，标量偏置/gain.logit
用 `[B,1,1]` 乘常值1表示；grouped evidence 偏置用 `[B,dreg,1]`。
它们是相同数学参数的明确打包，不引入额外可训练参数；初始化与别名由调用者绑定。

`WeightedRowGraph` 只组合固定局部算式，不能决定图事件就绪或选择。前向的投影、RMS、
SiLU、sigmoid、log1p、minimum、乘加/归约、拼接使用批量设备算子；guard kernel 处理
实际行、有限性和结构根。minimum 的前向承载 clamp_max，声明反向在相等边界取1。
该反向身份由 clamp 定义决定，不能套用两个可训练输入的 minimum 平分规则。

`append_weighted_describe_vjp` 接名称→(cotangent,connected) 的独立字段根。
静态反向按实际 tape 汇总全部使用者；零数值或零尺度保留连接，缺席保持 None。
各参数角色单独返回bank梯度和连接，跨角色别名由 canonical 归约层汇总。
所有输入和有效参数版本已保存，反向不读取原始输入/参数。精确状态解码不属于此局部代数层。

FP32 accelerator；局部tensor预算与vendor workspace独立准入，失败不得提交输出。
这部分不认证完整在线图、循环数值、retained训练、checkpoint或跨卡。

`test_resident_weighted_describe.py` 独立autograd核验标量链/边界与None/零累加。
`weighted_describe_formula.cpp` 不调用候选代数组合，逐行直接公式生成独立各字段根；
`weighted_describe_check.cpp` 选八种适用组合，含每字段独立根、三共享bank、动态输入/owner、
全部参数与输入VJP、固定证据常值、零match尺度、保存版本和容量/非法输入拒绝。
入口 `scripts/{build,verify}_device_control.py --checks weighted-describe`。

固定源码资格见[cd576d2运行证据](evidence/core5-resident-describe-20261027.md)。
