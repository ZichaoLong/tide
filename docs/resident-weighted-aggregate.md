# 常驻完整带权 Aggregate

固定 `0770593` [共同局部架构 §1–2](upstream/tide-core-5/local-semantics/weighted-message/architecture-learning-note.md)。
`weighted_aggregate.h::append_weighted_aggregate` 消费设备上的完整纤维，返回实际
`(r,u)`，其中 `r=sum(max(m_i,0))`，正总量时
`u=sum(max(m_i,0)*(z_i+alpha*B_i*SiLU(A_i*N(z_i)+e_i)))/r`，零总量时u为零。
载荷始终是 `(m,z)`；发送强度只乘一次。相同SrcClass的多个物理原子各贡献一次。
本组件没有状态/时钟，不能代替图就绪与状态更新。

输入为FP32 `payload[R,N,d+1]`、int64 `lengths[R]` 和 `owners[R,N]`。
长度0表示没有作用，非空但总量零仍是实际候选。只读真实前缀，padding可为poison。
各实际原子载荷有限；负质量取正部。owner只选择源函数参数bank，无同类唯一性限制。
`alpha>=0`、`epsilon>0` 是本批共同配置；alpha0和零总量分支不产生源函数依赖。
完整动态图根据当前消息生成这些buffer，不需要CPU预跑。

源参数bank为有效 `A[B,H,d]`、`B[B,d,H]`、`embedding[B,H]`。
region/node/source绑定及adapted的低秩矩阵合成/别名由参数owner层负责，
组件返回各有效角色的完整偏导供该层继续求导。没有把有效A/B梯度冒充低秩因子梯度。
不同隐藏宽度可由参数owner精确补零到H；无隐藏归一化，补零的神经元输出和梯度为零。
不同bank或角色的同一叶必须汇总全部贡献，不按数值相同合并计算。

设备先检查实际长度/载荷并归约质量，再为有实际源变换的行准备owner。
vendor RMSNorm、[批量双仿射/SiLU](resident-weighted-head.md)执行来源函数；
设备完整归约恢复每个纤维的实际(r,u)。同类共享参数，每原子输入/激活私有。
数值batch覆盖容量行；缺席行使用无依赖的安全零操作数，不声称所有填充计算都已消失。

实际tape保存原始载荷、长度、有效owner、变换值、归一化输出/rstd、参数版本及(r,u)。
`append_weighted_aggregate_vjp`接受每纤维完整(r,u) cotangent及独立root存在标志。
返回每原子完整(m,z)梯度/连接标志和全部源参数bank梯度/owner连接。
非空零总量仍有质量根，内容和源参数保持无连接；零cotangent不等于None。
正总量使用已声明的中心化均值VJP；单原子均值的质量偏导为连接零。
RMS使用保存的真实归一化/rstd执行normalized-coordinate一阶VJP，不重读可变原始参数。
跨作用保留和optimizer仍由训练owner负责；这里只声明一阶，未实现宿主autograd适配。

长度/owner非法、非有限值和反向溢出设置sticky error，外层事务不得提交部分结果。
前向/反向scratch分别准入，vendor workspace另设；有限容量不限制逻辑运行时长。
本组件当前FP32；参数低秩链、完整图、FP16、CUDA真机与性能分别验收。

| 对应 | 符号 | 独立测试 |
| --- | --- | --- |
| §2 完整原子归约/源类 | `weighted_aggregate_math.h::prepare/reduce`，登记prepare/reduce入口 | `test_resident_weighted_aggregate_math.py` 独立逐原子公式、同类不同载荷、零/负质量、singleton、padding |
| §1 RMS及§2 VJP | `reverse_mean/reverse_norm`，`weighted_aggregate_vjp.cpp` | 独立Torch公式/autograd，完整载荷根、None/零、零总量参数无连接 |
| vendor/共享owner/实际保存版本 | `weighted_aggregate.cpp` | `weighted_aggregate_check.cpp` 两alpha、4种隐藏宽度、完整参数根、重放、改写来源后独立反向、容量拒绝与失败不提交 |

目标入口 `verify_device_control.py --checks weighted-aggregate --device npu:0`。
CPU helper只能证明其算术；完整设备组合资格和固定源码证据独立记录。
