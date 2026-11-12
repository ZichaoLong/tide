# 常驻实际参数 bank 的分组矩阵边界

数学固定 `0770593838a37c790a568226930567ed60f99b83`；这是线性投影、ScalarHead/Query 和
KV Upd 的求值表示变化，不改变图、局部函数、指定一阶VJP或checkpoint。
[投影](resident-weighted-projection.md)、[head](resident-weighted-head.md)、
[KV更新](resident-weighted-kv-update.md)仍分别拥有其定义域与返回契约。

`weighted_bank_gemm.{h,cpp}`被这些生产调用者实际使用。输入行数R、输入/输出宽I/O和参数bank数B
是静态形状；int64 `owners[R]`及bool独立反向根是设备数据。一个bank对应同一参数角色的全部实际使用者，
不同bank即使数值相等也保持独立；跨角色/跨bank的真实Parameter别名由canonical reducer汇总。
节点私有状态不因参数共享而合并。

## 前向与转置

多bank在设备上稳定划分实际owner，得到int64 permutation[R]/offsets[B+1]、
行/bank bool连接。划分是O(R+B)，不读取主机实际控制/就绪或参考轨迹。
参数快照保持`[B,O,I]`及可选`[B,O]`，未使用bank置零且不读取其非有限源值。
输入和保存归一化值仍是`[R,I]`，不是每行携带一份矩阵。

每个投影只登记一个固定容量`T=min(R,64)`的数值body，设备int64游标保存`[bank,local_offset]`，
依稳定bank/行序遍历非空组及其tiles。64是有界基础候选，不声称对所有宽度、稀疏比和后端最优。
每个tile最多包含64条实际行，尾部不足部分填零；空bank完全跳过。相同scratch反复复用。
单bank快路继续使用一整块GEMM，其缺席行由mask清零，不经过多bank tile路径。
当前bank通过设备index_select进入单bank矩阵/bias scratch，矩阵复制粒度是tile而不是每个输入行。
本候选尚未跨同bank的相邻tile缓存权重传输，不把固定body称为最少运行时算子调用。
主要数值计算使用vendor GEMM/BMM、permute、bias add或sum，设备核只承担计划、复制、有限性和连接。

每组前向为`Y_b=X_b W_b^T+b_b`。独立根先排除不存在行，转置为
`dX_b=G_b W_b`、`dW_b=G_b^T X_b`、`db_b=sum_rows(G_b)`，实际行scatter回原顺序。
前向输出和dX各行只写一次；dW/db先统一清零，再按稳定tile顺序累加部分矩阵乘/行和。
vendor index_select→add→index_copy串行执行，无冲突scatter或原子加；dW累加复用随后加载
保存权重的同一bank scratch，db另有一个previous值向量。尾部padding不读源、不回填、不产生连接。
不会生成`[R,O,I]`逐行外积后再归约。零cotangent仍参与连接；未用根的poison不读取。
参数保存版本与输入保存版本由实际前向产生，独立reverse不读取可变原权重。

Head组合两次该投影与SiLU，完整保存preactivation/hidden；四个参数角色仍各返回偏导。
KV以正接收的实际owner直接调用base、right、left、summary四个投影，删除
“先复制成R份bank→每行投影→逐行参数外积→再归约原bank”的中间层。
低秩公式`B N(u)+U(V N(u))`、独立summary根和raw更新未变；三次输入RMS尚未共享。

## 容量、寿命与边界

矩阵部分从O(R OI)改为O(B OI)，完整输入/输出tape仍为O(R(I+O))，GEMM可复用行scratch
只占O(T(I+O))。`weighted_bank_layout.h`提供构造前的shape/溢出检查和保守前向/反向字节界，
计入tile操作数、矩阵partial及可选bias累加向量；projection/head再核对实际新保留storage。
KV更新、Aggregate及region调用者同步采用bank字节界；residual及KV qry/Full外层保守预算的矩阵项
也按实际bank计费，不能继续按已删除的逐行矩阵复制拒绝容量。
vendor workspace仍由DeviceProgram独立限制，不混入上述张量字节界。

令实际非空bank的行数为`n_b`，实际行总数`A=sum(n_b)`，非空bank数G，
数值填充行总数`P=T sum(ceil(n_b/T))`。有`A <= P <= A+G*(T-1)`，非空时`P < A+G*T`；
全空时P=0。本候选不保留旧容量桶的`P<2A`保证，小组可能承担明显填充成本；实际工作量
和性能对照须分别记录A、G、T及tile数，不能以静态body下降推断运行吞吐。
静态数值body为O(1)，其运行次数为`sum(ceil(n_b/T))`；没有主机读取实际group/count。
6011d599为每bank登记一套容量body，形成O(B log R)静态控制图；其完整消息块在CANN录入阶段
触发EE1019/507002（stream task buffer full），即使各局部组件设备数值检查通过。
1f0e6e43复用bank循环后仍为每个投影/伴随登记O(log R)容量body，完整消息块仍录入失败。
固定tile移除该容量树；新组合设备资格与规模任务资源仍需单独验收，不能以字节预算证明通过。
多个角色/区域、完整前反向的子program组合及共享KV历史是独立边界，本模块不改变它们。
ragged行复制由一个设备worker负责，避免奇数行宽跨worker共享cache line；参数bank快照按连续对齐块
分给worker。异步读者、互斥分支、共享scratch和保存反向均由DeviceProgram寿命覆盖。

调用者只能在统一error成功后发布结果；非法owner、非有限、容量不足保持拒绝，运行失败不换算法。
分组矩阵归约次序可能改变有限舍入，原严格阈值不放宽；同路径精确恢复仍须独立验收。

## 独立检查与资格范围

- `tests/test_resident_weighted_bank_gemm.py`编译实际计划/复制/快照语义核的CPU适配，配CPU矩阵算子，
  与逐行FP64公式/autograd比较投影、两层head及KV低秩/summary转置；覆盖真实共享叶贡献、
  相等值不同bank、独立根/零根/缺席poison、动态稳定分组、全空组、保存版本和容量拒绝。
  游标覆盖重复重放、跳过空bank、稳定多tile、奇数尾部、终止、非法游标/offset及sticky error；
  大于64行的独立公式检查累加dW/db、独立根与保存版本；它不模拟CANN任务池容量。
- `tide-weighted-bank-layout-check`是注册的CPU CTest，独立核对形状/溢出拒绝、tile填充界、
  前后向张量形状inventory及64行×3bank×32宽在128KiB张量预算内的结构界；
  64行以上scratch不增长，控制存储不随R/B增长。它不执行候选设备程序。
- `weighted_projection_check.cpp`在实际设备执行单/多bank、FP32/FP16、RMS/bias、完整VJP、
  动态重放、独立保存版本反向；保留64行×3bank×32宽的实际storage与逐行独立公式见证，
  新增同128KiB预算的193行、有bias见证，65/64/64分组覆盖跨tile dW/db和奇数尾部。
  新增用例的源码存在不等于新设备资格通过。
  原`weighted_head_check`、`weighted_kv_update_check`实际调用同一生产边界。
- 容量见证以实际FP32输入的独立逐行FP64公式核对，不把逐行FP32累加当作精确oracle。
  41c42759原NPU检查在193行的db比较失败；同材料CPU调查发现64行bank的旧FP32参考
  自身对精确和超过原容限。原失败保留，新检查仍用`atol=1e-6,rtol=1e-5`，分别报告
  candidate/旧参考对FP64的误差及精确首差。`test_resident_weighted_bank_oracle.py`用
  独立`math.fsum`核对原材料及取消相消贡献的反例；实际NPU误差仍须新资格确认。

CPU语义核/公式检查、C++语法、NPU/CUDA构建、真实设备数值、完整图训练/恢复与性能分别记录。
本契约和CPU检查不自动认证DeviceProgram的设备分支、CANN/设备队列、原预算FC16或其他规模成本。
