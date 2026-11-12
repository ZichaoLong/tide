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
每bank int32分支号以及行/bank bool连接。划分是O(R+B)，不读取主机实际控制/就绪或参考轨迹。
参数快照保持`[B,O,I]`及可选`[B,O]`，未使用bank置零且不读取其非有限源值。
输入和保存归一化值仍是`[R,I]`，不是每行携带一份矩阵。

构造时列出有限行容量`1,2,4,…,R`，最后一个容量为实际R；运行时每个非空bank根据设备count
选择最小容纳桶。每个多bank组的数值行数小于两倍实际行数，空组跳过；单bank继续使用一整块GEMM，
其缺席行由mask清零，不套用上述填充比例。组间复用同一scratch，支路间也复用存储。
主要数值计算使用vendor GEMM/BMM、permute、bias add或sum，设备核只承担计划、复制、有限性和连接。

每组前向为`Y_b=X_b W_b^T+b_b`。独立根先排除不存在行，转置为
`dX_b=G_b W_b`、`dW_b=G_b^T X_b`、`db_b=sum_rows(G_b)`，实际行scatter回原顺序。
不会生成`[R,O,I]`逐行外积后再归约。零cotangent仍参与连接；未用根的poison不读取。
参数保存版本与输入保存版本由实际前向产生，独立reverse不读取可变原权重。

Head组合两次该投影与SiLU，完整保存preactivation/hidden；四个参数角色仍各返回偏导。
KV以正接收的实际owner直接调用base、right、left、summary四个投影，删除
“先复制成R份bank→每行投影→逐行参数外积→再归约原bank”的中间层。
低秩公式`B N(u)+U(V N(u))`、独立summary根和raw更新未变；三次输入RMS尚未共享。

## 容量、寿命与边界

矩阵部分从O(R OI)改为O(B OI)，行工作区为O(R(I+O))。`weighted_bank_layout.h`提供
构造前的shape/溢出检查和保守前向/反向字节界；projection/head再核对实际新保留storage。
KV更新、Aggregate及region调用者同步采用bank字节界；residual及KV qry/Full外层保守预算的矩阵项
也按实际bank计费，不能继续按已删除的逐行矩阵复制拒绝容量。
vendor workspace仍由DeviceProgram独立限制，不混入上述张量字节界。

动态组的所有支路在构造时预先登记，实际分支由设备选择；静态控制图有O(B log R)个备选组，
一次执行每个非空bank只运行一个。这不是host捕获实际路线，也不宣称静态图体积或设备提交成本最优。
ragged行复制由一个设备worker负责，避免奇数行宽跨worker共享cache line；参数bank快照按连续对齐块
分给worker。异步读者、互斥分支、共享scratch和保存反向均由DeviceProgram寿命覆盖。

调用者只能在统一error成功后发布结果；非法owner、非有限、容量不足保持拒绝，运行失败不换算法。
分组矩阵归约次序可能改变有限舍入，原严格阈值不放宽；同路径精确恢复仍须独立验收。

## 独立检查与资格范围

- `tests/test_resident_weighted_bank_gemm.py`编译实际计划/复制/快照语义核的CPU适配，配CPU矩阵算子，
  与逐行FP64公式/autograd比较投影、两层head及KV低秩/summary转置；覆盖真实共享叶贡献、
  相等值不同bank、独立根/零根/缺席poison、动态稳定分组、全空组、保存版本和容量拒绝。
- `tide-weighted-bank-layout-check`是注册的CPU CTest，独立核对形状/溢出拒绝、桶界、
  64行×3bank×32宽在128KiB张量预算内的结构界；它不执行候选设备程序。
- `weighted_projection_check.cpp`在实际设备执行单/多bank、FP32/FP16、RMS/bias、完整VJP、
  动态重放、独立保存版本反向；新增64行×3bank×32宽的实际storage与逐行独立公式见证。
  原`weighted_head_check`、`weighted_kv_update_check`实际调用同一生产边界。

CPU语义核/公式检查、C++语法、NPU/CUDA构建、真实设备数值、完整图训练/恢复与性能分别记录。
本契约和CPU检查不自动认证DeviceProgram的设备分支、CANN/设备队列、原预算FC16或其他规模成本。
