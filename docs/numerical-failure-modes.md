# 数值失效模式与回归用例

新周期选定接收、Attention、记忆和区域分配的数学定义后，可以从下面相应的小输入开始实现。每个输入都给出旧实现首先算错的量，以及可以独立判断的值、支持集或一阶梯度。本版取自 2026-11-17 至 11-23 的 FC/HB 实例，数学快照为 `4f5b9f20`，最后一项实现修复为 `6e14a6cc`。具体函数与设备资格随版本定位。[采用的绑定](semantics.md) · [固定版本的数学核对指南](../math/reference/weighted-message-implementation-checks.md)

## 运行已有用例

在基座根目录的项目测试环境中运行下列命令。清单每行是一个 pytest 文件或用例，可取出相应行单独运行。输入、独立参考和断言都在基座内。文中的 r07、r09 等编号指下游实验轮次，验收链接指向同级的 `tide-learning-lab` 工作树。[回归清单](../tests/numerical-regressions.txt) · [测试环境](development.md)

```sh
TIDE_STABLE_DEVICE=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 xargs python -m pytest -q < tests/numerical-regressions.txt
```

2026-11-23 在实现 `6e14a6cc`、CPU Torch `2.12.0+cpu`、pytest `9.2.0` 上运行这份清单，319 项通过，3 项命中已登记的 strict xfail。仅修改 quad 时，可单跑 `python -m pytest -q tests/test_quad_total.py`。

CPU 可算解析参考、重建原始浮点输入和检查后端包装。验证 Ascend 内核时，可在已分配的设备上通过 `TIDE_STABLE_DEVICE` 选择逻辑卡，运行 `python -m pytest -s -q tests/test_attention_native_support.py tests/test_binary_parts.py`。前者包含公开 `auto` 与显式融合路径，后者对照标量二进制算术。另查实际分派、设备日志与主机回退。[精度与后端](precision.md) · [设备执行入口](execution-flows.md)

## 加权接收与 Attention

### 加权均值的商与反向

单边质量 `m=4.967183787530951e-38`、内容 `v=17.48714638`、输出余切 `g=0` 时，旧反向先算 `v/m` 就溢出。单边均值应精确等于 `v`，质量梯度为零，内容梯度为 `g`。多边在 `M=sum(m)>0` 时用 `dm_i=dot(v_i-u,g)/M` 核对。修复按行缩放质量、中心化内容，把二进制尺度保留到公开梯度形成，并先归约同一广播输入的贡献。另测全零行，当前分母取 1，`u=0`、`dv=0`，真实边仍有 `dm_i=dot(v_i,g)`。这是现行实例对旧商溢出的局部重建，修复始于 `70f4aa30`。[实现](../src/tide/weighted_mean.py) · [用例](../tests/test_stable_reception.py) · [CPU/Ascend 资格与保留偏差](evidence/stable-mass-20261119.md)

### KV Full 的质量振幅

输出是归一化读出乘 `a=min(M,1)`。在 `weak_fixture(1e-30, 1e-30)` 中，分开反传会使 `a*g` 提前归零，两键的质量梯度随之相同。由同一输入直接算复合函数，预期梯度为 `[64e-30,192e-30]`。一起求振幅和先验的导数，并对照 `M<1`、`M=1`、`M>1`。本实例 KV 在等于 1 时保留振幅导数，Pool 的总量项在相同端点取常数分支。`70f4aa30` 的局部用例含 CPU FP64 参考。[复合公式](stable-mass.md) · [KV 用例](../tests/test_mass_scaled_attention.py) · [Pool 端点用例](../tests/test_allocation_range.py)

### 普通 Attention 的弱先验

两键分数为零，质量均为 `1e-40`，值为 `[1,1+2^-20]`。输出为 `1+2^-21`，单位余切下两质量梯度约为 `[-2.3842e33,+2.3842e33]`。旧 math/tiled 路径在组合前溢出。`b4a8d937` 让弱正先验走稳定求值，指数项与余切组合后再存储。切换执行路径时保留每个先验原值。局部对照含真实独立零先验、共享输入、重复反向，以及真正超出根张量范围的导数。[实现](../src/tide/attention.py) · [用例](../tests/test_ordinary_attention_range.py) · [原 CPU/Ascend 对照](evidence/positive-chain-20261119.md)

### 空行、单个有效键与多个近同值的键

单键归一化读出直接等于 V，Q、K 和该正先验的梯度应为连接到原输入的精确零。空行按本实例返回零。五槽中只留一键也应相同。旧 tiled 和 NPU native/auto 都出现过假梯度，NPU 探针的 Q/K 残差最高 `6.72e-9`，零初始矩 Adam 首步改变量最高 `4.02e-4`。`885f7ef9` 合入空行和单键解析反向、混合行余切分流及 NPU FP32 多键的公共 V 原点。单键的未饱和质量振幅仍求导。对多键分别取 V 完全相同、仅相差 `1e-5`、两键分数相差 20，按 CPU FP64 商式比较梯度和首次更新，保留真实微小梯度。[单键与 Adam 用例](../tests/test_attention_singleton.py) · [混合行与多键用例](../tests/test_attention_native_support.py) · [原生内核反例](attention-singleton-repair.md) · [修复后验收](attention-native-support-repair.md)

## 区域分配与正支持

### 可表示的弱份额

Pool 取相同分数和质量 `[1e-40,1e-40]`，输出等于两个输入质量，余切 `[1,0]` 应给质量梯度 `[1,0]`。旧反向给出无穷。Shannon 取 63 个分数 2、一个 −98、单位接收和上界、预算 32，弱发送量 `32*exp(-100)/(63+exp(-100))` 约为 `1.89e-44`。旧实现先把单独的概率舍入为零。`b4a8d937` 修复 Pool 复合反向、Shannon 弱量重建及斜率加权的耦合梯度。等弱斜率、余切 `[0.25,0]` 时，接收梯度还可达到 `[0.125,0.125]`。这些输入是局部算术反例。特征的 RMS 归一化仍允许可训练读出和偏置产生大分差，各图可能出现的质量范围见资格记录。[分配实现](../src/tide/allocation.py) · [用例](../tests/test_allocation_range.py) · [公式与适用条件](positive-chain.md)

### Shannon 精确端点与可表示的剩余质量

R2 的 FP32 上界 `0.2345176637172699` 曾在重建时少 1 ULP，随后被判成内部解。`049ecdcb` 在命中时保留原上界。用单候选分数 1、接收 0.1、上界和预算 1 核对，输出精确为 1，上界梯度为 1、分数梯度为 0。另一真实故障发生在 603 次更新后，分数 `[6.040277004241943,-10.60705280303955]`、零下界、单位上界与总量应给弱项 `5.89055887e-8`。旧算法从总量中减去已舍入的强项，结果为零。`885f7ef9` 收录的活动约束求解保留剩余总量的尾数及舍入前的上下界归属。80 位 Decimal 独立求解并检查耦合的分数梯度，原故障批次有 CPU、Ascend FP32 记录。[精确端点用例](../tests/test_shannon_cap.py) · [弱项与约束用例](../tests/test_shannon_active_set.py) · [R2 原行](../../tide-learning-lab/docs/evidence/pilot-r2-shannon-repair.json) · [603 步故障](shannon-active-set.md)

### Shannon 真正的存储下溢

r09 分数 `[-73.42066192626953,32.303375244140625]`、单位上界与总量，闭式解的弱项约为 `1.21516e-46`，低于 FP32 最小正次正规数的一半。默认 `strict` 报告失去正支持。显式 `positive-floor-v1` 存 `[2^-149,1]`，按补前自由面和斜率反传。给弱项余切 `1e30`，两分数梯度约为 `[+1.21516e-16,-1.21516e-16]`。逐行核对角色上下界、补入总量和补值计数。两个不同的正尾都存成最小单位后，下游在相同分数下会读到相同权重。`885f7ef9` 的真实三批在 CPU 与 Ascend FP32 各补 23、56、111 个坐标。[实现](../src/tide/shannon_support.py) · [用例](../tests/test_shannon_positive_floor.py) · [策略](shannon-positive-floor.md) · [r09 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r09-foundation-shannon-acceptance.json)

### Pool 真正的存储下溢与极小预算

r16 第 3775 步两个单位先验的分数为 `[-80.67035675048828,25.64478302001953]`，弱发送量约 `6.73e-47`。`b93eeae7` 为 Pool 单独实现可选补值，默认 `strict` 保留原异常。加入第三个真实零先验后，补值输出为 `[2^-149,1,0]`，计数为一个坐标。反向仍取补前归一化，真实零保留右导数。令 `δ=2^-149`，先验 `[2δ,2δ]`、分数 `[0,3]` 的输出应为 `[δ,3δ]`。极小预算按整数个 δ 守恒。原批次双端核过，连续更新及新进程恢复在 CPU 验收。[实现](../src/tide/pool_support.py) · [用例](../tests/test_pool_positive_floor.py) · [r16 验收](../../tide-learning-lab/docs/evidence/pilot-r3-r16-foundation-pool-acceptance.json)

### Quad 在总量残差为零时选错约束

9B FP32 推理的原行在 `[8,32,32]`、步幅 `(32,1,256)` 下出现断点前缀消减，总量恰为 1，节点 1 的正质量却丢失。独立三节点公式给出 `c₁=2^-23/3`，对三个自由分数的梯度为 `[2/3,-1/3,-1/3]`。将一个分数向上移动 1 ULP 后，该节点应精确退出支持集。测试保存了一行 32 个节点的分数和接收量，可从这行重建两种布局。有限断点实现 `c17c613f` 后，`97c52960` 增加最优性边界复核和保护求解。[原行与解析梯度](../tests/test_allocation_face.py) · [9B 原形状取证](evidence/cleanup-numerics-20261119.md)

### Quad 不等式总量的来源与舍入前的自由面

r24 第 921 步取 `d=5.7220458984375e-6`，两个普通节点，分数约 `[34.98,-205.20]`，接收 `[0.5-d/2,0.5+d/2]`，预算 1。控制量应为 `[1-d,d]`，第二项对接收的梯度为 `[-2,0]`。再取 `d=0`，第二项和该梯度均为零。旧路径丢失正余量并错发恒零标记。`76b1c014`、`6e14a6cc` 保留预算、接收与上下界比较的来源和求和尾数，用原始分数差复检多个自由节点。用例还覆盖真实内点存成上界或零、第三个求和分量改写约束、远处已饱和节点干扰两个内点的分差。[实现](../src/tide/quad.py) · [闭式回归](../tests/test_quad_total.py) · [故障、修复和批次采用](../../tide-learning-lab/docs/requests/pilot-r3-r24-quad.md) · [独立有理数投影与验收](../../tide-learning-lab/docs/evidence/pilot-r3-r24-foundation-quad-total-acceptance.json)

## 零值来源与跨窗状态

| 来源 | 旧实现的首错与可重建的判断 | 实现或验收 |
|---|---|---|
| Reception 结构补齐，r07 | 第 7354 步把两个空槽当作可变零先验，形式导数约 `−4.16e38`，后续零乘法产生 NaN。`885f7ef9` 按拓扑成员屏蔽。取分数全零、V 为 `[1,3,7]`、质量 `[1,0,0]`、最后一槽缺席，归一化商对三质量的导数精确为 `[0,2,0]`。 | [掩码与右导数用例](../tests/test_reception_padding.py) · [CPU 原始反向](../../tide-learning-lab/docs/pilot-r3-r07-vjp-investigation.md) · [双端修复核对](reception-padding-repair.md) |
| KV 未写入槽，r10 | `039ad7d7` 使零接收在整窗和分窗都没有写入导数，窗口按原始逻辑时间取键。第 1137 步又暴露了读端给未写槽计算先验导数的问题，`17c7d675` 在 Attention 前排除该槽。反例令未写键分数高约 181，另一键质量分别取 `0.5` 和 `1e-30`。输出只取已写值乘其质量，两个质量梯度应为 `[0,已写值]`，分窗同解。 | [KV 实现](../src/tide/kv.py) · [读端用例](../tests/test_mass_scaled_attention.py) · [窗口用例](../tests/test_kv_memory.py) · [r10 原 NPU 接续](../../tide-learning-lab/docs/evidence/pilot-r3-r10-foundation-kv-acceptance.json) |
| 生产端确定的零，r11、r21 | `75d2a893` 将普通 quad 的常数下界沿真实边传给 Reception。取分数 `[-3,5]`、接收均为 1、预算 1，控制量为 `[0,1]`，第一项在生产端局部恒零。把它与独立零先验放在同一次 Attention 中，后者仍可有非零右导数。`68cf6d40` 将判断延伸到单区域和批量区域的节点先验，要求每条真实入边都已确定恒零。一条可变入边就能改变节点导数。若上下界在零处重合且输出随接收量变化，仍保留对接收量的导数。 | [区域归约](../src/tide/region.py) · [对照用例](../tests/test_reception_padding.py) · [r11](../../tide-learning-lab/docs/evidence/pilot-r3-r11-foundation-quad-reception-acceptance.json) · [r21](../../tide-learning-lab/docs/evidence/pilot-r3-r21-foundation-regional-prior-acceptance.json) |
| Timed 静默时钟与分支汇合，r13、r14 | 第 2727 步的窗口内，节点 0 在逻辑时刻 2690 没有外部输入，时刻 2691 的接收端却把它送来的零当作独立输入求导。第 3252 步还出现多个有限梯度在汇合处溢出。`29c4798f` 按注入、flush 及前窗 pending 的来源移送恒零信息。`3150ae64` 在各消费者汇总前投影恒零质量，并沿全部真实入边传递。构造两个各回传 `2e38` 的消费者，静默发送方的梯度应为零。相邻活跃 tick 和独立 pending 各接收 `1+1=2`。把 pending 重新设为独立求导输入后，按新输入求右导数。 | [路由实现](../src/tide/model.py) · [时钟与汇合用例](../tests/test_reception_padding.py) · [r13](../../tide-learning-lab/docs/evidence/pilot-r3-r13-foundation-ingress-acceptance.json) · [r14](../../tide-learning-lab/docs/evidence/pilot-r3-r14-foundation-route-acceptance.json) |

r11、r13、r14、r21 的 CPU 捕获或独立参考与原 NPU 的连续更新、新进程恢复分别记录在对应验收中。独立零先验也可能有真正越界的右导数。取质量 `[1,0]`、分数 `[0,100]`、V 为 `[0,1]`，第二项的真实右导数是 `exp(100)`，FP32 应在 Attention 的先验梯度处报告越界。对于已确定的恒零输入，沿路由定位最先产生非有限值的位置，后面的 allocation 保护检查可能收到已污染的余切。[零值与时钟契约](semantics.md)

## 精度、布局与设备

- **指数恢复。** Ascend 的 `ldexp` 曾为有限的 `−3e38` 先生成 FP32 的 `2^128`。`8fd5216a` 分两次恢复尺度，只有最后一次进入次正规范围。`3efcc6b1` 替换在 NPU 回退 CPU 的 `frexp/copysign`。以 `fraction=-0.8816207647`、`exponent=128` 对照标量 `math.ldexp`，再测最小次正规、带符号的零、真正的溢出和非连续输入。设备归属另查运行日志和算子轨迹。[实现](../src/tide/scaled_arithmetic.py) · [用例](../tests/test_binary_parts.py) · [原始回退与修复资格](evidence/positive-chain-20261119.md)
- **BF16 模式与 bias 舍入。** `050dafa3` 明确矩阵、累加、私有状态和优化器各自的精度。`778c65fb` 统一 BF16 推理与启用梯度时的先验表示。在同一 Q/K/V 上显式给 `[0,0.25,0.5,1]`，两种模式的前向应一致，真实零先验的梯度仍存在。`231f56eb` 使区域批量投影的 bias 留在计算精度内。独立 Decimal 点积定位了特定 CPU 布局的两次舍入 `0.2119140625` 与融合一次舍入 `0.2109375`。各自按其已舍入输入检查支持和梯度。[模式用例](../tests/test_precision.py) · [bias 用例](../tests/test_region_batch.py) · [CPU/Ascend 首个舍入差异](evidence/capacity-rounding-20261118.md)
- **与数值路径相邻的执行回归。** `d7437c21` 修正了 NPU 融合内核在没有 PSE 张量时的调用模式。`28941994` 修正推理模式初始化的填充缓存进入训练后的反向错误。前者可用 CPU 替身检查参数，后者可用同一模型依次做推理、前向和反向，覆盖区域批量与重计算。[适配器用例](../tests/test_attention.py) · [缓存用例](../tests/test_reception_execution.py)

## 带入下一个周期

1. **实现算子时**，按采用的公式挑用例。先确定空行、单键、两个真键，独立零与结构零，精确端点及两侧相邻浮点数，可表示弱项与真正的存储下溢。支持集、约束归属及逻辑时钟按离散值判断。连续量和一阶梯度用解析式、单侧导数、独立 Decimal 或有理数解，按条目原容差比较。
2. **接入快速实现时**，对同一份已舍入输入比较参考路径、默认分派和目标内核，保留原形状与步幅。测共享输入、混合支持行、真正的小梯度和零初始矩的首次 AdamW 更新。记录每条路径实际执行的设备与精度，以及哪些差异始于上游舍入。
3. **改到状态、路由或存储策略时**，用故障批次和真实后继输入检查优化器、梯度、私有状态、数据游标和 RNG，再在新进程载入一次并执行后续更新。对相同负载单独测整步代价及异常行占比。r24 在 CPU 与 Ascend 各自的轨迹上核了第 921 步的 40,960 行，分别载入第 922 步并复算第 923 步。该负载四组配对耗时的中位变化为 CPU +0.94%、Ascend +0.45%。[r24 独立参考实现](../../tide-learning-lab/scripts/r24_quad_reference.py) · [故障批次验收程序](../../tide-learning-lab/scripts/accept_r24_quad.py) · [恢复与计时记录](../../tide-learning-lab/docs/evidence/pilot-r3-r24-foundation-quad-total-acceptance.json)

基座还有一个独立于旧训练数据的 CPU 恢复用例。它在新 Python 进程中载入小模型，核对已保存字段和下一次前向，并执行优化器更新。[恢复判据与诊断入口](training-path-checks.md)

```sh
TIDE_STABLE_DEVICE=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_training_paths.py::test_fresh_process_restores_the_same_frozen_model_and_really_updates
```

### 新条目记录格式

| 字段 | 填写内容 |
|---|---|
| 适用条件 | 固定的语义版本、算子与绑定。新绑定改变了零边、端点或存储约定时重算预期值。 |
| 触发输入 | 写出值或可重建的生成方式，标明真实故障或构造输入，必要时给形状、步幅、时钟和前态。 |
| 首个错误与预期 | 写出旧实现第一次算错的位置及数值，再给正确的值、支持集和对指定输入的一阶梯度。 |
| 独立参考与回归 | 给出闭式解、单侧导数、Decimal 或有理数算法，注明容差和可直接运行的测试。 |
| 已验收范围 | 给出修复提交、后端、实际精度和证据入口。记录局部与整步的结果、有限超差及恢复范围。 |

字段沿用固定数学快照中从定义定位到代码、测试与运行证据的约定。[语义与证据的对应](../math/reference/weighted-message-implementation-checks.md)

### 已知偏差与待验证边界

**保留的有限超差。** 早期 Shannon 恢复、CPU 以 FP64 对照的私有 KV 状态，以及弱质量样本中一个很小的 V 梯度，均有超出原容差的记录。弱质量用例的 seed 为 773，超差坐标为 `dV[1,0,4,26]`，原判据为 `atol=1e-35, rtol=1e-5`。math、tiled、auto 各自保留一个 strict xfail。[弱质量记录](evidence/stable-mass-20261119.md) · [恢复与私有状态记录](attention-native-support-repair.md)

**待验证的边界。** RMS 平方在极端输入下的溢出、SSD 极端参数下的扫描乘积，以及共享参数在后续汇合时的消减或范围问题，可在选用相应机制时构造独立见证。NPU BF16 多键的精度与 CUDA 硬件路径仍需各自的实测依据。[算术范围](positive-chain.md) · [设备与精度范围](precision.md)

### 真正超出 FP32 的质量梯度与恒零边

在接收均值中，路由已证明恒为零的真实边只对该质量的可达切向量贡献零。其余真实零质量仍保留单侧导数。例如总质量为 `2^-148` 时，先为五条恒零边计算独立的质量导数会得到约 `−9.4e38`。先将恒零证明与结构掩码合并，再形成均值的质量余切。极小的正质量仍参与归一化。[接收回归](../tests/test_reception_padding.py)

正先验的真实导数也可能超出本地范围。一次接收自注意力的质量梯度为 `1.646e39`，一次 KV 已写权重的多次读端求和为 `4.251e38`。稳定的代数不能把这两个独立导数装进 FP32。公开算子在最终 VJP 的可表示性检查抛出 `GradientRangeError`。训练端可从相同批次、状态和 RNG，以二次幂缩小损失余切；真实参数梯度还原后才进行有限性检查、裁剪和更新。梯度根过大或缩放导致的小梯度下溢仍需单独检查。[注意力实现](../src/tide/attention.py) · [训练封装](../../tide-learning-lab/pilotlab/worker3.py)
