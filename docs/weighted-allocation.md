# 带权分配：正温度精确解

来源为固定 [区域分配 §4.1、§6、附录 S.2](upstream/tide-core-5/local-semantics/weighted-message/region-allocation-learning-note.md)，
提交 `0770593`。公共入口 [allocate_quad](../python/tidegraph/weighted/allocation.py)
接受一份完整候选的评分 `scores: [N]`、上限 `caps: [N]`、标量总量下界/上界及正温度。
输入为同设备 FP64/FP32；要求有限、`0≤caps≤1`、`0≤lower≤upper`、`lower≤sum(caps)`。
返回实际控制、乘子、分支、内部响应掩码与可行性/响应残差。无持久状态。

前向排序 `s−τμ,s` 的断点，扫描分段仿射总和，直接求目标总量对应的乘子；
复杂度 O(N log N)，存储 O(N)。无固定迭代近似、稀疏裁剪或硬 Top-k。
评分先减共同最大值，最后以实际响应重建内部段，减小累计舍入。
残差超过 `64*N*eps(dtype)` 明确失败；该界是浮点求值容差，不改变数学活动集合。
`c>0` 才是此函数的正支持。独立完整区域打包见 [区域批量契约](weighted-region-batch.md)，资格单列。

一阶反向身份为 `quad-exact-map-v1`：在实际最优响应点按 §6.2 隐式求导，
对评分、上限、实际起作用的总量及温度返回梯度，再经其依赖进行链式汇总。
它不是有限求解迭代的导数，也不是旧 HST。当前不声明二阶反向。

不可微边界约定：零端点取零侧，正上限端点取饱和侧；总和松紧交界取松弛侧，
但上下界重合时始终取等式分支。无内部响应时不除零，保持严格饱和分支。
全零单点分支返回无梯度根的零，全上限单点分支返回原上限（评分/温度无连接）。
其他分支的完整分配映射保留输入根，数值零梯度与无连接分开。
相等上下界使用 lower 作为同一独立等式参数；调用者应传入相同的总量张量。

[独立测试](../tests/test_weighted_allocation.py)枚举小多面体的所有面并比较目标最大值，
不复用排序求解器；覆盖教材 `(0.95,0.45,0)`、相同前向但不同总量/阈值导数、
温度响应点、评分平移、共享输入链式梯度、方向差分、平票、零/饱和与域拒绝。
独立求解检验公式，方向差分仅认证所选光滑分支；此增量不单独认证完整模型训练。

C++ 入口为 `tide::weighted::allocate_quad`：独立使用附录 S.2 的逆响应区间，
80 次有界二分后按实际内部响应作仿射校正，并核对同一残差界；其反向仍为精确映射，
不对这 80 次迭代求导。Python/native 均与独立多面体面枚举对照。

## Bernoulli、Shannon 与固定形状 ent

`allocate(scores, caps, lower, upper, temperature, regularizer=..., ent_alpha=...)`
扩展同一个 Alloc 定义域与返回。regularizer=bern 使用 μ*h_B(c/μ)；ent 使用固定
alpha∈[1,2]，alpha=1 是 Shannon。alpha 必须显式给出，其他正则不接受 ent_alpha；
它不是可训练变量。`RegionalConfig` 把这些函数接入 regional_total/regional_threshold/constrained，形状进入图身份；
默认 quad 的既有图/checkpoint 身份保持不变。C++ 公共入口为 weighted::allocate。

Python 以逆响应点建立区间，使用受区间保护的 Newton/二分，最多 80 轮；C++ 独立使用
80 次二分，并从区间端点/中点选择残差较小的响应。求解先平移评分，保存实际响应输入，
反向不由 `scores - rounded_global_multiplier` 重建。乘子字段是原坐标中的浮点表示，
残差在实际居中响应及原总量边界上计算。两端单点可行集与 quad 相同。

Shannon 在对数域比较上限；ent 用 log1p/expm1 在 alpha→1 时稳定求值。
Bernoulli 的斜率用 sigmoid(x)*sigmoid(-x)，避免用 1-sigmoid(x) 抹去上端附近仍可表示的导数。
`free` 字段记录实际浮点控制严格介于上下限的坐标；Bern 的反向直接用响应斜率，
不把舍入到上限误认为数学饱和。没有近零截断或改变正则的后备算法。

如果所选函数在某坐标数学上为正、请求 dtype 的结果却下溢为零，执行明确失败并报告
positive support underflow；不能据此悄悄删除活动节点/消息。此数值限制可由调用者显式选择
更宽 dtype 或适当参数处理，不自动提升精度。原始数值下溢、数学零和人为裁剪由此区分。
数学上真实零的 quad/alpha>1 ent 坐标保持零；纯实值浮点导数的极小非零仍可能下溢，
此实现不承诺任意有限实数参数都可在给定浮点格式中精确表示。

quad 同样执行该守卫：非单点可行集上 `caps>0` 且响应分子为正，实际控制却为零时
拒绝；不能用同一次已下溢除法重算出的零残差豁免。Python/C++ 的scalar和batch均适用。
`test_weighted_quad_support.py` 用松弛分支独立闭式解 `[1, tiny/2]` 验证拒绝，另核对
最小可表示正数、真实零、零上限与全零单点；没有人为最小发送阈值或自动dtype提升。

VJP 身份分别为 bern-exact-map-v1、ent-exact-map-v1，固定 alpha 另由配置标识。
使用实际响应点的 a=∂g/∂x、e=∂g/∂μ；总和起作用时减去 a 加权的根均值并返回总量导数，
松弛时均值为零。温度导数使用 -a*x/tau 后作相同等式修正。零上限取零侧、单点分支和
quad/ent 的零/上限转折沿用前节约定。Bern 舍入到上限仍沿数学内部响应求导。
只声明一阶；有限迭代自动微分、可训练 alpha 和硬代理反向是其他身份。
所有依赖输入保存版本，前向后非法原地修改评分/总量等会在反向时被检测。

[独立熵检查](../tests/test_weighted_entropy_allocation.py)用 80 位 Decimal 标量求解，
不使用 Tensor 求解器或逆响应区间；覆盖松弛、上下界、等式、零上限、端点、形状接近 1、
精确映射 Jacobian 和方向差分。非单位上限下 Shannon/softmax 的可行情形及越界反例、
ent(alpha=2) 与 scores+tau/2 的 quad 对应分别比较；不把不同函数称为完整模型等价。
[native 检查](../tests/test_weighted_entropy_native.py)核对独立解与 VJP，完整区域描述→总量→分配
链路、独立根、三步 AdamW、pending/checkpoint，以及下溢和版本变更拒绝。
standalone `cpp/test/weighted_entropy.cpp` 使用等评分 Bern 的手算控制/Jacobian 与独立 softmax。
当前增量的开发/不可变运行状态见 STATUS；不据此提前认证其他发送规则或新设备路径。

## 显式共同评分平移与稳定 VJP

`allocate(..., score_shift=eta)`（quad 入口同样提供）定义对 `scores-eta` 的同一 Alloc。
eta 是同 dtype/device 的有限标量，作为独立输入保存版本。
regional_threshold/constrained 通过该入口传入共同阈值。总量约束起作用时，
整个可微分支的共同阈值导数解析为连接零；松弛时为各评分导数之和的负值。
如果 eta 同时影响 bounds/caps 等，其他真实路径仍照常回传；没有按梯度数值截断。
全零/全上限单点分支仍按原约定返回无关零/原上限。

约束分支的加权根均值先对 cotangent 减去一个公共分量，再求加权均值，最后还原
总量根。这与 §6.2 的原式代数相同，减少大公共 cotangent 的相消；任意常数根偏移
只改变总量导数。没有改正则、求根次数、前向值、容差或梯度身份。
[独立平移检查](../tests/test_weighted_allocation_shift.py)核对四种正则的上下界/等式/松弛、
精确阈值零方向、bound 的其他依赖、共同 cotangent 偏移和版本检查。

此修正来源于新 match 训练见证：下界活动时约 5e-10 的浮点伪阈值梯度被 AdamW
放大成约 1e-5 的参数差异。原失败与后续资格分别保留，不将旧失败重标为成功。

本轮审查修复的固定源码、独立反例与后端范围见[审查修复资格](evidence/core5-review-fixes-20261028.md)。

## 共同偏移下的实际响应

`2da3f2c`/`0503593` 对 `[10000,9999]`、caps=`[1,1]`、lower=upper=`0.7`、tau=`1`
的手算解 `[0.7,0]` 曾在 FP32/FP64 都残差拒绝，原失败见
[固定报告](evidence/core5-performance-transfer-20261028.md)。原因是从原坐标舍入乘子重建响应。

当前 scalar、batch 和 resident quad 保存/使用实际居中响应 `x=(scores-center)-lambda_center`；
残差与温度 VJP 都在这个点求值。公开 multiplier 仍是原坐标的近似浮点报告，不能据它重建
实际浮点响应。原输入继续保存版本；松弛分支使用实际 `scores-score_shift`，单点分支不变。
正支持守卫在除温度前检查 `caps>0 && x>0`，原可行性容差、边界约定、精确映射 VJP 身份不变。
[闭式反例](../tests/test_weighted_quad_offset.py)独立验收单个/多个内部响应、饱和/零坐标、
评分/上限/总量/温度/共同阈值根及版本；resident checker 直接使用手算解与梯度，
不借用 eager 求解器。这项修复不证明其他 D2048 或 D64 的严格差异都已解除。
[07e56ca固定资格](evidence/core5-private-projection-20261028.md)记录CPU、NPU实机和CUDA构建分别通过的范围，
并保存原大型材料的残差拒绝；不得把手算共同偏移例的修复扩展成任意分数范围的数值保证。
