# 常驻带权摘要与 Attention 归约

固定 [KV §3.2、3.3、4.1](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
的局部归约由 `weighted_pool.h::append_weighted_pool` 实现。窗口/位置/查询投影由上层提供；
本组件不推断状态时间或充当完整 Mem/Full。sum 与 Attention 是不同函数，不宣称互相等价。

输入为 FP32 weights[R,C]、values[R,C,d]、可选 logits[R,C]、int64 lengths[R]、
bool present[R]。每行前 length 个槽是实际条目，独立于数值是否为零；缺席行完全不读其输入。
无 logits 时返回 sum(weights*values)/max(1,W)；有 logits 时仅正权重参与稳定 softmax，
返回 min(1,W)*Attention(logits+log(weights),values)。同时独立返回 W=sum(weights)。
全零权重的 Attention 向量恒零，无参数/输入路径；sum 的空张量运算保留连接零。

数值由 vendor sum、softmax、BMM、multiply 完成；设备内核负责ragged打包、支持/有限性、
独立根和标量边界。反向用 vendor BMM 计算 value梯度与各条目的向量内积。
Attention logits导数先对value减去一个实际条目及其加权均值，保持共同平移不改变logits梯度。
这与现有 `weighted::attention_mean` 声明的一阶函数对应，不借用HST。

`append_weighted_pool_vjp` 接向量/总量各自bool根，返回 weights/values/logits各自连接标记。
sum无logits参数角色；统计量根单独连接weights。Attention非空时，零权重条目也通过总量
衰减系数保留相应的weights导数，不能因它未进入softmax而删掉该路径。
max/min在W=1处分别使用既有clamp的导数1；全零Attention为显式常值分支。
所有actual数值及反向来源保存，修改原buffer不改变tape；只声明一阶。

浮点softmax可能下溢为零，但原正权重支持保存在weights/meta，不用数值probability推断
条目存在性或梯度连接。不添加裁剪。时间衰减及其数值政策由调用者声明，raw时间戳不在此层转换。
无效实际长度/负权重/非有限前向为error29，实际反向非有限为error30；scratch预算单独检查。
错误结果不可提交。窗口外或缺席padding允许poison；sum的实际value必须有限，Attention只读取正权重value/logit。

| 适用条目 | 独立验证 |
| --- | --- |
| 完整向量/总量与所有一阶根 | `test_resident_weighted_pool.py`：逐行FP64公式/autograd（clamp边界使用独立FP32求和）、平滑联合方向导数 |
| 空/零记忆、弱/单位/饱和W、None/零、共同行平移 | 同CPU测试；不将数值零当连接标记 |
| 设备vendor数值及保存版本 | `weighted_pool_check.cpp`：sum/Attention、3/129维、144行重放，独立逐行公式及全部VJP |

入口 `scripts/{build,verify}_device_control.py --checks weighted-pool`。当前为FP32局部组件；
NPU实测、CUDA构建、GPU真机及完整图分别记录。

固定源码资格见[afe7b5e运行证据](evidence/core5-resident-pool-20261027.md)。
