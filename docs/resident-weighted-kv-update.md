# 常驻 KV 接收更新与完整参数链

固定 [KV §1.1–1.3](upstream/tide-core-5/local-semantics/weighted-message/kv-memory-learning-note.md)
对应 `weighted_kv_update.h::append_weighted_kv_update`。本模块只实现 Upd；
Mem、Describe、Full 和图推进使用各自模块，不由此资格替代。

输入为 [raw 状态](resident-weighted-state.md)、FP32 content[R,d+1]=(r,u)、
int64 time/owners[R]、bool present[R] 及四个参数 bank：base[B,2d,d]、
left[B,2d,r_a]、right[B,r_a,d]、summary[B,d_s,d]。同组固定 d、d_s、r_a、epsilon、window 和缓存容量。
参数角色可经[canonical归约](resident-weighted-parameters.md)绑定到实际共享 owner；
节点状态仍私有。不同节点的 base 可以共享，适配因子和摘要不必共享。

raw旧态与结果可显式为FP32或FP16；content、四个参数bank、算术、保存值和所有伴随仍为FP32。
FP16旧态先精确拓宽，完成公式后仅在最终raw存储舍入；Cast声明一阶identity VJP。
`append_weighted_raw_half_store`检查舍入后的所有实际定长字段/有效ragged前缀，忽略未用padding，
保留精确int64元数据。有限FP32生成值若溢出half则以error24拒绝，不能提交live owner。
FP32正接收在存储时可成为零权重，但仍是已发生的接收：保留新条目、时间戳和计数及质量根。
这与r<=0的原样保留分支不同；没有添加人为裁剪。额外拓宽/存储/校验缓冲纳入组合预算。

设备依据实际 r>0 决定是否求投影。缺席行、零或负接收不读取 u 或参数 owner，
并保留完整旧 tuple；旧 tuple 本身仍必须有效。正接收计算
base N_d(u)+left(right N_d(u)) 与 summary N_d(u)，用 vendor BMM/RMS/index 操作完成数值，
再经 raw 更新按原始 int64 时间过滤、追加未加权 K/V/p 和 min(1,r)。
base、right、summary分别执行RMS投影，left接未归一化的低秩中间值；尚未共享这三次归一化计算。
不存在主机读取实际轨迹、逐行接收分支或参考递归解码。

VJP 先恢复 raw 独立字段根，再回传 K/V、summary 两个投影；低秩梯度使用保存的实际
left/right，按真实 bank 归约所有使用者。参数源或原内容后续改变不影响保存版本反向。
只使用 value/weights 根不会把 KV/summary 参数连接成零梯度；只使用 summary 根也不会
连接 base/left/right。content 是一个完整 (r,u) 根，任一分量连接就返回该根；饱和 r 的
导数为连接零。旧空/全被过滤的张量保持其 indexing 路径的连接零身份。
精确 r=1 延用 raw 层声明的 clamp_max 导数1；不提供高阶导数或硬函数代理。

前向/反向有独立保守 scratch 准入，FP32 数值/元数据错误置共享 error；只有统一检查
通过才执行 guarded raw adopt。缓存容量限制同时保存条目，不限制累计逻辑时间。
不得把中间结果当作失败事务后的有效 checkpoint。

| 定义与边界 | 独立证据入口 |
| --- | --- |
| 正接收、缺席/零接收、独立字段根 | `test_resident_weighted_kv_update.py`：分支、poison、None/零、合并溢出 |
| 完整低秩/摘要/私有内容梯度 | `weighted_kv_update_check.cpp`：每raw dtype在实际量化旧态上用独立逐行FP32公式/autograd，五个独立输出根及联合根；half结果仅在最终存储舍入 |
| 原始窗口、完整 tuple、强弱/饱和接收、保存版本 | 同 check：动态 bank、旧缓存清理、过期缓存保留、>2^54/MAX int64、失败不提交 |
| raw 子层与 RMS 导数 | `test_resident_weighted_raw_update.py` 与 `test_resident_weighted_projection_norm.py` |
| 半精度存储域 | 同standalone check：padding poison、正接收权重下溢仍追加条目/计数、独立质量VJP及生成字段溢出不提交 |

构建/运行入口 `scripts/{build,verify}_device_control.py --checks weighted-kv-update`。
当前数值域为FP32，raw存储为FP32/FP16。CPU helper、NPU 实际执行、CUDA 构建与 GPU 真机分别记录。

固定源码资格见[681a7e9运行证据](evidence/core5-resident-kv-update-20261027.md)。
FP16 raw更新已有[69bf4f0固定资格](evidence/core5-resident-half-kv-update-20261028.md)；完整half图与训练由[公共训练owner](resident-weighted-training.md)独立验收。

## 保留低秩因子的设备求值

前向使用`B N(u) + U(V N(u))`，分别保存base、V、U的真实projection tape，
不再先形成稠密`U V`。反向分别计算base、U、V贡献及两个归一化输入贡献后汇合；
共享参数仍按所有实际行归约，独立K/V/summary与None根、原始时间、half最终store守卫不变。
适配器前向及VJP中原有O(d²r)的稠密组合工作变为O(dr)，base的O(d²)仍在；
更多小projection提交也计入成本，因此不从乘法量减少直接推断加速。
现有保守预算继续生效，不凭估算收益扩大准入；实际retained buffers、完整窗口/训练与
checkpoint、单/双设备和两dtype由对应固定源码资格验收，旧报告不认证此增量。
