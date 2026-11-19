# 当前交接：FFN执行与CPU/NPU有限计划已完成

实现固定`05e8338b`，工具补充至`46317a8d`；本页及证据另行提交推送。
生产`src/tide` tree为`89f2707e71d989340c12de310797a4c7f4c99f64`，默认gather/rectangular不变。
数学`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`的108文件原字节核验通过。
完整结果、形状、阶段、资源与复现见[本轮证据](evidence/ffn-execution-devices-20261119.md)及其JSON。

实现了FFN直接权重bank访问W、有效行power2 packing P及组合WP，保持声明公式与参数归属。
CPU全套979 passed/1 strict xfailed/1 warning；工具19项检查通过，未合并为同一次资格。
小图CPU W/P阈内，WP同一参数字段两次比较超差；小图NPU P/WP各11次字段比较超差。
d768四项原公式输出超差仍在；补充FP64/同NPU路径诊断不追认原资格，不扩新候选到9B。
中尺度NPU baseline跨模式14字段超差，20GiB制品预算在恢复包前中止，W未开始。
中尺度CPU baseline/W各124跨模式字段超差，W后两更新392/533次字段比较失败；两者各自新进程恢复精确。
P初始推理/VJP有55/62字段超差，首更新计算后的载荷触及32GiB预算，未完成持久更新记录/恢复。
原支持、int64与有限性已保存比较通过，不消除值/VJP/更新超差；原阈值不变。
独立NPU恢复复用已有step4/position512 checkpoint，明确新空梯度/seed2718参考，
425个参数张量实际step4→5更新阈内非exact；saved data/RNG/next forward精确，不假装恢复旧未保存梯度/RNG。

中尺度B16/L128/chunk32三配对正常训练：NPU Tide883.923、TF4501.196位置/s，配对差5.034倍；
CPU32线程为93.050/239.461，配对差2.573倍。NPU W/P/WP配对吞吐比0.991/0.854/0.813，CPU W/P为0.838/0.862。
NPU normal allocated峰值22.739→21.517/20.598/19.273GiB；省显存没有变成同batch加速。
9B图仅比较既有KV128(7.268409B)与共同TF32(8.778371B)，不替代严格A–D。
同B8/prompt128/chunk32/decode16：NPU stable decode18.449/137.114位置/s，CPU3.843/6.506；
对应配对差7.435/1.686倍，CPU绝对吞吐下降。首主干输出不是完整语言模型TTFT。
同B8持续decode256为18.436/137.226位置/s，配对差7.443倍；未观察到长请求拉近差距。

正常计时、同步阶段、逐算子及独立ledger分列。P的KV FFN行减少约51%，调用28→131；
分配fallback请求324/执行49152行。大图NPU decode同步FFN169.700ms、KV历史49.669ms；
独立算子趟FFN bank gather150.055ms。不能把这些时间当正常wall占比或可回收加速。
CPU正常medium在NUMA6、large在NUMA0，诊断在NUMA3；相同初参数/输入不保证相同轨迹/后端。
CPU NUMA间差异未解释，不能给“非结构化X%＋复杂运算Y%”的可加因果比例。

【RESOURCE_STATUS】
四份用户未跟踪文档原字节保留、不提交；参考仓库、公共工具链及下游未修改。
本轮无9B训练、CUDA、多NPU、BF16新资格或模型质量结论；五类记忆绑定既有证据如下。
下一步机会在[ROADMAP](ROADMAP.md)，不自动续跑新矩阵。

---

