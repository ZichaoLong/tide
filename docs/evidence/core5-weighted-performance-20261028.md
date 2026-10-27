# Core-5 有限完整调用性能对照

在本组HB/KV中规模完整训练中，CPU standalone prefill的中位窗口耗时约350毫秒，
streaming约456毫秒；NPU mixed-c约1.41–1.61秒，resident约2.60–2.71秒。
这两个有限尺寸适合优先选择CPU；数据没有支持NPU在本组调用边界下更快。
尺度增加缓解了固定开销的相对影响，尚未抵消它。该选择不外推大型模型、稀疏路由或多卡。

固定benchmark脚本/fixture源码 `ceae7bf3173cc3c85250ff6ed46a6470b77a6971`；
CPU standalone消费者构建于 `3516bef9fef8d1571af0082402ebef71d88d8c32`，其C++消费者源码在ceae7bf未改变；
NPU standalone消费者构建于ceae7bf，resident生产库分别为Python-owned672e948和standalone7db0b08。
采用上游 `0770593838a37c790a568226930567ed60f99b83`，2026-10-28审阅。
[JSON](core5-weighted-performance-20261028.json)保存全部24项、逐窗原始时间/工作量、输入/二进制/日志哈希、
设备映射、环境、争用及profile摘要。它不需要旧运行目录才能理解结论；复现入口见文末。

24配置中22项通过事后完整记录、loss和最终参数对照；CPU反馈PDG两项严格浮点失败保留。
CPU组持久状态为failed/exit1，两个NPU组均passed/exit0；各配置自身状态分别记录。
没有重测取最好值，也没有将入口smoke或profile耗时作为正式吞吐数据。

## 测量域与计时

每配置一个独立进程、一个warmup窗口和三个连续测量窗口；表中是三个窗口的中位数，
JSON同时保留全部样本、构造、warmup和完整进程时长。窗口间状态持续推进，SGD每窗更新后显式detach。
FP32，seed7，SGD lr=.001、momentum=.3、weight_decay=.01；loss为实际输出数值投影cG平方和乘.001。
候选先完成测量，独立CPU调度随后核对，参考轨迹从未作为候选输入。

| 规模 | 内容宽 | 样本数 | 每窗位置数 | HB逻辑跨度 |
| --- | ---: | ---: | ---: | ---: |
| small | 4 | 1 | 2 | 10 |
| medium | 32 | 4 | 4 | 20 |

HB有8节点/4区域并保留同秩区域；DAG/PDG有3节点/2区域、重复边和非对齐延迟；PDG另有正时延反馈。
KV使用sum读取与首个正式区域组合；SSD含固定时间短卷积。具体函数、初态和共享来自
[公共绑定](../weighted-message.md)、[循环](../weighted-recurrent.md)与静态fixture导出。

计时含输入克隆/上传、在线调度、packing、完整CPU返回、loss、backward、finite checks、
optimizer、detach和同步。owner/optimizer构造、warmup、fixture生成、事后对照和最终参数导出单列或排除。
这些是完整公共调用的成本，不能当作裸设备kernel吞吐。
CPU/Python-owned使用Torch2.12；NPU Python为CANN9.1，standalone为SDK Torch2.10/CANN9.0。
这是实际可选的软件栈比较，不能从其差值单独推断语言或接口的因果收益。

## CPU结果

除两项标为失败的PDG外，下表配置都通过严格完整对照。单位：毫秒/窗口。

| 工作负载 | 纯PyTorch prefill | Python调用native prefill | standalone streaming | standalone prefill |
| --- | ---: | ---: | ---: | ---: |
| KV/HB small训练 | 128.9 | — | 87.0 | 83.6 |
| KV/HB medium训练 | 541.1 | 406.6 | 456.5 | 350.5 |
| SSD/HB medium训练 | 576.6 | — | — | 350.7 |
| KV/非对齐DAG medium训练 | 617.9 | — | — | 356.9 |
| KV/HB medium推理 | 218.1 | — | — | 109.9 |
| KV/反馈PDG medium训练 | 不作Python性能扫描 | — | 严格失败 | 严格失败 |

HB/KV medium本次prefill较standalone streaming减少约23%的窗口时间；small仅约4%，
单次流程不支持稳定收益的统计结论。DAG的非对齐依赖缩小可联合前沿，实际计数见JSON。
完整Python返回和适配成本保留在各接口里。

## NPU结果

两组都在物理9→逻辑npu:0串行运行；下表九项均通过完整记录、loss与最终参数对照。
单位：毫秒/窗口。空格表示本轮未测量，不表示不支持。

| 工作负载 | 纯PyTorch mixed-c | native mixed-c | Python resident | standalone resident |
| --- | ---: | ---: | ---: | ---: |
| KV/HB small训练 | — | — | 1077.4 | 984.6 |
| KV/HB medium训练 | 1611.9 | 1413.7 | 2706.1 | 2599.6 |
| KV/HB medium推理 | — | — | 485.4 | 445.2 |
| SSD/HB medium训练 | — | — | — | 3668.1 |

在这组NPU训练配置里，mixed-c的完整调用成本低于resident；CPU仍更低。
常驻路径的作用是设备内在线推进及明确的显式VJP/容量/续接契约，是否值得使用需要具体负载的测量。
本轮不覆盖NPU多卡通信、FP16吞吐或CUDA真机。

## 实际工作量与profile

| 每测量窗 | 候选更新/活动Full | 消息 | 输出 | pending |
| --- | ---: | ---: | ---: | ---: |
| HB small | 16/16 | 24 | 4 | 0 |
| HB medium | 128/128 | 192 | 32 | 0 |
| 非对齐DAG medium | 112/112 | 96 | 96 | 4 |
| 反馈PDG medium（严格失败） | 240/240 | 400 | 160 | 32 |

这些fixture全部候选活动，不能据此评价稀疏发送节省。
resident HB small有4个实际设备stage、32条物理local/Full行；medium为6个stage、192行。
逻辑Full数与物理计算行数分列，不能把32/192误读成额外语义作用，也不能只用16/128估计实际成本。
Python-owned KV medium训练设备分配器峰值约491MiB，进程生命周期RSS约2.54GiB；
mixed-c约37MiB/2.29GiB。allocator reserved、静态保留字节和operator workspace分别在JSON中，不混作实测峰值。

两个完整采集周期的medium profile使用独立进程，不能取代正式计时：

| profile阶段 | resident | 纯PyTorch mixed-c |
| --- | ---: | ---: |
| forward | .385秒 | 1.376秒 |
| decode/roots | .335秒 | 归于其调用阶段 |
| backward/finite | 2.510秒 | 1.173秒 |
| 完整返回/detach/sync | .076秒 | .229秒 |
| 设备kernel时间覆盖 | 1.440秒 | .187秒 |

resident profile的head guard/reduce分别约414/387毫秒，还观察到索引、scatter、zero与参数归约成本；
mixed-c约105毫秒落在host scalar事件中，resident约10毫秒。源码中每窗backward还构造实际adjoint程序。
这些观测支持把程序构造/提交、数据整理、guard与归约作为后续性能工程的具体检查点，不能仅凭kernel数量判定瓶颈。
嵌套host事件可重叠，kernel覆盖也不是独占关键路径；表中各行不能直接相加解释完整时间。

小规模profile2同样通过完整数值对照，但有vendor默认采集周期结束警告，其事件数不作完整工作量证明。
ceae7bf显式完成单个采集周期，medium两项没有该警告。
最初profile在设备初始化生成临时kernel_meta/后被洁净检查拒绝（exit2），未测量；
修复把运行cwd提前移到独立artifacts输出目录，原失败与诊断哈希保留。

## 严格失败与独立诊断

FP32反馈PDG两配置事后检查分别报maxabs 4.11272e-6与2.71201e-6，形状[33]。
原时间和全部原输入保留在JSON中，但不用于等价加速或选型的胜出声明。
另一次有限诊断使用相同FP32初值及其FP64提升，比较独立Python时间递归和native两种调度、四窗口SGD：
两dtype全部离散结构一致；FP32从pending到raw字段出现严格差异，最终参数仍在原容差内；
FP64全部完整记录和最终参数通过。这与有限精度误差相符，但不解除原FP32严格失败，未修改输入或容差。
独立公式/VJP、共享参数与精确checkpoint资格仍沿[语义索引](../semantic-map.md)追溯，本性能对照不替代它们。

## 资源、限制与复现

任务使用冻结源码、独立background.slice、Nice10、ATen/BLAS1；build2线程。
CPU每配置120秒、组1500秒；NPU每配置180秒、组960秒、排队180秒。全部结束且unit状态与持久记录核对一致。
本项目性能进程串行；CPU测量期间仍有单独NPU正确性资格，主机争用已记录。
其他用户未隔离，单次流程及持续演化窗口不支持统计稳定或普遍最快的结论。

构造与warmup已单列；runtime/loader冷启动未独立计时，完整进程时长还包含导入和事后验算，
不能把它当作纯冷启动延迟。standalone没有导出设备allocator峰值。以上测量缺口与数值失败分别保留。
源码/设备支持不因缺少这一小项指标变为不支持；总体单样本跨卡交付仍是独立待办。

复现使用[性能契约](../weighted-performance.md)中的三个公共脚本与standalone入口，选择
上述topology/family/size/mode/schedule，默认steps=3。静态导出只包含参数、输入和配置，不包含轨迹。
先安装匹配core和resident，再构建消费者；每次新建输出目录，固定source和输入hash。
Python PDG仅作正确性参考，不要求性能扫描。入口资格CPU三项、NPU一项均独立通过，
其1warmup+1窗时间未混入上述24配置。
