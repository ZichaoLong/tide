# 当前路线图与验收范围

本轮任务：固定 `0770593838a37c790a568226930567ed60f99b83` 的 core-5 与带权局部语义。
用户授权持续实施、私有环境、有限实验和当前分支分阶段提交推送；首个 KV 组合不代表完成。
[STATUS](STATUS.md) 是唯一当前交接，[execution-flows](execution-flows.md) 是持续契约。

| 阶段 | 交付与验收范围 | 当前状态 |
| --- | --- | --- |
| 0 起步与采用 | 迁移路径/工具链、最小 CPU 开发链路、固定原文与验收 §1.1–1.4 | 路径/来源核对与 Python 起步通过；原生基线构建与 93 项检查通过；core-5 19 文件逐字节校验通过 |
| 1 公共表示与一般图 | 同秩区域、任意端口日程、逐边/端口返回、原样嵌入；SrcClass 多原子、结构化 payload/content/Read、int64 元数据与精确编解码 | 一般 SettleGraph efb9d62 固定提交定向资格 247 项通过；结构化 Read/int64/SrcClass 表示 714d5f9 固定提交定向资格 212 项通过；具体带权绑定在下一阶段 |
| 2 首个完整组合 | kv/sum/+、cat/ind、regional_total/context/cat/quad；明确尺寸/初态/共享，FC 与 HB | Python/独立 C++ 首组合 6fb467c 固定提交定向资格 129 项通过；含 FC/HB、PDG 反馈、非对齐 TimedDAG、一般端口、完整记录/独立根、多步优化器与恢复；standalone 完整图训练/checkpoint a7106fc 固定源码 20 项通过（每 dtype 24 图案例） |
| 3 正式候选 | affine/delta/gated_delta/ssd、固定逻辑时间短卷积；全部适用发送/评分/证据/key、异构组合 | affine/delta、完整 GDN/SSD、固定时间短卷积与异构 Python/C++ 442e8b2 固定源码资格 544 项通过。Bernoulli/Shannon/固定 alpha ent 分配 a05d3ff 固定源码资格 226 项通过。十种发送/节点增益 4a08588 固定源码资格 320 项通过；match/att/grouped evidence 与两种 key cacb645 固定源码带权支线资格 593 项通过；旧同名内核不自动符合新定义 |
| 4 精确联合/packed/VJP | 在线贪心 prefill、完整记录与右切面、实际 packing、参数/计算共享和私有状态 | KV 实际 ragged 投影、四种循环仿射前缀及原始时间卷积联合计算已实现；26dad7d 固定源码相关资格 378 项通过；State 批量 VJP 7a017bf 固定源码相关资格 343 项通过；Aggregate 数值/batched VJP d90cfbc 固定源码 704 项通过；Read/Full 数值批量 7b8b469 固定源码相关资格 377 项通过。Read/Full 批量 VJP 67854b3 固定源码 647 项通过；区域实际数值/批量 VJP 及在线调度 5391017 固定源码 536 项通过；来源四种绑定/类别宽度及稳定 Attention/RMS VJP 503f609 固定源码 665 项通过；Aggregate 精确计算共享 52dee1e 固定源码 543 项通过；KV base 投影共享 6dc2665 固定源码 309 项通过；禁止轨迹预跑 |
| 5 设备与执行位置 | CPU、mixed、resident，前向/反向/优化器/跨卡/容量/续接分验收 | weighted placement 9234c13 CPU508项通过；470ce06 CPU200项、NPU Python/native mixed-a/b各2、mixed-c各19、显式同步两卡各2组合固定资格通过；默认异步两卡错误与standalone fc/affine严格失败保留，8c8824f半精度CPU99通过/2个GDN AdamW严格见证；01689c0 NPU half有限mixed/同步peer与master恢复通过；670ead3 standalone masters/schema2 CPU24项、NPU half显式SGD24图/AdamW2图及CPU handoff通过；2814b87常驻Alloc CPU29/NPU完整VJP、8e7298d发送CPU34/NPU八配置VJP及两者CUDA三架构构建通过；161ccea批量head CPU8/NPU共享VJP与保存版本/CUDA三架构构建通过；ce63aef完整Region组合CPU33/NPU14配置全部字段/参数VJP与保存版本、CUDA三架构构建固定资格通过；92751c0完整Aggregate CPU15/NPU物理原子/完整VJP与保存版本、CUDA三架构构建固定资格通过；weighted resident继续；670ead3 CUDA新栈core构建/CPU24项通过，真机未验；CUDA 源码、构建、真机独立记录，无 GPU 不阻塞其余工作 |
| 6 有限选型与交付 | 小中规模 profiling、代表性完整训练与推理；计入调度/packing/通信/同步等 | 待正确性后回答选型问题；默认每配置一次独立测量，不展开全开关矩阵 |

每项都维护固定定义→配置/参数/快照→代码→独立测试→固定源码运行证据。
同函数完整记录等价、指定投影等价、不同函数的性质比较分别列明；精确解 VJP、
有限迭代导数、硬函数代理反向分别命名。数学不适用、实现未完成、硬件待验不能混写。

已有 core-3 基座及历史通过范围保留于 [证据索引](evidence/README.md) 与 Git：
三图/独立调度、模块/训练/续接、CPU/NPU 与 CUDA 源码；不重新排队旧任务。
旧近似平票路由失败、容量拒绝与设备边界继续保留，不能用历史数字认证新实现。
当前图/checkpoint 版本由 [semantics](semantics.md) 统一维护。
