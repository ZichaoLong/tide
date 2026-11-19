---
type: implementation-review
status: reference
as-of: 2026-10-28
implementation-revision: 2c46aa7d0b77afe7a7b82ed428466e36f5162de4
upstream-revision: 0770593838a37c790a568226930567ed60f99b83
tags:
  - tide
  - implementation
  - design
---

# ascend 实现与记录审视：局部决策、参与范围与实际计算

**当前实现已采用 core-5，旧交接中“一般 SettleGraph、带权 KV 等尚缺”的判断需要更新。** 代码和留存记录已覆盖一般端口、多区域、五种记忆、十种发送及若干 CPU／NPU 执行与训练路径。相对本次已确认目标，主要待解决的是具体绑定如何限制整次输入的参与范围，以及数学上的少量 Full 如何转化为实际少计算。已有性能记录尚未证明同等任务效果下的稀疏收益。

本次按[设计讨论稿](../../memos/design/design-requirements-and-architecture.md)中的目标审视：先看模型的信息与决策局部性；在任务效果可比时，优先限制参与节点和连接，减少 Full 与消息。按[随后对固定端口扩展的澄清](../history/document-revisions.md#2026-10-28固定端口扩展与外部接入接口)，固定端口配置后的内部扩展需要多跳扩散与收拢，不同模型可以另选端口数；实验优先在 Next Token 预测上采用单输入单输出。内部直接通信对象数有界及固定主干或中心的依赖须分别检查。设备部署和学习规则另议。以下设计解释已据此更新，固定实现版本的源码观察与测量结果保持。

## 1. 版本、来源与本次核验范围

| 对象 | 本次观察 |
|---|---|
| 当前实现 | ascend 上 /home/zlong/llm/graph-execution-foundation；分支 graph-execution-foundation，提交 2c46aa7d0b77afe7a7b82ed428466e36f5162de4；工作树干净，无活动任务 |
| 实际采用 | [upstream.json][lock] 固定 core-5／0770593838a37c790a568226930567ed60f99b83；19 份上游快照。旧 core-3 的 5 份快照另存为 legacy |
| 昨日外部交接 | /home/zlong/tide-handoffs/tide-implementation-recommendations-2026-10-27.md 指向 35d63588f3a967e4b65da9c55be05a47f4d964f8；只作为该时点的历史判断 |
| 旧实验平台 | ascend 上 fractal-latcarf 工作树干净，HEAD 为 4883483；本次未扩大为其训练与历史实验审计 |
| 本次直接验证 | 将 19 份快照与本地上游 Git 固定对象逐字节比较，并核对 SHA256；9 个历史任务的 19 份日志、终态或测量文件与报告哈希一致；在当前实现上补跑 30 项定向 CPU 检查，全部通过 |
| 范围限制 | 抽查代码、测试与既有运行材料；未重新构建所有二进制，未重跑 NPU／CUDA 资格或性能矩阵，未进行任务质量实验 |

核对清单、文件摘要和本次测试命令保存在[伴随 JSON](2026-10-28-ascend-implementation-checks.json)。服务器的 check_upstream.py 在本次调用中检查了清单与快照，本次另用上游 Git 对象完成了源文件比较。哈希吻合证明所读记录与报告引用相符；对应算法的正确性仍须结合独立测试和限定的运行范围。

下文区分**实现事实**、**记录中的验证结果**与**设计判断**。一般定义继续以上游教材为准；截至此版本的实现能力从[语义对应][map]、[能力范围][capabilities]和[证据索引][evidence]进入。

## 2. 已实现的机制怎样对应设计要求

| 要求或接口 | 本次核对到的实现事实 | 意义与边界 |
|---|---|---|
| 节点从本地接收和历史形成描述 | [binding.py][binding] 为节点绑定私有状态、读取与 Full；[programs.py][programs] 的 Describe 从本节点接收与所选记忆取得字段，RegionalTotal 只处理本区域候选描述 | 支持模型中的局部信息依赖。参数共享与状态共享分别处理；信息可经消息累积，私有状态本身不限制远处信息的历史影响 |
| 多处独立协调、一般输入输出 | core-5 SettleGraph 允许同秩多个区域、平行边、内部输入与多个输出端口；[端口测试][ports-test] 以手算路径系数核对这些对象 | 旧“一层只能一个区域”的缺口已补齐。通用引擎能表达分散结构，具体模型仍要选定图和边界 |
| 选择参考历史，Full 使用正确快照 | Read 0／−／+ 分别取绑定初态、旧态、候选态；[常驻局部流程][flow-local] 先更新，再在所选快照上 Mem／Describe，Full 读取候选态；全部候选保留接收更新 | Mem 已在真实代码中承接“历史提供选择依据”。这属于当前带权绑定；其他一般局部函数可有不同状态采用规则 |
| 限制激活节点数 | [RegionalConfig.capacity][config] 对 hard_selection、regional_threshold_hard 返回 k，对其余发送规则返回区域规模；激活支持由控制量大于零决定 | 实现保留了数量容量与发送总量的区别。给任意发送模式配置小 k，不能统一解释成只算 k 个节点 |
| 减少参与连接 | [WeightedFull][programs] 向该节点全部出端口同值广播；[topology.py][topology] 控制固定邻域，通用 Full 接口另支持逐端口缺席 | 带权支线主要靠少激活与低出度减少消息，尚不能把发送强度理解成独立的逐边选择 |
| 多种局部记忆可替换 | 已有 kv、affine、delta、gated_delta、ssd 与固定逻辑时间短卷积；描述、评分、发送配置分开 | 支持比较记忆与描述成本；“接口可替换”不证明各记忆在同一任务上质量相当 |

已留存的[一般 SettleGraph 定向资格][settle-evidence]为 efb9d62 上 247 项 CPU FP64／FP32 检查通过；[发送函数资格][sending-evidence]为 4a08588 上 320 项通过。本次读取其原始日志并核对哈希。本次新跑的 30 项仅包含手算端口、KV 写入与快照、发送数量反例、容量配置、拓扑计数及 FC／HB 绑定与读出检查，不能替代完整 native 或设备资格。

**设计判断。** 统一执行器、设备协调程序或共同优化器的存在，不足以判定模型决策集中。按用户当前范围，应检查每个描述和区域选择实际能读到什么，以及边界职责之外的消息路径和选择依赖是否统一经过固定中心；物理调度的集中程度另议。

## 3. 对主线影响最大的三个问题

### 3.1 局部稀疏已经可表达，整次输入的参与范围仍需设计

**定义与实现事实。** FC 预设每层一个区域、相邻层全连接。HB 预设中间多区域，首尾各一个区域；其 relations 参数只接受包含递归基础关系的扩展，不能用它直接删掉基础分支。两者的 layered_inputs 都向全部入口节点注入，标准 readout 汇总指定输出。代码见[拓扑][topology]与[实例绑定][instances]，定义见[HB 实例 §3.1](../../instances/hb-lattice-weighted-message.md#31-每个区域都有正发送)。

在该实例的单位输入及非空发送条件下，每个输入位置到达所有区域，每个区域至少一次 Full。把每区激活上限设为 1 或 2，仍然经过全部区域，但节点和消息可以保持稀疏。按用户澄清，区域覆盖率本身不构成目标缺口；一个完整的小图计数见[设计讨论稿 §2](../../memos/design/design-requirements-and-architecture.md#2-在一张小图上完成两次计算)。

**设计判断。** FC／HB 可作为可复现的数量参照，扩展时需明确改变哪些参数。按上游拓扑定义，HB 固定区域宽度、分支数和偏移数量、采用基础区域关系，增加深度，可同时保持端口数、节点直接邻居和区域协调范围有界；FC 固定宽度加深与不断加宽有不同的通信后果。这些结构推导不证明任务效果。具体绑定须累计候选、Full 和消息，局部激活容量、邻域和发送规则均可参与控制；跳过区域保留为可选设计。

在给定端口配置下，现有 readout 汇总固定数量的末层端口，与逐步收拢后的边界相容；接入单输入单输出实验时，边界适配、入口注入、区域选择及读出的依赖仍须计入完整模型。其他任务可选不同的有限端口配置。

### 3.2 小发送总量不保证少量 Full，已在代码和测试中明确

[发送测试][sending-test]包含可直接检查的反例：四个节点上限都为 0.3，指定发送总量为 1，按评分排序填充得到 (0.3, 0.3, 0.3, 0.1)，四个节点全部激活。同一输入下 hard_selection 的 k=1 才只保留一个节点。两者是不同函数，前者的行为符合其定义。

**设计判断。** 本次“首先限制参与数量”应落实到容量、实际支持和候选覆盖的显式比较，不能用总强度小、温度低或载荷幅度小替代。硬选择提供当前可用的数量参照；其质量和学习行为仍待任务比较，尚未选为唯一方案。即使 Full 数有界，候选上的 Agg／Upd／Mem／Describe 仍须单列。

### 3.3 常驻设备路径保留数学稀疏，但仍执行填充行上的算术

**实现事实。** [weighted_flow_local.cpp][flow-local] 为静态节点和样本行构建局部与 Full 程序；不活动行经 guard 处理，不成为有效输出或消息。[KV Full][kv-full] 和[projection][projection] 仍在这些批次行上提交矩阵乘等算术。当前路径没有把全部数学上的不活动行压缩掉。其文档也明确说明未活动行可能占用厂商批量算术，见[常驻流程][resident-flow]。

[记录解码][flow-records]同时返回两种统计：

- candidate_updates／full_evaluations：实际数学候选和激活作用次数。
- physical_local_rows／physical_full_rows：作用轮数 × 样本数 × 静态节点数，即提交给局部组件的物理行数。

留存 HB 小配置每窗有 16 次数学 Full、32 个物理 Full 行；中配置为 128 次与 192 行。这些物理行不能直接换算成等比例耗时或 FLOPs，却足以说明仅报告活动数会遗漏工作。

**设计判断。** 这项差别不构成语义错误，但直接限制“少 Full 所以更快”的推论。它把下层执行成本反馈到上层设计：需要同时记录候选准备成本、描述读取成本和实际提交工作；也需要判断调整区域粒度、邻域或按活动行打包，哪一项能在完整调用中减少成本。后者均是待验证的实现或结构方案。

## 4. 现有实验究竟支持哪些结论

### 4.1 性能记录回答执行成本，尚未回答任务效果相当时的稀疏收益

[性能报告][performance]及同名 JSON 保存 24 个有限配置，22 个完整记录／损失／最终参数比较通过，2 个 FP32 反馈 PDG 比较失败。三个原始 measurements.json 和任务终态、日志均已核对哈希。测试脚本固定于 ceae7bf；所用消费者和库有各自的固定源码身份，不能把全部结果统称为当前 HEAD 上的一次运行。

各配置单独进程，1 次预热后计量 3 个连续窗口；训练窗口计时包括准备、调度、packing、前向、反向、optimizer、完整 CPU 返回与同步。HB 的“小／中”都只有 **8 个节点、4 个区域**；中配置扩到宽度 32、batch 4、每窗 4 个输入位置。

| 留存工作量，每个测量窗口 | 候选作用 | Full | 内部消息 |
|---|---:|---:|---:|
| HB 小配置 | 16 | 16 | 24 |
| HB 中配置 | 128 | 128 | 192 |
| 非对齐 DAG 中配置 | 112 | 112 | 96 |
| 反馈 PDG 中配置，严格比较失败 | 240 | 240 | 400 |

全部 24 个配置记录的候选都激活；[fixture][benchmark-cases] 未构成稀疏与稠密路由的对照。[loss][benchmark-runners] 是实际输出数值平方和乘 0.001，用于核对执行和优化器；它没有数据任务、效果参照或收敛结论。

在 HB／KV 中配置的完整训练窗口中，记录的中位数为：

| 路径 | 毫秒 |
|---|---:|
| standalone CPU prefill／streaming | 350.5／456.5 |
| Python 调 native CPU／纯 Python CPU | 406.6／541.1 |
| native mixed-c／Python mixed-c，NPU | 1413.7／1611.9 |
| standalone resident／Python resident，NPU | 2599.6／2706.1 |

这组调用边界和规模下 CPU 成本最低。其结果可指导进一步定位提交、数据移动、guard／归约与反向程序构造成本；样本很少、图很小且环境与工具链分列，不能外推大模型、纯设备输出或多卡吞吐。

### 4.2 训练与部署能力已有实证，资格按路径区分

[节点放置训练报告][training]记录了 0155b0f 运行库与 220052a 检查器的有限验证：双 NPU FP32 窗口 VJP、跨窗继续；FP32／FP16 各 20 次训练更新、各 6 次样本副本共同更新；拒绝后状态保持、换放置恢复及安装消费者。Python 路径另有原 30 项及修复后 8 项。此次读取了相关原始终态、日志及 corrected-training／single-window／consumer-checks 结果。

这些是执行、梯度、优化器和恢复机制的证据。学习规则是否应去中心化、模型能否学到有效分工、稀疏模型能否保持任务效果，均未由这些检查回答。CUDA 的 sm80／90／100 构建、loader 和 help 通过仍只属于构建与接口资格，没有 GPU 真机结果；aarch64 证据也不自动覆盖 x86_64。

### 4.3 “本机交付关闭”仍保留严格失败

当前 [STATUS][status] 明确保留以下限制，本次没有重新裁定或解除：

| 路径 | 保留的失败 |
|---|---|
| eager FP16 GDN，默认 AdamW | 约 0.001984 差异，strict xfail |
| 默认异步双 NPU eager VJP | CANN invalid event；显式同步路径通过不解除此失败 |
| standalone SDK 2.10 FP32 FC／affine | 第三步 evidence 差异 3.09944e-6；mixed-a 有同一见证 |
| FP32 反馈 PDG 性能配置 | 两项完整比较最大绝对差异 4.11272e-6／2.71201e-6；FP64 诊断通过不解除 FP32 失败 |

另有已定位并补验的历史失败，应与上表区分：全远端反向曾耗尽 CANN 任务缓冲，0155b0f 以每节点服务处理原复现；optimizer 拒绝检查器把数值错误码 20 写成 21，220052a 修正预期后补验通过。原 failed／cancelled 任务仍保留。抽查中未发现将旧失败覆盖成通过的做法，也未发现以 CUDA 构建冒充真机验证；这只描述本次检查到的记录。

## 5. 对下一版设计讨论的牵引

**已有能力：** 共同框架与带权局部接口已足以搭建新的可执行参照；Mem、区域协调和状态快照都有代码与定向检查，讨论可以落实到完整绑定及其轨迹。

**设计解释：** 上层“参与数量优先”要求下层分开报告候选、Full、实际用边与消息；下层静态填充和 Mem 成本又要求上层检查区域粒度、候选覆盖与描述复杂度。按[用户对区域覆盖的进一步澄清](../history/document-revisions.md#2026-10-28区域覆盖与输出读出的讨论澄清)，FC／HB 全区到达本身不排除节点、消息稀疏；局部选择与邻域等方式仍可共同控制参与数量，只使用部分区域保留为可选方向。这是当前机制之间的逻辑关系，不据此推断真实发明顺序。

**待验证假设与建议顺序：**

1. 先构造数量参照并声明扩展方向：明确输入／输出、区域、邻域、发送规则与观察窗口，检查直接通信和协调范围的统一上界；同时数候选、Full、访问区域、去重节点／边和重复消息。
2. 在同一参照上分开改变局部数量容量、邻域或发送规则，观察少发送是否减少后续候选，以及候选记忆读取是否抵消收益。跨区域覆盖可另作比较；若改为逐边选择，须声明与现有同值广播不同的绑定。
3. 优先在 Next Token 预测上采用单输入单输出，给定数据与“效果可比”的尺度后，再比较质量、上述数量与完整调用成本；常驻路径同时报告物理行数。其他任务和端口配置分别声明，当前没有足够证据预选某个记忆、拓扑或硬／软规则为最终方案。

这三项是依据本次审视提出的下一步设计工作，不构成已执行的实验，也不要求立即重启服务器上的旧任务。

[lock]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/upstream.json
[map]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/semantic-map.md
[capabilities]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/execution-capabilities.md
[evidence]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/evidence/README.md
[status]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/STATUS.md
[binding]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/python/tidegraph/weighted/binding.py
[programs]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/python/tidegraph/weighted/programs.py
[config]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/python/tidegraph/weighted/config.py#L209
[topology]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/python/tidegraph/weighted/topology.py
[instances]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/python/tidegraph/weighted/instances.py
[ports-test]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tests/test_settle_ports.py
[sending-test]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tests/test_weighted_sending.py#L86
[settle-evidence]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/evidence/core5-settle-20261027.md
[sending-evidence]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/evidence/core5-sending-20261027.md
[flow-local]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tools/device_online/weighted_flow_local.cpp
[kv-full]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tools/device_online/weighted_kv_full.cpp
[projection]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tools/device_online/weighted_projection.cpp#L60
[flow-records]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/tools/device_online/weighted_flow_records.cpp#L68
[resident-flow]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/resident-weighted-flow.md
[performance]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/evidence/core5-weighted-performance-20261028.md
[training]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/docs/evidence/core5-weighted-node-training-20261028.md
[benchmark-cases]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/scripts/weighted_benchmark_cases.py
[benchmark-runners]: https://github.com/ZichaoLong/tide/blob/2c46aa7d0b77afe7a7b82ed428466e36f5162de4/scripts/weighted_benchmark_runners.py#L31
