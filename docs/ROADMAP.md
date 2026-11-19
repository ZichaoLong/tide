# 当前有限计划：五类记忆/参数绑定

实际状态在 [STATUS](STATUS.md)，局部定义在 [memory candidates](memory-candidates.md)。

1. 审查草稿，接通反向混合、Q/K/V共同窄通道、仅记忆区域共享；FFN与其余归属不变。
2. 修复容量/统计宽度假设、公开训练入口、配置/checkpoint/放置；独立值/VJP、共享梯度、私有状态检查。
3. 提交固定实现后，冻结源码跑完整CPU及小尺度FP32/BF16多步/分块/实际新进程恢复；保留超差并有界诊断。
4. 约153M图七点FP32持续训练和实际恢复，再做9B图七点固定FFN的B1/B8比较。
5. 有限batch阶梯、kv128与memory_shared两项额外FFN近总参数、少量持续decode与设备成本/显存诊断。
6. 小尺度代表五类严格A–D；共同32层参考及旧证据保留身份。核对后代终态、原子交接、证据另提并推送。

本轮边界：最多6张空闲NPU并行、80个NPU子进程、每普通作业1800秒/128GiB RSS，
每卡48/56GiB allocated/reserved采样预算，新增产物80GiB；预计6小时内完成有限计划。
batch按预算拒绝、收益趋平或有限上限停止，不宣称全局最优。已知语义/状态/共享梯度错误阻止受影响路径。
有限数值差异保留原阈值、原失败与有界调查，不自动归因舍入。任务效果仍由下游验证。

---

# 当前周期交付与后续机会

本轮获准的 Reception、FP32训练与大参数推理七阶段有限计划已完成。
实际事实见[STATUS](STATUS.md)，持续约定见[project scope](project-scope.md)。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`；FP32优先，BF16次选。
实例产品路线继续；通用基座仍是另行明确启动的路线。

交付后的固定32/48层Transformer、Tide B2/B4/B8及17B两卡/四卡有限补测也已完成，
见[追加证据](evidence/transformer-depth-batch-20261118.md)。它保留原A–D与B1身份；
该次补测与下面另行授权的搜索均已终态，不重新启动原矩阵。较大batch提高总吞吐，四卡在本次
同batch比较中更慢但降低峰值；新的布局/运输优化仍需按实际需求另选有限候选。

## 追加交付：固定预算的 batch 吞吐搜索

12系列固定模型/FP32短请求搜索已经完成：9B单卡、17B双卡/四卡的Tide与
近总参数Transformer C/32/48层。各自48/56GiB预算内正式最佳、同B8对比、
预算拒绝/有限停止理由和原B1身份见[搜索证据](evidence/decode-capacity-20261118.md)。
全部作业及独立审计已终态；[STATUS](STATUS.md)保留当前交接。
这是有限已测最佳，不证明全整数batch、所有执行候选或硬件容量的全局极限。
没有重启原A–D矩阵，也没有增加训练/恢复或其他后端资格。

## 本轮交付索引

| 阶段 | 已交付内容及证据 |
|---|---|
| 基线与执行边界 | 冻结源码、配额/资源准入、原始输入/参数身份；首主干输出、完整prompt、首增量、稳定decode和有限尾部的独立计时契约。见[推理入口](inference-benchmark.md)。 |
| 接收self-attention | 三种incoming公式/VJP、真实零质量域；QKV融合、结构mask、相容合批和重算；九候选真实多步及新进程恢复。见[执行](reception-execution.md)及[数值证据](evidence/reception-numerics-20261118.md)。 |
| FP32热点 | SSD单位置直接递推、块内doubling；可选静态参数视图；保留block、gather和重算候选。见[成本与设备轨迹](evidence/reception-performance-20261118.md)。 |
| 大参数推理 | FC/HB约0.5B/1B、8.778B单卡、16.707B双卡；256增量、批量流、长prompt；17B合批与并发分开比较。见[推理证据](evidence/reception-inference-20261118.md)。 |
| 四类联合比较 | 三种incoming的推理/含图前向/完整训练，0.5B主力B4训练及推理A–D；实际触及目标、匹配偏差、工作/padding和双边归因。见[训练表](evidence/reception-training-20261118.md)及[比较契约](benchmark-contract.md)。 |
| 连续与完整验证 | 五组FP32百步和新进程第101步；独立缓存淘汰/原时钟检查；783项冻结CPU全套、后续34项入口检查、隔离wheel及双NPU。有限/支持超差仍保留，见[验证](evidence/reception-validation-20261118.md)。 |
| 分模式交付 | 五个实测FP32训练预设；首输出、decode、批量流、BF16及内存候选分别建议。见[执行建议](execution-recommendations.md)、[计算公式与伪代码](computation-and-throughput.md)、[经验](performance-lessons.md)。 |

完成有限计划不表示所有配置逐位等价或全部后端合格。FP32跨候选更新、双卡恢复及
BF16输出/支持/恢复超差按原阈值保留。正常时间、侵入式诊断和非等价Transformer
参考不混为同一结论；大小模型、Settle/Timed的相对表现确实不同。

## 具体后续机会

以下是下一次任务可选择的方向，不把它们伪装成本轮未跑完的测点，也不因列表而启动
全组合搜索或另一条产品路线。

1. **分配成本。** 0.5B的allocation约占一次侵入式前向27.69%；应先区分断点排序、
   解面重构、回退发生率和小算子/同步，再评估局部融合及专用VJP。当前证据没有把全部
   成本归给sort或fallback，也没有证明它们已达下界。
2. **Full补齐与索引。** 0.5B K packing执行/选择约5.25倍；可对有效行数分桶或缩小组，
   同时计入增加的launch。静态视图在9B推理有收益，但训练反向与跨更新差异还需单独
   权衡。保留动态稀疏Full、参数归属、索引梯度、原时刻KV与私有状态。
3. **有限数值归因。** 已有NPU内部捕获到界限；可用更窄的Reception Q/K或索引反向
   见证追踪原FP32更新超差。BF16组合路径的支持变化与placed恢复超差仍需各自处理，
   不用放宽阈值、仅exact下一前向或CPU局部解释代替完整因果证据。
4. **有需求的后端与尺度。** 当前完整模型是PyTorch，LibTorch为区域评分切片。
   真正需要更大训练或更低调度成本时，再选择相应连续训练/恢复测点及编译、C++、
   原生跨卡传输候选；CUDA、HCCL/peer DMA、全模型C++、compiled/resident训练仍无新增资格。

新的执行按[执行契约](execution-flows.md)冻结输入、限定时间/RSS/设备预算并核验终态。
共享NPU授权仅限原对话，不能成为仓库默认。参考、公共环境和下游仍保留各自权限边界；
任务头、数据、训练配方及模型质量归下游。现有复现依赖保留，退役须先核对活跃引用、
工作区与下游安装，遵守[保留规则](artifact-retention.md)。
