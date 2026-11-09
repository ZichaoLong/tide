# v6执行定位：大量小算子、动态索引和碎片化反向

2026-11-09，`TL-EXEC-v6-PROFILE`。**现有TIDE eager训练路径的主要问题是执行碎片化；
仅用局部宽度128解释CPU快于NPU并不充分。** 定位支持优先改进公共基座的批量前后向，
尚未交付优化补丁，也不构成语言质量、数学错误或所有TIDE实现性能的结论。

[研究表](language-execution-diagnosis-v6.md) · [完整测量/原始hash/Lens对应](../evidence/language-execution-diagnosis-v6.json) ·
[基座需求TL-REQ-004](../requests/TL-REQ-004.md)。原训练及冻结源码未改，基座只读。

## 完整训练步复核

同FC16/KV-qry/quad、width128、FP32、B8/L128、seed11，synthetic token输入。
CPU1与NPU/host2；时间包含前向、loss、反向、clip、AdamW，每段同步。
不含模型初次构造、输入准备、记录或profile；core前向包含每次调用内部的public runner构造和完整返回。

| 后端 | core前向 | 完整反向 | 完整步 |
| --- | ---: | ---: | ---: |
| CPU/native | 26.60秒 | 36.84秒 | 63.61秒 |
| NPU/native mixed-c | 134.93秒 | 129.63秒 | 265.00秒 |

均取一次预热后的一步。旧未分段校准CPU62.6–63.0/NPU232.5–236.6秒仍有效于其原时点；
本次共享主机负载和分段同步不同，不将差值误归因于代码回归。
新测CPU/NPU各16,384 candidate；Full为8,352/8,350，内部消息25,068/25,108，
保留数值与工作差异，不能宣布严格等价加速。任务readout、head和optimizer在此处不是主要耗时。

## 归因证据

小见证B2/L8仅16 token、256 candidate。native NPU未插桩完整步约4.09秒；
独立设备profile的6.155秒窗口中，Computing0.684秒、Free5.471秒，产生168,589个kernel记录。
有7,308次标量读取、6,875次索引和3,603次索引回写。**插桩有开销，这不是无插桩利用率百分比。**
Python设备trace以及native CPU算子trace独立复核同类形状/调用现象；不混用不同实现的耗时占比。

- **Read/Describe按行计算。** 基座C++及Python的batch入口内仍逐行完成归一化、score/evidence/key和小投影。
  `[128]×[128,32]`等三个典型向量投影各出现256次。源码注释说明这曾用于保持近零VJP的有限算术顺序。
- **KV可见性/正支持/分组需要大量动态索引和host判断。** Query signature含私有cache shape，
  逐行bool索引、nonzero和长度分桶；batch的部分收益随后被拆分。
  调度`max_full_batch`不能当作最终所有GEMM的batch，小见证有大量宽度128/512、batch仅1或2的矩阵调用。
- **反向同样碎片化。** 独立行/字段根的嵌套autograd与索引scatter带来大量小张量操作；
  native实际反向约占完整步49%。设备AICPU的Index/IndexPutV3各1,170次，共252.56ms。
  这里是设备侧AICPU，不是主机CPU fallback；剩余host、启动和同步成本不能全部归给某一函数。

## 隔离结果

宽度配对均为FC16、FP32、B8/L16，预热后一共两步的均值，当前独立物理batch为8。
更宽点只做三次synthetic诊断更新，不是新的语言训练或scaling成功证据。

| 局部宽度 / 参数 | CPU/native | NPU/native mixed-c | NPU耗时/CPU耗时 |
| --- | ---: | ---: | ---: |
| 128 / 5.09M | 6.94秒 | 23.54秒 | 3.39 |
| 256 / 17.64M | 7.98秒 | 26.25秒 | 3.29 |
| 512 / 65.55M | 11.64秒 | 27.45秒 | 2.36 |

增宽确实缩小设备差距，符合更大矩阵可能受益于NPU的判断；当前事件开销仍然明显，512以内尚未反超。
没有测出任意更大维度的交叉点，也不能将此表外推到其他记忆、HB或resident路径。

| width128下的public选项 | 完整步均值 | 与本组默认路径的关系 |
| --- | ---: | --- |
| mixed-c默认 | 23.54秒 | 参考 |
| mixed-a：Read/control/selection在CPU | 14.77秒 | 本次约减少37%；相关小计算放CPU明显缓解开销 |
| native紧凑事件/批量传输，关闭trace | 22.88秒 | 本次约减少2.8%，两步不足以认证稳定收益；返回策略也不同 |

两项与默认路径在各对应测量步的candidate、Full和内部消息计数一致；这不能代替逐事件/梯度等价检查。
mixed-a改善也不能全部归因于Describe：它同时改变Read、区域控制、selection、传输和反向路径。
这里只确认整体隔离效果；精确细分占比仍未知。

## 建议、边界与交接

优先由基座交付具名批量Read/Describe，减少KV的逐行动态索引/回读，再解决批量VJP与字段布局。
把可共享投影/FFN从ragged attention分组中分开。原严格路径、1e-6/1e-5阈值和失败保持；
优化路径单独做梯度、共享、None/零、离散支持、optimizer及新进程恢复，有限数值探索独立标记。
public resident可另行评估，不能先假定迁移即可解决。

本次执行全部完成，有限loss/梯度检查通过；未新增独立严格数值、离散等价、恢复或学习资格。
旧严格失败不解除。主要未知为原L128形状的逐算子critical path、其他语义族及resident表现；
本报告足以提出具体执行需求，不证明所有性能问题已找到或修正。

10个诊断服务均曾核对background.slice，终态后核对MainPID=0、无存活后代；
14个运行结果均完成。服务分配约0.81 CPU核时、0.36 NPU卡时，产物约2.42GiB，均在新增上限内。
源固定`d90e733`/`b4bc62b`，数学0770593、foundation74e7bbd及原native binary不变。
Lens项目`tide-learning-lab`，组`TL-EXEC-v6-PROFILE`，前缀`tl-v6-diag-`；所有终态投影读回核对成功，
精确显示名/ID见摘要。私有入口沿用`.local/LENS.md`，不写入公开文档。
