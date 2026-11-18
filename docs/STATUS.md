# 当前交接：训练默认、路径核对与性能归因已交付

本轮执行用户已确认的六项计划；原五类完善及容量/区域执行的证据仍保留原身份。
分支为 `graph-execution-foundation`，持续 FC/HB 实例路线不变，通用基座未启动。
数学固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件核验通过。
本轮起点 `3a49a789`；生产优化 `007a3828`，训练预设 `32ab13a5`，
forward-only完整计时修复 `f8235fe4`。生产 `src/tide` tree 为
`d3e8a6cb40eceaf487dc642a4283d58f90d5726f`；各实测源码见证据，不能用最终HEAD替代。

## 本轮结果与入口

- [训练建议](execution-recommendations.md)提供四个主力NPU训练预设及两个BF16备选，
  必须显式选择已准入设备；无参数CPU入口安全可运行。可覆盖参数、查看最终设置、
  保存和真实恢复；默认入口服务训练，其他模式分开给建议。
- [有限搜索](evidence/training-search-20261118.md)补齐batch-only及纯batch+整区域重算。
  新增Reception局部重算和SSD向量摘要prefix；同尺寸分别观察到约22–25%降峰和
  约6–9%缩短时间。原单区域、合批、math/tiled/fusion、整区域重算及放置候选均保留。
- [连续训练](evidence/training-continuous-20261118.md)六组各20次真实更新，历史窗口
  填满、optimizer step1→20、数据有限；另有六次新进程20→21更新。
  主力FC BF16 B128/L256约38249输入位置/秒；Settle FP32组合B24/L256约2075；
  Timed FP32组合B12/L128约317；0.5B FP32 B1/L128约96。allocated/reserved峰值
  和恢复数值分别判定，不能从allocated单独推断真实显存余量。
- [四类主表与其他模式](evidence/training-modes-20261118.md)包含正式A–D联合匹配，
  推理、含图前向、完整训练、首个主干输出和真实缓存decode。Tide当前完整训练仍
  慢于代表A/B；C/D的宽浅结构是不同计算参照。完整同步修复后的缓存结果单独归档。
- [成本证据](evidence/training-costs-20261118.md)有正常计时、固定anchor阶段/内存
  诊断和实际Ascend设备trace。差距包括数学工作、padding、矩阵SSD、索引和调度；
  Transformer自身chunk256有收益。重叠时间不相加，也不把诊断时间当正常占比。

## 数值事实与限制

[31候选路径核对](evidence/training-paths-20261118.md)结果为9 within、20 finite、
2 structural；31套保存/下一前向精确，恢复更新28 within、3 finite。原阈值不变。
Timed BF16的prefix组合在第二次更新出现实际支持和KV时间差异；不合批prefix隔离
也有后续支持差异，捕获界限内未建立完整因果链，该组合不作为Timed BF16默认。

定向FP32 Attention见证定位同输入PSE/rank-one两种表示的局部微差，独立FP64八行
均在原阈值内；不能据此解释全部状态或训练轨迹差异。两卡serial/per_device有反向
与恢复超差；单卡FC大batch BF16也保留配对恢复梯度/moment超差。
新进程实际执行成功、精确checkpoint和正常性能均不等于后续轨迹严格等价。

[集成与安装](evidence/training-validation-20261118.md)：冻结`ae247dee`完整CPU625项
通过；`32ab13a5`后续定向83项通过；同步边界回归2项通过。新wheel21文件与生产源
一致，安装API/真实恢复运行完成；placed CPU BF16仍有154梯度、140更新项超差。

完整模型仍是PyTorch，C++17/LibTorch是既有区域评分切片。CPU/NPU有实际证据；
CUDA、peer DMA/HCCL、全模型C++、compiled/resident训练没有新增资格。
全部合法主菜单可表达与元数据检查通过，不代表全笛卡尔组合、任意尺度或任务质量已验证。

## 工作区、资源及后续

本轮所有Tide实验已终态，当前无遗留运行服务或已记录存活后代；内层失败完整保留。
本线程允许共享NPU，本轮所列设备实验实际采用协作队列独占准入，仍共享主机；
不能称全机独占。配额从约83GiB增至约125GiB，低于300GiB软限，未清理复现依赖。

四份用户未跟踪文档保持原字节，不纳入提交；哈希见集成记录。参考、公共环境和下游
保持只读边界。没有下游发布或验收。冻结源、checkpoint、anchor、trace、私有wheel环境
及代理工作树保留；日常复用[性能经验](performance-lessons.md)，不查退役周期。

本轮没有待运行任务。有限后续机会与未解决数值边界列于[ROADMAP](ROADMAP.md)；
从[证据索引](evidence/README.md)可直接复核，不需要重新启动实验。
