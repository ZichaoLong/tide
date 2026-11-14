# v10修复接入与后台续训交接

2026-11-15 05:07 +08:00实际核对：三条均通过全部接入门槛并已在正式训练中继续更新。
[逐链真实记录](../evidence/direct-v10-repair-handoff.json) · [原20条快照](../evidence/direct-v10-observations-repair-start.json) ·
[授权执行表](direct-v10-repair.md) · [状态/日志/停止/恢复/Lens操作](direct-v10-operations.md)。

固定源码**d089dd0556798df55a0440ef471ffe254f8054e5**，`.local/source-v10-repair-v1`；
树SHA `f3c2917c9ff1e3f5aa4db6e9fd016eed0c5a2b012177230f9308a834225e6e16`。
三条配置只增加`implementation.aggregate_vjp=ordered-v1`；前向、global64、L256、physical8/16、LR/AdamW、数据和seed保持。
实际安装/数学仍74e7bbd/0770593，不采用基座工作树。原训练0ea05d61源码及三个failed保持原身份。

|配置ID|物理NPU|64更新先导前→后累计曝光（Mi目标）|131K validation NLL前→后|先导最大梯度范数|交接累计曝光/状态|
|---|---:|---:|---:|---:|---|
|hb-d128-r64-pool-s11|11|23.216934→24.191033|3.696425→3.683903|0.604362|24.741Mi，正式运行|
|hb-d128-r64-pool-s22|6|16.793482→17.771697|3.842901→3.823365|0.628041|17.970Mi，正式运行|
|hb-d128-r28-pool-s22|14|54.211791→55.189027|3.341698→3.333970|0.560797|58.269Mi，正式运行|

这些先导从各自不同的失败前曝光开始，表中NLL不能跨行作模型排名；64更新改善不等于长期稳定或架构质量优势。
先导是真实学习前缀，接续到正式128Mi；恢复的重复探针不计作学习曝光或新seed。

- 执行：三个生产算子见证、整网准入、状态迁移、恢复、64更新和Lens读回全部完成；正式续训仍running。
- 严格数值：三个实际算子见证的前向/VJP均过原atol1e-6/rtol1e-5；整网小型前向均过。
  整网参数VJP仍有有限严格失败，R64 s11/s22及R28 s22分别45/58/77个参数张量；原阈值与失败不变。
- 离散/机制：同进度的因果干预、实际图支持/可行性、共享head及optimizer更新通过限定检查。
  一般continuation/absent-zero fibre契约仍未验收。
- 迁移/恢复：父SHA与诊断证明核对；model/Adam/RNG/cursor保持，源码身份显式改变。
  在实际训练权重和完整L256/global64上，连续2更新与1+1新进程的参数、Adam、进度、梯度hash及CPU/NPU/Python RNG精确一致。
  只退役本链三份已比较的恢复payload；收据`recovery-retirement.json`保留。原checkpoint不删。
- 学习：三条短程全部保持有限；正式128Mi终点和独立262K confirmation尚未完成，test仍密封。
  后续终态需人工审阅，图中长期结果仍awaiting。

原20条此时12完成128Mi、5运行、3原失败。
修复不修改原控制器的256Mi成组决策，不计新候选/seed。接续到128Mi后各自自动做原262K独立确认，没有修复链自动256Mi。
共同截止仍**2026-11-16 21:51:33 +08:00**；全局12slot、每服务4CPU线程/72GiB、background.slice，
无累计CPU/NPU算时上限；修复24GiB、整个v10 64GiB目标/96GiB停止线、账号至少200GiB硬余量。
当前账号307.15/800GiB，硬余量492.85GiB。
先导NPU峰值分配R64约30321MiB（29.61GiB）、R28约25366MiB（24.77GiB）；共享设备，不把单卡观察当独占吞吐基准。

Lens项目`tide-learning-lab`：正式`tl-v10-repair-<ID>`，先导`tl-v10-repair-pilot-<ID>`，均TL-CONTROL-v10；
三条先导已verified/readback_matches，当前正式投影recording。
终点确认`tl-v10-repair-confirm-<ID>`属于TL-CONFIRMATION-v10。
私有入口保持`.local/LENS.md`。查看`python scripts/review_v10_repair.py`，原20条仍用`python scripts/review_direct_v10.py`。

回归：原直接特例/活动调度、局部算子、生产分派、配置迁移拒绝、提交/状态/图隔离相关检查通过。
新源码恢复与先导是实际NPU执行，不能由CPU单元测试代替；不承诺所有浮点范围或高阶导数安全。
