# 当前交接：稳定质量归约有限计划完成（2026-11-19）

本轮从 `a3fc8053` 接续，生产修复固定为 **`8fd5216a`**；成本工具为 `1aa26314`，
二者 `src/tide` tree 相同：`da223720c5c077178dde154d298fdbe73243e23c`。
当前文档/证据HEAD不扩大这两个身份的资格。固定math
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234` 的108文件原字节通过；四份用户文档未改、不提交。

## 已落实与验证

Reception采用缩放均值、中心化复合VJP及广播根汇总；KV Full采用质量缩放Attention，
保留真实零边导数、原时间与私有状态。两段指数恢复修复NPU将可表示 `-3e38` 误成无穷的错误。
不改floor、epsilon、原阈值、数学或参数归属。公式及执行边界见[stable-mass](stable-mass.md)，
正式资格和成本见[本轮证据](evidence/stable-mass-20261119.md)。

- 冻结CPU全套：**1140 passed, 4 xfailed, 1 warning in 603.14s (0:10:03)**。
- 第二轮空闲单NPU定向：**86 passed、3 strict xfailed、1弃用warning**，日志未观察到CPU算子回退。
- CPU与NPU各四个小FC/HB案例（3个FP32、1个BF16）：推理、含梯度前向、重复VJP、三步继续训练通过；
  新进程step4→5的saved data、next forward、actual update全部exact。
- 隔离wheel25包文件匹配；公开训练3步、checkpoint和新进程恢复更新通过。

首轮假溢出及CPU诊断wrapper接口失败保留原日志；旧FFN超差保留。
seed773三个Attention路径保留同一近零dV超差：相对约2.07e-5，原rtol1e-5/atol1e-35未改。
strict xfail仅接受该坐标与数值指纹，不能称全数值等价。独立公式和有界误差拆解见证据。

## 性能与推荐

固定926,185参数小HB X pool、独立B2、prompt16/chunk4/decode8、FP32、4线程，三组成对重复：
CPU完整推理耗时 **+7.25%**、完整训练 **+8.57%**；
单NPU分别 **+25.01%**、**+32.88%**。
实际活动、执行行、padding和矩阵工作一致；这是稳定计算、复合VJP和同步检查的合计成本，未分别消融。
分阶段耗时、吞吐、权重/优化器/私有状态/激活与allocator峰值全部在本轮证据，不外推9B。

优先FP32，建议包含本轮 `8fd5216a` 及原 `97c52960` solver修复的当前实现。
默认训练入口和gather/rectangular保留；BF16次选。
五类记忆绑定全部保留，见[候选定义](memory-candidates.md)和[五类证据](evidence/memory-candidates-20261119.md)。
[KV128/TF32拆解](evidence/kv128-transformer-breakdown-20261119.md)及
[FFN/CPU/NPU对照](evidence/ffn-execution-devices-20261119.md)保留原身份；本轮不重跑9B或Transformer。
共同32层辅助参考不替代A–D，工程性能不代表下游任务质量。

## 范围与待办

只声明一阶局部VJP。普通native极端logits/伴随、任意相消与概率下溢、后续共享梯度累计、
被mask的极端中心化锚仍有限制，不能把范围报错一概说成真实最终导数不可表示。
无新9B训练、CUDA、多NPU或模型效果资格；下游安装和接受状态未变。
此前整图FP32更新超差、BF16支持跨界及FFN路径超差继续保留，见
[数值定位](evidence/cleanup-numerics-20261119.md)；本轮局部修复不能替它们认证。
可选工作在[ROADMAP](ROADMAP.md)，不自动继续旧矩阵。

## 执行与存储

本轮10服务及记录后代均核验终态；无本轮活动任务。CPU服务累计
2448.299/3600秒，NPU 395.834/1800秒（含等待），
均为冻结输入、background.slice、采样进程树RSS上限。NPU只用空闲独占队列，未共享占用卡。
审阅证据独立提交为 `96a5dda8`，随后按[退休收据](evidence/stable-mass-retirement-20261119.json)
删除本轮临时目录，退休文件逐路径分配大小合计约3.782 GB；整个仓库含Git和主私有环境现约 **148.4 MB**，低于200 MB。
清理后主私有环境隔离导入及源码归属复核通过。源码已普通推送；远端与交接提交同步状态以 `git status` 为准。
历史大载荷继续不在线；[archive index](evidence/archive-index.json)可校验还原审阅JSON，
不重建旧tensor。主私有环境、固定math、Git、下游和四份用户文件均保留。
