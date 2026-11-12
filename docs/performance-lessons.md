# 性能经验与适用条件

本页按问题机制导航，详细观察保留原源码、配置、结果和失败身份。
当前工作安排只在[ROADMAP](ROADMAP.md)，当前交接只在[STATUS](STATUS.md)。
历史提交定位见[迁移映射](history-migration.md)，具体复现见[reproduction](reproduction.md)。

## 如何使用经验

以层级、计算粒度和接口管理组合复杂度。已有依据的局部优化在适用条件成立、相关语义与
模块检查通过后即可采用，不要求每个辅助函数先做完整模型benchmark或遍历配置组合。
端到端对照检查集成效果与实际收益；计时结论限于测量范围。遇到具体反例时修正适用条件。

这些方法同时用于上层调度、局部模块、状态、反向、检查、搬运、结果发布和优化器。
接口可为性能、通用性和模块替换而重新设计；保持数学与必要公共能力，版本化迁移历史调用形式。
不同函数、不同返回范围和不同计时边界的对照分别解释，不能把全部差距当作可消除开销。
完整方法见[仓库优化技能](../.agents/skills/optimize-tide-execution/SKILL.md)。

| 正在调查的问题 | 详细观察 |
| --- | --- |
| GEMV/GEMM/BMM、计算共享、packing、元数据及完整返回 | [批量与数据搬运](performance-lessons-batching.md) |
| typed 状态、KV/区域/消息列贯穿、默认 B×L 输入与 raw 恢复 | [跨模块张量交接](performance-lessons-prefill.md#从模块批量到跨模块张量交接) |
| B×L窗口、KV前缀、查询/区域VJP、结果发布 | [prefill与KV布局](performance-lessons-prefill.md) |
| 归一化、近零梯度、共享归约、独立公式与舍入 | [数值与梯度](performance-lessons-numerics.md) |
| 缓存所有权、设备工作区、多卡、checkpoint与恢复 | [状态与设备](performance-lessons-state-devices.md) |
| 配置身份、安装、计时分母、归档与失败复现 | [测量与证据](performance-lessons-evidence.md) |
| 外部Transformer/MoE、真实模型入口及计时范围 | [外部模型对照](performance-lessons-external-models.md) |

新近机制：[来源列与精确计算复用](performance-lessons-batching.md#来源变换也要跨物理边保留张量粒度)、
[跨过批量接口的静态形状审阅](performance-lessons-state-devices.md#静态形状审阅要越过批量接口)、
[旋转的整数域与原值守卫](performance-lessons-batching.md#旋转恒等式需要整数域与原值守卫)、
[单bank矩阵与服务任务容量](performance-lessons-state-devices.md#单bank矩阵与服务任务容量是两类问题)、
[相容节点、真实成员与固定值检查](performance-lessons-prefill.md#相容节点实际成员与固定值检查)、
[设备flow与反向算子归因](performance-lessons-evidence.md#从设备利用率追到实际反向算子)、
[精度迁移与默认调用链](performance-lessons-numerics.md#精度迁移需要覆盖默认调用链)、
[跨设备共享列伴随](performance-lessons-numerics.md#跨设备共享列的反向到达顺序)、
[公共任务边界与卷积转置](performance-lessons-prefill.md#公共任务边界与卷积转置)、
[区域列伴随与结构支持](performance-lessons-prefill.md#区域列伴随与结构支持)、
[隐藏前缀bank与块伴随](performance-lessons-prefill.md#隐藏前缀bank与块伴随)、
[外层批、状态块与实际算子](performance-lessons-prefill.md#从外层批到状态块和实际算子)、
[并行与临时存储](performance-lessons-evidence.md#并行临时存储与真实算子归因)、
[紧凑公式与共享bank的正负结果](performance-lessons-prefill.md#从紧凑公式到共享bank矩阵)、
[Describe列与按用途裁剪](performance-lessons-prefill.md#describe列与按用途裁剪)、
[局部首次借用计费](performance-lessons-state-devices.md#局部首次借用与全局存储余额)。

每条新经验给出观察、机制、改动、正负结果、适用条件、反例/测试以及来源链接。
采用局部改进的依据与宣称端到端加速的依据分别记录；必要数学工作、重复计算、可摊薄的
执行开销和未判定成本分别识别。完整记录等价、指定投影等价与不同函数比较不混称。
经验本身不能代替正确性门禁，也不要求重跑已有有效证据。

## 已有章节入口

以下标题保留原链接锚点；每节详细内容已按机制归位，旧证据无需改写。

## 本仓库 core-3

[详细观察](performance-lessons-batching.md#本仓库-core-3)。

## 参考仓库

[详细观察](performance-lessons-batching.md#参考仓库)。

## Core-5 已有观察

[详细观察](performance-lessons-batching.md#core-5-已有观察)。

## 本轮迁移与负结果

[详细观察](performance-lessons-batching.md#本轮迁移与负结果)。

## 实际私有矩阵批与数值修复后的观察

[详细观察](performance-lessons-batching.md#实际私有矩阵批与数值修复后的观察)。

## 局部数值修复与全部 eager 低秩

[详细观察](performance-lessons-batching.md#局部数值修复与全部-eager-低秩)。

## 共享准备顺序、训练观测和显式精度

[详细观察](performance-lessons-numerics.md#共享准备顺序训练观测和显式精度)。

## 近零梯度与归一化反向的工程经验

[详细观察](performance-lessons-numerics.md#近零梯度与归一化反向的工程经验)。

## D64算术边界与跨步误差

[详细观察](performance-lessons-numerics.md#d64算术边界与跨步误差)。

## 外层汇合精度和默认异步的有限退出证据

[详细观察](performance-lessons-numerics.md#外层汇合精度和默认异步的有限退出证据)。

## 独立输入矩阵批的覆盖与完整成本

[详细观察](performance-lessons-batching.md#独立输入矩阵批的覆盖与完整成本)。

## 连续局部算术与完整桥接的停止依据

[详细观察](performance-lessons-numerics.md#连续局部算术与完整桥接的停止依据)。

## 单行VJP、异形传输与常驻反向的实测边界

[详细观察](performance-lessons-state-devices.md#单行vjp异形传输与常驻反向的实测边界)。

## 紧凑局部纤维与执行资源数量

[详细观察](performance-lessons-batching.md#紧凑局部纤维与执行资源数量)。

## 自然维度、区域合批和大文件

[详细观察](performance-lessons-batching.md#自然维度区域合批和大文件)。

## 配置覆盖与证据的校验

[详细观察](performance-lessons-evidence.md#配置覆盖与证据的校验)。

## 下游安装与恢复的验证边界

[详细观察](performance-lessons-evidence.md#下游安装与恢复的验证边界)。

## 失败经验与临时产物的生命周期

[详细观察](performance-lessons-evidence.md#失败经验与临时产物的生命周期)。

## CUDA 真机迁移的限定经验

[详细观察](performance-lessons-state-devices.md#cuda-真机迁移的限定经验)。

## GPU 推理批次与完整返回

[详细观察](performance-lessons-batching.md#gpu-推理批次与完整返回)。

## 三平台B128/B512与诊断开销

[详细观察](performance-lessons-batching.md#三平台b128b512与诊断开销)。

## 同模型放置与通信的实际边界

[详细观察](performance-lessons-state-devices.md#同模型放置与通信的实际边界)。

## 本轮三平台六项打包处置

[详细观察](performance-lessons-batching.md#本轮三平台六项打包处置)。

## 完整checkpoint与恢复的成本

[详细观察](performance-lessons-state-devices.md#完整checkpoint与恢复的成本)。

## 八卡终态补充

[详细观察](performance-lessons-state-devices.md#八卡终态补充)。

## 终态产物与源码保全

[详细观察](performance-lessons-evidence.md#终态产物与源码保全)。

## 外部语言模型吞吐的可比性

[详细观察](performance-lessons-evidence.md#外部语言模型吞吐的可比性)。

## 外部Dense/MoE入口的初始实测经验（2026-11-02）

[详细观察](performance-lessons-external-models.md#外部densemoe入口的初始实测经验2026-11-02)。

## Dense/MoE后继的有限诊断（2026-11-03）

[详细观察](performance-lessons-external-models.md#densemoe后继的有限诊断2026-11-03)。

## Dense/MoE后继终态的正负经验（2026-11-04）

[详细观察](performance-lessons-external-models.md#densemoe后继终态的正负经验2026-11-04)。

## 正支持数值模式与有限域

[详细观察](performance-lessons-numerics.md#正支持数值模式与有限域)。

## 公共基线迁移与资格边界

[详细观察](performance-lessons-state-devices.md#公共基线迁移与资格边界)。

## TL-REQ-004密集查询与异形记忆分离

[详细观察](performance-lessons-batching.md#tl-req-004密集查询与异形记忆分离)。

## 联合KV准备的实际收益与剩余边界

[详细观察](performance-lessons-batching.md#联合kv准备的实际收益与剩余边界)。

## 张量窗口跨owner的语义边界

[详细观察](performance-lessons-state-devices.md#张量窗口跨owner的语义边界)。
