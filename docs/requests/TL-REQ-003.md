# TL-REQ-003 — 正支持与有限精度训练的可执行契约

状态：2026-11-09基座交付74e7bbd的`positive-floor-v1`已通过下游限定验收，见[报告](../studies/positive-support-acceptance-results.md)和[证据](../evidence/positive-support-acceptance-v1.json)。CPU原HB16边界/FC16干预、新模式恢复及CPU/NPU小型接入完成；原严格数值失败保持。完整log/缩放载荷暂缓，不阻塞当前稠密比较。基座保持只读，没有发送外部消息。

提出方为`ZichaoLong/tide`的`tide-learning-lab`分支；权威原文为本文件。原实验源码`aa76da565a4612b8d009a7abedbbd1187ef0c95a`，有界重放源码`9ebbc63784a29f2c785bb7568ddb9da63b63dd55`。研究和后续审阅见`docs/studies/allocation-support-v5.md`。

## 目标、材料和固定依据

目标是继续小型语言学习与实际稀疏/局部结构研究，不要求无限提升浮点精度。请求区分已有能力的正确使用、当前绑定的可表示域、未来表示/近似契约；不以吞掉严格断言为验收。

- 数学`0770593838a37c790a568226930567ed60f99b83`，foundation `85d161d5c9148eac2b164f89a0ba4204b4331898`，wheel0.3.0，SHA256`c40ba439d036f8b9d4149153aea8b89358fcebc2ac019c23b5c1e5b09587e075`；来源、wheel、安装和环境锁位于`dependencies/`。
- 实际Torch2.12.0+cpu、TorchNPU2.12.0.post3/CANN9.2；本次HB诊断只用CPU1/FP32，Python公共packed eager单host worker。
- 配置`configs/studies/language/a-hb16-cpu-v5.json`：HB16/4区域/width32/KV-qry/cat-ind/regional_total/Shannon/温度0.25，B8/L32、seed11，AdamW、128更新/512日程前缀。
- HB16 dense在86更新后失败，最后完整checkpoint64；浅宽FC4 NPU在56更新后同类失败；FC16 CPU训练128和正常test/推理完成后，context-reset干预评价同类失败。失败均为`positive allocation batch support underflowed in requested dtype`。
- 原始材料：`runs/lm-a-hb16-v5/train-tide-dense-s11/`，`runs/lm-a-controls-v5/train-tide-dense-s11/`，`runs/lm-a-fc16-v5/context-reset-tide-dense/`。原阈值和失败保留。
- 上游共同架构§4.1定义活动集为数学`c>0`；区域分配§S.3及实现检查§3.3区分数学零、浮点下溢零和人为裁剪零。现有基座按此严格拒绝，不能先认定为实现bug。

基座已经提供公共`AllocationFailure.record()`和有界调用观测；原实验入口遗漏异常张量落盘。该下游缺口已在新源码补齐，不请求基座重复实现。现有resident KV也区分“正接收后存储舍入为零”与“没有接收”，但该局部资格不认证本请求的Shannon图执行。

## 已取得的最小实际见证

[完整行与独立解](../evidence/hb16-positive-support-v5.json)来自第64步checkpoint重放，第72/80步loss和梯度范数与原记录一致，第87步再次失败，用时571.69秒。nodes=`[8,9,10,11]`、time=32、row=55；scores约`[4.610298,-0.114485,-21.564291,14.224824]`，caps全1，B=1.9164364337921143，tau=0.25。节点10的数学分配约3.1058796574e-46，低于FP32最小subnormal的一半。

80位Decimal独立闭式解验证了节点11饱和、其余三项内部解的全部面条件；与公共FP64单行相符。原FP32 scalar复现batch拒绝，残差1.1921e-6仍在基座3.0518e-5容差内，故不是总量求解残差超差。FP64及FP32的2倍/4倍温度单行探针通过，但没有完整训练/反向/恢复资格。另有[合成FP64反例](../evidence/shannon-fp64-range-counterexample.json)，证明仅换dtype不能保证任意评分范围。

本次定位已完成，原失败仍为失败。训练中评分差的具体形成路径、其他历史失败原因尚未全部区分。请求据此讨论可执行数值域与结构支持表示，不要求基座为这条超出FP32范围的精确正值伪造普通FP32解。

## 协同请求

1. **近期已有数学下的训练建议。** 对四候选真实失败行区分正常表示域下溢与求解器偏差；提供或确认公共评分尺度/温度/容量诊断，尤其`(s−λ)/τ`、最小正cap、控制对数和有效支持。不要求每个正式训练步骤同步导出张量。已有归一化match的有界部分可复用，但alpha_res残差、可训练bias以及区域query也须纳入范围分析。
2. **保留严格入口，评估另一种表示。** 可考虑结构支持独立于强度存储，正强度用log或尾数/指数表示。必须贯通SelStep、广播、Agg、零接收、KV追加、状态/质量、VJP及checkpoint。只保留bool支持却把后续正接收当成数值零分支并不充分。更换表示目标仍可为原数学；若数值近似影响后续分支/导数，要明确新的执行契约。
3. **由上游裁定的可选模型。** 若采用近零删除、正下界、稠密混合或新评分函数，应另立模型绑定和导数约定。不能把epsilon-clamp或dtype隐式flush称为原Shannon的严格实现。近零删除要有误差/工作量口径；“质量很小”本身不是整网影响小的证明。
4. **实验侧对照。** 允许先用公开proportional建立不依赖指数尾部的稠密参照，同时保留Shannon原失败。但proportional的总量/评分梯度与regional_total不同，只能作为附加控制。若要归因正则/稀疏差异，采用共同的稳定评分配方重训Shannon/quad配对，不拿旧cat-quad和新match-Shannon作单变量结论。

## 有界验收与停止条件

- 原失败行在原FP32严格入口仍应忠实失败；新范围配方或新表示另存身份。FP64行通过仅说明该行可表示，不能认证完整模型或任意后续参数。
- 已可表示域上以独立高精度数值/手算VJP核对，保留原1e-6/1e-5阈值、共享根、None/零梯度和边界约定。
- 新表示先用一个完整区域和一个两区域短序列，验证活动/消息存在性、零接收中性、KV元数据、完整梯度、optimizer和新进程checkpoint；不直接要求全后端矩阵。
- 再由实验侧在独立私有安装、固定同任务小配方验收学习和工作量。NLL、实际Full/消息、质量范围及端到端时间分别报告；硬件比例加速不是先决条件。
- 本次原重放最多1800秒、1 CPU线程/8GiB、32更新，计入原24计算worker小时及22:49共同截止；没有为基座开发或上游改写另行分配资源。

单行重放入口：`python scripts/replay_allocation_row.py --record <捕获JSON> --dtype float32 --output <新JSON>`；FP64或温度倍率是另一个探针。完整重放配置为`configs/studies/language/hb16-underflow-v5.json`，实际作业`lm-hb16-underflow-v5`。成功捕获是诊断完成，不改原失败状态。

基座交付：[固定74e7bbd回复](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/docs/evidence/positive-support-tl-req-003-20261109.md)及[数值策略契约](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/docs/positive-support-policy.md)。固定数学0770593未变；下游接受的是声明的有限精度前向/一阶VJP及所测使用范围，不是新熵函数或全平台严格资格。上方原请求与反例保留来源，不再把所有宽范围方案作为当前必做项。本请求不扩大跨仓库修改、推送或消息权限。
