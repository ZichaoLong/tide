# TL-REQ-007 当前实例基座的 Shannon 正支持与相关修复合入清单

2026-11-20，实验仓库 `tide-learning-lab` 提出，目标分支为 `ZichaoLong/tide:graph-execution-foundation`。A 阶段 r09 在区域 Shannon 分配处停止。用户已同意由实例基座实现选择性补值及其一阶反向，实验仓库负责真实任务接入和接续验收。[故障与方案](../pilot-r3-r09-shannon-underflow.md) · [基座协作约定](../../../graph-execution-foundation/docs/downstream-contract.md)

## 实现归属与交付状态

Shannon 分配、端点导数、正支持和接收这些模型机制由实例基座维护。实验仓库维护任务读出、数据、配方、调度、故障样本和验收程序。冻结程序从所选基座复制 `src/tide` 并校验每个文件。r09 的训练进程直接导入这个副本。R2 留下的 `.patch` 保存了精确上界修复的原始差异，R2 审计按文件名和哈希引用它。r09 已含这项改动，启动时应用补丁的次数为零。[冻结入口](../../pilotlab/prepare3.py) · [R2 审计附件](../evidence/pilot-r2-shannon-repair.json)

新的正支持策略已在 `graph-execution-foundation/src/tide` 实现。下游对基座的 26 个 Python 文件按内容固定版本，通过原训练 worker 的模型配置启用 `shannon_support_policy="positive-floor-v1"`。整个验收过程使用基座公开 API。冻结源码 SHA256 为 `cbf9c33157bd744e240fc4e662862cd147a5ba78861452838c053e8945d1e4f4`。11 月 20 日完成验收时，远端基座分支头为 `b3a02979`，本机分支头为 `049ecdcb`。[基座接口与实现范围](../../../graph-execution-foundation/docs/shannon-positive-floor.md) · [完整文件清单及验收记录](../evidence/pilot-r3-r09-foundation-shannon-acceptance.json)

实现现已发布到远端 `ZichaoLong/tide:graph-execution-foundation`。实现提交为 `885f7ef96be8cc31814e509c687017e6ff922d72`，文档提交为 `2906d544b8a51297a0c09b96f1cee998d94eb84c`。已回读远端分支头，并将 Git 对象中的 26 个运行时文件逐一与上述验收哈希对照。两次提交的运行时源码相同。[实现提交](https://github.com/ZichaoLong/tide/commit/885f7ef96be8cc31814e509c687017e6ff922d72) · [交付记录](https://github.com/ZichaoLong/tide/commit/2906d544b8a51297a0c09b96f1cee998d94eb84c)

## r09 已经运行的基座改动

北京时间 22:56 核对了远端基座分支头 `b3a02979276fbd1ca448318ef05c7e3a4533c67b`、本机基座提交 `049ecdcbc39c80297b37a99591906acbefa1bc13` 和 r09 实际采用的 r08 冻结 worker。冻结基座 `src/tide` 的 25 个 Python 文件均与本机基座工作树逐字节相同。远端已含 `b4a8d937` 的弱权重稳定求值及 `3efcc6b1` 的 NPU 算术适配。当前数学快照为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。[本机与冻结版本记录](../evidence/pilot-r3-r09-shannon-underflow.json)

| 应进入远端基座的修改 | 目前位置及配套回归 |
|---|---|
| Shannon 精确命中候选上界时保留上界值和边界导数 | 本机基座提交 `049ecdcb`，含 `src/tide/allocation.py` 和 `tests/test_shannon_cap.py`。将实验仓库唯一的独立源码补丁施加到其父提交 `b3a02979`，得到的运行时文件与 `049ecdcb` 逐字节相同。可直接采用含测试的提交。[原补丁](../../patches/pilot-round2-shannon-cap.patch) |
| Shannon 活跃约束判断、补偿求和及可表示弱权重的耦合梯度 | 本机基座未提交的 `src/tide/allocation.py`，对应 `tests/test_shannon_active_set.py`。该修复保住此前约 `5.89e-8` 的真实弱项。[基座说明](../../../graph-execution-foundation/docs/shannon-active-set.md) |
| Reception 的结构填充掩码，区分缺失边和有右导数的真实零质量边 | 本机基座未提交的 `src/tide/config.py`、`src/tide/model.py`。配套 `tests/test_reception_padding.py`、`tests/test_reception_execution.py` 及示例 CLI 的旧配置字段说明。[下游回放](../pilot-r3-reception-padding-repair.md) |
| Attention 的单有效键、空行及多键梯度消减 | 本机基座未提交的 `src/tide/attention.py`，包含 tiled 单支持梯度、NPU auto/fusion 的混合行反传和 FP32 多键公共 V 中心化。配套 `tests/test_attention_singleton.py`、`tests/test_attention_native_support.py`。[tiled 见证](../pilot-r3-attention-risk-screen.md) · [NPU 验收](../pilot-r3-native-auto-repair.md) |

本次在基座工作树的 CPU 环境重跑了 `test_shannon_cap.py`、`test_shannon_active_set.py`、`test_reception_padding.py`、`test_attention_singleton.py`，共 89 项通过。NPU 融合路径沿用上表所列固定源码的真机证据。上表列出与本周期数值和执行修复有关的文件，基座工作树还包含其他主题的未提交文档。

## 修复前的故障和基线

实际故障发生在 `multiscale_settle-d128-k-singleton7` 的 A2，累计 1,255 步后的下一次前向。CPU 和原 4 号 NPU 在第 9 次区域分配出现相同的 23 个丢失坐标。当前的实验原型只覆盖双候选、两端上界均为 1、总量为 1 的情形，用进程内替换接入了一次真实故障 batch。它给出了可迁入基座测试的行和参考梯度。模型前反向在两个后端完成，优化器更新数为 0。[23 行原始数据](../evidence/pilot-r3-r09-shannon-underflow.json) · [补值与梯度对比](../evidence/pilot-r3-r09-shannon-floor-comparison.json) · [实验原型](../../scripts/probe_r09_shannon_floor.py)

在含上述活跃约束修复的基座工作树，从仓库根目录、已激活的 PyTorch 环境，可用公开入口直接复现。这里给 `w`、`mu` 取全 1，以重现原行相同的角色界和总量。

```sh
PYTHONPATH=src python - <<'PY'
import torch
from tide.allocation import allocate

for dtype in (torch.float32, torch.float64):
    scores = torch.tensor([-73.42066192626953, 32.303375244140625], dtype=dtype, requires_grad=True)
    ones = torch.ones_like(scores)
    try:
        c = allocate(scores, ones, ones, torch.zeros_like(scores), ones, 1., kind='shannon')
    except ArithmeticError as error:
        print(dtype, str(error))
        continue
    root = torch.tensor([1e30, 0.], dtype=torch.float32).to(dtype)
    ds, = torch.autograd.grad(c, scores, root)
    print(dtype, c.tolist(), ds.tolist())
PY
```

本次复核，FP32 的 strict 路径报 `Shannon allocation underflow would erase a positive sender`。CPU FP64 弱项为 `1.2151616428374501e-46`，分数梯度为 `[+1.215161661122554e-16, −1.215161661122554e-16]`。选择性补值的预期 FP32 输出是 `[2⁻¹⁴⁹, 1]`。强候选在实数解中仍小于上界，只补前向会把两个分数梯度都算成零。[独立 120 位 Decimal 参考及仅改前向的反例](../evidence/pilot-r3-r09-shannon-floor-comparison.json)

## 原验收条件

1. 在公开的模型配置和角色约束 Shannon 分配入口提供显式数值策略。strict 继续用于严格对照。补值只作用于数学上为正而最终 FP32 舍入为零的坐标，检查单项上界和按行的补入总量，并给出可查询的补值计数。结构零、精确端点和已可表示的正数维持各自语义。[旧基座策略](https://github.com/ZichaoLong/tide/blob/74e7bbd62341d2ccd382194a2c52ad4fea033665/docs/positive-support-policy.md)
2. 用补前的响应和实际约束计算一阶梯度。基座回归应区分舍入到上界的内部解与真正饱和的上界，覆盖可变总量、非零角色下界、零上界、舍入边界和广播输入。核对补后质量在接收聚合、注意力和 KV 中的行为，包括多个极弱发送者汇合时按存储值计算的比例。
3. 交付可定位的基座提交和策略身份，注明适用的设备与精度。下游在新的冻结版本上重放故障 batch，用真实后继输入比较 AdamW、私有状态及同策略新进程恢复。旧 strict 检查点按显式策略转换登记。任务读出、数据、训练调度和实验诊断由实验仓库维护。

## 2026-11-20 的下游验收

用两端各自捕获的 23 行独立核对，所有应补坐标均存入 FP32 最小正次正规数 `2⁻¹⁴⁹`。弱、强分数与总量的一阶导数，相对双候选闭式解的最大相对误差分别在 CPU 为 `8.35e-8`、NPU 为 `1.10e-7`。原故障批在两端用默认 strict 均复现原异常，用显式新策略均完成优化器更新。[逐行参考和同源对照](../evidence/pilot-r3-r09-foundation-shannon-acceptance.json)

| 实际更新步 | CPU / NPU 补值坐标 | CPU / NPU loss | 数据和状态 |
|---|---|---|---|
| 1256 | 23 / 23 | 5.55780554 / 5.55780506 | 从正式 step 1203 的数据游标前推，输入和目标与故障捕获逐位相同 |
| 1257 | 56 / 56 | 5.98353863 / 5.98353720 | 原数据流的下一个 batch，按文档边界重置私有状态，完成后保存新策略检查点 |
| 1258 | 111 / 111 | 6.21113539 / 6.21113682 | 使用上一步 KV，连续执行与新进程恢复各做一次更新 |

每步有 123 个参数梯度，模型、AdamW、私有状态和梯度均有限。新进程加载 step 1257 的模型、AdamW、私有状态、数据流、RNG 和曝光计数逐位一致。随后 step 1258 的输入、重置标志、补值计数和私有状态在两端各自与连续运行一致。CPU 整步逐位一致。单卡 NPU 的参数、AdamW 和裁剪后梯度最大绝对差分别为 `1.86e-9`、`9.31e-10`、`3.73e-9`，loss 差 `4.77e-7`，通过原有 `atol=1e-5, rtol=1e-4`。[逐字段比较](../evidence/pilot-r3-r09-foundation-shannon-acceptance.json)

故障捕获标记为不可直接恢复，缺少 RNG 和数据游标。上述三步以验证过的原输入、精确加载的捕获参数、AdamW 和私有状态，以及显式初始化的诊断 RNG 为起点。另在 NPU 从正式 step 1203 检查点精确加载全部状态，显式切换策略后成功执行真正的 step 1204。该检查点到故障步之间的整段训练轨迹仍属未验证范围。新策略把异常行送到主机复核，本次未测训练吞吐及长期质量。A 仍停在 r09，B 关闭。[验收程序](../../scripts/accept_r09_shannon.py)

在已加载 `ascend/dev-workspace-9.2.0` 的实验环境，可在本仓库执行以下 CPU 验收。NPU 使用同一命令与内容哈希，在队列分得一张卡后指定 `--device npu:0` 和新的输出目录。

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 \
  .venv-pilot-r3/bin/python scripts/accept_r09_shannon.py \
  --foundation-src ../graph-execution-foundation/src \
  --sha256 cbf9c33157bd744e240fc4e662862cd147a5ba78861452838c053e8945d1e4f4 \
  --device cpu --output .local/pilot-r3/tl-req-007-cpu-repeat
```
