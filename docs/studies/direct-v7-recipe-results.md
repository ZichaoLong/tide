# direct-v7配方：fan-in成立，最终RMSNorm没有赢得短程筛选

**问题。** 在普通Torch直接HB图中，固定std0.15和最终读出尺度是否影响学习？
源码d30559c，Wikitext103固定训练/validation，4096 BPE、L128、B32、FP32，seed11。
所有候选262,144有效训练目标、同32,768目标验证，始终使用4M LR日程；不是各自最佳超参。
本实现无TIDE路由、payload、Read/Describe和发送语义，不能直接诊断旧TIDE整体函数。

| 直接模型 | std0.15 / 无norm | std0.15 / RMSNorm | fan-in / 无norm | fan-in / RMSNorm |
| --- | ---: | ---: | ---: | ---: |
| HB40 | 6.8665 | 6.8568 | **6.3921** | 6.7762 |
| HB88 | 6.9284 | 6.8718 | **6.4003** | 6.7526 |

数值均为validation NLL，越低越好。fan-in按投影真实输入维度使用std=1/sqrt(d_in)，
RMSNorm仅作用于最终hidden，epsilon1e-6，其余配方不变。
在无norm条件下，改fan-in分别改善0.4744、0.5281 NLL；在fan-in条件下加最终norm反而损失0.3841、0.3523。
选择规则取两尺寸平均，因而选中**fan-in、无最终norm**，没有为了符合假说改阈值或换问题。

最终norm确实让std0.15模型的初始NLL从HB40/88约9.234/9.312回到8.344/8.340，
但它没有带来这次短程最优学习。初始输出校准和训练后的质量是两件事。
交互效应、LR/梯度尺度与更长曝光仍可能改变排序；本轮没有对四种配方都跑4M。

Transformer12/28层的原位置尺度NLL为6.9524/6.9516，平衡尺度为6.9432/6.9512。
差异都很小，按预设0.02平局规则选平衡尺度；**没有证明位置尺度已经修复Transformer配方**。
后续[主训练](direct-v7-scale-results.md)显示其容量排序仍不理想，需要独立改进对照。

**验收与边界。** 12格均完成训练、全验证、有限greedy推理和Lens读回；因果干预、绑定权重/optimizer及
新进程三步恢复通过。CPU/NPU严格1e-6/1e-5的部分前向/参数VJP仍有限失败，原阈值未改；
没有将学习效果当作数值等价。筛选仅一seed、共享validation、有限候选，不是独立test确认。

**建议。** 在直接图的后续容量研究复用选中配方；迁移到TIDE需按角色、分配/读取语义重新桥接。
这里支持“这套直接模型配方能改善短程训练”，不支持“最终norm一般有害”或“旧TIDE问题已修复”。
原std0.15与未选norm结果、旧HB88负面结果和Transformer反证全部保留。

[原始检查/配置/运行/Lens映射与逐块统计](../evidence/direct-v7-observations.json) ·
[研究表](direct-language-v7.md) · [查看与复现](direct-v7-operations.md)。

Lens：项目`tide-learning-lab`，group `TL-DIRECT-RECIPE-v7`，
`tl-v7-screen-hb40/hb88-<std015|fan_in>-<raw|rms>-s11`及4条`tf12/tf28-<original|balanced>-s11`。
