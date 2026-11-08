# 语言v3终态：小曝光下的质量与实际稀疏折衷

2026-11-09主代理审阅，待用户验收。A于北京时间00:47、B于00:42完成；七候选均完成128次更新、独立test与推理，全部29个指标投影verified/readback_matches。实际cgroup和已知后代均已终止，当前无活动/排队作业。本次检查没有追加训练。

## 结果

同路线固定WikiText文档、词表、L32、B8、seed11和采样顺序。A各32,677、B各32,503有效训练目标；独立test为16文档/4,096目标，validation为8文档/2,048目标。NLL越低越好，**A/B词表和预训练背景不同，表间不直接比较**。

| A：从零训练 | test NLL | test PPL | test Full事件 |
| --- | ---: | ---: | ---: |
| TIDE dense | 6.779286 | 879.44 | 16,384 |
| TIDE sparse | 6.802554 | 900.14 | 8,236 |
| 参数接近的小Transformer | 7.022139 | 1,121.18 | 不适用 |
| train-only unigram | 6.946099 | — | 不适用 |

| B：预训练模型的最后MLP | test NLL | test PPL | test Full事件 |
| --- | ---: | ---: | ---: |
| 冻结原模型 | 4.361484 | 78.37 | 不适用 |
| 原MLP继续训练 | 3.684206 | 39.81 | 不适用 |
| 简单MLP替换 | 4.147179 | 63.26 | 不适用 |
| TIDE dense完整替换 | 4.234056 | 69.00 | 16,384 |
| TIDE sparse完整替换 | 4.236845 | 69.19 | 8,351 |

稀疏减去稠密的NLL差：A **+0.023268**，B **+0.002789**；配对文档bootstrap单侧95%上界分别0.037871/0.011343，均低于原0.05容差。Full事件分别减少**49.73%/49.03%**。这是当前checkpoint、文档子集和单种子下通过的质量/活动筛选；不是正式跨种子非劣或通用语言能力认证。区间采用整文档重采样20,000次、seed20261109，不含训练种子不确定性或多重比较校正。

A的两种TIDE均优于当前小Transformer与unigram，但小Transformer尚未超过unigram，PPL也很高，不能据此宣称普遍架构优势。重置早期上下文使TIDE dense/sparse NLL增加0.038864/0.046996，说明有有限短上下文贡献；zero-core增加约1.48/1.46，说明训练模型依赖该分支，不证明其独特性。

B在alpha=1替换时优于冻结原模型（稀疏ΔNLL=-0.124639），完整模型的小型推理检查确认包装器中的旧MLP零调用。**它仍落后于同预算继续训练原MLP（+0.552639）及参数接近的简单MLP（+0.089665）**。因此局部承接与限定保持成立，TIDE相对简单替换的架构优势未成立。

## 六轴验收和范围

| 轴 | 审阅结果 |
| --- | --- |
| 执行 | 29阶段完成、7训练均128步；独立test与推理产物齐全 |
| 严格数值 | A两个训练后TIDE的小型core/分块/streaming检查通过；B五模型full/cache均在原atol1e-5/rtol1e-4下有限失败。B dense core另有1个梯度元素超差：1.3933e-6，对应阈值1.0186e-6；原因未知，保留失败 |
| 离散/机制 | 所测因果干预、TIDE离散支持/未来VJP检查通过；B五模型full/cache argmax一致。只覆盖记录的小型输入 |
| 恢复 | 复用七候选B2/L3三步新进程精确恢复；本次7个终点checkpoint均CPU解码、有限张量、optimizer/RNG/游标/身份核对通过；未执行长轨迹独立恢复比较 |
| 学习 | 七候选验证NLL均下降；独立test和原0.05质量容差单列；没有把退出码或严格数值资格当成学习结论 |
| 性能/结构 | Full约减半，但candidate/Upd/Read不减；FC4单区域、内部消息0，无多区域/规模证据。训练时间未呈同比例改善，不影响上述质量/活动筛选 |

Full是公共执行器实际执行事件，不是mask标签；它也不是总FLOPs。B训练采用冻结前缀cache，不能将尾部计时称为整个Qwen吞吐。16个采样更新的A dense/sparse平均约13.51/14.07秒，B约13.65/13.58秒；有限观测不是正式后端加速基准。

## 身份、成本与查看

执行源码`6cd4f856df02e9c90645198cf5998a27a877b4ec`，只读快照`.local/source-language-explore-v3`，树SHA256 `2c44b51421077624867ee27fa6fa6c72d8c95e60a28736be2565083260da8e77`已复核。数学`0770593838a37c790a568226930567ed60f99b83`、foundation`85d161d5c9148eac2b164f89a0ba4204b4331898`、248安装文件保持一致；Torch2.12.0+cpu/TorchNPU2.12.0.post3/CANN9.2/Transformers4.57.1。配置[fc-explore-v3.json](../../configs/studies/language/fc-explore-v3.json)，每路线resolved配置与完整argv/hash保存在原作业记录中。

v3保守累计占卡预算扣8,438.933秒；含v1/v2旧失败共**11,990.574秒=3.3307NPU小时**，原8小时余4.6693小时。共同截止仍为北京时间11月9日07:28:46；剩余额不是新作业授权或延长截止。当前不自动启动FC8、第二seed或下一批。

[原始终态与配对统计](../evidence/language-v3-terminal.json)、[checkpoint检查](../evidence/language-v3-checkpoint-inspection.json)、[查看/停止/恢复](../RUNNING.md)。Lens项目`tide-learning-lab`，A/B运行前缀`tl-lm-a-language-v3-`/`tl-lm-b-language-v3-`，29个对应位于终态JSON的lens；私有入口`.local/LENS.md`。研究图离线页面为`web/index.html`。

一页结论：[数据](TL-LM0.md)、[接入与正确性](TL-LM-EXEC.md)、[A语言](TL-A-LM.md)、[B承接](TL-B-LM.md)、[规模未检查](TL-LM-SCALE.md)、[联合审阅及下一步](TL-JOINT.md)。后续选项须经用户验收后再安排，不靠改问题获得成功标签。
