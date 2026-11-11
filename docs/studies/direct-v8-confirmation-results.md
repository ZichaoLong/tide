# v8 有限种子复核：结论

问题：按预声明类别选出的seed11候选，在新初始化seed22上是否保持方向？
controller从157个完整首轮结果中选19项：FC10、HB9；HB递推形状类无合格项，所以没有凑足20。
配置无checkpoint继承，沿用1M目标、260更新及共同validation。18完成、1失败；全部选择/拒绝依据保留。
逐项参数、配置、记录、配对文档区间见[审阅JSON](../evidence/direct-v8-results.json)。

| 候选 | seed11 NLL | seed22 NLL | 均值 |
| --- | ---: | ---: | ---: |
| FC anchor | 5.3621 | 5.3709 | 5.3665 |
| FC budget-div1 | 5.3326 | 5.3613 | 5.3469 |
| FC textbook-mixed | 5.2149 | 失败 | — |
| FC delta-state | 5.2980 | 5.3262 | 5.3121 |
| FC FFN8x | 5.2800 | 5.3025 | 5.2912 |
| FC Read-zero | 5.3554 | 5.3721 | 5.3637 |
| FC SSD mem-off | 5.4246 | 5.5067 | 5.4657 |
| FC temperature1 | 5.3113 | 5.3090 | 5.3102 |
| FC direct-attention | 5.2454 | 5.2824 | 5.2639 |
| FC source-adapted | 5.3618 | 5.3875 | 5.3746 |
| HB anchor | 5.3854 | 5.4213 | 5.4033 |
| HB budget-div1 | 5.2915 | 5.3087 | 5.3001 |
| HB textbook-KV | 5.4073 | 5.3790 | 5.3931 |
| HB gated-delta/sum | 5.3323 | 5.3784 | 5.3554 |
| HB FFN8x | 5.2939 | 5.3847 | 5.3393 |
| HB Read-old | 5.3685 | 5.3778 | 5.3731 |
| HB temperature1 | 5.2871 | 5.2881 | 5.2876 |
| HB pooled-attention | 5.2136 | 5.2044 | 5.2090 |
| HB source-adapted | 5.3915 | 5.3935 | 5.3925 |

FC direct-attention相对anchor两seed分别改善0.1167/0.0885 NLL，HB pooled-attention改善0.1718/0.2168。
各自配对文档bootstrap区间均不跨零；区间只条件于固定seed/checkpoint及被选择的validation，不包含模型选择或seed总体不确定性。
对照[已调参Transformer](direct-v8-transformer-results.md)，TF6/12双seed均值为5.7874/5.8340，
相应FC/HB候选分别为5.2639/5.2090。参数为4.955M/11.596M，对照5.792M/10.530M，只有近似匹配。
TIDE候选和TF使用不同架构及各自有限选择配方，搜索预算也不同；这支持本轮1M短曝光质量比较，不能声称普遍架构优势。

quadratic温度1是后续真实稀疏的明确候选：FC/HB两seed平均NLL为5.3102/5.2876，
相对anchor均有同向改善；其validation逻辑Full/候选比例分别约58.0–60.3%和59.7–61.6%。
HB相对pooled-attention仍差0.0736/0.0837 NLL，而pooled-attention逻辑比例100%。
当前二者都物理计算全部Full，因此此处只支持质量/逻辑支持的取舍，尚不能称为真实稀疏或对应吞吐收益。

FC Read-zero、来源适配、HB textbook-KV的两seed方向并不一致，不能普遍推广；
FC SSD mem-off两seed都落后anchor，且并不等于关闭所有Describe记忆。FFN8x增加参数（FC8.239M、HB19.804M），
不能把其收益直接归给局部语义或称为已建立规模规律。
FC mixed在距离1M仅1,584目标时失败，仍然是失败；不补填最终NLL、不拿最近checkpoint代替完整对照。

本批保留严格CPU/NPU数值失败与小型恢复边界。推荐把FC direct-attention、HB pooled-attention作为后续质量参照，
FC/HB温度1作为稀疏候选；单独补齐未完成数值路径。新任务尚未提交，test仍密封。
