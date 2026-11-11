# direct-v8 终态交付

2026-11-11 19:24北京时间复核：本批于18:49结束，40个后台服务全部终态、无存活后代，当前没有本批活动或排队作业。
[固定研究表](direct-coverage-v8.md) · [首轮广覆盖结论](direct-v8-coverage-results.md) ·
[有限种子复核结论](direct-v8-confirmation-results.md) · [Transformer调参结论](direct-v8-transformer-results.md)。

| 部分 | 已提交 | 完成完整1M及验证/推理 | 原失败 |
| --- | ---: | ---: | ---: |
| FC/HB首轮候选并集 | 182 | 157 | 25 |
| 局部候选seed22 | 19 | 18 | 1 |
| 单项quad修正r2 | 1 | 1 | 0 |
| Transformer调参与seed22 | 18 | 18 | 0 |
| 合计 | 220 | 194 | 26 |

没有因外层worker退出零而忽略内部失败。完整运行均有32,768目标独立训练划分validation及有限输出推理记录；
194项的初始/最终评价在相同block和目标数交集上的NLL均下降，支持该范围内的学习。
生成检查使用短greedy全前缀重算，只验执行/有限输出，不能据此宣称开放式生成质量已足够。

最清楚的复核方向是FC direct-attention、HB pooled-attention以及quad temperature1；
两seed结果及近参数TF对照详见复核页。广覆盖没有证明所有局部候选都可训练，递推形状组存在明显执行缺口。
相对坐标求根仅有一个成功的具名修正复跑；25个其余未完成身份不因它而自动变为已修复。

## 可复现身份与分轴验收

TF固定源码2702d91，校准/anchor为778f9e9，广覆盖/确认5db28ae，单项修正cd85637；
数学0770593838a37c790a568226930567ed60f99b83，规划/安装74e7bbd，私有环境`.venv-positive-floor-v1`。
解析配置位于`configs/studies/language/direct-v8/`，包括controller产生的19份confirmation配置。
每项配置SHA已与实际训练identity核对；data lock一致，安装身份保留在每个stage，未采用基座新优化。

220项均有接入记录及三步新进程同栈精确恢复通过；这不认证整条260步轨迹的独立重放或跨后端恢复。
220项均保留至少一个CPU/NPU严格数值失败（原1e-6/1e-5阈值），不得把可学习标成严格等价通过。
小型因果、共享optimizer、支持及公式VJP范围独立记录；实际26次分配可行性失败保留原2e-5阈值，范围约2.04e-5至0.652。
历史失败result中的部分axes文字停留在running；外层/内层终态和错误记录证明它们已经失败，审阅JSON明确列execution及原recorded_axes，未修改原运行。
固定静态单位输入函数、positive-floor策略、实际稠密Full及公共continuation未认证边界见[实现说明](direct-v8-implementation.md)。

## 资源、记录与查看

监督区间64.7分钟；保守计费8.204 NPU卡时、20.289 CPU核时（含NPU host及2核时开发预留），
新增产物约54.28GiB，终态配额余量约577.57GiB；均低于240/768核时、128GiB新增及200GiB余量上限。
计费包含设备服务等待/启动，不能当作纯训练kernel用时或跨实现加速比。训练和评价分项时间见逐配置记录。

220个训练投影均verified/readback_matches，包括失败运行。Lens项目`tide-learning-lab`，
`TL-DIRECT-TF-v8`查看调参，`TL-DIRECT-COVER-v8`查看首轮与r2，`TL-DIRECT-CONFIRM-v8`查看种子复核。
私有实例与SSH查看方式仅在本机`.local/LENS.md`；没有部署或改变现有服务。
[运行/来源/Lens对应](../evidence/direct-v8-observations.json) · [质量/配对统计/工作量/失败](../evidence/direct-v8-results.json) ·
[研究图](../../web/index.html) · [日志、停止和恢复方法](direct-v8-operations.md)。

只读重建：`python scripts/review_direct_v8.py`、`python scripts/review_direct_v8_results.py`。
项目记录为权威，查看不重跑训练；本批已关闭自动提交。重启/延长训练必须新身份并另列预算。

## 后续可选包（建议，尚未提交）

以下是本批终态时的历史建议。用户随后明确允许稠密遮罩继续训练，并要求更实质的数据/容量/活跃比例阶梯；
当前建议已由[v9联合阶梯表](language-scale-activity-v9-proposal.md)替代。保留原表，不将它们当作当前前置条件。

| 包 | 依据与有限范围 | 建议资源上限 |
| --- | --- | --- |
| 数值补齐 | 捕获真实失败行，区分根求解消减、评分尺度和递推问题；保留阈值，最多补齐其余25个未完成身份，不扩大候选 | 6h wall、4 NPUh、8 CPU核时、16GiB新增 |
| 真实稀疏 | 优先HB temperature1；同函数物理稠密/跳过未选Full，各2seed，加pooled-attention两seed质量参照，共6条1M；状态/Describe工作单列，不以理想吞吐倍率作为资格 | 12h wall、8 NPUh、32 CPU核时、32GiB新增 |
| 数据与容量阶梯 | 以已复核KV候选为起点，HB40/88 pooled、HB40 temperature1及近参数TF12/28，各2seed，最多10条4M/16M轨迹；先有限校准新增深度配方 | 24h wall、24 NPUh、32 CPU核时、64GiB新增；HB88成本仍需先导确认 |

预算是后继包上限建议，尚非已启动任务；精确表需在选择后冻结。
数值补齐只阻塞受影响候选，已完成KV候选足以支持后继稀疏/规模设计。首轮排序不决定普遍优劣；
新长度、曝光、深度和计算路径均是迁移条件。B预训练路线没有新任务，本轮证据不代替其局部承接结果。
