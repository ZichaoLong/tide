# 联合prefill有限代表矩阵

固定实现`a553beae90374ccf1e68feee45e011d3d7d35679`；数学0770593。
[审阅配置、工作量与结果](prefill-representatives-20261110.json)保留34项请求和精确身份。
CPU16项、NPU18项均完成运行及真实选定路径下一窗精确恢复；所有点独立数值/离散为not_checked。
独立小门禁见[分量窗口](component-window-20261110.md)、[整序列VJP](kv-sequence-vjp-20261110.md)、
[query根](query-root-boundary-20261110.md)，不能把小门禁变成34项规模数值资格。

共同设置：B4、每输入端口16个原始时间位置、staggered-v1稀疏日程、FP32，
CPU1线程/NPU2主机线程、SGD、一次预热后一次完整训练计时，outputs返回、packed局部计算、
query auto，native默认窗口与显式greedy对照。完整步含准备/传输、调度、packing、
forward/loss/backward、有限检查、optimizer和同步；构造、checkpoint写盘/恢复另列。
预热延续到测量窗口，不与原FC16重置预热的任务混算。无独占计时承诺，不作统计稳定结论。

| 配置（native，auto / greedy） | 节点 / d / 参数 | CPU秒 | NPU mixed-c秒 |
| --- | --- | ---: | ---: |
| wide-fc/kv | 4 / 512 / 16,223,730 | 0.446 / 0.432 | 0.687 / 0.660 |
| many-hb/kv | 176 / 32 / 5,220,726 | 7.618 / 7.391 | 22.090 / 22.537 |
| many-lh/kv | 176 / 64 / 14,904,826 | 18.257 / 19.493 | 51.729 / 53.530 |
| sum-fc/kv | 16 / 64 / 1,304,092 | 1.106 / 1.069 | 2.211 / 2.217 |
| layer-chain/ssd | 4 / 128 / 1,153,816 | 0.651 / 0.670 | 2.024 / 1.883 |

| 其他覆盖 | CPU秒 | NPU秒 |
| --- | ---: | ---: |
| layer-chain-gated_delta-auto | 0.850 | 1.970 |
| query-fc-kv-auto | 1.046 | 2.321 |
| query-fc-kv-pytorch | 1.862 | 2.818 |
| layer-chain-ssd-pytorch | 1.143 | 2.417 |
| query-fc-kv-libtorch | 1.082 | 2.110 |
| sum-fc-kv-libtorch | 0.832 | 1.799 |
| query-fc-mixed-a | — | 1.623 |
| query-fc-mixed-b | — | 2.379 |

auto/greedy在这些配置的端到端差异有限；本轮不能宣称全窗口策略普遍提速。
单次小规模NPU仍慢于CPU，主要矩阵已部分合批不等于装配、检查、递归VJP和
读取快照投影已经消除；需要结合新profile与同公式层链基线定位。纯PyTorch和独立SDK
各有真实点，不从Python-native推断；其他dtype、resident、双卡和大参数另行验收。

常驻a553库在CANN9.2实际构建后，另在物理NPU7完成6项小检查：KV反馈PDG原始
checkpoint、一般FC/HB端口，以及KV FC/HB和SSD PDG训练/磁盘恢复（FC含新进程）。
这些是独立CPU参考检查，不是常驻整序列批量或规模性能资格。

复现使用安装CLI与`scripts/prefill_regression_config.py --suite representative`；
额外客户端用`--case query-fc|sum-fc|layer-chain --client pytorch|libtorch`，mixed位置用
`--placement mixed-a|mixed-b`。每项准确请求保存在JSON的requested，site/binary须换为
按[development](../development.md)从固定源码构建的匹配路径；不依赖本次临时驱动。
