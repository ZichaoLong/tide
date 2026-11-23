# TL-REQ-005 接续需求：稳定正量链路、声明VJP与有限正支持

2026-11-11，用户已讨论并同意修复方向，明确授权实验线程向本基座目录提交需求。
状态：**已提出，待基座线程接收/回复；未实施、未交付、未下游验收。**
本文件是本地交接入口，权威请求由提出方`tide-learning-lab`维护；不建立第二套待办或结果台账。

## 权威请求与固定证据

- [完整需求 TL-REQ-005](https://github.com/ZichaoLong/tide/blob/69be6d0eab0179d568988a85e51171db1016b568/docs/requests/TL-REQ-005.md)，实验分支已普通推送。
- [实际失败、静默见证与链路审计](https://github.com/ZichaoLong/tide/blob/d6582a172c3d76f26e67d9c4824b11fc619d4b36/docs/studies/positive-path-audit-v6-results.md)。
- [实际完整局部行、100位Decimal及公共接口结果](https://github.com/ZichaoLong/tide/blob/d6582a172c3d76f26e67d9c4824b11fc619d4b36/docs/evidence/positive-path-v6-local-analysis.json)。
- [运行/恢复、每项产物hash与Lens对应](https://github.com/ZichaoLong/tide/blob/d6582a172c3d76f26e67d9c4824b11fc619d4b36/docs/evidence/positive-path-v6-review.json)。
- 本机权威文件：`/mi/data2T/zlong/home/llm/tide-learning-lab/docs/requests/TL-REQ-005.md`。

数学`0770593838a37c790a568226930567ed60f99b83`，被测基座及协议提交
`74e7bbd62341d2ccd382194a2c52ad4fea033665` / `tide-downstream-contract-v1`。
本地基座提交时HEAD为`e3452962b928d8cb82d73b1b08fd1ef69e41a119`；本需求不是对该HEAD的回归判定。
请先对照当前源码和固定反例，复用已经交付的部分；实验线程没有更新安装锁或接入性能优化。

原任务均为HB40/width32/region4、CPU1/FP32/native，局部公共Python API另验。
私有安装`.venv-positive-floor-v1`；Torch2.12.0+cpu/TorchNPU2.12.0.post3/CANN9.2/aarch64。
native binary SHA256：`27ecaa1315d250d9ab2de2a42f2a2487dc4d8225d739c61ffb333b7d66b08289`。
依赖/安装/环境锁及完整源码、配置身份见权威请求。所有旧源码、安装、失败和阈值保留。

## 为什么需要这一项

| 实际失败 | 关键值 | 应区分的原因 |
| --- | --- | --- |
| constrained Bern，129更新后的验证 | c约1.27845e-40 | 结果可表示，当前sigmoid提前归零；应先修稳定公式 |
| regional-total Bern，训练前向142 | c约1.49804e-48 | sigmoid有提前归零，但最终cap×概率也超出FP32范围 |
| direct attention / received，训练前向75 | cap约3.90e-21、share约3.69e-26、乘积约1.43837e-46 | 两因子均为正常正数，最终发送控制归零 |

FP32最小正次正规数为2^-149，不能与finfo.tiny混淆。
此外，公共小图证实strict Aggregate能在正质量/正常KV写入次数下丢失内容；
Describe前置FP64→FP32 cast能静默改变Full支持；Bern局部VJP能丢失可表示的约8.19e-10梯度。
内部Attention尾部×大value和epsilon转换另有公式/配置见证，未证明发生在原训练。
三项原失败已重现，四个CPU诊断终态，无新学习比较；局部FP64通过不代表完整FP64训练资格。

## 已对齐的交付方向

1. **稳定公式与完整反向。** Bern使用稳定sigmoid/log响应；以log斜率求KKT耦合和复合VJP，
   覆盖score/cap/temperature/target/shift，不只修前向或score梯度。稳定Aggregate推广到适用strict/native路径。
   KV/attention的小量×大value/伴随应按复合函数求值，不能统一给内部概率补最小值。
2. **显式正支持存储策略。** 默认strict仍拒绝真正范围下溢；扩展具名策略到本次Bern/attention发送及实际Python/native消费路径。
   只修失去的数学正支持，保留合法零、cap/总量/相对扰动限制。前向存补值，反向按未补值响应声明，
   不能使用普通clamp导数或用补后c重建斜率；Bern不能照搬Shannon VJP。
3. **跨生成、转换、消费的契约。** 既有末端control cast检查不足，补前置Describe/state质量cast的语义检查。
   保持raw KV质量、内容、时间/次数及Full活动一致；bool或次数不能恢复加权内容。
   epsilon/tau等必要尺度在实际dtype确定后验证正性，不能只在host验证。
4. **连续量许可与后继边界。** 保留短通道衰减归零许可；门值/梯度不统一floor。
   log-mass或更大范围融合用于需要尾部比例、或中间质量伴随超范围而最终参数梯度可表示的情形，
   属于有条件的表示/接口后继，不要求它先完成才交付前述修复。

每项的具体公式、边界、来源及验收表均在权威请求中。总补入质量小不保证下游只聚合尾部时的比例误差小。
新数值/图/恢复身份明确，原strict失败不改写；跨模式权重引入不冒称原轨迹精确恢复。
按CPU FP32真实反例、CPU FP64局部参考、拟交付NPU FP32路径做有限验证；不要求全平台全候选矩阵。
执行、严格数值、离散语义、恢复和学习分开。保留原atol1e-6/rtol1e-5，有限超差经有界排查后可标记探索。
必要检查应批量化，不借逐算子同步或全量日志重新制造TL-REQ-004的执行成本。

## 复现与原始输入

最小实际局部行已入上述JSON，不需要大型checkpoint即可复现scalar分配/发送：
`actual_bern.*.captured.tensors`和`actual_direct`包含实际行；公共API为
`tidegraph.weighted.allocate`及`tidegraph.weighted.sending.attention_send`。
本机已有固定环境下的小型补证入口（新输出，不能覆盖原证据）：

```sh
cd /mi/data2T/zlong/home/llm/tide-learning-lab
# 先按原项目入口激活匹配环境；本文件没有提交新计算。
python scripts/probe_positive_formulas_v6.py --root "$PWD" \
  --output docs/evidence/tl-req-005-local-replay-new.json
python scripts/job.py status artifacts/jobs/v6-positive-direct-received
```

完整边界重放配置在`configs/studies/language/positive-path-v6/`，原精确argv/env/source/hash在
`artifacts/jobs/v6-positive-*/job.json`。最多16次更新，各CPU1/30min/8GiB，额外边界前向/局部补证15min；
原总额度1.75 CPU核时/1GiB产物，已用约0.2395 CPU核时/49MiB。这是历史预算，不自动启动新的上游作业。

direct仅捕获native局部row5/col0及实际max-centered scores，不是全局节点5；小received上游来源尚未完全追溯。
原三项训练恢复输入、四份诊断输出及固定源码/安装/native均列为当前磁盘审计的保护项。
另行清理仅只读调查，没有删除这些输入；取得独立小型复现后再明确原大型材料的保留需求。

## 接收与边界

由基座线程按`docs/downstream-contract.md`回复接受/部分接受/暂缓及范围，登记其ROADMAP/STATUS，
交付实现/证据提交、公开调用、实际客户端/设备/dtype、构建身份、拒绝边界、复验命令和迁移说明。
下游随后在新私有安装验收；上游交付与下游接受独立。

本轮只新增此交接文档，未修改基座执行源码、STATUS/ROADMAP、共享环境或已有交接文件，也未提交/推送基座分支。
需求原文提交不等于基座已接受或修复；不创建外部GitHub issue，不自动开展实现或训练。

权威请求快照SHA256：`beea81d2ff4cb6de5141bdba834ba70970b4b487869180cccb34cd2e59a4dbb1`（69be6d0eab0179d568988a85e51171db1016b568）。
