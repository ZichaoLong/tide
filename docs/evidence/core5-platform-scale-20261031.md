# 三平台规模阶段：已审阅的GPU约1B大batch与三拓扑（2026-10-31）

本报告包含五项GPU正式流程：1B大batch两项及FC多节点/HB/LH各一项；CPU/NPU规模与容量尚不在此归档。多卡见[独立报告](core5-platform-multicard-20261031.md)。
这些配置均完成执行、独立完整记录和离散检查、磁盘精确续接；原FP32数值超差在explore下被显式接纳。
**没有严格合格吞吐。** 不是将原FP64资格改写为FP32通过，也不认证训练。

源码`1b3e39f4f58378272d5afe41f970c92dd0e4fed7`，Python安装内容hash
`6af9b3981aa512982618b31b02ad69db5efd20c31438b6cb273038d877c68d29`；GPU core源de999c2。
四张RTX3090环境的GPU0、Torch2.12.1+cu130、sm86、IEEE FP32/TF32关闭，ATen1/interop1/worker1。
模型为4节点/2区域FC、KV、d4096、FFN4d，1,010,339,302个去重可训练参数，B为独立staggered-v1样例。
相同P2/span8、native mixed-c、packed算子/返回scalar、完整返回；warmup1窗、连续2窗独立计时，另验精确恢复窗。
GPU1曾有小型人类入口演练，非本项目争用未排除；没有重型构建重叠。不声称统计稳定性。

| B | 两个完整窗秒 | 中位秒 / 样例每秒 | candidate每窗 | Full | 消息 | 最大State样例批 / Full时间行 |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 128 | 3.689730 / 4.550663 | 4.120197 / 31.07 | 672 | 386/385 | 386/386 | 128 / 168,167 |
| 512 | 14.495737 / 14.404902 | 14.450319 / 35.43 | 2688 | 1549/1543 | 1546/1544 | 512 / 679,678 |

这些State/Full实际行不是所有MM/BMM都已充分合批的证明；后续独立profile另测算子形状。
B1同模型中位0.162938秒已在[首组证据](core5-platform-baseline-20261031.md)记录；B128/512成本随B增长，样例吞吐约31.07/35.43每秒。
4节点大局部模块不能外推16/20节点或HB/LH；新增材料不能只换层级名称。

| B | 构造秒 | 精确恢复检查秒 | 独立参考秒 | 计时窗最高allocator bytes | 恢复峰值bytes | RSS高水位bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8.620 | 73.144 | 123.854 | 4,559,988,736 | 4,788,428,288 | 19,431,018,496 |
| 512 | 8.654 | 103.484 | 469.732 | 5,278,416,896 | 5,505,466,368 | 24,999,137,280 |

计时含必要输入、传输、调度、packing、forward、finite检查、完整返回及同步；构造、逐窗checkpoint出版、检查/恢复分列。
设备allocator不含驱动/厂商隐藏内存；RSS为进程生命周期峰值。静态B128/B512设备预算6.424/12.764GB、host35.552/61.446GB、disk10.660/11.923GB；实测不等于通用预算公式可直接删减。

B512首差为window0、sample0/node0 KV keys：137/4096坐标超原atol1e-6/rtol1e-5，示例最大绝对差约2.74e-5。
全部报告的离散差异计数为0；数值差异计数123100保留。原严格阈值未修改，输入/精度没有替换。
完整返回、自身精确恢复与独立数值是分开的事实。

[可查询v2归档](core5-platform-scale-20261031.json.gz)保留配置、实际来源/峰值、逐窗/warmup成本、首差与诊断。
远端外层任务于2026-10-31T05:56:14Z终态exit0，supervisor/workload/remaining_live均空；外层passed不改变两个内层strict-failure。
原checkpoint可以按保留规则退役；可由配置、固定来源及确定输入重新生成。

```sh
# 干净安装/实际site按docs/human-workflow.md；从跟踪矩阵选择一个case
python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(next(c["config"] for c in m["cases"] if c["name"]==sys.argv[2]),indent=2))' configs/matrices/cuda-scale.json cuda-scale-fc-few-1b-b512 > selected.json
tidegraph plan selected.json --site site.json
tidegraph job start selected.json --site site.json --output-dir jobs/1b-b512-new
tidegraph results docs/evidence/core5-platform-scale-20261031.json.gz --batch 512 --format table
```

## 更多节点与HB/LH的约1B实测

这三项固定bd2ff2e（安装内容hash`afc35286eae8dca7744134dcc966cde82ddf5b30e07720cd0ef491ee30783ff8`），相同FP32/TF32关闭、mixed-c/native、packed/scalar完整返回与一流程口径。B1/P2，FC16的span20，HB/LH20的span14；不同物理模型不能互称等价加速。均完成独立完整记录/离散及精确恢复，数值超差仍保留。

| 拓扑 | 节点/区域 | 参数量 | 中位秒 | candidate | Full两窗 | 消息两窗 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| FC/d2048 | 16/8 | 1,014,118,392 | 0.437248 | 31 | 16/16 | 28/28 |
| HB/d1792 | 20/10 | 971,589,310 | 0.498007 | 39 | 20/22 | 48/50 |
| LH/d1792 | 20/10 | 971,589,822 | 0.418530 | 39 | 21/21 | 69/69 |

相较4节点大模块的同量级参数B1/0.162938秒，更多节点/边增加调度、组装与局部调用成本；本证据不能仅由参数量预测速度，不能将FC4的结果外推到三拓扑。全参数owner、来源hash、实际批长、分桶观察和恢复峰值在归档。任务platform-cuda-finite-scale-bd2ff2e于06:08:29Z终态exit0且无后代；其finite六项在另一证据中，本归档不重复累计。

归档SHA256：`fdf3615e02f4cc10166f9b233934aa3e0221b062ccfd1729df720e4c086b16f7`。
