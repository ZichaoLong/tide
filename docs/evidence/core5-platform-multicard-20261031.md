# 同模型多卡：CUDA实际放置、通信与训练（2026-10-31）

固定Python实现`bd2ff2ed6786c57f492b6ac8b83b4b3a4b0d821e`；core/resident构建源de999c2，standalone源eca070f，本组仅实际运行Python/native eager。旧库来源/hash与安装内容分开保存在归档，不声称重新构建。RTX3090×4、Torch2.12.1+cu130，FP32/FP32 scoring、TF32关闭、ATen/interop/worker均1、mixed-c/packed/scalar完整返回。每配置warmup1窗、连续2窗、独立参考与磁盘精确续接。一次有限流程，不宣称统计稳定性。

六项全部执行完成、离散/完整记录检查及精确续接通过。三个1B推理点保留数值超差探索结果；三个小型训练严格通过。原严格阈值、旧gated-HB多卡恢复失败及异步边界不因此解除。

## 同一模型的推理放置

FC16节点/8区域、KV、d2048、FFN4d，1,014,118,392个去重可训练参数，8个不可拆共享组，B1/P2/span20。各卡数的model_id/workload_id相同；参数不复制成独立变量。按共享组确定性平衡放置。三次每窗均31 candidate、16 Full、28消息；State最大样例批1、时间长2。

| 卡数 | 实际节点分配 | 窗口中位秒 | 全程逐卡allocator峰值bytes | 数值/离散/恢复 |
| ---: | --- | ---: | --- | --- |
| 2 | 8/8 | 0.416659 | 2288798208 / 2288798720 | 超差 / 通过 / 精确 |
| 3 | 6/6/4 | 0.416816 | 1748088320 / 1748088832 / 1207379968 | 超差 / 通过 / 精确 |
| 4 | 4/4/4/4 | 0.370152 | 1207378432 / 1207379968 / 1207379968 / 1207378944 | 超差 / 通过 / 精确 |

相同单卡模型在规模组中位0.437248秒；多卡2/3/4分别0.416659/0.416816/0.370152秒，仅为有限成本观察，未严格等价，不称可推广加速。逐节点参数owner、实际计算回调统计、state slots与逐目标消息位置都记录并审阅，与请求映射相符；空pending明确为空，不据空集合推断其他窗口。节点分配2卡为8+8，3卡6+6+4，4卡4+4+4+4。完整共享梯度贡献另由小训练验收。

三个推理点每测量窗的native跨卡传输均7个copy group/28行/229488 logical bytes，普通复制记录同量tensor-leg bytes；Python边界另2次/16392 bytes。该计数不包含厂商隐藏复制或返回packing全部流量，也不是完整传输带宽。当前12方向直接P2P及原生跨卡原子仍不可用，ordinary不认证P2P/NCCL/capture或resident协议。

三点首差均为window0、sample0/node0 KV keys（原atol1e-6/rtol1e-5），不能用恢复成功替代独立正确性。完整差异保留在归档。

## 三种明确通信的小型训练

同一FC8节点/4区域KV模型、16,388参数、B2/P3/span18，2卡各4节点，原weighted-output-square×0.001、SGD lr0.001/momentum0.3/weight_decay0.01。三个通信配置的model/workload/task身份一致。每窗39 candidate、39 Full、58消息；独立记录、输入/参数根、梯度、参数、optimizer槽及精确续接分别验收，8份比较报告各自零差异。

| 通信 | 中位秒 | native逻辑/物理传输bytes | native端点同步 | Python边界复制/同步 |
| --- | ---: | --- | ---: | --- |
| blocking | 0.592957 | 4176 / 4176 | 24 | 10 / 20 |
| cpu_bridge | 0.668933 | 4176 / 8352 | 24 | 10 / 20 |
| ordinary | 0.583081 | 4176 / 4176 | 0 | 10 / 0 |

Native三种请求均为6组跨卡复制；cpu_bridge的6组明确经主机、物理字节翻倍。Python边界360 bytes是设备到主机输出/目标准备，因此其cpu_bridge_copies=0不代表native未中转。计数含声明路径的前后向同步边界；不能外推为所有ATen/driver流量。所有卡共享一个optimizer窗口边界，未将共享参数拆成副本绕开约束。

## 来源、生命周期与复现

任务`jobs/platform-cuda-multicard-bd2ff2e`于06:17:00Z exit0，SSH独立检查supervisor/workload/remaining后代全部为空；外层passed不覆盖三项strict-failure。没有项目重型构建重叠，其他系统负载未隔离。allocator范围不含驱动/厂商隐藏内存；详细构造/计时/恢复峰值和RSS高水位在归档。

```sh
tidegraph plan configs/matrices/cuda-multicard.json --matrix --site site.json
tidegraph matrix configs/matrices/cuda-multicard.json --site site.json --output-dir jobs/new-multicard --detach
tidegraph results docs/evidence/core5-platform-multicard-20261031.json.gz --format table
```

安装/实际site见[公共手册](../human-workflow.md)。约1B三卡并不代表靠近多卡总容量；后续容量准入分别核对逐卡峰值、host和checkpoint磁盘，不直接加总空闲显存。

归档SHA256：`4dbc8fc617ab3bfe01ad3d58fee479d658353a993fbc860bdd4af833985ef439`。
