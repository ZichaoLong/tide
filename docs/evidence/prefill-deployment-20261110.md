# 联合prefill的部署代表、容量修正与停止证据

这轮补齐宽FC/CPU线程、FP16、异构LH、约1B/9B、双NPU及resident的有限代表。
成本结果仍不支持“默认入口已消除碎片化”或“加速器普遍快于CPU”。
[审阅记录](prefill-deployment-20261110.json)保存每项配置、源码/包/二进制身份、
实际工作量、恢复、容量拒绝和取消记录；不是全部开关的笛卡尔资格。

数学固定0770593；CANN9.2、Torch2.12.0+cpu、TorchNPU2.12.0.post3、aarch64。
普通CPU/单NPU部署成本复用f15c4a5包/库；双卡、B32、层链复用3d88b0b包/库。
后继485a3e2只改记录预算/诊断与共享trace判定，新安装复用3d C++，对应C++源码diff为空。
resident包/库固定a553bea，复用core原manifest身份413faf6；已核对C++内容hash与a553一致，原身份不重写。目录预算在f609/1f82143分别修订。
逐项原始身份以JSON为准，不能把新驱动提交当作重新构建了所有后端。

## 有限成本与实际活动

均使用默认auto分量窗口、packed、按需outputs返回、一次预热和一次测量。
任务为生成器的staggered输入、内置weighted-output-square；训练scale=1e-6、SGD。
除FP16行外payload/scoring为FP32；设备行除resident外为mixed-c、普通ATen，
客户端均为Python调用LibTorch。计时包括输入、搬运、调度、packing、检查和同步，
训练另含反向及optimizer；构造、磁盘保存和恢复在计时外。

| 拓扑/模块 | 节点 / d / 参数 | B / L | 执行 | 模式 | 完整窗秒 | 候选 / Full / 消息 |
| --- | --- | --- | --- | --- | ---: | --- |
| 宽FC/KV | 4 / 512 / 16.224M | 4 / 16 | CPU16线程 | 训练 | 0.332410 | 214 / 122 / 122 |
| LH/KV | 176 / 640 / 1.051B | 2 / 8 | CPU32线程 | 推理 | 1.368072 | 1687 / 377 / 2246 |
| 同上 | 同上 | 2 / 8 | 单NPU，host2线程 | 推理 | 3.397598 | 1687 / 377 / 2246 |
| FC/KV qry，FP16 | 16 / 128 / 4.565M | 4 / 16 | 单NPU，host2线程 | 训练 | 8.137528 | 915 / 306 / 908 |
| LH/五记忆异构 | 176 / 64 / 15.435M | 4 / 16 | 单NPU，host2线程 | 训练 | 88.001672 | 8384 / 2713 / 15926 |
| FC/KV qry | 16 / 128 / 4.565M | 4 / 16 | 双NPU，显式launch blocking | 训练 | 16.780382 | 915 / 306 / 912 |
| LH/KV | 176 / 1920 / 9.315B | 2 / 8 | 双NPU，普通异步复制 | 推理 | 13.613896 | 1665 / 346 / 2071 |
| LH/KV | 176 / 640 / 1.051B | 32 / 16 | CPU32线程 | 推理 | 12.262388 | 55030 / 12206 / 73474 |
| 同上 | 同上 | 32 / 16 | 单NPU，host2线程 | 推理 | 33.965264 | 55029 / 12204 / 73469 |

实际活动率应取Full/候选，例如B32约22.2%；名义参数量、发送强度和活动数量不能互代。
CPU/NPU的实际选择与消息数有差异，不能报告为同轨迹等价吞吐；大尺度独立数值/离散
均为not_checked。共享主机存在其他工作负载，NPU队列逐次记录实际物理卡；没有独占计时保证。
这不是口述旧LH9B的复验：图、局部模块、B/L、硬件及计时范围均不同。

普通七项小batch部署完整窗及公共exact-next已通过。CPU B32原计时完成后在保存阶段被
取消，只有未发布staging文件；后继不计时重建两窗，公共exact-next已逐值通过。
该后继汇总脚本随后误对dict调用record()而failed，原resume通过与wrapper失败分别保留，
不据此重复计时，也不把原取消任务改写为通过。NPU B32完整raw解码、实际下一窗及参数/目标的逐值恢复通过；checkpoint为5,776,264,344字节，恢复检查127.320秒，均不计入33.965秒成本。

## 新预算为何可以重启B32

首次NPU B32在执行前被115,339,688,680字节估算拒绝，其中99,470,671,872字节来自
全图全部事件记录。此前输出推理已经不保留这些记录；该拒绝没有真实OOM或性能数据。
485a3e2依据Python/C++分量窗口按区域顺序求值的生命周期，只为已审核的ranked-local、
无梯度、无插件、outputs/metrics PyTorch/Python-native路径预留最大区域事件块的两倍加输出。
保留完整估算可查询，不减少参数、raw state、消息、恢复及host预留，也不关闭容量门禁。
训练、旧greedy/streaming、完整trace、插件、standalone/resident及其他图继续原保守估算。

新实际准入估算24,922,405,608字节；原同配置已进入运行，测量阶段peak allocated
5,488,138,752字节。它不是所有临时工作区的精确上界；allocator、厂商workspace和
通信仍有未知分量，含恢复的peak allocated/reserved分别6,066,335,744/6,396,313,600字节，分阶段记录保留。没有把不检查数值等同豁免资源错误。

485a3e2新私有固定安装46/46检查通过，包含不分配1B模型的原拒绝见证、异形区域/放置、
保守后备、五记忆实际所选返回/训练/恢复、独立engine检查及未返回raw状态错误注入。
恢复诊断不再临时开启完整trace，比较实际返回和下一窗完整raw状态/参数/目标。

## 异步和常驻的具体终态

默认ordinary双NPU FC16在warmup反向多分钟无推进；gdb显示NPUStream::stream/
_index_put_impl_等待。只停止对应本项目unit，PID/cgroup消失，原inner running快照、
堆栈和取消/143保留。显式ASCEND_LAUNCH_BLOCKING=1是独立配置，执行/恢复通过不解除默认异步。
9B普通双卡推理通过也不认证跨卡反向、HCCL、直接P2P或常驻。

resident依次暴露目录静态预算、optimizer取得总预算1/16、反向四份view预留等边界。
修正每样本L×R阶段上界（journal自身另含B）及分区后，两项仍在construction拒绝：

| 代表 | 最后拒绝 | 已知实现位置 / 下一项有依据的工作 |
| --- | --- | --- |
| FC16/KV qry | resident KV qry tensor budget exceeded | weighted_kv_query.cpp的rows×capacity×d×summary保守守卫与当前remaining；先导出逐节点剩余和实际query/tape占用 |
| 四层SSD | local row graph tensor budget exceeded | weighted_row_graph.cpp按表达式与retained tensor双重计量；先分解表达式保存/共享workspace生命周期 |

两项尚未进入warmup，没有训练、续接或吞吐结论；不能把整体18.52GB/6.24GB预算视为
所有数值工作区足够。此轮不再统一加预算重试。新方案应保留数值tape、独立根和精确恢复，
先验证分区/复用，再运行这两个原代表。小图resident六项独立通过仍见
[先前代表证据](prefill-representatives-20261110.md)，不能扩大为整序列常驻优化已完成。

## 从干净仓库重建

按[开发入口](../development.md)安装对应固定提交；新预算无需新C++算法，仍须匹配库身份。
[回归目录](../prefill-regression.md)的生成器连接现有公共CLI，不依赖此轮运行目录：

```sh
python scripts/prefill_regression_config.py --case lh-1b --samples 32 --positions 16 \
  --mode inference --device npu:0 --threads 2 --output NEW_CONFIG.json
tidegraph plan NEW_CONFIG.json --site SITE.json
tidegraph run NEW_CONFIG.json --site SITE.json --output-dir NEW_RUN
```

本次只在最终边界保存，需将生成配置的measurement.checkpoint_interval设为2；其余原配置
逐项见JSON。宽FC选wide-fc；五记忆LH选many-lh/family=mixed；9B选lh-9b、B2/P8及
peer-device=npu:1；resident选query-fc或layer-chain/ssd并指定placement=resident。
如重建原CPU B32的独立恢复检查，固定安装下使用diagnose_experiment_resume.py、windows=2，
其新见证不能冒称恢复了被中止任务未发布的字节。未解除项的唯一当前状态/重启条件见ROADMAP。
