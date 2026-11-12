# 区域均值、写入合批与resident录入边界

实现`1f0e6e43a540a78f2fe810e97b7562b73481fb22`，数学固定`0770593`。
完整身份、38点结果、实际工作量、测试命令与静态审阅见[机器记录](region-tensor-20261112.json)。
本报告不包括后继固定tile实现，不宣称P1—P6完成。

## 实施与独立资格

Python/C++ Aggregate改成接收者×行×物理来源的单次装配、均值和中心化显式VJP；
重复边仍独立参与，质量/内容根、None与连接零、原始版本守卫保持。C++正KV写入整组计算。
跨节点Read按目标device/dtype统一映射真实共享角色；相同值的独立参数不合并。
resident移除全参数layout占位、optimizer/8，按实际13类buffer及存活program workspace计费；
设备bank游标复用容量分支体，数学和独立FP32根保持。

- 两套独立Python-owned/SDK core及resident构建通过；SDK六项定向CTests通过。
- 固定CPU定向304通过，9项原严格数值xfail保留；没有新失败。旧四项计数断言已迁移到真实tensor指标。
- CPU FP64/NPU FP32 × Python/native，各自位置、mixed-a和LH176，共12个安装消费者完成三步AdamW及新进程精确恢复。
  小型位置/mixed-a另有独立标量检查；LH176生命周期不是规模数学oracle。
- 新独立SDK NPU五记忆×三混合放置列检查通过：独立streaming、raw状态、VJP和两次SGD窗口。
- resident投影与其他8个局部组件真机通过。optimizer另通过32条轨迹/256次更新、实际存储盘点与边界拒绝；
  这不是完整新resident owner训练资格。Python resident仅构建，依赖完整组合的运行未启动。
- CUDA核映射生成/静态适配审阅不等于CUDA编译或GPU实测；本轮没有GPU资格。

## 有限成本

CPU1线程、NPU主机2线程；NPU成本为物理5。每配置独立进程，一次warmup、一次测量；
CPU池12–15，NPU主机24–27。重型构建先终态；其他卡上有本项目资格与下游进程，主机非独占。
profile另开流程，未把插桩时间当作吞吐。它们是有限观察，不是峰值或全部规模结论。
默认native为Python调用C++/LibTorch、主机日程，NPU采用mixed-c；不是resident计时。

FC16沿用B8/L128/d128/H2/FFN512、1024输入位置及固定下游任务；推理只做最后位置head/首token返回，
grad-forward/训练做全序列head/loss。训练含准备、检查、反向、clip、AdamW、清理及同步。
FC测量发生于warmup更新之后；材料身份、实际Full/消息量逐点保存，不把不同数值工作量冒称完全相同轨迹。

| FC路径 | CPU推理 / 含grad前向 / 训练（秒） | NPU推理 / 含grad前向 / 训练（秒） |
| --- | --- | --- |
| 默认native | .416107 / .447188 / 1.185442 | .146660 / .152745 / .285554 |
| 默认Python | .530182 / .573785 / 1.213215 | .279095 / .283445 / .525157 |

FC公共native完整训练CPU/NPU为1.152264/.269344秒。默认native训练吞吐为863.8/3586.0位置每秒，
推理首token返回时间为.4161/.1467秒。FC直接公式未重测，保留[6011d599原对照](tensor-kernels-20261112.md)；
Transformer数学不等价的原数据也不重标为本轮实测。

LH176为176节点/22区域/1058体边、d64、B4/L16、staggered-v1，61个实际输入位置、14,902,694个训练参数。
默认路径实际8368个候选、2572个Full、15340次边访问；直接公式物理计算11264个Full行后mask。
LH每次reset材料，完整前向/输出loss/检查；训练增加反向、SGD(momentum .3)和同步。

| LH路径 | CPU推理 / 含grad前向 / 训练（秒） | NPU推理 / 含grad前向 / 训练（秒） |
| --- | --- | --- |
| 直接公式 | .202424 / .234421 / .779556 | .176716 / .203998 / .612110 |
| 公共native | .511421 / .574303 / 1.285649 | 1.215598 / 1.336716 / 2.630771 |
| 默认native | .617892 / .675150 / 1.390021 | 1.414563 / 1.460060 / 2.840692 |
| 默认Python | 1.486842 / 1.603058 / 2.671841 | 2.371578 / 2.441078 / 4.177551 |

默认native LH训练吞吐CPU/NPU为43.9/21.5位置每秒，直接公式为78.3/99.7。
相对上版1.5954/3.5953秒有所改善；NPU仍慢于CPU及直接公式，不能宣称差距已闭合。
直接/公共的Full物理工作、根和解码职责不同，差额不是全部可消除的调度时间。

## 静态与动态诊断相互核对

默认native外层节点由14858降为9071；计入native私图的总节点由23333降为18140。
IndexCopyBackward由2066降为8。两套统计重叠，不能相加。ATen记录数632049→572242；
直接公式仍94543。节点下降说明该类逐源反向装配已消除，不等于全部装配或数值成本消除。

固定38份文件hash绑定的静态审阅见机器记录。整组写入和均值已经落实，剩余差距不能沿用旧解释：
Aggregate仍物化`[receiver,row,physical-source,D]`，直接实现保留唯一来源块、用incidence归约；
KV写入/Read/Full每阶段仍发布节点字段，之后重装配bank/区域/下一查询。G=8时仅写入/Read就有
32与32/48个index-select源码表达式。它们不是设备kernel数量或各项耗时占比。
默认checkpoint解码是额外职责，不能解释公共接口与直接公式的主要差额。

## resident失败保持，停止相同候选

projection成功，但完整block-kv-messages在录入时仍EE1019/507002。独立trace原样复现，
未执行模型；厂商日志确认stream task buffer full，不是HBM或数值求解失败。
依赖的sequence/flow/session、Python resident安装生命周期及FC/LH规模未运行。

失败fixture仅4帧×3节点=12行、2bank、Read cache2/Full cache3、宽度2。
Read/Describe/context/Full共有14投影；仍静态登记70个前向容量body及70个反向body。
源码bank BMM为210次；实际失败前记录929个vendor操作，其中224 BMM、230 IndexSelect、116 IndexCopy。
最后记录task2043，随后snapshot guard提交失败；它只是触及累计容量的提交点，2043不是跨环境的固定容量。
自定义核、label和branch未计入vendor统计；静态录入数也不是实际执行次数。

下一次修正应去掉投影内部的容量树，或使用设备长度驱动的grouped GEMM；不能只继续放宽字节预算。
固定tile会改变padding界，须验跨tile参数伴随。更大flow仍需要有界程序组合，不能把局部消息通过当作规模通过。
本报告不认证任何后继方案。未完成项仅见[ROADMAP](../ROADMAP.md)。

## 复现与维护

按[development](../development.md)在固定提交分别构建两个运行时。CPU定向完整argv嵌入机器记录；新增独立入口为`tests/test_aggregate_tensor.py`、`test_column_read_transport.py`、
`test_resident_optimizer_storage.py`；SDK用`ctest -R 'weighted-(columns|kv-tensor|kv-tensor-prepare|read-transport|aggregate-tensor|column-checkpoint)'`。
安装消费者用`tensor_window.py --positions`、`--placement mixed-a`和`planned_windows.py --plan PLAN`；PLAN已嵌入记录。
成本和诊断用`benchmark_lh_tensor.py`、`benchmark_dense_query.py`、`profile_lh_tensor.py --census`，完整argv已嵌入记录。
resident原失败用`TIDE_CANN_TRACE_OPS=1 tide-block-kv-messages-check --device=npu:0 --dtype=float32`，
需合法设备分配及有限独立服务。源码、能力、独立资格、有限成本、后继候选分别导航，不依赖旧raw目录。
