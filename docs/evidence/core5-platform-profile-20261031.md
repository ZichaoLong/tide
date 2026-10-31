# CPU/NPU约1B大batch的实际算子观察（2026-10-31）

这是两项独立profile，**不是新增吞吐或数学资格**。执行固定bd2ff2e，配置b495f10；
CPU与NPU的正式数值/离散/恢复结果引用[规模阶段](core5-platform-scale-20261031.md)，
GPU同材料profile亦在该报告。公共native实现与独立调度均未修改。
FC4/KV、1,010,339,302参数、B128独立staggered样例、P2/span8，FP32 payload/scoring，
packed/scalar完整返回。CPU为aarch64/ATen4，NPU为Torch/TorchNPU2.12/CANN9.1、ATen1/HF32关闭；
两者interop/worker1。warmup1后只采一窗，不执行CPU参考预跑，不把profile成本放入正式曲线。

## 实际批与小调用并存

| 观察 | CPU | NPU | 含义 |
| --- | --- | --- | --- |
| 最大FFN矩阵 | `[16384,4096]@[4096,168]` | 相同 | B128在线形成168时间行，168不是逻辑batch |
| 同权重projection计数 | 44次/3084行 | 相同 | 平均70.09行；只是被观测的projection接口，不是全部MM |
| 私有BMM | 864次，全部2行 | 相同 | 576次`[2,32,4096]@[2,4096,1]`和288次`[2,8192,32]@[2,32,1]` |
| KV Read/Full可见长度桶 | 各12桶，无单行桶 | 相同 | Read672行、最小32；Full386行、最小5；长度2/3/4 |
| 独立投影预算后备 | 1组/96行 | 相同 | 8MiB保护的是堆叠参数/一阶缓冲；数学作用仍执行 |
| 单行描述/评分 | 两形状各672次 | 两形状各672次 | 内容4096或4227→32；仍有小调用 |
| 共享base逐输入投影 | 288次`[8192,4096]@[4096]` | 相同 | 同输入在两节点间共享计算，不等于跨不同输入合成大MM |

CPU的288次大base GEMV在profile内`aten::addmv_` self CPU累计1.733秒；
全部mm self累计0.980秒、bmm 0.108秒。该成本表明跨不同输入、同一base的准备阶段值得后续研究，
但不能把CPU self时间外推为GPU/NPU占时或预期加速。源码边界为
`cpp/src/state_preparation.cpp`、`weighted_kv_preparation.cpp`、`weighted_kv_projection.cpp`，
Python对应`weighted/kv_projection.py`、`kv_projection_batch.py`和`kv_independent_batch.py`。
当前共享准备按相同输入身份组装，增加8MiB上限会重复堆叠大base，不能作为安全的大batch修复。
未来方案应按实际同权重/独立输入分组，并保持独立根、共享贡献与固定顺序；本轮不据调用数重写这一接口。

## 提交、复制与诊断

NPU捕获204,289个设备kernel，设备区间并集1.129秒；完整profile标记9.787秒。
host inclusive nonzero2.872秒、index1.231秒、copy0.653秒、有限值检查0.559秒。
这些有嵌套，不能相加成独占成本。9,413次`_local_scalar_dense`只是该接口观测，不代表全部取回。
CANN导出API表另外记录`aclrtMemcpy`64,168次/累计943.886ms、异步copy390次/3.325ms，
`aclrtSynchronizeStream`54,949次/437.411ms、带timeout的stream同步41,549次/135.308ms，
device同步2次/0.030ms。API累计时间不保证互斥，本报告不把它称为同步墙钟占比；
GPU已有显式同步区间并集占比2.861%，两个口径不能直接相除。复制字节不能从这些次数补推。

Read/Full桶在大batch已经有实际填充，本材料没有统一padding重写依据。
小操作与返回/检查成本依然存在，有限值packing已经有[三平台同材料对照](core5-finite-packing-20261031.md)；
减少kernel未减少显式同步，默认scalar保持。A/B/C完整返回策略见[首组对照](core5-platform-baseline-20261031.md)，
不混入较小返回契约。跨节点/多head数值合批、resident布局和反向模板的约束仍由
[性能经验](../performance-lessons.md)及[唯一待办](../ROADMAP.md)管理。

## 观测修正与复现

CANN本次二维/向量乘法直接记录为`aten::matmul`：共2,360次；原解析器只收mm/bmm/addmm，
因此漏掉了这些形状。c59688c补上该实际入口，保留原`Input Dims`字符串及独立接口计数，
并明确matmul与子mm/bmm可能嵌套，不能求和当kernel数。
开发及冻结c59688c源码的13项离线解析检查均通过；没有改变运行、数学、精度、checkpoint或活跃安装。

原CPU trace SHA256为`3c15132c15132a31c40097ba0132fd2cd39d31f8e24a3d9067814da807af0710`；
NPU为`0b13a1609623d4568c4f0c22862b76198620376237ae8ad6fbf2dc5cb911da13`。
后者在独立background.slice服务中以固定解析器重读，07:04:11Z exit0、控制组空；
原result/trace未修改。归档同时保留原遗漏形状表、重解析来源与原运行身份。

```sh
# 匹配安装/site按human-workflow准备；只有需要新profile时才运行。
tidegraph matrix configs/matrices/npu-scale-observation.json --site site.json --output-dir jobs/new-profile --detach
tidegraph results docs/evidence/core5-platform-profile-20261031.json.gz --format html --output profile.html
```

原始trace退役后，从跟踪配置重新采集；不依赖旧绝对路径。
[可查询归档](core5-platform-profile-20261031.json.gz) SHA256：
`cc16a3a51900a937ae8aa1d8c707afee016f68442895c16917c7cd89162b9bbf`。
