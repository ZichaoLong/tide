# 常驻因子KV、实际训练观测与分配失败资格

固定实现`15c3e65fe3a737324c90bdf26b60c28a34ad2dcf`；上游0770593不变。
[审阅记录](core5-resident-boundary-20261029.json)保存固定来源、原始失败终态、构建身份和独立初始化探针。
本报告覆盖相关实现和正确性；有限性能另行记录。

## 实现与对应

常驻KV §1.2按`B N(u)+U(V N(u))`保留低秩因子，保存实际B/V/U projection tapes并汇合其独立输入伴随。
不再形成稠密UV；base仍需O(d²)，适配器组合由O(d²r)降为O(dr)，新增小投影提交计入完整成本。
共享参数、私有状态、原始int64时间、独立K/V/summary根、None/连接零与half最终存储守卫保持。
现有保守内存准入保留。[KV契约](../resident-weighted-kv-update.md)和[总对应](../semantic-map.md)
给出域、尺寸、快照、实现符号与独立逐行dense公式/autograd检查。FP16指raw存储；该常驻算术/VJP为FP32。

实际allocation错误20–22在设备服务结束后导出保存的输入、节点/int64时间和错误行，快照上限64MiB。
早退输出叫`partial_controls/partial_statistics`，`output_valid=false`；仅残差22附完整controls/residual。
未算出的残差不填零；运行时错误无可读行时保留原错误。失败owner拒绝继续执行，须从旧checkpoint重建。
公开Python与standalone真实Bern反例用scores±2000确认code21、候选节点、部分行和失效owner；
[分配契约](../weighted-allocation.md)说明码20–22的实现域，单个code21反例不冒充三种错误的完整矩阵。

`WeightedGradients.statistics`分列主机微秒construct/seed/run/bridge/reduce，run包括等待，
不是设备kernel时间。现有adjoint绑定具体journal、根和参数地址/版本；本增量未缓存跨窗口的旧adjoint。
[训练契约](../resident-weighted-training.md)定义这些观测与生命周期。

benchmark解码候选实际梯度connection、SGD动量、步数及master，并核对实际选项/可训练集合。
原生trainable列全部别名，offset/槽按物理owner一行、无尾哨兵；解码按真实canonical顺序与参数尺寸。
比较注册Parameter时使用模型实际别名，完整checkpoint仍保留legacy buffers并作逐值恢复核对。
磁盘save/load后换owner执行下一窗，完整返回、参数、梯度与SGD槽要求逐值一致。
显式peer放置计入各卡同步/通信；容量JSON只接受正int64容量，不接受算法或精度替换。
[工作负载契约](../weighted-workloads.md)给出公共命令、计时边界和故障注入入口。

## 固定提交资格

本机aarch64 Ascend910_9392；HF32关闭。SDK Torch/LibTorch-NPU2.10/CANN9.0和Python-owned
Torch2.12/Torch-NPU2.12.post2/CANN9.1分别验收。构建来自已归档的bdd7f6c dirty快照；
运行前对固定15c3e65核对完整C++/resident内容hash、所有binary hash及相互依赖。
未把旧build的来源改记为clean15c3e65。CPU核心23、SDK核心22/常驻7、Python核心3/常驻3个二进制分别核对。

| 固定源码检查 | 结果与范围 |
| --- | --- |
| CPU消费者设施 | 10 passed；选项/精度/非法容量、坏梯度/槽、真实eager恢复及SGD布局反例 |
| Python-owned公共常驻 | 14 passed；KV四图/两存储dtype的完整训练和恢复、前向、真实失败快照，以及新实际梯度/SGD/resume消费者 |
| Python/native eager | FP32/FP16各FC、HB、非对齐端口三组，完整前后向、独立根、多步更新与恢复通过 |
| standalone SDK组件 | KV-update、allocation、flow、training、node-training共9个组件/dtype检查通过；后两项含五组合/20更新、三窗含空窗、独立根/冻结/SGD/AdamW与恢复 |
| standalone eager | FP32/FP16各FC、HB、nonaligned KV三组，各4次更新、完整native checkpoint及独立CPU参考通过 |

CPU生产eager未再变化，继承[e0d6e27资格](core5-eager-boundary-20261029.md)的517 passed/2原FP16 strict xfail；
这不是在15c3e65重跑517项。以上均是相关限定资格，不覆盖所有配置/尺度的笛卡尔积。
新源码没有CUDA构建或GPU真机资格；GPU依用户延后，x86_64没有目标机器。
原D64/D2048/反馈FP32、FP16 eager GDN与默认异步双NPU限制仍由[逐项表](../ROADMAP.md)保留。

## 已保留的失败与初始化边界

开发中的数组类型编译错误、CANN9.1旧合并对象链接错误分别保留；后者用新的resident构建目录解决。
首次新消费者误用精确Python浮点loss比较，后续暴露native buffers/canonical、无尾哨兵offset、
全别名trainable与物理owner槽的区别。修正按原公开布局和原数值容差进行，原failed记录没有改为passed。
最终真实NPU消费者同时核对梯度、槽和精确恢复，fake布局测试不能单独认证它。

首次固定Python资格把CPU设施和NPU测试放在一个进程，15 passed/9 failed：
CPU autograd已启动后才注册torch_npu，NPU反向报`device_ready_queues_`内部断言。
随后同源码独立NPU进程14项+6组eager全部通过；原failed/1保留。
另两个全新进程的标量探针确认：late顺序exit1且同断言；先`resolve_device("npu:0")`
再运行CPU与NPU反向的early顺序exit0。探针程序原文/退出码在JSON，诊断状态confirmed不表示late通过。
应用在第一次autograd前注册所需NPU；晚注册仍是该运行时限制，当前门禁分进程。

服务均在background.slice、单线程ATen/BLAS：SDK上限2100秒/16GiB，Python上限1200秒/16GiB，
排队180秒；初始化小探针360秒/8GiB、每顺序45秒。SDK实际资格约713秒设备占用，Python约197秒。
所有服务已终态；systemd属性不证明独占资源或硬隔离。

## 从干净仓库复现

按[开发](../development.md)构建匹配CORE/RESIDENT；[设备入口](../device-control.md)区分两个ABI。
用新输出目录、冻结源码和上述有限资源运行：

```sh
python scripts/verify_device_control.py --build-dir RESIDENT --device npu:0 \
  --checks weighted-kv-update weighted-allocation weighted-flow weighted-training weighted-node-training \
  --output-dir NEW_SDK
python -m pytest -q tests/test_weighted_training_observation.py tests/test_weighted_benchmark_precision.py
```

SDK节点训练需要两张逻辑卡。公共NPU测试在另一个新进程，设置`TIDE_BUILD_DIR=CORE`、
`TIDE_RESIDENT_LIBRARY=RESIDENT`、`TIDE_RESIDENT_DEVICE=npu:0`，选取JSON记录的四个pytest入口及KV条件。
两套eager的完整命令及三图/两dtype由JSON的stages保存；使用本机当前匹配库替换旧构建绝对路径。
核对源码与build hash，保留CANN PYTHONPATH；无需依赖旧运行目录或修改公共工具链。
