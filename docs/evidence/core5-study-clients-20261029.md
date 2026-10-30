# core-5 配置目录与三个消费者资格

固定数学0770593。设施实现`372eb874245858de727376cb2d7981970a732631`；
resident半精度参考与串行恢复修正`e50b6d8e90fd71879e860250baa81d294b10ee95`。
两次均先提交，再运行只读checkout；[小型机器记录](core5-study-clients-20261029.json)保存
源码、build/binary/log hash和每阶段终态。性能单独验收，不从小图资格外推。

| 固定源码/环境 | 完成的检查 |
| --- | --- |
| 372eb87，aarch64 Torch2.12 CPU | 新core构建46.53s；365 passed，0 skipped，153.61s；含静态目录、独立公式、参数/共享、真实拓扑训练和恢复 |
| e50b6d8，同CPU环境 | 43项相关检查通过；fresh standalone客户端build/loader；CPU6个消费者配置通过 |
| e50b6d8，standalone Torch2.10/CANN9.0 | fresh客户端build/无Python loader；NPU8个消费者配置通过 |
| e50b6d8，Python/Torch-NPU2.12/CANN9.1 | 实际resident FP32单卡、FP16单卡、FP32双卡3项通过；真实梯度/SGD/master、磁盘换owner逐值恢复 |

CPU6组为KV/SSD各FP32 packed prefill/2 workers、FP64 scalar streaming、FP16 packed prefill。
NPU8组为KV FP32 mixed-a/b/c/resident，KV FP16 mixed-c/resident，SSD FP32 mixed-c/resident。
每个消费者warmup后执行实际SGD更新，核对全部记录/参数、最后梯度/动量/适用master，
另保存checkpoint、重建owner并精确核对下一窗口。整数/路由/None连接严格，未调整容差。
CPU43项由原存储参考9、消费者精度10、自然recipe完整训练24组成，和365有重叠，不累加为独立总资格。

自然recipe24项覆盖FC/HB/LH启发连接×五记忆及混合，另有HB教材位置/qry/att、
混合、bern/ent、分数零温、稠密Attention小组合。每项用CPU FP64、两步SGD、独立参考调度和新owner恢复；
局部公式、独立梯度根仍由对应模块测试负责。目录所有候选静态构造通过不意味着所有组合实测。

2026-10-30覆盖审计补注：上述24项中的FC统一使用depth=1，主体没有层间边，FC mixed实际为KV。
原通过数、源码和结果保留，但不能据此认证多层FC或FC异构组合。HB/LH mixed实际含KV/GDN/SSD。
后续修正使用多层FC、两个中间区域及实际模块/事件断言；新资格单独记录，不能追溯提升本报告范围。

实施没有更改数学/图/checkpoint公共身份。参数、输入和解码来自候选实际计算；
性能参考只在计时之后运行。三个调用入口保留dtype/scoring、线程、packing、执行位置与peer。
resident的FP16存储/FP32算术与eager半精度分开参考；每类公开字段先检查dtype再无损比较。
独立参考只显式添加教材State/Full存储边界，用CPU调度与PyTorch/LibTorch SGD，不调用resident求解或解码。

原失败如实保留：开发编译缺头文件；SDK私有启动环境缺入口；runtime未启用half；
旧resident构建器hash不匹配；Python以eager半精度中间记录对照resident FP32算术。
修正均经过新目录验证。开发SDK另一次编译/loader完成后source-state守卫失败，终态源码一致，
没有将其标为构建通过；后续设备cwd移出源码树，并重新干净构建/验收。不能据此断言旧守卫原因已完全证明。

长任务使用background.slice，half=160构建线程（320可用CPU）、256GiB构建上限，
Python设备检查64GiB，单任务有限超时；SDK1卡/Python2卡由队列选择空闲设备。
共享栈保持只读。CUDA新客户端未构建/真机未验，x86未验，既有数值/异步/资源边界不由本资格解除。

干净checkout复现（按[开发环境](../development.md)分别激活配套Python与SDK，长任务使用本机持久服务）：

```sh
git checkout e50b6d8e90fd71879e860250baa81d294b10ee95
python scripts/build.py --backend cpu --build-dir build/cpu --jobs half
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q tests/test_weighted_storage_reference.py \
  tests/test_weighted_benchmark_precision.py tests/test_core5_recipe_training.py
python scripts/build_weighted_benchmark.py --core-build build/cpu --output-dir build/client-cpu --jobs half
python scripts/check_weighted_benchmark.py --binary build/client-cpu/client/weighted-benchmark \
  --native-library build/cpu --output-dir artifacts/new-cpu-check
# 在独立SDK环境构建core和resident；不能把CPU/Python-owned库混入SDK。
python scripts/build_weighted_benchmark.py --core-build SDK_CORE --resident-build SDK_RESIDENT \
  --output-dir build/client-sdk --jobs half
python scripts/check_weighted_benchmark.py --binary build/client-sdk/client/weighted-benchmark \
  --fixture-dir artifacts/new-cpu-check --device npu:0 --output-dir artifacts/new-sdk-check
```

Python设备检查设置`TIDE_BUILD_DIR`、`TIDE_RESIDENT_LIBRARY`和`TIDE_RESIDENT_DEVICE=npu:0`，
显式提供两张可用逻辑设备，在新进程先`resolve_device("npu:0")`再运行
`tests/test_weighted_benchmark_resident.py`。首次CPU autograd之后才注册NPU仍属已知运行时限制。
完整配置/命令与修改维护映射见[core5-study](../core5-study.md)，无需历史运行目录才能构造这些输入。

`d6d2fa7`后续仅扩展结果查询：`index_core5_study.py`可读取仓库审阅JSON，不要求历史run目录。
固定只读checkout的5项相关检查通过（23.79s），包含无Torch CLI、非法配置与缺梯度/恢复拒绝、
已退役路径及“失败记录有计时也不能通过”的真实子进程反例。没有改执行器或扩大性能资格。
