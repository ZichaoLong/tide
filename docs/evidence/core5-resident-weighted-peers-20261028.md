# Core-5 常驻样本副本与master断言复验

固定standalone检查源码 `7db0b08e33c90b3fc278f2acdbce2ba738297310`；Python-owned
源码 `672e9484ea9ca6033f2b09ab1efaf57db56eefea`。固定上游
`0770593838a37c790a568226930567ed60f99b83`，2026-10-28审阅。
[JSON](core5-resident-weighted-peers-20261028.json)保存源码、构建、二进制、loader、日志及XML哈希。

| 独立门禁 | 实际结果 |
| --- | --- |
| CPU exact-identity guard | 4通过、0 skip，pytest .724秒；int64大值、signed zero、细小FP32与half次正规差异 |
| standalone NPU公共训练，FP32/FP16各自 | 5配置、20更新，每轨迹3个连续窗口；修正后的独立master断言通过 |
| standalone NPU双卡样本副本，FP32/FP16各自 | 2配置、6次共同更新；完整私有根、共享梯度和、两优化器、恢复及error25全体拒绝通过 |
| Python-owned双卡 | 4项通过、0 skip，62.496秒；两dtype、KV/PDG/SGD与SSD/HB/AdamW、三更新及磁盘恢复 |
| CUDA standalone | 264步构建、两入口loader/help通过；选定68条CUDA编译均为sm80/90/100，无真机结论 |

三个任务终态passed/exit0，对应unit复核inactive/success/0，源码和实际二进制身份一致。
NPU standalone为Torch2.10/CANN9.0，Python-owned为Torch2.12/CANN9.1；两批均物理9/13→逻辑0/1。
两种runtime独立构建，不能混用扩展或外推资格。每任务1800秒、2个build worker、ATen/BLAS1、
Nice10和独立background.slice；两卡排队180秒、设备资格600秒。

[样本副本契约](../resident-weighted-peers.md)对应公共
`WeightedTrainingSession::step_replicas` / `WeightedResidentTraining.step_replicas`。
各卡拥有完整图及私有状态；canonical参数梯度求和，连接位取或，不按卡数平均。
完整optimizer状态先精确比较，所有proposal经过共同拒绝决策才提交，FP16发布经过存储舍入。
单样本的区域、Attention、窗口和私有状态未被此接口切分。

standalone检查在`weighted_training_peer_check.cpp::run`使用两条独立CPU轨迹，覆盖非对齐KV和反馈异构图。
它比较所有状态、控制、逐边/端口、pending、独立输入/左切面根和canonical参数贡献；
输入只含静态图、实际参数及外部记录，参考没有向候选提供事件序列。
独立NamedOptimizer核对共同更新、FP32 master与实际存储值，恢复后继续更新；
None/连接零、先累计再共同更新、通信容量拒绝、同代optimizer槽不一致拒绝均实际执行。
Python独立时间递归、torch.optim和磁盘恢复提供另一调用边界的证据。

梯度/master容差3e-5/3e-6，实际named参数1e-5/1e-6；FP16完整记录.02/.001。
离散身份、bool、int64和None/连接零严格比较；各卡optimizer与参数发布要求精确一致。
这沿用[公共训练精度契约](../resident-weighted-training.md)，没有因本次修复放宽容差。

原`core5-weighted-peer-npu-qualify`仍为failed/exit1：672e948检查器两处借用
`registry.owners()[j]`临时vector元素，FP32首例触发std::bad_alloc，FP16未运行。
GDB在原固定binary抓到检查器run()/std::string::reserve异常；该诊断exit0只表示抓栈完成。
遗留CANN scratch已核对进程终止后，带哈希移到原诊断artifacts目录保留，未删除或改写原结果。
7db0b08只修上述检查器及`weighted_training_trajectory.cpp`的最后master断言；生产数值/通信/optimizer代码未改。
新资格补齐[21ed86d报告](core5-resident-half-training-20261028.md)中该断言的可靠证据，原批次的审计补注保留。

从干净固定源码和匹配工具链复现（本机持久服务与环境见[开发入口](../development.md)）：

```sh
python scripts/build_device_control.py --core-build CORE --build-dir NEW_BUILD \
  --runtime standalone --ascendc-soc SOC --jobs 2 --checks weighted-training weighted-peer
python scripts/verify_device_control.py --build-dir NEW_BUILD --device npu:0 \
  --output-dir NEW_CHECKS --checks weighted-training weighted-peer
TIDE_BUILD_DIR=PYTHON_CORE TIDE_RESIDENT_LIBRARY=PYTHON_RESIDENT TIDE_RESIDENT_DEVICE=npu:0 \
  python -m pytest -q tests/test_resident_weighted_peers.py
```

两卡必须在进程外正确映射到逻辑0/1。CUDA构建另用`--cuda-architectures '80;90;100'`，实际GPU验收待硬件。
本报告不认证单样本节点切分、性能或新安装消费者；eager异步双卡、GDN FP16 AdamW及SDK affine严格见证仍单列。
