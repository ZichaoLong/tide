# 常驻存储预算与服务诊断

本页解释 `tools/device_online/` 的构造诊断，不改变数学、预算限额或拒绝条件。
固定旧实现的审阅范围及历史失败见
[47704c9审阅](evidence/resident-workspace-audit-20261110.md)。
[554f2f5固定资格](evidence/interface-heads-qualification-20261110.md)已完成CANN构建、服务检查和设备小见证；
CUDA本增量仅有源码审阅，未新增构建或真机资格。

## 预算含义

`DeviceProgram::retained_tensor_bytes` 对实际底层 storage 去重，同一 storage 的多个
view 不重复收费；其字节数包含完整 storage。程序、owner 持有的额外张量及合作程序
共同参与 union。算子 workspace 单独预算，不能与张量余额混用。

`WeightedFlow::Impl::budget_snapshot` 将 owner 限额分成实际唯一 storage 和尚未分配的
journal 预留。`remaining` 保持原计算及拒绝条件。构造失败会附加以下可复查信息：

- `owner_limit`、`owner_unique_used`、`journal_unallocated`、`owner_available`；
- coordinator 本地的 `node`、`phase` 或 `region`，Full 的额外输出预留；
- KV query 的 rows/banks/cache/width、原 coarse reservation 及 half raw widening；
- RowGraph 的 expression/kind/role、累计 expression bytes、local program delta。

`local_program_delta` 包含首次进入该程序的借用 storage，不能直接叫“新分配量”。
RowGraph 的两项检查取逻辑 OR，不相加。当前实现仍保留这一局部计费规则；诊断
不把尚未确认的历史失败原因归为重复收费，也不自动降低保守 KV guard。

## 服务与 workspace

`TaskServices::statistics()` 输出 phases、实际独立 worker programs、最大构造嵌套
和共享 workspace 高水位。CANN 每个 service 有独立 worker；CUDA 的 phases 位于
同一个条件图，因此 worker programs 为零，不能用相同 phase 数推断相同流资源。

成功返回的 flow stats 增加 `local_service_phases`、`local_service_worker_programs`、
`local_service_maximum_nesting` 和 `local_service_workspace_high_water`。这些只描述
coordinator 本地服务；peer workspace 仍由 `operator_workspace_bytes` 的既有总量报告。
CANN 已按严格请求/完成次序共享一个高水位 arena；本改动未改变服务调度或生命周期。

## 独立小见证与入口

`weighted_budget_check.cpp` 用一行、64个32×32 FP32参数库、32 KiB局部余额，比较
局部程序增长与从开始就持有 input/owner/error/参数库的全局 union 增长。
它要求保留原 RowGraph 拒绝，同时证明实际新 owner storage 小于余额、局部增长
大于余额，差额至少包含完整参数库。输出各项实测字节数，**复现拒绝不等于已修复**。
无需运行真实模型、预跑轨迹或耗尽设备内存。

`weighted-budget-construction` 注册为构造见证，验证结果中的
`evidence_kind=construction` 和 `scope` 保持此边界。成功意味着捕获预期拒绝且
上述存储差额断言通过；它正常退出0，不制造资格流程失败。错误的拒绝原因、
断言失败或非零退出仍失败。`profile_device_control.py` 在读构建/启动profiler前拒绝
该用例，不能以初始化时的ATen操作冒充常驻数值执行。入口契约由
`test_resident_construction_gate.py` 在不加载设备的条件下核验。

最小构建与诊断入口为（在固定源码、匹配standalone core和当前工具链中运行）：

```sh
python scripts/build_device_control.py --core-build NPU_STANDALONE_CORE \
  --build-dir NEW_DEVICE_BUILD --ascendc-soc Ascend910_9392 --jobs all \
  --checks services weighted-budget-construction
python scripts/verify_device_control.py --build-dir NEW_DEVICE_BUILD \
  --output-dir NEW_DIAGNOSTICS --device npu:0 \
  --checks services weighted-budget-construction
```

目标仅在 standalone device build 中生成；必须使用项目当前工具链、指定空闲设备
和主任务统一的资源安排。`task_services_check.cpp` 另核对三层嵌套的实际计数、
workspace 共享与重复窗口。构造见证、真正执行、CUDA源码/构建及GPU真机证据分别记录。

修改预算公式时同步维护本页、受影响组件契约、相应设备小见证及固定源码运行证据；
当前状态和未解除项由 STATUS/ROADMAP 维护。
