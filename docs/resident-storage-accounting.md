# 常驻存储预算与服务诊断

本页解释 `tools/device_online/` 的存储计费与构造诊断，不改变数学或预算限额。
固定旧实现的审阅范围及历史失败见
[47704c9审阅](evidence/resident-workspace-audit-20261110.md)。
[554f2f5固定资格](evidence/interface-heads-qualification-20261110.md)保存修复前的CANN构建、服务检查和拒绝见证；
该历史证据不认证本页后续修复；后继的独立构建、真实计费修复/Describe执行、
half拒绝见证与保留失败见[e6f7bd5固定资格](evidence/additive-interface-qualification-20261110.md)。

## 预算含义

`DeviceProgram::retained_tensor_bytes` 对实际底层 storage 去重，同一 storage 的多个
view 不重复收费；其字节数包含完整 storage。程序、owner 持有的额外张量及合作程序
共同参与 union。算子 workspace 单独预算，不能与张量余额混用。

`WeightedFlow::Impl::budget_snapshot` 将 owner 限额分成实际唯一 storage 和尚未分配的
journal 预留。`remaining` 保持原计算及拒绝条件。构造失败会附加以下可复查信息：

- `owner_limit`、`owner_unique_used`、`journal_unallocated`、`owner_available`；
- coordinator 本地的 `node`、`phase` 或 `region`，Full 的额外输出预留；
- KV query 的 rows/banks/cache/width、原 coarse reservation 及 half raw widening；
- RowGraph 的 expression/kind/role、累计 expression bytes、owner storage increment及local program delta。

`local_program_delta` 包含首次进入该程序的借用 storage，不能直接叫“新分配量”。
局部构造现由 `WeightedStorageIncrement` 固定调用者借用集合，计费为
`bytes(program_after ∪ borrowed) - bytes(program_before ∪ borrowed)`。
Describe、Recurrent、KV Full在构造前列出输入/元数据/参数；RowGraph反向列出实际保存的
前向tape及独立cotangent。多个view与重复声明在同一storage上去重，不会反复获得抵扣。
遗漏借用声明仍保守收费，不能让参数逃出计费；开始后不能再添加新临时量作为借用。

RowGraph的累计expression bytes与实际新增storage两项检查取逻辑OR，不相加。
调用者仍负责将所有借用量计入owner总预算；`WeightedFlow::remaining`保留原union、
尚未分配journal及完整总限额检查。算子估计和workspace守卫不放宽。
Recurrent外层的前向/反向组合守卫也使用同一增量含义，避免子模块修复后又被外层重复收费。
这只解除已复现的首次借用边界，不把尚未确认的历史失败归为同一原因，也不降低保守KV query guard。

后继单bank投影按[整块矩阵契约](resident-weighted-projection.md)保存参数一次；KV query、Full及
Residual的构造预留因此将该矩阵项从行数乘积中分离，其他窗口/向量/元数据项仍保守计费。
构造末尾另检查实际新增storage union，反向通过`weighted_saved_tensors.h`枚举既有typed tape的借用集合。
该枚举只负责计费，不复制、恢复或重放数值。预留减少不改变owner总预算、journal预留或operator workspace，
不得把源码公式的减少写成实测HBM或旧规模已通过。缺席行、共享根、保存值及真实容量不足分别验证。

## 服务与 workspace

`TaskServices::statistics()` 输出 phases、实际独立 worker programs、最大构造嵌套
和共享 workspace 高水位。CANN 每个 service 有独立 worker；CUDA 的 phases 位于
同一个条件图，因此 worker programs 为零，不能用相同 phase 数推断相同流资源。

成功返回的 flow stats 增加 `local_service_phases`、`local_service_worker_programs`、
`local_service_maximum_nesting` 和 `local_service_workspace_high_water`。这些只描述
coordinator 本地服务；peer workspace 仍由 `operator_workspace_bytes` 的既有总量报告。
CANN 已按严格请求/完成次序共享一个高水位 arena；本改动未改变服务调度或生命周期。

CANN服务停止沿构造时的父子树传播：coordinator只通知直接子服务，各父服务在空闲入口收到
停止请求后，依次停止自己的直接子服务并等待完成，最后才确认自身完成。所有后代仍由设备通知
关闭，正常数值调用、窗口顺序和共享workspace的先后关系保持；不把全部后代的关闭命令累加到
coordinator同一捕获流。流数量没有因此减少，也不表示任意大图不再有任务数/流数限制。
`task_services_check.cpp`用双根、兄弟及三层嵌套，依次执行3/0/5次循环并重新提交窗口，检查
实际数值、计数、workspace和完成通知。完整图反向与恢复另验。

`scripts/trace_resident_step.py CONFIG --site SITE --output-dir NEW`在真实python-native resident
初始step上持久记录construct/advance/cotangents/backward/optimizer及错误堆栈，保存配置/安装身份。
它不重写数值步骤；advance后多读一次metrics，仅用于诊断。按同一空闲设备准入和持久服务规范运行。
可显式加`TIDE_CANN_TRACE_OPS=1`观察捕获流和最后任务号；诊断host时间不是吞吐测量。

## 独立小见证与入口

`weighted_budget_check.cpp` 用一行、64个32×32 FP32参数库、32 KiB局部余额，比较
局部程序增长与从开始就持有 input/owner/error/参数库的全局 union 增长。
它要求该原失败组合成功构造，同时证明实际新owner storage小于余额、原局部增长
大于余额，差额至少包含完整参数库。随后将余额设为实测新增量减1，要求真实不足拒绝；
遗漏参数借用声明的另一个构造也必须保守拒绝。包含重复bank与view，输出实际字节数。
无需运行真实模型、预跑轨迹或耗尽设备内存。

`weighted-budget-construction` 注册为构造见证，验证结果中的
`evidence_kind=construction` 和 `scope` 保持此边界。成功意味着构造修复及上述
存储差额/真实不足断言通过；它正常退出0，不制造资格流程失败。错误的拒绝原因、
断言失败或非零退出仍失败。`profile_device_control.py` 在读构建/启动profiler前拒绝
该用例，不能以初始化时的ATen操作冒充常驻数值执行。入口契约由
`test_resident_construction_gate.py` 在不加载设备的条件下核验。

`weighted_describe_check.cpp`另有真实数值见证：64个参数bank、128KiB局部额度，
参数总量大于局部额度；完成Describe后两次改变设备owner，核对实际score/w/mu/evidence。
它与原有八类完整字段/一阶VJP检查一起通过`weighted-describe`执行，不能把构造检查冒充数值执行。

最小构建与诊断入口为（在固定源码、匹配standalone core和当前工具链中运行）：

```sh
python scripts/build_device_control.py --core-build NPU_STANDALONE_CORE \
  --build-dir NEW_DEVICE_BUILD --ascendc-soc Ascend910_9392 --jobs all \
  --checks services weighted-budget-construction weighted-describe weighted-recurrent weighted-kv-full
python scripts/verify_device_control.py --build-dir NEW_DEVICE_BUILD \
  --output-dir NEW_DIAGNOSTICS --device npu:0 \
  --checks services weighted-budget-construction weighted-describe weighted-recurrent weighted-kv-full
```

目标仅在 standalone device build 中生成；必须使用项目当前工具链、指定空闲设备
和主任务统一的资源安排。`task_services_check.cpp` 另核对双根/兄弟/三层嵌套的实际计数、
workspace 共享与重复窗口。构造见证、真正执行、CUDA源码/构建及GPU真机证据分别记录。

修改预算公式时同步维护本页、受影响组件契约、相应设备小见证及固定源码运行证据；
当前状态和未解除项由 STATUS/ROADMAP 维护。
