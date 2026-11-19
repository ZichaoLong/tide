# 新 wheel 隔离安装、可选绑定与真实恢复更新

固定源码 `8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee`，`src/tide` tree
`e4ee19932499ed75fce73531331ded317b95ef3d`；数学仍为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。详细检查、原始状态、命令、脚本和
产物 SHA256 在 [JSON](capacity-installed-20261118.json)。本页仅声明这些源码与测点。

## 安装身份及边界

从固定归档复制构建目录，离线构建新 wheel，再以 `--no-deps` 安装到全新私有 venv。
使用公开 `ascend/dev-workspace-9.2.0` 的只读依赖；实际 Python 3.12.13、
Torch `2.12.0+cpu`，禁用 vendor autoload。公共环境、参考及下游均未改动。

wheel SHA256 为 `72e3eba6bb3d572c2fec9a3afe16617331f37c2fa2c616bffc36b3c5100f0e27`。
从 `/`、移除 `PYTHONPATH`、Python `-I` 执行公开 API；21 个 wheel Python 文件
与冻结源码/安装文件逐一匹配，最终加载的 21 个 Tide 模块均来自新 venv 并匹配 wheel。
每例恢复都启动独立 Python 进程并重新检查安装身份。源码归档 315 个文件运行前后相同，
108 个固定数学文件核验通过。构建只修改单独副本。

## 四个实际案例

共同配置是 `multiscale/compact-9`、n2、hubs2、d64、X memory、region_shared、
Timed period3、terminal_hubs；显式时钟为绝对位置 period7、KV window11、
short_decay0.09、SSD stride3。B1/L4，每次训练保留 `2+2` 两 chunk 间的完整梯度，
随后 flush，再 detach；三次更新后保存，原进程和新进程各执行真实第四次更新。
状态位置为 `0→7→14→21→28`，输入位置数与 flush 逻辑时刻分开。
这是安装/机制检查，不是完整训练性能计时。

- **batched**：fixed_pass Hub、region_batch_size4、checkpoint_regions、local math；
  实际分组含 1/2/4 区域，3,301,216 参数。
- **placed**：bounded Hub、checkpoint_regions、两个 CPU 逻辑组与 `per_device`；
  3,544,192 参数。最多两 worker，挂钩投影确实在 caller 与 worker 上执行。
  两逻辑组最终都在同一 CPU 后端，不能据此声明双 NPU、通信或设备加速资格。

| 案例 | 推理/含梯度前向最大绝对差 | 新进程第四次梯度 | 新进程第四次更新与状态 |
|---|---:|---|---|
| batched FP32 | 3.8146973e-6 | 精确 | 精确 |
| batched BF16 | 0 | 精确 | 精确 |
| placed FP32 | 6.6757202e-6 | 原阈值内，最大1.0430813e-7 | 原阈值内，最大参数差8.3371997e-6 |
| placed BF16 | 0 | 161 个张量超差，最大1.5869141e-3 | 134 个参数/slot等张量超差；最大参数差1.2435019e-4，最大 moment 差1.5869178e-4 |

恢复比较保持 `atol=1e-6, rtol=2e-3`。四例保存的参数、optimizer、私有状态、配置，
加载的数据和下一次前向全部精确；实际更新后的浮点参数、梯度、slot和状态均有限。
batched 的6项、placed终端Hub的9项 None 梯度在恢复后保留，不能把这些断开项当零梯度
补齐。各轮实际更新的参数张量数分别为204和409，optimizer step均推进到4。

FP32/BF16参数、梯度与AdamW slots实际为FP32。BF16案例的32个KV payload张量为BF16，
其余浮点私有状态为FP32，48个时间/索引张量为int64。单独的挂钩观察确认被检查的
`nn.Linear` 投影输出在两精度下分别为FP32/BF16，覆盖推理、含梯度前向、训练与恢复。
输出dtype观察不能认证未观测算子的硬件累加实现；稳定子计算政策见[精度契约](../precision.md)。

## 原检查分类问题与保留的失败

原 harness 为 `9b220bb5`。它把推理/含梯度前向的零容差观察同时用于“结构失败”分类，
因此两项有限FP32差异被标为 `structural_failed`，服务最终 `failed/exit2`。
其整数/布尔元数据检查均无差异，也无非有限值。原四例 JSON、0/0阈值、标签和退出状态
全部保留，不能因修复分类而改写为通过。

`cd131fc2` 只修正今后 harness 的分类：浮点非精确与 support 差异单列，元数据、
非有限、None/梯度连接、保存数据和恢复下一前向仍有各自严格检查；9个相关边界测试通过。
0/0是额外逐位观察，不能替代仓库FP32/BF16正式容差，也没有据此降低原阈值。
没有为改变标签而重跑四例矩阵。原 harness 未单独枚举浮点strength的support；
BF16的精确前向建立相同support，FP32这两次非精确前向不能由本记录认证support等价。

原 shell 在exit2处结束，冻结源码不可变检查和数学核验随后另行完成，命令及原因在
`postflight.json`。这不修改原服务结果。

## 一次限定 BF16 恢复诊断

以原 placed BF16 第三次更新后的 checkpoint 和输入为唯一数据，在同一个新进程中加载
两个owner A/B。参数与初态不变，顺序执行 A并发、B并发、A并发、B并发、A串行、A串行
六次完整反向；串行 witness 使用公开 `serial_region_audit`，其上下文覆盖重计算反向。
之后 A 使用最后一次串行梯度、B 使用最后一次并发梯度，各真实执行一次 AdamW。
诊断依然使用同一 wheel；冻结 driver、源码和完整命令包含在JSON中。

| 检查 | 观察 |
|---|---|
| 六次前向及状态 / None语义 | 全部与原进程第四次精确；9个断开参数一致 |
| A同owner两次per_device反向 | 158张量超原阈值，最大梯度差1.7089844e-3 |
| B同owner两次per_device反向 | 梯度精确 |
| A同owner两次serial反向 | 梯度精确 |
| A serial / A per_device | 169张量超差，最大梯度差1.4648438e-3 |
| B随后真实更新 / 原进程第四次 | 参数、slot及状态精确 |
| A随后真实更新 / 原进程第四次 | 142张量超差，最大参数差1.3186541e-4 |

该点已经在**同owner重复反向**中出现差异；改变线程构图/累加次序可改变BF16梯度，
而checkpoint数据、前向和None语义精确。新进程的B更新后来可以精确复现原进程，说明
checkpoint损坏不是原超差的必要解释。这里没有逐算子归因全部梯度差异，也没有证明
串行调度在所有配置/后端都精确或消除原超差。

四个选定坐标用独立标量FP64 AdamW、实际梯度和保存moments预测更新差；预测与实际
更新差的最大偏差为 `2.6461e-11`。它只说明这些梯度差足以解释所选更新差，不统一解释
其他误差，不提供整个模型的FP64资格。原 placed BF16超差仍然存在于原记录中。

## 执行、资源及复现

两次任务均在 `background.slice`，CPU affinity80–83、Torch/interop线程1、最多两区域
worker、CPUQuota400%、RuntimeMax600s、MemoryMax16GiB及14GiB采样进程树RSS界限。
本机MemoryMax不是可靠的cgroup硬限制，实际使用项目supervisor的采样界限。
初次配额约80.5GiB/300GiB软限、诊断前约81.3GiB/300GiB；主机可用内存约1.8TiB。
没有NPU准入或设备占用。其它作业仍在运行，共享主机身份保留；本页没有速度结论。

原安装任务实际103.121s，采样进程树RSS峰值1.517GiB，终态exit2；诊断34.131s、
1.171GiB，外层exit0且内部所有数值结果分别列出。终态均MainPID0、无剩余cgroup。
这些时间包含构建/安装或侵入式诊断，不能充当正常模型延迟或吞吐。

复现原失败时，从 `8a22530d` 冻结源码，并取主分支 `8fa6690e` 的
`scripts/check_installed_capacity.py`（与原工作树 `9b220bb5` 文件逐字节相同）。在新输出目录用
`pip wheel --no-deps --no-build-isolation --no-index --no-cache-dir` 构建副本、
`python -m venv --system-site-packages` 建新环境，`pip install --no-deps --no-index`
安装所构建wheel；再从 `/` 移除 `PYTHONPATH` 并以 `python -I` 调用harness，传入
`--wheel`、`--venv`、冻结文件SHA256的 `--source-manifest` 和新 `--output-dir`。
原完整shell与manifest身份在JSON；新工作使用修复后的harness，保持原失败记录独立。

限定诊断driver在JSON的 `reproducer_files.diagnostic_driver`；把它与主分支 `22bc86af`
的harness（原工作树 `cd131fc2` 的相同文件）复制到新的冻结输入目录，使用相同已安装解释器从 `/`、`-I` 调用：
`diagnose.py --input ORIGINAL_INSTALL_JOB --output NEW_RESULT.json`。
`--input` 依赖原 `dist/`、`env/`、manifest和placed BF16 checkpoint/expected数据；
退休这些产物前需先重建对应输入。每次新执行重新核对配额、活动任务与资源，并按
[执行契约](../execution-flows.md)提交有界服务。
