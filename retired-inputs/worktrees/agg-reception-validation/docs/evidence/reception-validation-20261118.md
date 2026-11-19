# Reception 交付：最终集成、隔离安装与后端验证

CPU 全套 783 项通过，新 wheel 的隔离安装与实际恢复完成。双 NPU、BF16 及 static_view
路径的有限超差仍分别保留；安装完成和外层任务通过不代表全部数值等价。

[精简 JSON](reception-validation-20261118.json)保留全部作业身份、原文件 SHA256、命令、
逐项比较、服务退出与记录后代核对。本页没有启动新实验或改写原结果；
[三种接收 FP32 数值记录](reception-numerics-20261118.md)保留此前独立范围。

本页记录两个固定源码身份下的有限检查：`1fbe15c9` 的 CPU 全套、隔离 wheel 和静态
参数索引路径，以及 `1553b301` 的双 NPU FP32 放置和 light64 BF16 路径。
完整模型是 PyTorch，双卡采用 host-staged transport；本页没有全模型 LibTorch、
CUDA、peer DMA/HCCL 或 compiled/resident 训练新资格。

## 最终 CPU 集成

冻结 `1fbe15c9` 的完整 CPU 套件为 **783 passed，1 warning，501.13 秒**，XML 中
failure/error/skipped 均为 0；外层服务 passed/exit0，运行 503.443 秒。
warning 为 `tests/test_costs.py::test_cost_cli_keeps_normal_and_restored_trace_separate`
触发的 Torch profiler 周期清空事件提示，不是算子执行错误。资格属于这份冻结源码，
不能将计数重标到后来修改了 harness/配置的 HEAD。

生产 `src/tide` tree 为 `539c5eb15176827ca14a19c7de82f08df5b6c1d1`。
所有 394 个该源码归档常规文件逐字节匹配 Git archive；生成/未跟踪文件不在该核验中。
此前首轮 715 passed / 11 failed 及其修复仍由 Reception 数值证据保留，不抹去原结果。

## 新 wheel 与公开入口

冻结 `1fbe15c9b0ce900bfffbf5c148ca61c81cb34046`，从归档的单独副本离线构建 wheel，
以 `--no-deps --no-index` 安装到新私有 venv。依赖使用原有只读基线，禁用 NPU vendor
自动加载；从 `/` 移除 PYTHONPATH、使用 Python `-I` 运行。21 个安装 Python 文件与
wheel/冻结源一致，最终加载的 21 个 Tide 模块也一致；每次新解释器恢复重新校验。
wheel SHA256 `86d913952e9953acc101f004f6e7db4e3eb76150a803e3669898dfefb444b745`。

安装机制案例共同使用 compact-9 Timed，n2/h2、d64、X、region_shared、incoming_light_64，
period3、terminal Hubs，显式绝对位置时钟 period7、KV window11、short_decay0.09、
SSD stride3。B1/L4，2+2 分块保留完整梯度后 flush/detach，三次更新后保存，原进程
和新解释器各做第四次更新。全部选择 fused QKV、静态 padding mask、Reception 重算、
SSD prefix/doubling、static_view 及 region_batch_size4。

batched 为 fixed_pass Hub，4,095,584 标量参数；placed 为 bounded Hub、两个逻辑
CPU 组、per_device，4,338,560 标量参数。逻辑组都落在同一 CPU 后端，不能替代双 NPU
资格。两种路径的实际活动 Parameter 张量分别为 217 和 435，step 真实到 4；
6/9 个 None 梯度语义保留。各次完整有限窗口的状态位置为 0→7→14→21→28。

| 案例 | 推理/含图前向最大绝对差 | 新进程第4次梯度 | 新进程第4次参数/slots/state |
|---|---:|---|---|
| batched FP32 | 4.7683716e-6 | exact | exact |
| batched BF16 | 0 | exact | exact |
| placed FP32 | 6.4373016e-6 | 阈内，max_abs 5.2154e-8 | 阈内，max_abs 2.58535e-6 |
| placed BF16 | 0 | 153 字段超差，max_abs 9.765625e-4 | 132 字段超差，max_abs 1.4002621e-4 |

四例保存数据和下一前向均 exact；已比较的 support/元数据/非有限检查未出现差异。
模式前向采用额外 0/0 逐位观察，因此两项 FP32 非 exact 被记为 finite_deviation；
不能将该观察误解为统一 FP32 容差失败。恢复阈值原样为 atol1e-6/rtol2e-3，placed BF16
是真实恢复超差。外层安装任务 passed/exit0，内层仍 finite_deviation。
独立挂钩看到被观察 Linear 输出在 FP32/BF16 案例分别为相应 dtype，不能认证每个算子
的硬件累加精度。

同一隔离安装环境还从公开 examples/train.py 用 light64 Settle n8/h4、d128、
17,880,464 参数、static_view/doubling、B1/L2 做三次真实训练，保存/下一前向 exact、
配对更新阈值内；另一个 `-I` 进程从保存点做一次更新，state_position 6→8。
该入口重新生成本次 seed 的合成输入，不声称恢复随机数据流或任务训练配方。

## 两张 NPU 的区域放置

冻结 `1553b301`；物理 NPU 1+14，进程内 npu:0/1，host_stage、区域分区、per_device，
每个参数分区内兼容区域 batch4。配置是 compact-9 Settle、n8/h4、d128、X、light64，
B1/L4，FP32。源码共 391 个归档常规文件逐字节匹配 Git archive。
基准是相同初始 CPU 参数/输入的一张 NPU 普通模型；放置模型接收 owner wrappers
采用显式 separate QKV fallback。逻辑参数总数仍为 17,880,464。

| 比较 | 原阈值超差字段数 | max_abs / 说明 |
|---|---:|---|
| 初始 whole result | 1 | 全结果 5.1558e-6；失败在 KV features |
| 初始输入/参数梯度 | 0 / 0 | 输入 rtol1e-2，参数 rtol2e-3，atol 均1e-6 |
| 分块单卡/放置 state | 1 | 7.27177e-6 |
| 放置 whole/分块 state | 9 | 9.05991e-6 |
| 第1→3次参数更新 | 32→37→36 | max_abs 1.56959e-4→2.55880e-4→3.31731e-4 |
| 保存数据、下一前向 | 0 | exact |
| 恢复后实际参数更新 | 13 | 5.59695e-5 |
| 恢复后 optimizer/state | 0 / 0 | optimizer 阈内；state exact |

其余常规阈值 atol1e-6/rtol2e-3。已比较的 support/元数据均无不等，全部检查值有限，
仍标 finite_deviation。恢复是在同一进程新建 owner 后进行，不能称为新进程恢复。
阶段时间包含记录 I/O、没有阶段设备同步，不能用它推导吞吐或热点占比。
单独串行审计见证在两张 NPU 上实际执行 FP32 mm/bmm 及 npu_fusion_attention_v2；
该审计显式暂停并发，不能将其算子次序等同真实并发轨迹。

## BF16 light64 路径与活动差异

冻结 `1553b301`，物理 NPU1。compact-9 Settle、n8/h4、d128、X、region_shared，B1/L4，
chunk2，有一次 anchor AdamW 更新，随后三更新再实际新进程恢复。baseline 为单区域、
separate QKV、不启用结构 mask，Reception 重算及 SSD prefix 已启用；optimized 再选择
region batch4、fused QKV、结构 mask、SSD doubling。主阈值 atol0.002/rtol0.05，
恢复阈值仍 atol1e-6/rtol2e-3；不得混用。

baseline 为 within_threshold，optimized 为 finite_deviation。20 项比较中 10 exact，
6 项含主阈值超差，共 2027 次字段超差；无非有限、dtype、整数元数据或 None 连接差异。
两者各自 inference↔grad exact，同 owner 重复 VJP 阈内；跨候选 inference/VJP1/VJP2
分别 189/240/240 字段超差，三个更新分别 342/446/570。

第2次更新返回 strengths 的 16 个元素中 2 个支持不同：一处 0 对 0.01368936896，
另一处 0.02131661773 对 0。这是实际记录到的输出支持差异，不能被 finite_deviation
总体标签覆盖。该步输出 values 的 max_abs 为 6.1281247、最大 scaled 为 2904.57。
内部工作审计从首次比较已不同：两个 chunk 合计 positive_messages 为 baseline3781 /
optimized3783；三个更新分别为 3781/3783、3611/3610、3785/3781，最后一步 Full 总数
509/510。候选槽位均 752；按模式/更新还应查看逐 chunk 记录。

内部捕获达到 2,000,000 元素上限，未建立因果起点；组合比较没有拆出各优化的独立
贡献。本页未作新高精度重放，不把该 BF16 轨迹自动归因为已经证明正常的舍入。
两候选各自新进程恢复的保存/下一前向 exact，实际 364 个活动 Parameter 的 step4→5
更新都在恢复原阈值内、非 exact；这不消除上述跨候选活动与数值差异。

## FP32 static_view 路径

冻结 `1fbe15c9`，物理 NPU 2，进程内 npu:0；
compact-9 Settle、n8/h4、d128、X、light64、FP32、B1/L4、chunk2。输入配置选择
fused QKV、结构 mask、Reception 重算、SSD prefix/doubling；路径检查器将两候选
都解析为 region_batch_size=1。实际配置只在参数索引上不同：baseline 使用 `gather`，
views 使用 `static_view`。本次单独比较参数索引方式，不覆盖 batch4 的组合数值资格。

baseline 为 `finite_deviation`，views 为 `finite_deviation`。主比较/恢复原阈值均为
`atol=1e-6, rtol=2e-3`；20 项比较中 9 exact，4 项含超差，共 165 次超差字段。

| 检查 | baseline | views |
|---|---|---|
| 同owner重复VJP | 非 exact，0 超差字段 | 非 exact，0 超差字段 |
| 推理/含图前向 | 非 exact，4 超差字段 | 非 exact，4 超差字段 |

views 对 baseline 的无梯度推理 exact，两次 VJP 均非 exact、在原阈值内；
三个连续更新的超差字段数为 0→8→149。初次 VJP 返回树的首个差异为
`receive.embedding.weight` 梯度，同一 owner 重复 VJP 也非 exact。
这说明没有恢复或索引切换也能产生有限反向差异，但没有证明后续全部参数超差由此导致。
最后一次最大 scaled 的张量是 `root.parameters.receive.k.weight`，scaled 31.0677，
该张量 max_abs 6.72191381e-05；不将其当成全局最大绝对差。

已比较的整数元数据、dtype、非有限值、support 和 None 连通性均无差异；
单独核对的模式工作计数及连续更新工作计数也全部一致。

两候选各自新进程恢复均为 `within_threshold`：保存数据和下一前向 exact，
364 个活动 Parameter 的 step 真实从 4 到 5，实际更新非 exact、在原阈值内。

views 的内部见证达到 262,144 元素上限，返回 `capture_limit`，未建立因果归因。
返回树的首个字段仍不等于因果起点；本页没有额外高精度/算子重放，不能据这项小图
结果推定任意合批、尺度或后续轨迹等价。

## 终态、身份及复现

数学仍固定 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件核验通过。
路径记录里的 `source_dirty` 原值保留；本页对冻结归档全部 Git 常规文件的字节核对
分别建立实际源码身份，不借文档 HEAD 或外层 exit0 替代内部资格。

| 作业目录（artifacts 下） | 冻结源码 | 外层终态 | 内部结果/范围 |
|---|---|---|---|
| agg-integration-cpu-final-v1 | 1fbe15c9 | passed / exit0 | 783 passed / 1 warning |
| agg-installed-final-v1 | 1fbe15c9 | passed / exit0 | finite_deviation |
| agg-placement-light64-fp32-v1 | 1553b301 | passed / exit0 | completed；finite_deviation |
| agg-paths-light64-bf16-v1 | 1553b301 | passed / exit0 | finite_deviation |
| agg-paths-static-fp32-v1 | 1fbe15c9 | passed / exit0 | finite_deviation |

上述任务均使用 `background.slice`、有限运行时间和采样进程树 RSS 界限。CPU 全套
固定 affinity56–63，安装64–67；BF16 路径184–187，双 NPU 放置188–191，static 路径
220–223。外层均请求4线程，安装机制 harness 实际 ATen/interop 为1；具体资源、
时间/RSS界限、采样峰值和队列身份保存在 JSON。采样 RSS 不是设备显存或可靠 cgroup
硬界限，任务墙钟包含排队/记录开销，不能用于性能对照。

NPU stderr 保留 CANN/SoC 的 32 对齐额外分配提示；放置任务还保留 TypedStorage
弃用提示，均未使这些任务中断。完整原始日志随哈希保存。

NPU 使用协作队列准入，身份为 `exclusive_cooperative_queue; external jobs not controlled`；
主机共享，不声称控制队列之外的使用者。终态核对时，这五个服务均 inactive/dead、
MainPID=0，已记录的 PID/start 身份中没有仍存活的非僵尸后代；这不表示其他项目任务
全部结束。没有清理冻结源、checkpoint、wheel、环境或复现工作树。

复现使用 JSON 的 `jobs[].requested.command` 及各原 `launch.json`/`run.sh`；安装 driver、
构建/安装/恢复命令都随哈希记录。冻结源码和私有环境保持对应身份，所有结果写新目录，
设备重新按[执行契约](../execution-flows.md)准入。CPU 全套入口为：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q --junitxml NEW-tests.xml
```

隔离安装使用 `--execution-variant reception`；公开入口示例/config 属于 checkout 工具，
wheel 安装 `tide` 包。测试、安装和各个硬件点分别给出资格，不能将 CPU 双逻辑组、
恢复下一前向 exact 或 source 菜单可表达性替换为所有设备/配置的训练等价。
公共工具链、参考仓库和下游未修改，本页不声明下游发布/验收。
完整模型仍是 PyTorch；[LibTorch 区域评分切片](native-regional-performance-20261118.md)
沿用自己的原资格。不同数学模型的性能对照及各模式建议由单独证据负责。
