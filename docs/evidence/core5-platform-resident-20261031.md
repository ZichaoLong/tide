# 单卡resident大batch与布局边界（2026-10-31）

本报告含CUDA六项resident及两项同材料eager、NPU六项resident。固定Python bd2ff2e、公共设备库源de999c2；RTX3090/sm86、Torch2.12.1+cu130，FP32/FP32评分、TF32关闭、ATen/interop/worker1。resident通过公共设备库执行，不称纯PyTorch设备调度。设备就绪/选择/合批/推进协议未修改，实际CUDA时间线证明引用[固定完整流程](core5-cuda-finite-20261031.md)；本轮新增的是大batch实测计数与完整检查，不将主机提交称设备进度。

## 默认与显式有界布局

FC4/2区域、KV、d16、31,990可训练参数、B1/128/512独立staggered样例、P2/span8。warmup1、连续2窗、独立完整记录/离散检查与磁盘精确续接。原默认B128的物理布局约16.87GiB，B512约269.61GiB，默认tensor_bytes=1GiB不足；两项在分配前明确拒绝。B512另超过现场可用HBM。这不是硬件逻辑batch上限。

对同一FC4/P2图，外部+物理体消息结构上界12B，正延迟/封窗阶段上界显式stages32，不需要乘B；queue=12B，tensor_bytes为2/4GiB。依据结构而非CPU真实轨迹，原输入、cache、batch和数学不变；完整配置与原拒绝均保留。

| 配置 | 执行/数值/离散/恢复 | 中位秒 | 实际candidate / Full / 消息 | 物理局部行 | allocator峰值bytes |
| --- | --- | ---: | --- | ---: | ---: |
| cuda-resident-b1 | 完成 / 通过 / 通过 / 精确 | 0.029189 | 7 / 7 / 6 | 12 | 8160256 |
| cuda-resident-b128 | 准入拒绝 / 未检查 / 未检查 / 未检查 | — | — | — | — |
| cuda-resident-b512 | 准入拒绝 / 未检查 / 未检查 / 未检查 | — | — | — | — |
| cuda-resident-bounded-b128 | 完成 / 超差探索 / 通过 / 精确 | 2.324339 | 672 / 672 / 576 | 1536 | 340269568 |
| cuda-resident-bounded-b512 | 完成 / 超差探索 / 通过 / 精确 | 9.472509 | 2688 / 2688 / 2304 | 6144 | 1342168576 |
| cuda-resident-training | 完成 / 通过 / 通过 / 精确 | 0.334966 | 19 / 19 / 18 | 32 | 33324544 |

B128/B512实际3阶段，物理局部/Full行1536/6144，实际候选672/2688，缺席填充各56.25%。这是物理行比例，不能直接当作56.25%的耗时可消除。logical batch未切成独立小模型；设备窗口完整执行。B1物理12/实际7行，3阶段。B128/B512 resident_tensor_bytes约331.773/1326.635MB，parameter_role_bytes144536、unique bank127960、operator workspace4MiB。总HBM、内部布局、服务资源分别验收。

有界B128/B512离散与精确续接通过，原严格数值超差保持。小训练是d8/8190参数、B2/P3/span12，4阶段，完整独立记录/原loss参数梯度/SGD槽和恢复通过；不能外推大batch训练。原默认拒绝不改成通过；有界配置有自己的身份。allocator不含驱动隐藏内存；详细构造、恢复峰值和严格差异见归档。

d16 resident与旧d128 eager不是同一模型，因此此阶段不计算其加速比。同材料eager结果见下表；NPU补充另列。没有仅凭填充比例重写布局或多head调度；重启需要实际提交/算子时间和保持在线就绪、缺席与完整字段解码的方案。

## 复现

```sh
tidegraph plan configs/matrices/cuda-resident-bounded.json --matrix --site site.json
tidegraph matrix configs/matrices/cuda-resident-bounded.json --site site.json --output-dir jobs/new-resident --detach
tidegraph results docs/evidence/core5-platform-resident-20261031.json.gz --format table
```

同组独立1B eager profile放在[规模报告](core5-platform-scale-20261031.md)，不计resident或吞吐。外层任务platform-cuda-resident-observe-bd2ff2e终态exit1，无后代；两个预期容量拒绝导致整体failed，不覆盖其他项的分别验收。多卡resident仍受P2P/原子协议门禁，未获此单卡资格。

## 同材料eager对照

固定bd2ff2e，配置9a1cd96；d16/FC4/B128或B512/P2的模型、输入、任务、FP32、完整返回、线程与resident逐项身份核对一致。
eager的B128严格通过、B512数值超差；两项执行/离散/完整检查/精确续接均通过。不能以自身恢复称B512严格等价。

| B | eager mixed-c中位秒 | resident中位秒 | eager / resident全程allocator峰值bytes |
| ---: | ---: | ---: | --- |
| 128 | 2.154304 | 2.324339 | 18,004,992 / 340,269,568 |
| 512 | 8.680192 | 9.472509 | 45,964,800 / 1,342,168,576 |

相同材料下resident当前没有成本优势；这是一次有限流程的成本观察，不是严格等价加速比。
两者candidate/Full/消息相同；resident保留缺席行和设备服务成本，不能从56.25%填充直接推断可优化的时间。
任务platform-cuda-resident-match-bd2ff2e于06:38:06Z终态exit0，无supervisor/workload/remaining后代。

## NPU实际resident与服务任务容量

固定Python bd2ff2e、公共设备库eca070f，Torch/TorchNPU2.12、CANN9.1，队列物理11→npu:0。
FP32 payload/scoring、HF32关闭、ATen/interop/worker1，与CUDA同model/workload/task及完整返回。
本轮未修改设备协议或把主机逐事件推进称为resident；原完整NPU协议/独立根资格仍按各自源码与规模引用。

| NPU配置 | 执行 / 数值 / 离散 / 恢复 | 中位秒 | 实际candidate / Full / 消息 | allocated峰值bytes |
| --- | --- | ---: | --- | ---: |
| npu-resident-b1 | 完成 / 通过 / 通过 / 精确 | 0.034157 | 7 / 7 / 6 | 20945920 |
| npu-resident-b128 | 准入拒绝 / 未检查 / 未检查 / 未检查 | — | 未执行 | 未采得完整峰值 |
| npu-resident-b512 | 准入拒绝 / 未检查 / 未检查 / 未检查 | — | 未执行 | 未采得完整峰值 |
| npu-resident-training | 完成 / 通过 / 通过 / 精确 | 0.647399 | 19 / 19 / 18 | 112547840 |
| npu-resident-bounded-b128 | 完成 / 超差探索 / 通过 / 精确 | 1.653417 | 672 / 672 / 576 | 354291712 |
| npu-resident-bounded-b512 | 构造失败 / 未检查 / 未检查 / 未检查 | — | 未执行 | 未采得完整峰值 |

B128两个测量窗1.640338/1.666496秒，3阶段，1536物理局部/Full行与672实际候选，缺席比例56.25%；
operator workspace16,777,728 bytes，resident tensor331,771,837 bytes。独立首差window2/trace380/descriptor evidence，
1/16坐标差1.13947e-6，原atol1e-6/rtol1e-5不变。完整离散与磁盘精确下一窗通过，不称严格等价吞吐。
小训练只验本任务loss梯度/参数/SGD槽及恢复；不能由该loss推断任意独立VJP根已重新验收。

默认B128/B512同CUDA一样被原物理布局预算拒绝；显式有界B512已通过现场HBM准入：
估计1,102,675,652 bytes、当前usable61,348,970,496 bytes，随后构造报aclnnInplaceCopy/361001。
对应进程厂商日志明确：`The number of pending tasks in the stream exceeds the limit. ErrorCode=EE1019`，
`MEMCPY_ASYNC`、`rtMemcpyAsync ErrCode=507002 / stream task buffer full`，本次失败复制count=1。
这定位到构造所捕获的流任务容量，尚未进入warmup；只采得runtime_ready零allocated，不能称构造峰值为零或显存不足。

源码`cann_program.cpp:DeviceProgram::finish`将延迟命令录入绑定模型；
`weighted_flow_copy.cpp:weighted_copy`对非连续bool目标保留静态连续段复制，以避开旧bool ScatterUpdate失败。
该路径会生成很多小copy，但当前日志不足以将第3688个CopyNpuToNpu唯一归到某个调用者；不能伪造排他profile。
本轮停止扩大该resident配置，保留eager B512结果和原默认拒绝。后续先对静态copy调用者/段数定点观测，
再比较有界的bool批量搬运（例如独立验证整型暂存与还原）或设备服务拆分；必须验证未选择字段不变、
int64索引、重复owner、完整解码、独立根/窗口/恢复及暂存预算。绑定模型内不能靠主机逐事件flush冒充修复。
增加HBM预算或减少逻辑batch不是本错误的修复；相同配置不自动重试。

父任务07:31:56Z exit1、systemd控制组空；六子任务均终态且remaining_live_pids为空。
三个失败保留原配置、结果和log；厂商短日志摘要及原hash保存在本归档最后一项details.post_run_diagnostic。
原plog SHA256 `cd81603b9377d9cedd8a2381726d4301e50e73c4d19c3531deb5f3e57e7f1462`。

## CPU/NPU同材料eager补齐

固定bd2ff2e、native eager、d16/FC4/KV/B128或B512/P2，packed/scalar完整返回，ATen/interop/worker1。
原目录resident-match与resident模型/任务身份逐项对应；这里CPU表示CPU候选，不将其当resident实现。

| 平台/执行 | B128中位秒 | B512中位秒 | 独立数值 / 离散 / 精确恢复 |
| --- | ---: | ---: | --- |
| CPU/eager | 1.023296 | 3.724563 | B128全部通过；B512数值超差、离散与恢复通过 |
| NPU/mixed-c | 3.641601 | 21.963012 | 两项全部严格通过 |

每窗B128有672 candidate/672 Full/576消息，B512为2688/2688/2304；逻辑样例数与时间/跨节点行数分列。
NPU resident B128观测1.653417秒，小于同材料eager的3.641601秒，但resident数值超差，不能称严格等价加速。
NPU resident B512构造任务资源失败，而eager B512完成；选型必须同时看数值、服务容量和成本。
这四个子任务均终态exit0、remaining_live_pids为空；父链仍运行其他容量项不改变其验收范围。

归档SHA256：`336986f8557fcc542fc5cfaec53810b378a6e107cb7dedf1abe9909d4bbc5a25`。
