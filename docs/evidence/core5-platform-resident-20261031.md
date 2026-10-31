# 单卡resident大batch与布局边界（2026-10-31）

本报告目前含CUDA六项resident及两项同材料eager；NPU仍待实际验收。固定Python bd2ff2e、公共设备库源de999c2；RTX3090/sm86、Torch2.12.1+cu130，FP32/FP32评分、TF32关闭、ATen/interop/worker1。resident通过公共设备库执行，不称纯PyTorch设备调度。设备就绪/选择/合批/推进协议未修改，实际CUDA时间线证明引用[固定完整流程](core5-cuda-finite-20261031.md)；本轮新增的是大batch实测计数与完整检查，不将主机提交称设备进度。

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

有界B128/B512离散与精确续接通过，原严格数值超差保持。小训练是d8/8190参数、B2/P3/span12，4阶段，完整独立根/梯度/SGD槽和恢复通过；不能外推大batch训练。原默认拒绝不改成通过；有界配置有自己的身份。allocator不含驱动隐藏内存；详细构造、恢复峰值和严格差异见归档。

d16 resident与旧d128 eager不是同一模型，因此此阶段不计算其加速比。新`*-resident-match.json`已固定同材料eager配置，完成后再比较。没有仅凭填充比例重写布局或多head调度；重启需要实际提交/算子时间和保持在线就绪、缺席与完整字段解码的方案。

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

归档SHA256：`4fd0bee5de7bc458cd99157989ae102a0461cf2f8fbf860e8844498eaffb3648`。
