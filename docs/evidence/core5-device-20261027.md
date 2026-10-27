# Core-5 带权 eager 设备资格 — 2026-10-27

固定实现 `470ce06112b888ed0b08866c3d791501b3bcd240`；固定上游 `0770593`。
**CPU 200 项通过；NPU Python/native 的有限混合矩阵与显式同步两卡配置通过。**
standalone SDK 完整门禁严格失败，单列如下；[命令入口、配置、运行与摘要](core5-device-20261027.json)。

| 固定提交范围 | 纯 Python | Python 调用 native |
| --- | --- | --- |
| 单卡 mixed-a | 2 组合通过 | 2 组合通过 |
| 单卡 mixed-b | 2 组合通过 | 2 组合通过 |
| 单卡 mixed-c | 19 组合通过 | 19 组合通过 |
| 两卡 mixed-c，ASCEND_LAUNCH_BLOCKING=1 | 2 组合通过 | 2 组合通过 |

各资格 source 为上述完整提交、dirty 为空、workload exit0、服务 inactive/success；
CPU 200 passed/0failed/0skipped，FP64/FP32；NPU FP32，Ascend910_9392/aarch64，
Python3.12.13/Torch2.12.0+cpu/Torch-NPU2.12.0.post2/CANN9.1，matmul HF32=false。
所有状态、消息存在性、路由、int64 元数据、参数别名与 None/零连接严格比较；
FP32 atol1e-6、rtol1e-5。此表是代表范围，不是所有开关的笛卡尔积。

[设备契约](../weighted-device-validation.md)连接固定教材、实现与独立检查。
19组合覆盖 FC/HB、反馈重复边 PDG、非对齐 TimedDAG、独立稀疏端口、五记忆、
异构 Attention、三种 Read 快照、十种发送、quad/Bernoulli/Shannon。
独立 CPU 标量参考仅提供比较结果；候选自行 streaming/在线 greedy prefill。
比较完整返回、实际 owner、独立输出/描述量/control/原始 State 根、共享参数和输入 VJP，
保留图切点、三步 AdamW、精确 checkpoint 与恢复后的第四步、CPU handoff。
两卡按 region 分置并验证实际跨卡消息、共享叶、反向与交换 owner 恢复；不称为 HCCL 数据并行。
局部 literal 反例覆盖 >2^54 int64 平票、fractional fill VJP、全非正 hard fallback、
packed copy 的别名/None/连接零；一般公式与独立调度证明边界见各模块既有证据。

加速器标量硬发送使用实际设备 int64/score 稳定排序和 fractional fill，CPU 标量排序独立。
厂商将 int64 Argsort 放到设备 AiCPU；这不是主机 CPU fallback，也不是 AiCore 核。
窗口内就绪、合批和推进仍在主机，不能认证 resident。

有限 profile：一次 warmup、一次完整训练，包含输入上传、调度/packing、forward/loss、
backward/finite、AdamW、通信和同步；模型/optimizer构造单列。Python开发tree2a1af
记录28301 kernel、2733主机scalar、1243 memcpy；native固定提交记录31531 kernel、
1641主机scalar、1372 memcpy。两者均32 candidate/31 Full/44消息，20个设备AiCPU事件。
保留 profiler 警告；有限 trace 未见命名 host fallback，不证明任意算子/形状都无回退。
不同进程/负载下的核数不能推断速度；此项只回答算子归属和主机交互问题，不作吞吐排名。

## 保留的失败与边界

- 默认异步的完整两卡 Python all-VJP 出现 CANN `rtStreamWaitEvent` invalid event。
  显式 current-device0/1、关闭 autograd 多线程均未解除；普通 P2P/两个根的小探针通过
  不认证完整训练。同步配置通过不表示默认异步已修复。native默认异步两卡尚未认证。
- standalone SDK 与 Python-owned native 分栈。LibTorch-NPU2.10/CANN9.0 编译/loader通过；
  首轮 CANN 内嵌 Python 找不到 NumPy，在launcher PATH选择配套Python后，相同binary的
  feedback/delta、HB/heterogeneous 开发检查四步与完整TIDEGCK1通过。
  随后的固定提交24图门禁在 fc/affine 严格失败：前两轮 forward/VJP/更新通过，第三轮
  evidence描述量超容差（shape[2]，最大绝对差3.09944e-6）。独立诊断副本重现，未放宽容差；
  不能把两个开发组合或编译通过当成全部 standalone 资格。其余未执行格子不推定通过。
- CPU开发首次198通过/2失败是新增报告字段与旧dict断言冲突；修正后的固定200项通过，
  原failed记录保留。FP16草稿、CUDA构建/真机、weighted resident与性能选型不在本证据。

复现：检出固定实现，按 [development](../development.md) 配置匹配环境；CPU运行JSON中的
pytest列表并加 `--dtype both`。设备运行 [入口](../../scripts/check_weighted_device.py)，
mixed-c省略case执行19组合；mixed-a/b与两卡选择 `delta-pdg` 和 `heterogeneous-attention`。
各输出目录新建、source只读；长任务background.slice/Nice10、ATen/BLAS1线程、
2 build workers、每项1800秒（manifest刷新600秒）、动态队列/独立设备可见性。
原始路径仅作运行标识；上述入口与fixture可从仓库复现，无需旧运行目录。
