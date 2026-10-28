# 非GPU已知边界：固定复现与定向解除 — 2026-10-29

诊断实现`aadd3b4b0a73f075923c4dbf299c57a987436be4`；数学固定0770593。生产公式、默认dtype/optimizer与容差均未改。
[结构化证据](core5-nongpu-boundaries-20261029.json)保存精确来源、原输入hash、每步结果、恢复与作业终态。
本报告既有通过，也有严格失败，不是“全部限制已解除”。性能与后续KV合批另验收。

| 原问题 | 本轮结果与可用边界 |
| --- | --- |
| standalone SDK FP32 affine | **原fc/affine mixed-c和mixed-a均通过**：ead6f34与ad6d1e7 SDK构建的C++/22 binary hash完全相符；原默认AdamW四次更新、完整TIDEGCK1、CPU交接。旧470ce06失败不改，不外推全24图，也不将此归因于本次诊断代码 |
| D64外层反向汇总 | 同hash原材料，加原三组局部FP64干预；外层汇合暂存FP64仍失败。第一步完整事件逐值相同，但2个梯度张量/动量槽超容差；第二步完整状态、9个梯度张量/动量槽失败。两步参数更新仍在容差内，不替代训练验收 |
| 原反馈PDG FP32 | W32/B4/每窗4位置/seed7，native streaming和prefill四窗均有记录、梯度/动量槽严格差异；参数值均通过。三个独立调度各自换owner后第四窗记录/梯度/参数/槽**逐值恢复通过** |
| 同材料反馈PDG显式FP64 | Python独立参考、native两种调度的四窗完整记录/梯度/SGD槽/参数及逐值恢复全部通过；不替代原FP32，也没有新增计时 |
| FP16 GDN默认AdamW | ead6f34的Python/native均再次在第二次更新见原坐标`nodes.5.kernel.conv.weight[0,2]`差.001983642578125；保留严格失败。原FP32、显式SGD和resident的独立资格不互相冒充 |
| 默认异步双NPU | 固定aadd3b4纯Torch小反例：stock和显式device guard仍EE1017，关闭autograd多线程及经CPU桥通过小反例。ead6f34关闭多线程的**完整图仍failed/134**；不接入生产替换，不改公共环境 |

CPU设施固定资格**14 passed/0 skipped**：独立相消、广播梯度、Python非Tensor参数、None/连接零、
算术边界原检查及两种调度的小图完整训练/真实恢复。复用的23个CPU binary在前后完整hash核对，
旧manifest来源1850434保留，资格来源是aadd3b4。D64诊断总体exit2；NPU纯Torch组总体exit2。
外层汇合使用私有Torch2.12接口，内部保存VJP图仍用原引擎；它只检验该有限干预，
不是完整高精度反向或独立数学正确性证明。两条D64轨迹各自的精确恢复也通过。

NPU纯Torch栈：aarch64、Python3.12.13、Torch2.12.0+cpu/Torch-NPU2.12.0.post2、CANN9.1.0、
Ascend910_9392、HF32关闭，物理1/9映射逻辑0/1，未设launch blocking。
SDK affine另用正式LibTorch-NPU2.10/CANN9.0及配套Python运行时，不能与Python-owned栈合并认证。
已有显式launch blocking的完整图资格继续有效；它不是默认异步修复或吞吐结论。

## 从仓库复现

检出上述诊断提交，按[开发入口](../development.md)构建匹配CPU/SDK。
固定原D64计划与join命令见[算术契约](../weighted-arithmetic-diagnostics.md)；
记录计划hash应为`6c8cc36ae97287133d40e67ccc8e137574b80e2df420feff6adfef3395397bb3`。
无需任何旧运行目录：

```sh
python -m pytest -q tests/test_weighted_join_probe.py tests/test_weighted_arithmetic_probe.py \
  tests/test_weighted_workload_training.py --dtype both
python scripts/diagnose_weighted_feedback.py --native-library CPU_CORE --dtype float32 --output-dir NEW_FEEDBACK32
python scripts/diagnose_weighted_feedback.py --native-library CPU_CORE --dtype float64 --output-dir NEW_FEEDBACK64
python scripts/diagnose_weighted_precision.py --implementation python --output-dir NEW_HALF
python scripts/diagnose_weighted_precision.py --implementation native --native-library CPU_CORE --output-dir NEW_NATIVE_HALF
SDK_CORE/tidegraph-weighted-training-check --device npu:0 --dtype float32 --placement mixed-c --case fc/affine --output-dir NEW_AFFINE_C
SDK_CORE/tidegraph-weighted-training-check --device npu:0 --dtype float32 --placement mixed-a --case fc/affine --output-dir NEW_AFFINE_A
python scripts/diagnose_torch_peer.py --device npu:0 --peer-device npu:1 --policy stock --output-dir NEW_PEER
```

后三种peer政策分别用`single_thread`、`device_guard`、`cpu_bridge`和独立新目录/进程。
完整图多线程诊断可在同一入口调用前用`torch.autograd.set_multithreading_enabled(False)`包住
`scripts/check_weighted_device.py`的main，参数为`--device npu:0 --peer-devices npu:0 npu:1
--implementation python --placement mixed-c --dtype float32 --case delta-pdg --case ports-sparse
--case heterogeneous-attention --output-dir NEW_FULL`。环境显式unset launch blocking。
小反例能排除“必须依赖本项目图核才失败”，不能定位到厂商栈的某个内部组件。

全部长检查使用冻结树、background.slice、Nice10、算子1；CPU固定任务24GiB/10min/单项240s，
peer16GiB/10min/每个子进程90s/排队180s，SDK各16GiB/7min/排队180s/计算180s。
早期两个私有join诊断器错误、反馈导入路径错误及一次资格启动目录碰撞分别保留原failed，
没有将这些基础设施失败解释为数学结论。原始路径只是运行标识，仓库入口和seed/配置足以重建输入。

## 本轮停止条件

已解除的SDK具体见证可按上述范围使用；数值失败保留原输入/容差和可复现见证。
外层FP64单项干预不足以解除D64，暂不采用；不继续重复既定九项精度探针。
FP16近零梯度与默认AdamW、原FP32反馈及D64/D2048仍需要新的有依据算术方案，
不能靠改输入、裁剪梯度、换optimizer或提高默认dtype宣称原配置通过。
默认双NPU异步保留为可独立复现的运行时阻塞；新栈或修复须同时过小反例与完整图训练/恢复。
GPU真机依用户决定延后，x86_64和大型设备训练/跨卡吞吐未验；这些不阻塞已验证CPU/显式设备路径。
