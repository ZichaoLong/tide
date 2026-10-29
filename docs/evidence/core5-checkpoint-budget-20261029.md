# 大模型native checkpoint文件预算

固定实现`b255e270eab38faa9c4348e19ed435ce02bbc4b4`，数学仍为0770593。
[机器记录](core5-checkpoint-budget-20261029.json)保存构建、消费者检查、原FC1B复验与失败。
原始问题来自[e50自然规模测量](core5-natural-study-20261029.md)，不是新的数学定义。

## 修复与资格

native参数/优化器和完整图codec原先有固定4GiB文件上限。现在公开API与standalone消费者
接受显式正int64文件预算，默认仍4GiB；格式、数值、张量/集合限制及排他发布保持。
加载在分配文件buffer之前检查文件尺寸；拒绝不修改live参数/槽。保存检查内外层最终字节数。
host codec仍整体驻留内存，不宣称分片/流式保存；详细契约见[GraphCheckpoint](../native-graph-checkpoint.md)。

| 固定源码资格 | 结果 |
| --- | --- |
| aarch64 Torch2.12 CPU | fresh core50.35s；32项相关检查通过；fresh standalone客户端及6配置通过 |
| standalone Torch2.10/CANN9.0 NPU | 配套core、9392目标resident及客户端build/loader通过；8个真实消费者配置通过 |
| 独立预算反例 | 精确边界与原字节不变；预算小1字节拒绝；无发布/无参数槽修改；默认>4GiB稀疏文件预读拒绝；非法int64拒绝 |
| 完整图与训练 | C++完整State/History/pending/int64及外层预算；CPU FP32/FP64/half训练/优化器/恢复，NPU8组完整梯度/SGD/master与磁盘换owner |

32项包括既有完整训练测试，和其他证据有重叠，不累加成一次全矩阵资格。
构建half=160线程；正式CPU与SDK服务各256GiB/2600秒，计时与构建不重叠。
SDK最初driver误填9391目标，审计后停止未验收构建；下一服务拒绝复用已存在目录。
两个终态分别cancelled/failed保留；最终使用新目录和已验证9392目标通过，没有改公共环境。
源码构建与真机证据分别记录；未取得新的CUDA/x86目标资格。

## 原FC约1B用例的复验

同一969,962,344参数模型、FP32/FP32 scoring、standalone NPU mixed-c，原B1/P1、种子、
SGD与原严格容差；只增加`--checkpoint-max-bytes 17179869184`（16GiB）。一次进程，
32核/512GiB/1800秒，空闲单卡，原输入静态材料hash相同。没有自动重试。

- 两个测量窗1.258203/1.193136秒，candidate64/Full9/messages64。它们是失败成本，不是合格吞吐。
- 实际保存了**5,253,356,896字节**文件，sha256见机器记录。新owner的完整文件加载返回，原4GiB阻断解除。
- 随后原owner执行第4个训练窗（warmup、2测量窗之后）时，finite检查发现非有限梯度。
  代码顺序在`save → 创建并load新owner → 原owner step → 新owner step → 完整比较`；
  错误在原owner step发生，不能归因于解码后的恢复计算。
- 第二个测量窗的loss已达约3.68034e20。这是本固定大尺度、平方观测loss和SGD设置的数值边界，
  尚未定位首个非有限中间值；不是已证明的通用KV失效。没有调整loss/lr/输入来制造通过。
- **没有通过下一窗精确续接，也未进入事后独立CPU全记录/训练核对。** 小图资格不能覆盖这一缺口。

因此“文件大小限制解除”和“该1B完整训练/恢复通过”是两个结论。此原候选停止；
重启需要独立定位非有限伴随来源或明确的新训练数值方案，不能只重复放大文件/显存预算。
文件与原失败记录保留为复现输入，状态及停止条件集中于[ROADMAP](../ROADMAP.md)。

## 独立复现

```sh
git checkout b255e270eab38faa9c4348e19ed435ce02bbc4b4
python scripts/build.py --backend cpu --build-dir build/cpu --jobs half
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_cpp_checkpoint.py tests/test_cpp_master_optimizer.py \
  tests/test_cpp_weighted_training.py tests/test_core5_study.py
# 按development分别构建SDK core、resident和benchmark。使用新目录和显式空闲设备。
python scripts/run_core5_study.py --case fc-1b-kv --execution cpp-npu-mixed-c \
  --standalone-binary SDK_BUILD/client/weighted-benchmark --fixture STATIC_FC1B_FIXTURE \
  --max-materialize-bytes 34359738368 --checkpoint-max-bytes 17179869184 --output-dir NEW_RUN
```

静态fixture用[公共目录和导出器](../core5-study.md)重新生成，无需下载旧运行目录；
不得把CPU真实轨迹、选择或梯度输入候选。GPU真机按用户约定延后。
