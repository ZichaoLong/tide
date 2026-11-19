# 大实例连续运行与恢复 · 2026-11-18

0.5B/1B 在 FP32、BF16 的推理、含梯度前向及完整训练共 12 个子任务、36 个连续段均完成。四组训练 checkpoint 的成对续接检查均在原阈值内；
另外四个独立新进程完成加载和第一次真实更新。两组恢复证据的输入、终态和计时边界不同，分别列出。
全部原始分段字段、阈值、命令、哈希及资源记录保留于 [JSON 证据](capacity-continuous-20261118.json)。本轮只读 50 个 JSON，没有打开或 stat 大 checkpoint，也没有启动实验。

连续运行/成对检查源码与设施：`efca7198570e8ee7a21ddccd81165eb36bdb269a`。新进程恢复源码与设施：`062a8ab6d6ba0faa191d8900420b697711b7d21d`。
两者数学均为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`。模型实际唯一参数分别为 494,110,704 和 976,768,720。

## 三段连续执行

每段 B1，prefill128 由两个 chunk64 执行，随后真实历史 decode8，合计 136 个输入；三段时钟为 136→272→408。
本次两个规模均为 Settle，flush 秒数/额外输出槽均为零，不能将这份结果扩展为 Timed flush 资格。
预热为零，第一段可能包含首次调用成本；后两段延续前段私有状态。三段是有序连续观察，不是独立重复或长期稳态样本。

`grad_forward` 在段内保留完整图、段间 detach，没有反向与参数更新。`train` 在段内完整 BPTT，整段时间包括
prefill/decode、返回结果与 loss、反向、完整 AdamW、同步及更新后 detach。输入传输、构造、checkpoint 和检查不计入三段完整时间。
表中峰值直接对应三段原始 `peak_device_bytes`；CPU RSS 是进程生命周期高水位，另存 JSON，不能混同这些 NPU 峰值。

| 模型 | 精度 | 模式 | 三段完整时间 s | 三段 NPU 峰值 GiB |
|---|---|---|---|---|
| HB 0.5B | FP32 | inference | 13.338 / 12.552 / 12.678 | 2.346 / 2.612 / 2.610 |
| HB 0.5B | FP32 | grad_forward | 16.019 / 14.241 / 14.152 | 18.770 / 19.668 / 19.677 |
| HB 0.5B | FP32 | train | 28.568 / 26.332 / 27.716 | 20.709 / 25.211 / 25.099 |
| HB 0.5B | BF16 | inference | 14.928 / 13.720 / 13.714 | 2.169 / 2.374 / 2.374 |
| HB 0.5B | BF16 | grad_forward | 16.630 / 14.721 / 14.970 | 12.742 / 13.293 / 13.258 |
| HB 0.5B | BF16 | train | 30.111 / 30.003 / 32.502 | 14.795 / 18.969 / 18.903 |
| HB 1B | FP32 | inference | 6.962 / 6.053 / 5.946 | 4.176 / 4.506 / 4.504 |
| HB 1B | FP32 | grad_forward | 8.227 / 6.736 / 6.492 | 28.798 / 29.390 / 29.397 |
| HB 1B | FP32 | train | 14.535 / 12.466 / 12.563 | 32.781 / 40.299 / 39.973 |
| HB 1B | BF16 | inference | 8.139 / 6.574 / 6.541 | 4.045 / 4.254 / 4.256 |
| HB 1B | BF16 | grad_forward | 8.716 / 7.242 / 7.182 | 18.708 / 19.178 / 19.169 |
| HB 1B | BF16 | train | 14.987 / 12.978 / 13.123 | 22.798 / 30.366 / 30.187 |

训练最大已报峰值：0.5B 为 FP32 25.211 / BF16 18.969 GiB；1B 为 40.299 / 30.366 GiB。
这些观察中 BF16 省内存，但没有表现为更短的完整训练时间。两种精度的活动轨迹可能随更新分化，不能宣称严格同轨迹加速。
0.5B 与 1B 的拓扑和宽度不同，较大参数模型在这里更快不构成一般规模规律。

原记录还保留每个 prefill/decode 调用、首个主干输出、输入窗口、flush、反向、优化器和完整时间，以及私有状态存储量。
`stable_decode` 字段仅按脚本规则排除八个 decode 中的第一个：中位数与 `7/其余七次秒数总和` 吞吐分别报告，不能把该名字当作统计稳定证明。
首主干输出不是含任务头的语言模型 TTFT。

## 原进程内 checkpoint 成对续接检查

只对四个 `train` 子任务执行：在三次更新后的时钟408保存并加载，原 owner 与恢复 owner 分别对 **64 个输入**做下一前向/反向/AdamW，
两个终态均为472。这里没有记录该成对下一步的性能时间，`save_seconds`/`load_seconds` 仅属于原检查的保存/加载边界。
原输入哈希对应完整136输入张量，64输入 witness 没有单独哈希字段；完整输入哈希和冻结源码仍保留。

| 检查 | 原 atol | 原 rtol | 四组结果 |
|---|---:|---:|---|
| saved_data / next_forward / next_state | 0 | 0 | 全部 exact，最大绝对差 0 |
| input_gradients | 1e-6 | 1e-2 | 全部 exact，最大绝对差 0 |
| parameter_gradients | 1e-6 | 2e-3 | 均非 exact，最大绝对差 3.7253e-9 |
| updated_parameters | 1e-6 | 2e-3 | 均非 exact，最大绝对差 2.9104e-11 |
| updated_optimizer | 1e-6 | 2e-3 | 均非 exact，最大绝对差 4.6566e-10 |
| updated_state | 1e-6 | 2e-3 | 全部 exact，最大绝对差 0 |

全部32项检查为 finite、within_threshold，metadata/support/numerical failure 列表为空；两 owner 的 loss 相同。
参数梯度/参数更新/优化器的非零差异未抹去、未改阈值，也未改称 exact。JSON 保留各组 max_abs、max_scaled_error、浮点张量数与原阈值。

| 模型 / 精度 | 保存 s | 原检查加载 s | checkpoint GiB |
|---|---:|---:|---:|
| HB 0.5B / FP32 | 28.700 | 7.382 | 5.744 |
| HB 0.5B / BF16 | 29.694 | 6.688 | 5.650 |
| HB 1B / FP32 | 59.531 | 12.391 | 11.134 |
| HB 1B / BF16 | 53.068 | 11.822 | 11.040 |

## 新进程加载及首个真实更新

每个观察只有一个恢复 owner，无模型/更新预热。恢复原 checkpoint 的时钟408、AdamW step3，然后执行
**prefill128/chunk64 + decode8，共136输入**，终态544、step4。它使用新输入哈希，既不是上面的64输入成对检查，也不是同一条后续轨迹。

| 模型 / 精度 | 完整加载 s | 首次完整更新 s | 加载峰值 GiB | 更新峰值 GiB | 更新输入位置/s |
|---|---:|---:|---:|---:|---:|
| HB 0.5B / FP32 | 7.996 | 27.654 | 5.749 | 25.278 | 4.918 |
| HB 0.5B / BF16 | 7.603 | 32.043 | 5.674 | 19.016 | 4.244 |
| HB 1B / FP32 | 13.811 | 14.258 | 11.157 | 40.516 | 9.539 |
| HB 1B / BF16 | 14.463 | 15.177 | 11.066 | 30.494 | 8.961 |

加载完整时间包括 model/state 与优化器恢复及同步；更新包括 zero_grad、前向/readout/loss、反向、AdamW、detach 和同步。
两段完整时间包含各阶段 allocator 探针和 Python bookkeeping，阶段秒数则不包含其后内存读取/重置。
模块/设备上下文建立、输入生成/传输/哈希、checkpoint SHA 扫描、finite/storage/slot/sample 检查和磁盘记录写入不在上述完整加载/更新时间内。
原始分阶段峰值与合并峰值均在 JSON；文件系统缓存未控制，新进程不等于冷文件缓存。每格只有一次观察，不是稳态吞吐估计。

加载及更新后的参数、优化器和私有状态 finite；更新后梯度、输出 finite，私有状态已 detach。
0.5B 的860个、1B 的432个优化器参数槽全部 step3→4，增量错误列表为空，原 AdamW 组配置恢复并保留。
实际参数、梯度、一二阶矩及 step 张量均为 FP32；BF16 私有状态混用 BF16/FP32/int64。
每个 Parameter 前最多16个标量的更新抽样分别改变 13,756/13,760（0.5B FP32）、13,759/13,760（0.5B BF16）、
6,910/6,912（1B FP32）、6,912/6,912（1B BF16）。这只验证抽样发生更新，不是完整轨迹比较。
计时恢复过程未加入算子 dtype 诊断；本页不据此扩大 BF16 算子或跨精度数值资格。

## 哈希、执行身份与复现

四份 checkpoint SHA 来自各新进程在加载后完成的流式哈希记录；四次 `expected_checkpoint_sha256` 实参均为 null。
加载前后 size/mtime/device/inode 记录一致，且字节数与原保存记录一致；本轮继承这些 SHA，没有重读 checkpoint 或宣称进行了预先期望哈希对照。
`checkpoint_source` 是调用方提供的生产源码身份，checkpoint 格式本身没有编码源码提交。

- HB 0.5B FP32：`c352854bb734a55f2d6b35896ef2960ad3839abf440362ec609dd144e120e10d`。
- HB 0.5B BF16：`69b33d535b08c6dab73bedfc0deb05037a065124186ede8dd28f6dd53989d8b5`。
- HB 1B FP32：`848b6faa9903290eb794cf4095fd5890c2a10a89b66bf3f014039a857fd83a66`。
- HB 1B BF16：`92af1f898418b1331caed99a6bfa0ee764c60d325e0e2bf4808a24f633728972`。

全部作业使用协作队列独占准入，记录为 `exclusive_cooperative_queue; external jobs not controlled`，不能据此假定整机无争用。
连续作业在 UTC07:30–07:40、新恢复作业在07:40–07:42执行，候选前后 host load、服务终态与 PID 保留。
每进程四线程；Torch 标签 `2.12.0+cpu`，实际设备为 NPU，模块 `cann/9.2.0:ascend/dev-workspace-9.2.0`。

| 作业 | 物理 NPU | CPU affinity | 外层时限 s / 采样 RSS GiB |
|---|---:|---|---|
| capacity-continuous-hb05b-v3 | 5 | 132-135 | 3600 / 128.0 |
| capacity-continuous-hb1b-v3 | 1 | 128-131 | 3600 / 160.0 |
| capacity-restore-hb05b-bf16-v1 | 0 | 116-119 | 1200 / 96.0 |
| capacity-restore-hb05b-fp32-v1 | 5 | 108-111 | 1200 / 96.0 |
| capacity-restore-hb1b-bf16-v1 | 1 | 112-115 | 1200 / 96.0 |
| capacity-restore-hb1b-fp32-v1 | 7 | 104-107 | 1200 / 96.0 |

连续作业单子进程限时900秒，外层队列等待上限600秒；新恢复单卡 allocator 预算52 GiB，队列等待600秒。
RSS 是采样进程树监督，非 cgroup 硬内存；连续原记录没有显式单卡 allocator 预算字段，不补造预算。

仅复核 JSON 身份，不接触 checkpoint：

```sh
python - <<'PY'
from pathlib import Path
import hashlib, json
record = json.loads(Path('docs/evidence/capacity-continuous-20261118.json').read_text())
for item in record['verified_json_files']:
    path = Path(item['path'])
    assert path.suffix == '.json'
    payload = path.read_bytes()
    assert len(payload) == item['bytes']
    assert hashlib.sha256(payload).hexdigest() == item['sha256']
print('verified', len(record['verified_json_files']), 'JSON files; checkpoint payloads untouched')
PY
```

若需要新实验，原始冻结源码与精确命令在 JSON 的 `runs[].launch_arguments`，连续子进程另保留在 `runs[].children`。
使用新输出身份并按当时资源情况准入。源码身份、两精度输入/初始参数哈希、checkpoint 哈希和所有原阈值都应继续保留；
本页的有限连续运行、同精度恢复检查和单次恢复性能，均不替代长期训练质量或其他配置的数学/数值资格。
