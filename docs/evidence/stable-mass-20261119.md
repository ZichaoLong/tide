# 稳定质量归约：实现、数值资格与成本（2026-11-19）

本轮将历史修复中的稳定求值方法落实到当前 FC/HB 实例：Reception 使用缩放均值和中心化复合
VJP，KV Full 使用质量缩放 Attention。未引入 positive floor、epsilon、新数学、参数或共享状态。
固定数学仍为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108 文件原字节通过。
局部公式及执行范围见 [stable-mass](../stable-mass.md)。

## 源码与历史关系

| 身份 | 内容 |
|---|---|
| `a3fc8053c48b730e2414245961701c1c7f86d91c` | 本轮起点与成本比较 before |
| `70f4aa308c46daef9692b9378c26eb7e5f667fb1` | 第一冻结实现；保留其失败 |
| `8fd5216a1a7833cf4c81975a0aaa5724465f9c38` | 有界二进制指数恢复、诊断接口补齐后的生产源码 |
| `1aa26314f7300dcbf7839e63d4146332cc5c4fb3` | 成本编排工具；生产树与 `8fd5216a` 相同 |

生产 `src/tide` tree 为 `da223720c5c077178dde154d298fdbe73243e23c`。
`169bf675` 的中心化 Aggregate VJP、`2c11695c` 的缩放均值与复合质量 Attention 是方法来源；
后者的 `positive-floor-v1` 存储策略没有迁入。`d21658c6` 的下游 pool 反传溢出见证和
`d089dd05` 的乘法次序修复说明历史问题确曾发生，但该独立实现不能算当前库已验证。
本轮重建有界算术见证；退役的完整训练 checkpoint 没有重放。

首轮 NPU 发现可表示的 `-3e38` 被 `ldexp` 的中间 FP32 `2^128` 变为 `-Inf`。
当前实现先规范化尾数，再以两个有界幂恢复尺度，首段保持正规数，最后才舍入到次正规。
这修复了实测的中间溢出；不代表任意最终值/梯度都可在 FP32 表示。
首轮 CPU 全套另发现诊断 wrapper 未接 `mass_scaled`；已补捕获、重放、独立公式与实际 dispatch。

## 资格与保留失败

冻结CPU全套结果：**1140 passed, 4 xfailed, 1 warning in 603.14s (0:10:03)**。首轮为1 failed、1131 passed、4 xfailed；唯一失败是上述诊断接口，原日志保留。

第二轮空闲单 NPU 定向检查为 **86 passed、3 strict xfailed、1 TypedStorage 弃用警告**。
原假溢出回归通过，日志未观察到 CPU 算子回退。FP64 oracle 始终在 CPU。
NPU 原始失败、CPU 首轮训练、近零 VJP 有界调查完整保存在
[r1 审阅 JSON](stable-mass-20261119-r1.json)；CPU 首轮全套原失败日志另收入最终记录。

三个新 strict xfail 是同一 seed773 的 `dV[1,0,4,26]`，分别在 math/tiled/auto 路径：

| 项目 | 值 |
|---|---:|
| CPU/NPU 实际值，二者相同 | `1.043685510987748e-30` |
| 独立 CPU FP64 oracle | `1.043707131749261e-30` |
| oracle 转 FP32 | `1.043707140083803e-30` |
| 绝对/相对误差 | `2.162909605513009e-35` / `2.072333882097155e-5` |
| 原阈值，未改 | atol `1e-35`，rtol `1e-5` |

严格标记只接受这个坐标和精确数值指纹，不吞掉其他字段、非有限或新增超差。
将实际 FP32 乘积改为 FP64 归约得到 `1.0436930341515932e-30`；将实际 FP32 概率/alpha
改为 FP64 乘积与归约得到 `1.0436919635826823e-30`。选中坐标同时受概率/alpha及乘积/归约影响，
这不是全部整网差异的归因。既有 FFN 超差仍保留，不能称全数值等价。

### 训练、续接与实际恢复

下列四例在 CPU 和 NPU 各完成一次 anchor 更新、三步继续训练、checkpoint及新进程 step4→5：

| 案例 | 核心形状/路径 | 参数 | 两平台结果 |
|---|---|---:|---|
| FC KV128 FP32 | 单节点单层，d192、KV128、FFN128 | 338,809 | within_threshold |
| HB X pool FP32 | compact-5-dd4，n1/hub1，d64/KV32/FFN128，region_shared，incoming_light64，fused QKV与padding mask | 926,185 | within_threshold |
| HB Timed K FP32 | 同小 HB，d64/KV32/FFN64，region_shared，checkpoint_reception | 490,619 | within_threshold |
| HB X pool BF16 | 与第二例同配置 | 926,185 | within_threshold |

均为 B2、L8、chunk4；推理、含梯度前向、重复 VJP、输入/参数梯度、多步状态和实际更新均有记录。
各平台各案例的 saved data、next forward、actual update 全部 exact；这是各自同候选恢复结果，
不要求 CPU/NPU 输出互相 exact，也不证明修复前后整网训练轨迹相等。
私有状态、原始 int64 时间、零接收、窗口淘汰、分块继续及共享梯度由定向/既有组合检查覆盖。

隔离 wheel 的 **25 个包文件**逐文件匹配生产树；去掉 PYTHONPATH，用新私有环境的 `python -I`
调用公开 `examples/train.py`，三步训练和新进程恢复再更新通过，优化器 step 从3→4。
wheel SHA256 为 `f39df3186a68ca06a90ffded7c838c6c29a5d89c06e265e99d547a3b786cfec9`。
新环境仅用于验证；主私有环境、公共工具链及下游环境没有被替换。

## 修复成本：固定小图、普通计时

采用上表 926,185 参数的 FP32 HB X pool，固定独立 B2、prompt16/chunk4/decode8、4 Torch线程。
before/after 各三组成对重复，每组1次 warmup、3次普通样本，第二组倒置先后顺序。
表中是三组样本中位数的中位数；百分比是三组成对比值的中位数，两者不必严格相除一致。
训练包括图跨 chunk 的前向、loss、backward、AdamW、同步与每次更新后的状态 detach；
推理包括已驻留输入至输出就绪/释放和同步。首主干输出不含任务头、采样或完整语言模型 TTFT。
工作/内存观察另跑，未纳入普通吞吐计时。

### CPU：分阶段耗时

| 模式/阶段 | before，秒 | after，秒 | 成对耗时变化 |
|---|---:|---:|---:|
| 首主干输出 | 0.100958 | 0.107909 | +6.89% |
| prefill（32个输入位置） | 0.400219 | 0.429064 | +6.99% |
| 稳定decode（排除首步，共7步/14个位置） | 0.669028 | 0.718447 | +7.31% |
| 完整推理请求 | 1.164870 | 1.249286 | +7.25% |
| 训练前向 | 0.470204 | 0.497763 | +5.56% |
| backward | 0.389184 | 0.437592 | +12.12% |
| AdamW | 0.007396 | 0.007506 | +1.28% |
| 完整训练更新 | 0.867617 | 0.941972 | +8.57% |

| 吞吐 | before | after | 成对变化 |
|---|---:|---:|---:|
| prefill输入位置/s | 79.9562 | 74.5809 | -6.53% |
| 稳定decode输入位置/s | 20.9259 | 19.4865 | -6.81% |
| 独立完整序列/s | 1.7169 | 1.6009 | -6.76% |
| 完整训练输入位置/s | 35.5639 | 32.9029 | -7.48% |
| 独立训练序列/s | 2.2227 | 2.0564 | -7.48% |

三组成对完整耗时比（after/before）：inference [1.07247, 1.09595, 1.06175], train [1.07571, 1.10013, 1.0857]。

### NPU：分阶段耗时

| 模式/阶段 | before，秒 | after，秒 | 成对耗时变化 |
|---|---:|---:|---:|
| 首主干输出 | 0.144624 | 0.178677 | +24.80% |
| prefill（32个输入位置） | 0.577604 | 0.712538 | +25.02% |
| 稳定decode（排除首步，共7步/14个位置） | 0.998769 | 1.252892 | +25.44% |
| 完整推理请求 | 1.722952 | 2.150659 | +25.01% |
| 训练前向 | 0.651488 | 0.807726 | +23.34% |
| backward | 0.693009 | 0.964487 | +39.39% |
| AdamW | 0.009472 | 0.009644 | +1.96% |
| 完整训练更新 | 1.361310 | 1.785346 | +32.88% |

| 吞吐 | before | after | 成对变化 |
|---|---:|---:|---:|
| prefill输入位置/s | 55.4013 | 44.9099 | -20.01% |
| 稳定decode输入位置/s | 14.0173 | 11.1741 | -20.28% |
| 独立完整序列/s | 1.1608 | 0.9299 | -20.01% |
| 完整训练输入位置/s | 23.5222 | 17.9730 | -24.33% |
| 独立训练序列/s | 1.4701 | 1.1233 | -24.33% |

三组成对完整耗时比（after/before）：inference [1.2447, 1.26915, 1.25013], train [1.30046, 1.32876, 1.33131]。

### 内存边界

| NPU普通计时峰值 | before | after |
|---|---:|---:|
| inference allocated，MiB | 36.1196 | 36.2168 |
| inference reserved，MiB | 58.0000 | 58.0000 |
| train allocated，MiB | 68.1953 | 68.3203 |
| train reserved，MiB | 160.0000 | 160.0000 |

单独内存诊断：权重3,704,740 B，AdamW状态7,410,180 B（含CPU step槽700 B），
结束时私有记忆190,128 B，均未变化。前向额外保存激活的可见独占存储从
13,523,328 B变为 13,645,920 B。
saved-tensor hooks及分阶段存储观察有自身开销；allocator残差还含未观测张量与后端开销，
不能称纯临时工作区，也不能将持久记忆或分阶段差额当峰值显存。CPU没有等价设备allocator计数；
完整记录另列进程生命周期RSS和可见张量存储，RSS不作为算子临时空间。


每对初始参数、输入 hash、参数量、配置以及完整工作记录一致。
推理 Full 为528次：KV选中/执行288行，SSD240行；训练观察 Full352次：KV192行、SSD160行。
这组小图没有 Full padding；各 NodeLinear 矩阵行数与乘加量也一致。训练工作观察来自第一次测量的
anchor，仅覆盖单独含梯度前向，不是对每次后续更新都做内部活动审计。

新增稳定归约、复合 VJP、范围检查和主机同步共同增加成本；本轮没有消融出每一项的独立比例。
NPU 增量集中在前向和反向，不能归因于 FFN padding、更多参数或更大持久状态。
弱质量路径有数值资格，未单独给出全网弱质量分布下的吞吐。
这不是9B、Transformer对照、batch最优点或下游质量测量，旧性能保留其原源码身份。

## 适用范围与后续建议

优先使用包含 `8fd5216a` 的 FP32 当前实现，接受已记录的稳定性成本；BF16作为次选。
默认训练入口、参数绑定、私有历史及 FFN 执行策略保持原设置。
不迁入质量 floor，不用放宽阈值或改训练 epsilon 消除失败。

仍有限定：普通 native 任意极端 logits/伴随、非 mass-scaled Attention、任意中间相消、
概率下溢及最后不可表示的公开结果不获全域保证。Attention 的检查在展开根上，后续共享参数
汇总可能仍有范围问题；极端被mask的首 value 仍可能影响中心化锚并造成中间溢出。
后一项为静态审阅边界，未新增实验或作为已实测失败宣称。只声明一阶 VJP。
RMS/SSD极端参数化、log质量存储、9B训练、CUDA、多NPU及任务效果不属于新增资格。

下一步可在有界普通域内实现更便宜的稳定归约，或融合尾数/指数计算与范围检查以减少设备同步；
必须先独立验证范围判据、真实零边和共享梯度，不能用默认截断代替稳定计算。
跨路径旧超差和整网更新偏差继续独立保留。

## 复现与产物

完整命令、环境、配置、输入/包 hash、原始普通计时、工作/内存记录、训练与恢复记录和终态在
[压缩审阅 JSON](stable-mass-20261119.json.gz)，经 [archive index](archive-index.json) 校验解压：

```bash
python scripts/materialize_record.py --index docs/evidence/archive-index.json \
  --record docs/evidence/stable-mass-20261119.json --output /tmp/stable-mass-review.json
```

新冻结资格使用新任务名和新输出目录，例如在[私有环境](../development.md)中：

```bash
python scripts/launch.py --name stable-mass-cpu-recheck --revision 8fd5216a \
  --threads 4 --affinity 0-7 --seconds 600 --rss-gib 32 -- \
  python -m scripts.qualify_stable_mass --device cpu --stage all \
  --threads 4 --output-dir '{run}/results'
```

NPU 重做须重新检查当时空闲资源，使用 launcher 的空闲协作队列；旧物理卡号不构成未来占用许可。
成本入口为 `scripts/measure_stable_mass_costs.py`，冻结到 `1aa26314`；before 用 `git archive a3fc8053`
重新创建只读源码目录。配置原文在解压记录的
`files["artifacts/stable-mass-20261119/cost-config.json"].text`，不要使用已退休的绝对输入路径。
原始训练张量、重复冻结源码和临时安装环境在证据提交并核验全部后代终态后退休；
保留记录可审阅，不承诺旧 `.pt` 路径在线或原完整训练 checkpoint 直接重放。

本轮10个服务及202个记录后代身份均核验终态。CPU服务累计2448.299/3600秒，NPU服务累计395.834/1800秒（包括等待），峰值进程树RSS分别为1.651和5.014 GiB。
服务时长不等于模型计时。CPU使用固定亲和性与4线程，共享主机；NPU使用Ascend910空闲独占单卡协作队列，定向r2为物理7，训练/成本为物理11；未共享占用卡。CANN9.2、Torch2.12、TorchNPU2.12.post3。
主机为HiSilicon aarch64，320逻辑CPU、4 socket、8 NUMA节点；lscpu型号字段为unknown。每个测点仅4 Torch线程，不能视为整机CPU最优吞吐。
最终退休与目录体积见[收尾收据](stable-mass-retirement-20261119.json)及[STATUS](../STATUS.md)。

审阅证据提交 `96a5dda8` 后，本轮11个临时目录已退休，退休文件逐路径分配大小合计约3.782 GB，
未转移到其他路径。仓库含Git和主私有环境约148.4 MB；主环境隔离导入仍指向当前源码，
四份用户文档SHA256未变。公共工具链、下游安装和两工作树均保留。
