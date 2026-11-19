# 清理、数值定位与quad修复（2026-11-19）

本轮从 `c5bfebaf` 接续。诊断工具固定 `2360728d`、`e1c6aaa0`；完整更新重放固定
`aab9a4d2`，删除后摘要重放固定 `72c1a9d0`；生产修复及新资格固定 **`97c52960`**。
原资格、诊断与修复后的测量保持各自身份。数学仍为
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，108文件原字节不变。
完整审阅摘要、文件身份、终态与清理清单见[配套JSON](cleanup-numerics-20261119.json)。

## 清理与保留

| 范围 | 文件数 | 释放 allocated bytes | GiB |
|---|---:|---:|---:|
| 已成功恢复或完成用途的旧载荷 | 70 | 31,598,239,744 | 29.428154 |
| 六组成功profiler原件及导出 | 1,539 | 5,706,231,808 | 5.314342 |
| 完整重做后删除的两份update2 | 2 | 6,329,466,880 | 5.894775 |
| 以上旧产物合计 | 1,611 | 43,633,938,432 | 40.637272 |

首批固定清单SHA256为 `e8c280818e7402f1390ede8a58ef75b67bbedda5610babb54a8c49019feb93f0`。
逐文件stat/路径守卫、依赖审查、fsynced删除收据和终态在
`artifacts/cleanup-numerics-20261119/`。12份新冻结源码的6036文件逐字节匹配已推送提交，连同缓存、成功测试临时文件、
两份本地恢复bundle和安装临时产物，另释放2.990738 GiB。本轮累计删除43.628010 GiB。
新临时产物与上表旧产物分开计数，避免把新写入当原有配额净减少。
仓库实测91.150→51.045 GiB，净减少约40.105 GiB；账户quota约91.006 GiB，
软限300/硬限700 GiB。账户目录du有不可读Docker子目录，不作为完整账户总量。
旧profiler完整时间线已不在线，`costs.json`、分类和主要算子汇总、trace身份仍保留。
152份小结果原字节副本、所有原阈值失败及其身份继续保留。

除明确退役的两份update2，其余290个旧`.pt`、6734个`.f32`身份不变，约45.604 GiB。
其中FFN medium约17.868、memory资格7.920、training paths/placement7.759、
execution候选3.884、Reception3.846、native见证1.777、其他张量2.551 GiB。
这些主要是已结束实验的失败复现输入，不表示仍有这些实验待运行；各组尚未全部完成
最小化，不能整组删除。74个旧失败profiler仅约1.43 MiB，也继续保留。
旧CPU anchor、两份update1仍是已删update2的复现依赖。
新训练anchor和两份checkpoint保留跨路径失败前态；两份成功本地恢复bundle可退役。
四份用户文档、主math/.git、唯一ignored源码、下游依赖和公共环境保持原样。

## FP32训练更新：小梯度被AdamW放大

`numerics-saved-update-v1` 以只读mmap和分块比较处理7份当时存在的原载荷，全部重新核验SHA。
原阈值仍为 `atol=1e-6, rtol=0.002`；W为actual，baseline为reference。

| 观察 | 原超差字段数 | 本轮重现 | 分类 |
|---|---:|---:|---|
| 初始VJP1 | 0 | 0 | 390字段非exact，但阈内 |
| update-1 | 392 | 392 | 391前向/状态、1梯度 |
| update-2 | 533 | 533 | 529前向/状态、2梯度、2参数 |

原失败字段集完整重现，没有结构、非有限、支持或int64差异。
update2使用真实update1前态，对64个坐标做独立80位Decimal AdamW回放。
优化器step3→4，lr=1e-4、eps=1e-8、betas=(0.9,0.999)、weight_decay=0.01。
下面两个坐标自身均确实超出原阈值；方向为W减baseline。

| 参数坐标 | 实测参数差 | Decimal预测 | 实测−预测 | 原scaled error |
|---|---:|---:|---:|---:|
| `regional.4.second.bias[47]` | 2.859975211e-6 | 2.859942655e-6 | +3.255677280e-11 | 1.058229 |
| `regional.26.second.bias[58]` | 1.918087946e-6 | 1.918091889e-6 | −3.943034000e-12 | 1.041615 |

第一行的baseline/W梯度为−8.699656273e-11/−1.933256888e-10；AdamW分母仅为
1.014224/1.021438倍eps。固定历史只换入当前梯度的割线增益约−2839.786。
按“旧参数→旧矩→当前梯度”换入的差异贡献为
1.593083878e-6、9.649067740e-7、3.019520021e-7。这解释了所选实测参数更新差，
不是凭最终超差猜测；分解顺序依赖，不能当唯一因果比例，也不解释全部533字段。
anchor+VJP1另有首步反事实预测，没有把已删除的update0伪装成实测。

区域第二层bias产生公共评分偏移。在严格激活的总量约束、固定face上，评分VJP和应为零；
松弛总量或声明的边界导数处不保证成立。因此不能全局删bias或强制小梯度清零。
本轮没有修改AdamW epsilon、训练配方或原阈值。

两份update2均先从各自真实update1完整重做，整个返回树对原件exact；随后释放5.894775 GiB。
删除后又用 `72c1a9d0`、anchor+各自update1及旧完整exact证明重新执行，完整树摘要再次exact。
四次均step3→4、position384→512，且该路径RNG不变。update1没有RNG sidecar，工具重置
anchor RNG并核验无随机消耗；这不是普适RNG恢复资格，也不证明baseline和W互相等价。

## 实际FFN局部VJP

`numerics-ffn-vjp-v1` 在原CPU32线程、NUMA6、相同生产树重做一次真实baseline反向。
主干输出/loss、输入梯度及K-down owner160–191行对原载荷exact。
保留第28次、即最后一次K-down调用的 `X=[32,16,25,768]`、`W=[32,192,768]`
和实际incoming VJP，见证约112 MiB。

同X/W/dY下，gather/grouped输出和输入VJP逐位相同；单调用dW最大差5.093170330e-11。
对独立CPU FP64显式收缩公式，gather/grouped的dW最大差为4.996355141e-11/
3.867764377e-11，均在原阈值内。旧全图所选dW最大差1.164153218e-10。
单调用贡献与全图累积梯度分开记录；没有重新执行完整W反向，也没有解释全部上游bias差异。

## 共享9B：求解器丢失正量及修复

旧生产树诊断在空闲物理NPU11执行region batch1/4两个请求：B8、prompt128/chunk32/decode16。
同一个参数owner、各自从空私有状态开始。参数/输入SHA和旧输出support/time SHA均重现。
唯一内部support差异位于第三prefill块、region43/layer4/node817/local1、sample5、原始int64时间509。
Full总数为180209/180208。

| 原路径 | 生产控制量 | 对各自已舍入FP32输入的独立Decimal解 |
|---|---:|---:|
| region batch1 | 1.311302185e-6 | 1.311302185e-6 |
| region batch4 | 0 | 3.973642985e-8 |

两者该区域提交均为 `[node,B,L]=[32,8,32]`；allocator内部为 `[8,32,32]`，stride
`(32,1,256)`。原shape同输入重放exact；单行连续重放会得到正值，不能代替失败形状。
`upper.sum + breakpoint prefix` 相消使排序求解器选择了错误的两个内点face；首次归零在
`_response`，之后只重建既选face。总质量残差为零，使旧valid检查漏掉错误的最优性边界。

修复 `97c52960` 给最终face补KKT检查。设内点集I、剩余质量R，
L=max(下界face的s−lower)，U=min(上界face的s−upper)。非空I检查
`sum_I(s−L) >= R`、`sum_I(s−U) <= R`；空I检查L≤U，固定坐标不约束乘子。
避免先构造再舍入绝对乘子；不加epsilon。错误face走已有同设备、同dtype的保护二分，
没有生产FP64/CPU回退，不改固定数学。依据为固定allocation note的边界/VJP与最优性残差要求。

在原完整提交形状、值及独立score VJP的NPU检查中，两条路径均通过。
原region4同输入的控制量现在为5.960464478e-8，独立解3.973642985e-8，支持一致、值阈内。
以该控制量为loss，正确score VJP在索引[1,20,27]为(2/3,−1/3,−1/3)；旧错误face给零VJP。
因此问题会影响训练梯度，不能只视作一个很小的输出误差。CPU单元测试另覆盖相邻精确零边界。

修复后同参数/输入、同物理卡的9B两请求复查：Full均为180209，880个区域调用的内部
支持无差异，公开输出support/time SHA相同，输出和私有状态有限。仍有298个
received/raw-score/score/control元素比较超出原阈值（分别16/119/132/31）；不是全输出/全状态值等价检查。
region4的旧输出support SHA因修复而改变，region1保持一致。原资格失败保留。
该局部修复不认证所有FP32近边界情形；新生产树没有新的吞吐测量，不能套用旧性能数字。

## 修复后的训练与恢复

冻结 `97c52960` 的小X、d192、compact-5-dd4/n2、B2/L8/chunk4、FP32，
baseline为region batch1，combined为batch4+whole-region checkpoint；都从同一个真实
非空AdamW anchor开始。两条路径各完成推理、含梯度前向、重复VJP、三次继续更新、
checkpoint及独立新解释器恢复后的实际第五步更新。

baseline自身检查在原阈值内；combined相对baseline三次更新分别有1、1、8个字段超差，
涉及regional second.bias及KV/卷积状态，没有支持、int64、结构或非有限失败。
两条路径各自恢复的saved data/next forward精确，step4→5的实际参数/梯度/状态更新
阈内但非exact。不能用本地恢复通过覆盖跨路径失败。
冻结全CPU套件1043 passed、1 strict xfailed、1已知profiler warning；
隔离wheel的23包文件匹配修复源码；公开CPU训练/checkpoint数据和下一前向exact，
配对实际更新阈内，另一个新解释器从position8推进到16并真实更新。临时安装与checkpoint已退役。
这些是小CPU安装冒烟，不替代上述NPU资格；旧资格不追认为通过。

## BF16小训练见证

原生产树 `numerics-bf16-training-routing-v1` 从旧小X非空anchor执行baseline/combined
各三次继续更新。anchor/input匹配，六个观察的旧support/discrete SHA全部匹配，
完整浮点观察SHA六个均不exact，不称旧全数值轨迹逐位重现。

第三更新在region10/node16/sample1/int64时间191：baseline控制量0.004970729351，
combined为0。两个节点、总量1，此节点的独立解是 `clip((1−评分差)/2,0,1)`；
两条路径评分差分别0.9900585413、1.0018478632。各自Decimal支持与生产一致，同输入
分配重放exact。这是上游评分跨过真实支持边界，与9B求解器丢失正量不同。
第二更新另有内部支持差异，但旧内部轨迹未保存，不能称新增旧公开输出失败。
上游描述、梯度和历史的完整因果链仍未定位；本轮不重跑BF16全矩阵。

## 当前可复现入口

各run的 `launch.json` 保存实际命令、环境和源码；用对应提交的git archive恢复工具。
冻结source绝对路径是历史身份，不是长期在线承诺。以下输出必须指向新路径；正式长任务
仍使用 `scripts/launch.py`、空闲NPU队列及background.slice。
git archive不包含ignored artifacts，下面所有产物输入均须展开为主仓中的绝对路径；
launcher会设置冻结源码的PYTHONPATH，手动复现也须明确选择对应冻结src，避免载入主树editable安装。

```sh
# 删除update2后的当前可运行复现：固定72c1a9d0、CPU32线程、原affinity240–271。
# ORIGINAL_QUALIFICATION = artifacts/ffn-medium-cpu-qualification-v1/qualification
python -m scripts.replay_saved_training_update --qualification-dir ORIGINAL_QUALIFICATION \
  --verified-inputs artifacts/numerics-saved-update-v1/diagnosis/result.json \
  --owner baseline \
  --expected-observation artifacts/numerics-update2-baseline-v1/replay.json --output NEW_JSON
# W对应 --owner weight_access 和 numerics-update2-weight-v1/replay.json。

# 原局部FFN诊断固定e1c6aaa0，launcher需 --affinity 240-271；
# 其原生产identity守卫不能改为修复后的HEAD。
python -m scripts.diagnose_ffn_training_vjp --qualification ORIGINAL_RESULT_JSON \
  --output NEW_JSON --threads 32 --capture-mib 128

# 修复资格固定97c52960。
python -m scripts.check_allocation_face_witness \
  --witness artifacts/numerics-shared-routing-v1/routing.data/allocation-witnesses.pt \
  --device npu:0 --threads 4 --output NEW_JSON
python -m scripts.check_training_paths --config configs/memory-candidates/qualification.json \
  --device npu:0 --precision fp32 --candidates baseline combined \
  --batch 2 --length 8 --steps 3 --chunk 4 --threads 4 \
  --restore-mode new_process --no-internal-witness --output-dir NEW_DIRECTORY
python -m scripts.diagnose_routing_divergence --config configs/memory-candidates/9b-memory-shared.json \
  --device npu:0 --precision fp32 --batch 8 --prefill 128 --prefill-chunk 32 --decode 16 \
  --threads 4 --capture-mib 64 \
  --historical-results ABS_ROOT/artifacts/memory-shared-region1-b8-v1/inference.json \
    ABS_ROOT/artifacts/memory-infer-memory-shared-v1/results/common-b8.json --output NEW_JSON
```

原七载荷的 `diagnose_saved_training` 命令只保留历史身份：两份update2已退休，不能原样再次运行。
仍可直接阅读已保留result/witness及全部原字段比较，并使用上述删除后续更入口。
本轮没有9B训练、多NPU、CUDA或下游任务质量新增资格。训练优先FP32、固定已验证设置；
W/P/WP、BF16与区域分组的严格等价限制继续保留。通用基座未启动。

## 资源与终态

15个background.slice服务和110个记录后代均已核验终态。CPU服务合计
0.262616/4小时，NPU服务0.115923/2卡时（含队列/依赖等待）；
最大采样进程树RSS为30,912,393,216 B，约28.789/128 GiB。
新产物清理前观察约3.522/4 GiB，清理后约0.560 GiB。最多一张空闲NPU并发，
所有新NPU观察在队列准入的物理11，未使用共享占用卡。软件为公开dev-workspace9.2.0
及私有overlay（Torch/TorchNPU2.12、CANN9.2）；公共环境未改。
这些服务时长及RSS用于资源审计，不是吞吐或allocator显存测量。

训练新载荷含anchor合计1.020882 GiB，比内部1 GiB准备估算多21.383 MiB；
整体新产物仍低于4 GiB上限。成功恢复bundle已退役，跨路径失败anchor/checkpoints保留。
