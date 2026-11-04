# Dense/MoE后继终态审阅（2026-11-04）

本次只读审阅作业、逐case结果与进程；未新建下载、构建或模型执行。
源码保持下载8b45cb4、后继472d94f、第二实现/放置983da72；数学hash通过。
最后一项于**2026-11-03 21:29:46北京时间**结束。11个后继service记录全部终态，
remaining_live_pids均空；核对原supervisor/workload PID+start identity已退出，
系统无活跃gef-lm unit，排除当前检查祖先进程后无命令行引用实验根的进程。
旧[启动观测](llm-continuation-20261103.md)保留原时间身份，不继续称running。

## 57项精确清单

对照跟踪recovery/engine生成器核对57个唯一case及完整有效配置，没有遗漏或重复计数。
详见[JSON](llm-terminal-audit-20261104.json)，包含配置、测量、工作量、原记录hash和失败。

| 类型 | 执行完成 | 依赖失败未执行 | 超时 | 预算拒绝 |
| --- | ---: | ---: | ---: | ---: |
| 44项测量/观察 | 26 | 16 | 1 | 1 |
| 13项检查/诊断 | 10 | 3 | 0 | 0 |
| 总计 | 36 | 19 | 1 | 1 |

26完成项含2个profile，不能全当正式吞吐；其余24成本流程为20推理、4训练。
4个新进程精确续接全部通过：OLMoE CPU/单NPU B1、OLMoE双NPU B1、Qwen1.5B双NPU B1。
它们不代替独立数值或MoE路由资格。外层queue exit0仅表示清单处理结束。

## 实际成本观察

下列推理均BF16、16 ATen线程、输入512/输出128、Transformers。
计时含上传、在线生成、有限值检查、token/统计返回与同步；构造/warmup/磁盘另列。
有限数值超差及离散差异保持；这些是探索性成本，不是严格等价加速比。

| 模型/放置 | B | 生成秒 | 输出token/s |
| --- | ---: | ---: | ---: |
| Qwen1.5B CPU | 512 | 2344.508 | 27.953 |
| OLMoE CPU | 1 | 107.961 | 1.186 |
| OLMoE CPU | 8 | 128.458 | 7.971 |
| OLMoE CPU | 32 | 282.996 | 14.474 |
| OLMoE CPU | 128 | 635.409 | 25.785 |
| OLMoE 单NPU | 1 | 54.589 | 2.345 |
| OLMoE 单NPU | 8 | 62.199 | 16.463 |
| OLMoE 单NPU | 32 | 69.815 | 58.669 |
| OLMoE 单NPU | 128 | 93.415 | 175.389 |
| OLMoE 双NPU跨层 | 1 | 76.030 | 1.684 |
| OLMoE 双NPU跨层 | 32 | 102.232 | 40.066 |
| Qwen1.5B 双NPU跨层 | 1 | 6.177 | 20.723 |
| Qwen1.5B 双NPU跨层 | 32 | 6.358 | 644.187 |

Qwen1.5B CPU B512正式生成约39.1分钟，另warmup837.722秒；全进程RSS峰值25.966GiB。
原1200秒超时未覆盖；新7200秒上限完成。B8/L8192另完成1259.049秒正式生成，warmup957.724秒。
CPU两个原L2048全参训练超时点均完成：globalB4/micro1和micro4的3更新中位分别155.515/534.987秒，
FP32参数/梯度/Adam槽、BF16 autocast配方未变。大micro在这次CPU观察中没有收益。
OLMoE Q/V LoRA（globalB4、micro1、L512）CPU/NPU各3更新中位49.616/4.284秒，
41.196/477.166目标token/s；仅机制和成本，没有模型效果声明，也不是全专家训练。

双NPU为同模型显式跨层复制、非TP/EP：这些B1/B32点比单NPU慢，不能推广到容量受限尺度。
OLMoE B128单NPUallocated峰值23.205GiB、reserved60.910GiB；allocated与保留池不是同一容量指标。
变长B128与两个独立profile均完成，详见JSON；不由profile计时宣称吞吐。

## 严格检查和容量

5个执行完成的独立check/training-check：CPU FP32/eager小训练1160字段全通过；NPU对应307字段失败。
OLMoE CPU FP32小检查36字段中5失败（首logits.0最大2.34e-5）、token相同；
单/双NPU BF16各36字段失败且各4个token不同。四个因素诊断和一个显存诊断独立记账。
旧严格阈值、旧Qwen/BF16失败不修改；未对各大点运行独立CPU参考，不写passed。

NPU Qwen0.5B L2048/micro1显存诊断完成：一次globalB4更新，
实际dispatch记录96次npu_fusion_attention_v3及96次对应反向；allocated峰值12.151GiB，reserved14.180GiB。
这证明该见证使用融合Attention，不能单凭micro1比例认证原micro4容量；原保守拒绝保持。
这次instrumented单更新和旧含warmup/连续更新的15.853GiB峰值范围不同。

OLMoE B512 CPU在1200秒整流程上限终止，未形成完整正式测量；NPU估算63.080GiB，
当时安全可用54.725GiB，分配前拒绝。后者不是实际OOM或已证实的实现上限。
需要另列时限/多设备或预算方案，不能改batch或沿用小batch倍率填补。

## 下载、引擎与剩余

- OLMoE完整12.892GiB、3分片与其余文件均publisher hash通过，实际MoE首次闭环。
- Qwen7B首分片保留2.930GiB partial；第7次临时失败耗尽6次重试预算，后几次为读超时。
- Qwen30B-A3B前4/16个权重分片已核对，完整文件14.899GiB，第5片保留1.578GiB；同样耗尽有限重试。
  这两个模型及Qwen MoE四设备点没有执行结果，19个依赖失败不可称模型失败或性能通过。
- llama.cpp固定源码归档下载在5MiB处read timeout；**未进入CMake编译**。流程phase虽写building，
  实际故障层是准备中的网络下载。4成本点与2小检查未执行，无GGUF/BF16 KV运行证据。
- vLLM/TP/EP原A3 NNAL配套边界保持，本次没有新安装、降级公共工具链或修改驱动。

主要待续是两模型/llama源码下载、OLMoE B512的时限/容量决策及数值/路由差异诊断。
状态查询不自动开启新的重试预算。可复用已完成实验；后继只选缺失项。
本次账户约106GiB，实验根du约55GiB；project160/account300GiB新增写入停止线、800GiB硬限保持。
当前瓶颈不是磁盘容量。精确接续位置与查询命令见[STATUS](../STATUS.md)。
