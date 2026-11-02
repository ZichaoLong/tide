# Qwen 吞吐公开资料调查（2026-11-02）

本报告收录 **83条外部公开记录：CPU15、NPU13、CUDA55（52推理、3训练）**，
对应30份引用资料。它们不是本仓库新跑的83个实验，也没有新增本机正确性、续接或性能资格。
本轮仅读取文档、公开结果、配置和少量框架源码；没有下载权重、导入模型、安装框架或执行训练/推理。
仓库审阅基点为`37b567da553e988a35c65a9746a2c466aaa5ace0`；core-5固定数学未变。

[结构化JSON](qwen-throughput-survey-20261102.json)是本次调查的完整记录，
[CSV](qwen-throughput-survey-20261102.csv)是按指标展开的可筛选视图。
每个来源保留URL、读取时间、文档SHA256、版本及限制；每条结果保留原表行或原数字位置。
`external_report_not_locally_validated`表示外部报告，不能导入为`tidegraph`的合格run。
CSV一条测量可有多个指标行；统计测量数量须按`id`去重。空值表示未披露，不是零。

## 覆盖和资料可靠性

| 范围 | 本次取得的资料 | 仍未建立的结论 |
| --- | --- | --- |
| GPU推理 | Qwen2.5 0.5/1.5/3/7/14B，Transformers/vLLM；Qwen3 0.6/1.7/4/8/14B，Transformers/SGLang；BF16为主，少量FP8对照 | 官方选表都是B1；不能得到B128/B512曲线、3090/T4或本机NPU性能 |
| CPU推理 | llama.cpp的Pi5线程曲线、Kirin980局部配置、Xeon AMX候选对照；Q4不能与FP32混算 | 没有本机HiSilicon服务器实测；没有充分指定的服务器Arm大batch表 |
| NPU推理 | 910B3发布基准、Atlas A2限流例、910B4问题及后继；含固定B8和动态服务 | 非本机910_93/A3；动态请求数不是逻辑batch；发布基准有dummy权重 |
| GPU训练 | 8×A800、Qwen2.5-14B全参SFT、8K，Megatron与DeepSpeed三实现 | 原运行锁及有效token计数不足；不能冒称本机训练token/s |
| CPU/NPU训练、LoRA/QLoRA | 核对MindSpeed、SWIFT、LLaMA-Factory、Unsloth的支持、配方和公开相对声明 | 本次有限调查未取得条件足够完整的Qwen绝对吞吐点；支持清单不是实测 |

优先采用模型作者/框架发布资料；CPU社区记录单列出处和限制，不能当厂商保证。
`latest`页面按本次读取的内容hash归档身份，不能把其当前版本当原实验的源码版本。
不把搜索摘要、营销倍率、其他模型的MFU、教程输出、未来版本支持声明填成Qwen实测。
Qwen2.5/Qwen3在本报告中是有资料可核的固定调查组，不声称它们是最新一代Qwen。

## 指标口径

| 指标 | 分子/分母 | 必须同时说明 |
| --- | --- | --- |
| prefill | 处理的输入token / prefill耗时 | 是否含tokenization、传输、padding；实际计算行数 |
| decode | 生成token / decode耗时 | 是否扣除首token；上下文长度；单请求还是全服务聚合 |
| 全程输出吞吐 | 输出token / 整个请求或运行耗时 | 包含prefill、排队等，不能叫纯decode |
| 总token吞吐 | 输入+输出token / 运行耗时 | 输入很长时可显著大于输出吞吐 |
| 训练 | 有效token / 完整optimizer更新耗时 | microbatch、DP、梯度累积、全局batch、padding/packing、loss mask |

逻辑`B`是独立样例数。llama.cpp的`-b`/`-ub`是token处理/物理批上限；`pp512`是512个prompt token，
不是B512。`pp512`与`tg128`是分开的微基准，不是先prefill512再decode128的一次完整请求。
vLLM的请求总数、QPS、最大并发、实际就绪batch、`max_num_batched_tokens`分别记录。
训练全局batch满足`microbatch × DP × 梯度累积`；TP卡数不能再次当DP相乘。

以下所有数字均为原来源报告或明确标出的静态计算。没有跨机器性能倍率和core-5/Qwen加速比。

## GPU：模型大小、长度、实现与精度

来源：[Qwen3官方速度基准](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)，
JSON来源`qwen3-source`。单张H20 96GB，B1，输出2048；Transformers4.51.3与SGLang0.4.6.post1。
表中单位是官方定义的 **(输入+输出) token/s**，不是纯decode。

| Qwen3，BF16 | 输入1，Transformers | 输入1，SGLang | 输入6144，Transformers | 输入6144，SGLang |
| --- | ---: | ---: | ---: | ---: |
| 0.6B | 58.57 | 414.17 | 154.82 | 1426.46 |
| 1.7B | 59.83 | 227.80 | 238.53 | 838.28 |
| 4B | 45.94 | 133.13 | 159.95 | 466.19 |
| 8B | 45.32 | 81.73 | 146.12 | 296.25 |
| 14B | 40.66 | 47.10 | 108.52 | 174.85 |

数据文件还保留输入30720的同精度点。以1.7B/SGLang、6144输入为例，
`838.28 × 2048 / (6144+2048) = 209.57`是**条件换算的全程输出token/s**，仍含prefill，不能叫decode速度。
长度增加后“总token/s”上升，不代表单个输出token的延迟下降。

同一1.7B、B1、输入1/输出2048：SGLang的BF16→FP8为227.80→333.90；
Transformers为59.83→23.83。量化效果与具体实现相关，不能泛化为“FP8必然更快”。
SGLang跳过tokenizer、设置140000 context与85%静态内存比例；未报告实际占用，不能把空值当零。

[Qwen2.5官方表](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
提供另一组A100 80GB、B1、输出2048的对照：1.5B/BF16、输入1时
Transformers4.46.0为39.68、vLLM0.6.3为183.33；输入6144时为40.88、176.68。
该页没有明确给出速度分子和计时起止，本调查原样记录为“定义未完全披露的Speed(tokens/s)”，
不借用Qwen3的公式补写，也不与H20表合并排序。五个规模共20条BF16记录在JSON中。

## CPU：可参考的线程与算子经验

| 来源与材料 | 配置 | 原报告结果 | 使用范围 |
| --- | --- | --- | --- |
| [Pi5 Arm记录](https://github.com/ggml-org/llama.cpp/discussions/30069)，文件名Qwen2.5-0.5B-Instruct/Q4_0 | llama.cpp `836d571`，CPU，2线程，r5，单序列 | pp512：173.68±0.08；tg128：33.84±0.05 token/s | Cortex-A76/2GB；不代表本机服务器 |
| 同材料 | 4线程 | pp512：289.92±0.11；tg128：27.28±0.02 token/s | prefill受益而decode下降；不是“线程越多越快” |
| [Kirin980记录](https://github.com/ggml-org/llama.cpp/discussions/30549)，Qwen3-4B/Q4_K_M | `45451c3`，B1，context2048，8线程默认 | decode6.0 token/s；TTFT2.4s | 4个prompt，输入长度未给；仅社区记录 |
| 同材料 | 4个大核亲和性＋ngram-mod推测解码＋ub128 | decode8.2 token/s；TTFT2.2s | 作者报告greedy四例输出相同；多个策略同时改变，不能单独归因 |
| [Xeon AMX记录](https://github.com/ggml-org/llama.cpp/pull/30429)，Qwen3-8B/Q4_K_M | 8488C，固定4线程，5对交错；`23b0202`与候选补丁 | tg128中位6.09527→6.90782；pp512中位64.7596→62.8325 token/s | x86 AMX路径不能迁移成aarch64性能承诺 |

Pi5原表显示`qwen2 1B / 630.17M`，与文件名规模不同；两者均保留，未下载GGUF核定独立参数量。
llama-bench明确排除tokenization与sampling，不能与包含完整返回的core-5墙钟直接相比。
Kirin的“reply time”由`TTFT+128/decode速度`构造，不能标成实测HTTP整段耗时；Q4_0结果另组保留，作者报告输出/质量变化。
Xeon[原始五对数据](https://gist.github.com/ToukoUrsin/339887c6fec9b5dce8b01811ef654e51)
有两个较慢prefill候选；prefill均值下降14.89%，不能只保留decode收益而宣称全流程提升。

## NPU：动态服务、固定批与失败范围

来源：[vLLM-Ascend0.7.3发布基准#776](https://github.com/vllm-project/vllm-ascend/issues/776)。
Qwen2.5-7B-Instruct、TP1，作者后续注明910B3(64)；文本模型命令明确`load_format=dummy`。
有效dtype没有明确披露，不补写成BF16。200个ShareGPT请求、变长输入输出，实际动态batch未给。

| 到达率（请求/s） | vLLM-Ascend输出token/s | 加MindIE Turbo后输出token/s | 原路径TTFT中位ms | Turbo后TTFT中位ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 206.113 | 213.524 | 97.8581 | 69.5214 |
| 4 | 623.829 | 705.608 | 104.405 | 72.2935 |
| 16 | 978.005 | 1261.63 | 114.836 | 85.6201 |
| 一次性提交全部 | 1132.80 | 1485.51 | 2845.75 | 3198.48 |

这是发布方的结构/服务成本数据，**不是本仓库真实权重语言效果或独立正确性通过**。
一次性提交时吞吐增加而TTFT变差，服务选型必须结合延迟目标。
同报告还有固定**B8、输入32/输出128**：原路径完整batch延迟中位4284.9ms，warmup5、测量15次。
离线200请求、总88257 token、40.8233s、2161.93总token/s亦单列；200不是B200。

[v0.9.0rc2例#1167](https://github.com/vllm-project/vllm-ascend/issues/1167)
声明Atlas A2 313T 64GB、TP1、dummy权重、输入200/输出128、200请求、到达率1/s，
报告136.62输出token/s、350.09总token/s。它受请求到达速率约束，不能解释为饱和吞吐。

[910B4真实模型加载的问题记录#3112](https://github.com/vllm-project/vllm-ascend/issues/3112)
采用Qwen3-8B、开启prefix caching、请求输入1024/输出2048、200请求、10请求/s：

| 公开记录 | 输出token/s | 平均TTFT | 实际输入/输出token |
| --- | ---: | ---: | --- |
| Torch2.7.1、vLLM0.10.2、Ascend0.10.2rc1，原严重延迟 | 45.03 | 3754.759s | 204065 / 369494 |
| Torch2.9.0、vLLM `4034c3d`、Ascend `01d3515`，后继记录 | 631.76 | 209.864s | 204800 / 409600 |

两条记录均保留，原问题不改成通过。版本、实际工作量、提交者环境都变化，不能把它们称为同材料单项优化倍率；
后继仍有约210s TTFT，也不等于满足在线服务延迟要求。未给实际计算batch、有效dtype、模型revision或独立数值验收。
上述910B/A2资料都不能替代本机910_93/A3、TorchNPU2.12/CANN9.1资格。

## 训练：已有绝对数据和明确缺口

[Megatron-SWIFT官方表](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html#benchmark)
报告单机8×A800、Qwen2.5-14B全参SFT、8K context：

| 实现 | 原报告s/it | 原报告每卡内存 |
| --- | ---: | ---: |
| Megatron-LM | 9.04 | 64GB |
| DeepSpeed ZeRO2 | 10.32 | 80GB |
| DeepSpeed ZeRO3 | 10.56 | 58GB |

关联脚本固定读取于SWIFT `680df5c670d7e8add906ac49381d603869410612`：
[Megatron](https://github.com/modelscope/ms-swift/blob/680df5c670d7e8add906ac49381d603869410612/examples/megatron/base_to_chat.sh)
是TP4、microbatch1、global16、packing、selective recompute；
[DeepSpeed](https://github.com/modelscope/ms-swift/blob/680df5c670d7e8add906ac49381d603869410612/examples/megatron/benchmark/deepspeed.sh)
是8进程、microbatch1、GA2、BF16、packing、max_length8192，脚本显式选zero2。
这些是**当前关联配方**，不是原9.04/10.32/10.56秒运行的完整锁；不把它们猜填成原实际batch。
若另行假定每步确有16×8192有效token，才可用`16×8192 / s_per_iteration`条件换算；本数据集不把该推导写成实测token/s。

MindSpeed提供Qwen训练脚本，例如固定`4db0f8c`的Qwen3-8B配方为8设备、TP1/PP2、micro1/global64、L4096/BF16。
它证明存在可审阅入口，不提供相应吞吐。v2.2.0 README中的MFU和samples/s表属于Llama/Mixtral，不能挪给Qwen。
SWIFT NPU和LLaMA-Factory支持文档同样不等于性能表。
[Unsloth Qwen3](https://unsloth.ai/blog/qwen3)给出相对速度/显存声明及QLoRA B2、GA4、rank32，
但缺少完整的硬件、序列、绝对token/s组合；其30B表亦含MoE语境，不纳入dense绝对吞吐行。
因此CPU/NPU全参训练、各平台LoRA/QLoRA及训练规模曲线都仍需后续实测。

## 参数与大batch预算：仅静态下界

本次读取两份固定config及Transformers4.51.3结构源码，按共享embedding只计一次计算参数，未实例化模型：

| 模型 | 固定模型revision | 结构推导的独立参数 | 层/KV heads/head_dim | 每样例每token的BF16 KV |
| --- | --- | ---: | --- | ---: |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1,543,714,304 | 28 / 2 / 128 | 28,672 bytes |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 1,720,574,976 | 28 / 8 / 128 | 114,688 bytes |

Qwen3 Hub元数据给出的safetensors条目计数是2,031,739,904，与结构去重数相差一个embedding矩阵。
不能直接把Hub条目数称为独立参数，也没有检查实际权重存储/加载别名。配置、公式所据源码和两种计数都在JSON。

KV下界公式：`2(K,V) × 层数 × KV头数 × head_dim × B × (输入长度+输出长度) × dtype字节数`。
下表为**BF16唯一权重＋完整KV**，假定全部B同时活动、各样例KV独立、无前缀共享，输出128：

| 模型 | B128，输入512 | B512，输入512 | B512，输入2048 | B512，输入8192 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | 5.06GiB | 11.63GiB | 32.63GiB | 116.63GiB |
| Qwen3-1.7B | 11.95GiB | 38.20GiB | 122.20GiB | 458.20GiB |

尚未计入激活、logits、块取整、workspace、graph池、runtime、物理副本、主存及恢复峰值，不是资源准入或实测峰值。
FP32对应权重/KV字节翻倍。prefill切块可降临时激活，不能让同时活动样例的全部KV消失；分波执行必须报告真实并发。
以FP32参数/梯度/Adam两个FP32槽的`16P`为例，1.5B仅这四项就约23.00GiB，训练还需激活等；
混合精度master、梯度dtype、ZeRO/TP/DP和checkpoint会改变实际预算，不能套推理下界。

## 待统一授权的有限实测方案

当前没有模型作业，也没有默认排队。先做本机CPU/NPU；GPU资料继续独立保留，不依赖已释放的3090。
以下是建议实验表，不是已安装支持或现成可运行的`tidegraph run Qwen`入口。

| 分组 | 建议材料与变化 | 要回答的问题 |
| --- | --- | --- |
| batch曲线 | Qwen2.5-1.5B，输入512/输出128，B1/8/32/128/512 | 真正逻辑batch的吞吐/延迟/峰值，是否分波或小算子退化 |
| 长度曲线 | 同模型B8，输入128/512/2048/8192，输出128 | prefill与decode、KV和Attention成本分别如何变化 |
| 规模曲线 | Qwen2.5 0.5/1.5/3/7B，同B8、输入512/输出128 | 参数增大后的成本；先0.5/1.5，再准入3/7B |
| 架构复核 | Qwen3-1.7B，B1及B128，同输入512/输出128 | 同量级不同KV结构；不与Qwen2.5混成同一模型倍率 |
| 实现对照 | 主引擎共同点B1/B32；CPU PyTorch与llama.cpp，NPU TorchNPU/Transformers与vLLM-Ascend；MindIE作为预算允许的后续 | 同一材料的框架成本；量化、speculation、prefix cache分别成组 |
| 小训练 | 0.5B/1.5B，全参与LoRA分别选点；L512/2048，micro1/4，固定global batch并调整GA | 有效token/s、optimizer边界和真实内存，不要求收敛 |

基础三条曲线去掉重复点为每平台11点；额外架构/实现对照及训练分阶段准入，不做笛卡尔全扫描。
CPU先小材料比较4/16/64线程再固定，分别记录ATen/BLAS、进程和调度设置；不能继承旧core-5单线程结论。
NPU部署优先独立BF16组，FP32用于小型对照及与基座精度讨论；FP16/量化另组，不混算倍率。
本机公共基线只读；外部引擎依赖是否兼容TorchNPU2.12/CANN9.1须先解析验证，不能由文档支持推断安装成功。

实测授权后先实现最小外部runner/报告适配，复用项目持久作业、站点和资源准入，固定模型/引擎revision及完整配置。
准备不同内容、真实变长的独立样例；同材料固定seed、prompt、输出策略、thinking模式、cache/speculation开关。
另设定长合成计算组与变长真实文本组，标注截断/EOS/实际输出；真实轨迹不由CPU参考提前喂给候选。
记录实际并发、KV块、token批长/切块、MM/BMM形状、有效/填充token、TTFT/TPOT、吞吐、逐卡峰值及主机争用。
profile、warmup和正式计时分开；模型构造/编译/下载另计，计算计时包含声明范围内的准备、传输、调度、采样、返回和同步。
训练还包含loss、反向、梯度汇总、optimizer及同步；数据准备、checkpoint单列。

先在小例核对logits、loss、独立梯度及optimizer状态和保存后下一步；性能按需返回与完整诊断返回分开。
不同精度/量化的质量验收另列；自身恢复不替代独立正确性。非有限值、容量或运行时异常明确失败，不能吞成性能结果。
初始建议每配置20分钟计算上限，默认一次独立测量流程；达到时限保留实际进度，不外推大batch速度。
首批仅取0.5/1.5B资产，建议项目新增磁盘软预算60GiB；纳入环境、唯一权重缓存、checkpoint暂存/恢复双份后重新准入，
再决定3/7B扩展与旧资产退役。本机800GiB硬配额必须保留余量，tmpfs同主存联合预算。

## 与core-5比较的边界

现有core-5“样例·窗口/s”不是LM token/s，逻辑时间与prefill长度也不是Qwen上下文长度。
当前core-5合成向量任务没有对齐tokenizer、词表头、loss、模型效果或Attention可见性；参数相近不等于工作量相同。
推荐将后续结果分成两层：匹配尺寸/计算预算的稠密串行向量基线，以及实际Qwen部署基线。
前者用于研究图调度和packing，后者用于部署成本/吞吐选型；二者均需清楚声明实际工作量及返回口径。
本报告不改变原core-5严格失败、NPU B512精确续接失败、resident/多卡硬件边界。

## 查询、核验与保留

无需Torch、原运行目录或联网即可查询；例如列出NPU输出吞吐：

```sh
python -B - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('docs/evidence/qwen-throughput-survey-20261102.json').read_text())
for r in d['measurements']:
    p = d['profiles'][r['profile']]
    if p['platform'] == 'NPU' and 'output_tokens_per_s' in r['metrics']:
        print(r['id'], r['model'], r['batch'], r['metrics']['output_tokens_per_s'], r.get('weights'))
PY
```

CSV可按platform/model/task/precision/metric筛选；比较前同时检查JSON中的profile缺失字段、source caveat及timing。
CSV的`linked_recipe_*`不是原运行参数；`output_tokens`在问题记录中是请求长度、在reply估算中是公式长度，实际总量在JSON。
每条测量的原行/原数值和来源hash已保留；大网页及发现过程临时文件不作为长期依赖。
本轮核验仅包括表格转录、来源身份、JSON/CSV一致性、公式/预算算术、链接和固定数学校验，不是模型正确性测试。
后续可按source URL重取；内容hash不同须另记新调查版本，不能覆盖这次证据。
