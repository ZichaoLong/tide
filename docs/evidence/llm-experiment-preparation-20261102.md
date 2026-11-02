# Dense / MoE 实测准备与待授权计划（2026-11-02）

本次完成环境只读审计、三条网络路线的有界测速和有限实验设计。
**没有安装软件、下载模型权重、导入Torch、执行模型或提交后台作业。**
用户要求先审阅本计划，再授权执行；以下矩阵均为计划，不是结果或已可运行的配置。
审阅源码为`5d0e99e37cddf9db982fc9ad5aae13cd33ed0b54`，固定core-5数学未变。

[准备记录JSON](llm-experiment-preparation-20261102.json)保存9次网络观察、环境元数据、来源、预算和计划上限。
既有[Qwen公开调查](qwen-throughput-survey-20261102.md)的83条外部数据保持原身份，不增加本机模型资格。
当前任务状态只见[STATUS](../STATUS.md)，待办只见[ROADMAP](../ROADMAP.md)。

## 网络：先用c28，按实际目的站分别判断

实际入口为直连、`proxy route=relay`（当前relay为aura）、`c28 route=direct`。
后者表示通过c28服务器访问网站，与本机原生直连不同。没有切换共享路线、relay或修改全局配置。

每个对象/路线一次，顺序执行，禁止并行测速互相争用；socket超时10秒、整体25秒。
读取HF固定config、公开WikiText数据文件前16MiB、PyPI NumPy wheel最多16MiB。
二进制数据边读边计算SHA256后丢弃，没有安装wheel或保存数据集。

| 路线 | HF小型config | HF数据/CDN样本 | PyPI wheel样本 |
| --- | --- | --- | --- |
| 本机直连 | 10秒后连接失败 | 10秒后连接失败 | 读取0.125MiB后超时；未完成 |
| netpath proxy / relay | 10秒超时 | 10秒超时 | 25秒仅读取0.3125MiB；未完成 |
| netpath c28 | 660 bytes，2.47秒 | 16MiB，15.05秒，**1.063MiB/s** | 14.89MiB，13.95秒，**1.067MiB/s** |

速度包含连接、重定向与读取。c28两项纯body观察约1.30/1.16MiB/s；不能挑body速度冒称全程速度。
完整wheel的SHA256与PyPI公布值相同。9次样本总计读取32,849,324 bytes，约31.33MiB。
权重URL另做一次HEAD，返回200并指向同一`us.aws.cdn.hf.co`，声明文件3,087,467,144 bytes；**权重body读取为0**。
这仅确认该URL的元数据/重定向可达，不证明真实权重的持续速度、断点恢复或全部分片可下载。

按本次约1.06MiB/s短样本作排期参考：0.5B+1.5B BF16权重约一小时，OLMoE约3.5小时，
Qwen30B MoE约15小时。不是下载时长承诺；分片、缓存、CDN和波动都会改变结果。
初始只开一个下载流；授权后根据首个真实分片观测，最多比较两个流，不默认大量并发。
当前不继续重复失败路线、不改relay；后续若路线已调整，再独立记录新的测速。

复核可从JSON读取`network.samples[].source_url`，例如用curl执行同样的上限：

```sh
# c28；改为--entry proxy前先确认proxy status仍为relay。
/home/zlong/.local/share/netpath/skills/use-netpath-network/scripts/netpath exec --entry c28 -- \
  curl --fail --location --range 0-16777215 --connect-timeout 10 --max-time 25 \
  --max-filesize 16777216 --output /dev/null --silent --show-error \
  --write-out 'http=%{http_code} bytes=%{size_download} seconds=%{time_total}\n' PUBLIC_URL_FROM_JSON
# 直连用curl --noproxy '*'加同样参数；不输出认证环境或重定向签名URL。
```

原观察使用Python流式读取和SHA256；上述curl为有界复核入口，不能写成原程序的重复测量。

## 环境与当前资源

| 项目 | 实际观察 | 执行时的决定 |
| --- | --- | --- |
| CPU | aarch64，320个affinity CPU，8个NUMA域；user.slice未见CPU配额 | 计时先比较4/16/64线程，记录NUMA和亲和性；构建线程与计时线程分开 |
| 主存 | 总计约2TiB，可用约1.8TiB | 不把快照视为预留；每个计算任务初始主存上限384GiB |
| 磁盘 | 账户32438MiB，软限400GiB、硬限800GiB | 项目首阶段80GiB、全部阶段160GiB上限；账户到300GiB暂停新大写入并审查 |
| 临时内存盘 | /tmp与/dev/shm均为tmpfs，各显示约1006GiB容量 | 二者共享主存，不相加；项目临时周转先限32GiB，并计入任务主存 |
| NPU | 8块双芯片板，npu-smi列出16个物理芯片，每芯片65536MiB HBM，健康OK | 按物理芯片映射到逻辑设备；不把8块板误写为8个64GiB执行设备 |
| NPU占用 | 第一次无进程候选0/1/3/5/7；第二次为0/3/5/7，1已被其他进程使用 | 尚未获取任何锁；正式启动仍须两次检查、全组协作锁和实际HBM准入 |
| 后台设施 | systemd用户管理器和background.slice在线 | 复用现有持久作业，不使用会话内nohup替代 |

公共Python基线仍为Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post2、CANN9.1。
本次只通过`importlib.metadata`检查安装身份，未初始化设备。
公共环境已有safetensors0.8.0，但未安装Transformers、tokenizers、huggingface-hub、accelerate、PEFT或vLLM。

**两条环境路径分开：**

- Transformers参考/训练：复用只读公共2.12基线，在项目私有overlay补齐增量依赖并锁定；先依赖解析，再安装和小型算子检查。
- vLLM-Ascend：先核对一个明确版本对和Ascend专用构建方式，使用独立环境。已读v0.30.0 README要求CANN9.1、Torch2.10、TorchNPU2.10.post4；PyPI该版本wheel要求post7。普通PyPI vLLM0.30.0又要求Torch2.13，与插件元数据不一致。必须据实际Ascend安装方案解释这些差异后锁定，不能直接把两个普通wheel强装在一起，也不降级公共2.12。

版本README、两份PyPI元数据及hash保存在JSON。猜测的原始`docs/source/installation.md`路径返回404，
未据此认定产品不可用；安装来源路径、post版本和实际构建依赖需在授权后的配套准备阶段核实。
最多尝试一个有官方依据的配套方案；若仍不兼容，记录具体阻碍，继续独立的Transformers/CPU工作，停止版本遍历。
需要修改公共环境、宿主驱动或超出项目预算时再请求额外授权。
NPU进程在首次autograd前注册后端；CPU与NPU检查分进程。双芯片先验实际HCCL/并行小例，
再进入模型分布、训练和恢复；旧普通复制证据不能替代该引擎的集体通信资格。

## 模型与容量

选择已有资料充分、结构清楚的固定模型，不默认追随新一代模型。

| 模型 | 总参数 / 每token激活参数 | 专家 | BF16权重 | B128 / B512的权重+KV下界 |
| --- | --- | --- | ---: | ---: |
| Qwen2.5-1.5B | 约1.544B / dense | — | 约2.88GiB | 5.06 / 11.63GiB |
| OLMoE-1B-7B-0125-Instruct | 约6.919B / 约1.3B | 64选8 | 12.89GiB | 22.89 / 52.89GiB |
| Qwen3-30B-A3B-Instruct-2507 | 约30.532B / 约3.3B | 128选8 | 56.87GiB | 64.37 / 86.87GiB |

下界采用输入512、输出128、全部逻辑样例同时活动、BF16完整KV，无前缀共享；不含workspace、激活、
runtime、分块取整、物理副本、checkpoint和恢复峰值。OLMoE的B512单芯片能否容纳还需校准，不能仅凭52.89<64就准入。
Qwen30B MoE从双芯片预算开始；不反复OOM尝试单芯片BF16边界，不把两芯片空闲HBM简单相加。
每芯片至少预留`max(6GiB,10%总HBM)`，根据实际空闲量、模型放置和完整峰值逐项准入。

OLMoE固定`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`，原生context4096；其Instruct配置`use_cache=false`，
推理需显式启用并记录KV策略。Qwen MoE固定`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`，非思考版本。
模型卡/config URL见JSON；尚未检查权重实际别名和加载后的物理副本。

MoE按总参数预算存储/全参训练，按实际路由和算子分析计算，不能只用激活参数量估计全部成本。
例如Qwen每层B512单token decode平均只有`512×8/128=32`行/专家，B128只有8行；
真实负载不均和填充还会降低效率。专家6.25%的选择比例不是整模型FLOP或内存比例。

## 待授权的有限矩阵

主要推理采用BF16；CPU FP32用于小型数值锚点。FP16、量化和不同返回契约独立记录，不能静默替换。
CPU先确定合理线程与NUMA配置，再固定整条曲线。下表主推理曲线固定输出128token，长度为输入token数。

| 批次 | 配置 | CPU/NPU唯一主测点 |
| --- | --- | ---: |
| 首批dense | Qwen1.5B：L512，B1/8/32/128/512；B8，L128/512/2048/8192；补Qwen0.5B B8/L512。重复点只运行一次 | 9+9=18 |
| 首批MoE | OLMoE：L512，B1/8/32/128/512；再补B8/L128和L2048 | 7+7=14 |
| 后批dense规模 | Qwen3B/7B，B8/L512 | 2+2=4 |
| 后批Qwen MoE | CPU B1/8；NPU双芯片B1/8/32/128/512，补B8/L2048 | 2+6=8 |
| 合计 | 首批32点；后批12点 | **44点** |

主曲线使用不同内容、长度明确的独立样例；不是将一个prompt复制B份。
另加**最多4个真实变长输入点**：Qwen1.5B与OLMoE的B128，各CPU/NPU一次；
输入长度覆盖128/256/512，允许自然EOS，记录实际输出/活动样例/KV长度，单列计时，不与固定长度口径混算。
主曲线固定生成长度的EOS策略、采样和seed提前定义；不由CPU预跑生成候选轨迹。

**实现/放置对照最多10个新增点**：CPU的Qwen1.5B和OLMoE各B1/B32优化实现（4）；
NPU Qwen1.5B和OLMoE各B1/B32第二实现（4）；Qwen MoE双芯片TP/EP在B32/B128的另一放置（2）。
CPU优先评估llama.cpp，NPU优先vLLM-Ascend；以实际支持、配套和独立检查为前提。
同dtype/输入/返回下才计算实现倍率；只存在量化实现时单列成本，不能冒称同精度收益。
若优化引擎受阻，可有界检查Transformers/Accelerate的跨层放置；它是另一种配置，不称TP/EP或多模型副本容量。
不新增core-5算子实现来承载Qwen，也不将llama.cpp称为standalone LibTorch。

**训练最多14点**，统一global batch4，梯度累积为`4/microbatch`：

- Qwen0.5B全参：micro1/4 × L512/2048 × CPU/NPU，共8点。
- Qwen1.5B全参与LoRA：micro1/L512 × CPU/NPU，共4点。
- OLMoE LoRA：micro1/L512 × CPU/NPU，共2点。

建议固定配方为非padding token上的移位交叉熵均值，AdamW lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0。
全参使用FP32参数/梯度/optimizer槽、BF16 autocast计算和FP32 loss；实际生效精度另记录。
LoRA使用冻结BF16基座、FP32可训练adapter/optimizer，r=8、alpha=16、dropout=0，只作用于Attention Q/V投影。
运行前解析并冻结具体模块名单；OLMoE router和专家基座参数冻结，原辅助loss系数0.01显式保留。
这是一套新的固定测量配方；LoRA不认证全专家参数训练，不为避免失败暗改loss或更新方式。
每点建议1次warmup更新、3次完整计时更新，检查有限值、实际有效token与optimizer边界；
小代表另做checkpoint后的下一步/新进程恢复。CPU小例对照logits/loss/梯度/optimizer，未检查的大点明确标未检查。
不默认开展30B全参训练；例如FP32参数、梯度、Adam两槽的16P仅权重相关就约455GiB，还没有激活。

另有CPU线程校准最多3点、精度校准最多2点、独立profile最多4点。
上述上限合计81个流程，包含profile/校准，**不是81条性能结果或严格通过保证**；小型正确性检查另列。
profile与正式计时分离，重点观察专家token直方图、grouped GEMM、dispatch/combine、通信、padding和同步。
单配置一次独立流程，20分钟计算时限；超过时限保留实际进度，不按小batch倍率外推。

## 实施、后台与停止条件

授权后按以下依赖推进，阶段完成后继续，不为每个正常动作重复请求许可：

1. 补最小已安装CLI的外部LM配置/预算/检查/运行适配，复用现有任务与CSV/HTML报告；目前`tidegraph`尚没有该LM入口。
2. 固定私有环境、源提交、模型revision、输入、精度与完整配置；先提交实现，再对固定源码运行检查。
3. 后台先下载Qwen0.5/1.5B，完成CPU/NPU小例和首批dense；下载OLMoE后进入其矩阵。下载等待不持有NPU锁。
4. 首批资源/时间实测校准后，再下载3B/7B和Qwen MoE，进入后批；独立路径受阻不阻塞其他路径。
5. 审阅终态、失败与计时范围，提交小型证据和可查询报告；保留配置及必要复现，清理获准退役的中间产物。

下载、环境/构建、CPU、NPU使用独立`background.slice`服务和固定源码，复用仓库持久记录。
队列等待放服务内，单任务等待最多6小时；计算服务最长24小时，大资产下载服务最长30小时。
初始下载并发1，最大2须有首分片实测依据。权重共享一个内容缓存，不为每个引擎复制完整原始资产；格式转换独立预算。
下载可与轻量准备/正确性检查交叠；正式计时串行且避开项目重型构建/转换，记录同时存在的其他负载。
构建按实际CPU和内存约束选all/half/显式线程数，不能继承测量的4/16/64线程作为构建上限。

任务持久保存状态、日志、展开配置、依赖和进程身份；停止用完整control-group，确认没有遗留后代后才清理输入。
失败配置默认不自动重跑；恢复以明确checkpoint/source/input身份启动新run。启动队列不代表完成，也不承诺自动通知。
网络凭据只由netpath注入下载子进程，不写进配置、命令行或持久日志；不切换全局路线。

首阶段80GiB和全阶段160GiB是项目磁盘峰值上限，包含环境、唯一权重、转换暂存、checkpoint双份/恢复和报告，
不等于每项自动准入。账户总量达300GiB即停止新增大写入，保留400GiB软限和800GiB硬限之前的余量。
tmpfs只作可丢弃周转，最多32GiB且与主存联合预算；小型记录和必要checkpoint保留在持久盘。

非有限、非法配置、运行时/通信错误、预算不足和超时均明确终止相应点。
数值超差、离散输出/路由差异、自身续接、执行完成和吞吐分列；未检验不得填通过，探索不能吞任意异常。
保留全部既有core-5严格失败。core-5样例窗/s与LM token/s仍不直接相除。

**本次准备到此停止，等待用户对这份范围的执行授权。** 当前没有模型、下载或排队作业，gpu3090已释放且不在计划中。
