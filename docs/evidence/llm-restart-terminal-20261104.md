# Dense/MoE重启后终态审阅（2026-11-04）

本次为状态审阅，没有启动新下载、构建、模型或数值复验。
固定源码79197ad557b33bfa82c1914a02038f8944351247的全部后台服务已终态，
最后一项于**北京时间2026-11-04 15:37:11**结束。7个job记录（含资格）后代为空，
核对supervisor/实际workload身份均退出；系统无active gef-lm unit，排除检查自身祖先后无实验根相关进程。
[启动观测](llm-restart-20261104.md)保留原时间身份，不继续称running。
完整[21项JSON](llm-restart-terminal-20261104.json)逐项对照跟踪生成器与原提交清单。

## 清单终态

| 状态 | 数量 | 含义 |
| --- | ---: | --- |
| 执行完成 | 6 | 4个推理成本点、2个独立小检查；不等于严格全通过 |
| 1200秒整流程超时 | 2 | llama.cpp Qwen1.5B/OLMoE B32；没有完整metrics |
| 资产依赖失败未执行 | 3 | Qwen30B-A3B四NPU小检查及B32/B128 |
| 调度器失败而未提交 | 10 | Qwen7B CPU/NPU各1项，Qwen MoE CPU2/NPU6项；没有新case结果 |
| 总计 | 21 | 外层queue/job exit0不能代替上述逐项事实 |

## 新的成本结果

下表均16线程、输入512/输出128。Transformers是原BF16配置；llama.cpp是固定GGUF
MOSTLY_BF16（1D FP32）、BF16 KV、GGML native ARM/OpenMP且BLAS关闭，不能混称相同算术实现。
计时含在线生成、所声明的准备/上传、有限值归约、返回及同步；构造、warmup、磁盘分别记录。
各点一次流程，未排除其他项目负载。结果用于有限成本观察，不宣称严格等价加速。

| 模型/客户端/放置 | B | 正式生成秒 | 输出token/s | 本次恢复 |
| --- | ---: | ---: | ---: | --- |
| OLMoE Transformers CPU | 512 | 1962.182 | 33.400 | 未选择 |
| OLMoE Transformers双NPU跨层 | 512 | 183.145 | 357.837 | 未选择 |
| Qwen1.5B llama.cpp CPU | 1 | 51.258 | 2.497 | 新C++进程精确通过 |
| OLMoE llama.cpp CPU | 1 | 37.278 | 3.434 | 新C++进程精确通过 |

OLMoE CPU B512另有728.533秒warmup、构造1.656秒、RSS峰值66.136GiB；
双NPU另有64.295秒warmup、构造8.546秒、RSS峰值6.388GiB。
物理5/9映射逻辑0/1，参数实际分别3,562,602,496和3,356,559,360，合计6,919,161,856。
这是同一模型的跨层显式复制，不是TP/EP；卡9按原PID/start绑定共享，争用记录保留。
两卡allocated峰值31.738/31.175GiB，reserved峰值56.908/55.705GiB，不能混为同一指标。

B512实际在线prefill为8次×每次512样例×64时间，每次32,768输入行，decode127次×512样例。
这说明客户端向模型提交的批次，没有测量本次内部MM/BMM形状或MoE专家桶，不能冒称所有算子充分合批。
两大点独立数值、离散及checkpoint均not_checked；此前小点的数值/token失败不被执行成功覆盖。
原单NPU B512预算拒绝仍保留，本次双卡成功不称原单卡通过。

llama两个B1分别另有warmup33.265/26.775秒；新进程比较完整序列化状态及下一步logits逐字节通过。
两项独立B2/L16检查均执行完成，但四步logits均超原atol1e-5/rtol1e-4：
Qwen1.5B token也不同；OLMoE token相同。GGML独立KV未暴露，仍not_checked。
两个B32流程达到1200秒时限后停止，没有完整metrics，不从B1外推吞吐或判断具体耗时阶段。

## 下载与调度故障

Qwen7B的4个权重分片及全部资产已在14:39:59完成发布hash验证，合计14.196GiB；
本次只核对完成记录和size/mtime，未重复大文件hash读写。它的CPU/NPU测量尚未执行。
Qwen30B-A3B在第5/16分片再次耗尽10次有限重试，11次连接错误均记录URLError；
已验前4片保留（完整文件14.899GiB），第5片partial从1.578增长至2.875GiB，未运行该模型。
不重新下载已验分片，不把下载失败写成模型失败。

新的调度故障已定位：NPU执行末段在固定检出当前目录留下648字节的
`fusion_result.json`（graph_fusion诊断，mtime10:59:01，SHA256
10a47c1bd11eefdb667d57f66ab5d36c26fde12cc19b8c9a41db41b2a0c4e993）。
跟踪源码无diff，仅此未跟踪输出。14:40:20 dispatcher调用`submit_lm_job.py`时，
干净源码检查正确拒绝了该目录，dispatcher随之退出，10项后续清单没有创建新job/case。
这属于启动器未隔离运行工作目录的缺口，不是用户修改了固定源码，也不是Qwen7B执行失败。

后续修复应让模型运行在各自输出/scratch目录，使用固定源码的绝对入口、输入与PYTHONPATH；
保留源码干净保护，不全局忽略任意产物、不删除输出后盲目重跑。回归需覆盖真实子进程写cwd、
延迟提交下一个独立队列仍成功，以及异常/超时后代清理。再新attempt只补未执行项。
当前只记录问题和重启方案，没有借状态查询开启新执行或重试预算。

## 可复用材料与资源

llama完整构建和Qwen1.5B/OLMoE两份GGUF均保留，来源、二进制/共享库/转换hash见JSON。
原失败、partial、输入、checkpoint及运行时小文件均未删除。固定实现/安装的118项资格不因此升级为模型通过。
账户约136GiB，实验根du约84GiB；项目160/账户300GiB新增写入停止线与800GiB硬配额不变。
当前没有GPU依赖、NPU等待或活动下载；瓶颈为工作目录隔离、Qwen MoE网络和llama B32时限/性能。
接续位置和命令只见[STATUS](../STATUS.md)，未完成项只见[ROADMAP](../ROADMAP.md)。
