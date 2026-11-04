# Dense/MoE失败重启提交与构建（2026-11-04）

用户明确授权重启失败作业。本报告为北京时间10:10的启动观测；
[JSON](llm-restart-20261104.json)包含完整21项配置、来源、资格、服务/cgroup、网络和资源记录。
运行/等待不是模型通过，旧[57项终态](llm-terminal-audit-20261104.md)保持原失败身份。

## 实现和固定资格

实现79197ad557b33bfa82c1914a02038f8944351247，固定检出source/restart-g，私有wheel已安装。
开发相关46项、固定安装/设施118项通过；两者重叠，不累计成模型资格。
新增retry-matrix复用原生成器和配置校验；dispatcher与引擎后继可接明确清单/新依赖身份，
站点shim固定既有共享适配器的policy/helper hash。未修改模型、数值、训练配方或公共工具链。
测试核对19项完整配置与原证据一致，已完成项不进入重启清单；超时和双设备变更独立显式标记。

| 清单 | 项数 | 范围 |
| --- | ---: | --- |
| compute | 11 | 10个Qwen7B/30B MoE未执行点，OLMoE CPU B512新7200秒时限 |
| llama | 6 | 原2小检查与4个Qwen1.5B/OLMoE CPU成本点 |
| placement | 4 | 原Qwen MoE四设备3项，新增OLMoE双NPU B512容量替代 |

OLMoE单NPU B512原63.080GiB拒绝保留。减小prefill_chunk到32/16仍估60.080/58.580GiB，
没有通过调小默认预算强行运行；双NPU每卡估36.828/36.444GiB。
双卡保持同一模型/B512/512输入/128输出/BF16/prefill64和完整KV，属于另一放置配置。
原双卡OLMoE小检查及精确恢复按原源码身份复用；数值/token失败不改写。

## 网络、构建与实际后台状态

首1MiB试测含连接开销：权重直连及当前proxy各15秒超时，c28精确206耗3.681秒；
llama源码直连1.125秒、c28 2.210秒、proxy 6.572秒。不能外推稳定大文件带宽。
模型改8MiB Range、每文件10次临时错误重试，先Qwen7B再Qwen30B-A3B，保留已验分片和partial。
未更改netpath共享路由或写入认证。Qwen7B断点已实际增长；Qwen MoE新服务等待前者终态。

llama源码改走直连，固定2cbca389f515df455412dafa07e2b1359cc72a6d完整归档38,909,915 bytes取得，
旧5MiB partial另名保留。**完整C++构建已完成**，实际320 affinity线程；编译器、archive/binary/共享库hash
及foundation source见JSON的engine_build。独立转换overlay已准备；GGUF转换等待计算锁，尚无本次模型资格。
源码下载成功与C++构建成功分别有事实，不能据此认定BF16 KV、模型执行或恢复通过。

6个服务实际active，supervisor的name=systemd cgroup均在background.slice、与focus.service分离：
两下载服务、llama、placement、dispatcher和CPU-ready。CPU B512已取得compute锁并进入计算；
双NPU B512在站点队列等待。其余4个CPU/NPU服务依赖资产，尚未提交，之后由dispatcher自动提交。
所有计算/构建由同一个compute.lock串行，网络读取可并行于计算；用户可继续编辑主工作区。
具体unit、状态/日志、查询和停止命令只在[STATUS](../STATUS.md)维护。

NPU只对一张重新观测的低占用物理9复用既有共享授权：固定PID/start/UID，48小时有效，
HBM≤8GiB、AICore≤5%、空闲≥52GiB，保留两次采样、队列顺序、整组锁和整组生命周期。
其他卡要求空闲，新增进程或超预算拒绝；每case保留共享审计。尚未从队列等待推断真实跨卡执行。

small下载30小时；moe/llama/placement42小时；依赖等待30小时；设备/计算锁6小时；
自动计算service24小时/队列23.5小时。仅CPU OLMoE B512改7200秒，其余case仍1200秒。
模型资产、输出和环境保留在仓库外；项目160/账户300GiB新增写入停止线及800GiB硬配额不变。
当前账户约106GiB，新增剩余资产/GGUF/构建约70GiB预留。MemoryMax为请求，不夸大v1控制器隔离。

后续审阅需分别检查执行、独立数值/离散、精确恢复、容量拒绝和吞吐；不能把outer exit0或构建通过
提升为内部全部通过。vLLM/TP/EP、原数值失败及已释放GPU的边界不变。
