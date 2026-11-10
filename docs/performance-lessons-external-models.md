# 外部模型执行与对照经验

本页按机制集中保存已有观察，来源与范围仍由各固定证据报告定义。
源码定位与历史提交的关系见[迁移映射](history-migration.md)。
当前待办与停止/重启决定只在[ROADMAP](ROADMAP.md)，下列历史决定不自动成为新任务限制。
从[经验索引](performance-lessons.md)选择相关主题；新工作按[优化技能](../.agents/skills/optimize-tide-execution/SKILL.md)进行。

## 外部Dense/MoE入口的初始实测经验（2026-11-02）

[固定来源/启动证据](evidence/llm-background-execution-20261102.md)与core-5结果分列。
Qwen1.5B BF16 B8/L512/输出128，同一材料下CPU4/16/64线程单次完整窗161.754/67.664/184.702s；
选择16线程继续，不能由更多可用CPU推断更快，也不由这一次外推其他模型/NUMA配置。
CPU FP32小型SDPA/cache与独立eager完整前缀token相同，但logits/KV在原阈值下超差；数值失败、
有限吞吐观察和同栈精确恢复必须分列。MoE激活参数量不用于代替总权重/KV/恢复预算。

任务间用同一计算锁避开项目自身争用。非阻塞轮询会反复错过短交接，设备已经获锁却无法计算；
后继用有超时的内核等待并保留原等待/取消记录。不在活跃检出上补丁，不把已安装旧wheel冒称新源码构建。
本机systemd生命周期隔离已实查，但v1 memory controller仍在共同user service；MemoryMax请求不是已证明硬隔离。
网络断流是资产失败；已校验文件/partial、显式续传与模型运行结果分别保存，不让下载外层终态伪造模型资格。


[11-03终态审阅](evidence/llm-terminal-audit-20261103.md)补充：同一Qwen1.5B BF16/输入512/输出128，
NPU B128→B512整批时间6.339→24.924s，输出吞吐2584.46→2629.45 token/s（约+1.7%），
allocated峰值5.591→13.668GiB。该单次配置的更大batch主要增加时长/内存，不能由逻辑batch推断线性吞吐收益。
CPU B512触及1200秒流程上限，没有完整窗口吞吐；不从B128外推。0.5B NPU长序列micro4是保守静态预算拒绝，
并非OOM；micro1实测峰值可供后继校准，但不直接把两者按倍数替代正式准入。
OLMoE/7B/30B MoE的多次长响应提前结束阻断了32项依赖点，优先完善可验证的分段续传；
下载失败、未检查、严格超差和自身精确恢复分别保留，队列exit0不等于模型全部通过。

## Dense/MoE后继的有限诊断（2026-11-03）

[后继证据](evidence/llm-continuation-20261103.md)将两个容易混淆的事实拆开：
CPU FP32/eager手写CE与完整梯度/第一步Adam新配置通过，并不修复旧BF16候选；
同eager完整前缀/缓存从第二个输出有严格数值差异，CPU/NPU小点token仍相同。
同前缀全部/最后logit投影本次通过，而eager/SDPA可在首logit不同；NPU还改变token。
因此先定位日程、算子和精度因素，不能把所有超差笼统归为设备或降低阈值。

NPU SDPA开关在此版本只设置全局变量；不能据开关标签删掉显存预算中的math后备。
micro1峰值可指导新诊断，不证明micro4的最坏情况安全。原73.23GiB预算拒绝保持。

大文件长连接失败后，严格32MiB Range及每文件有限重试可在实际URLError后继续偏移；
仍须完整publisher hash才算资产通过。原失败应归档，不能覆盖成running丢失证据。
续跑需要显式case清单和原配置身份；只换attempt重跑全矩阵既浪费资源又混淆结果。

第二实现要记录实际算术与库：GGUF MOSTLY_BF16含FP32 1D张量；n_ubatch只是容量，不是MM观察。
A3不能从普通A2 wheel或其他Torch栈推断可用；有界跨层后备是另一放置配置，不冒充TP/EP。
新引擎构建、实际执行和独立/精确恢复仍应逐项验收，不能由适配器单元测试推断模型通过。

在含torch_npu的公共Python基线中，Transformers导入检查也可能加载NPU库；`TORCH_DEVICE_BACKEND_AUTOLOAD=0`不替代CANN激活。此次未加载module的CPU机制测试报libhccl缺失，正确加载只读公共module后通过；不是模型或数学失败。

## Dense/MoE后继终态的正负经验（2026-11-04）

[57项审阅](evidence/llm-terminal-audit-20261104.md)确认：原1200秒限制包含构造/warmup/测量。
Qwen1.5B CPU B512正式生成2344.5秒之外还有837.7秒warmup，B8/L8192也有957.7秒warmup；
超时不能当作该配置不可执行。明确7200秒的新后继完成，原超时记录仍保留。
相同globalB4的CPU Qwen0.5B全参L2048，micro1/4三更新中位155.5/535.0秒；
更大micro并不保证更快，不能从减少累加次数推断整体收益。

OLMoE CPU/NPU首次真实推理和LoRA均完成，双NPU跨层B1/B32比单NPU更慢；
复制后的容量收益与小点速度是不同问题。独立CPU数值失败、NPU token改变，不能把成本倍率称严格等价加速。
OLMoE B128 NPU allocated23.205GiB而reserved60.910GiB，两种峰值都需报告；
B512静态估算63.080GiB被拒绝，不是已发生OOM，也不能直接从allocated线性推到安全容量。

NPU L2048/micro1实际dispatch有96次融合Attention及96次对应反向，这是算子证据，
强于仅检查SDPA flags。单instrumented更新的12.151GiB峰值仍不认证更大batch或连续更新峰值。

32MiB分段帮助OLMoE完整下载并最终hash通过，但Qwen7B/30B MoE仍因读超时耗尽预算；
分段不是网络稳定性保证，需保留partial及已核对分片。引擎phase=building也不证明开始编译：
llama实际停在源码归档5MiB的网络读取，下一步应修准备层而非猜测C++编译问题。


11-04[失败重启](evidence/llm-restart-20261104.md)中，权重仍需c28，但llama固定源码直连完成，
随后完整C++构建通过。模型权重与源码站点应分别试测，不能由一个域名的代理经验推广其他来源；
首块可达性与持续带宽也不同。OLMoE B512静态KV本身40GiB，prefill64→32/16只将总估算
63.08→60.08/58.58GiB；减小时间批不能消除常驻KV，需另列显式多设备配置，不降低安全余量冒充单卡通过。


[11-04重启终态](evidence/llm-restart-terminal-20261104.md)进一步显示：固定Git源码不应同时作为
模型运行cwd。NPU graph_fusion产物fusion_result.json落入检出后，下一次延迟提交的clean guard
正确拒绝；需要隔离每case运行目录，而不是放宽干净源码要求或把所有输出加入ignore。
OLMoE B512 CPU/双NPU确实完成，后者allocated约31GiB但reserved约56GiB，两种峰值仍须保留。
llama两个B1的完整状态精确恢复不能代替独立数值，两B32整流程超时不能从B1外推吞吐。
Qwen7B用8MiB/单模型流完成，Qwen MoE同策略仍连接失败，分段和串行不是网络稳定性保证。


[目录隔离后继](evidence/llm-isolation-launch-20261104.md)将相对厂商输出放在各case工作目录，
同时保留严格源码准入；真实写入、延迟准入和超时后代测试覆盖这条边界。
计时超时先增加有限阶段进度，把构造/warmup/正式生成分开，再决定是否值得优化或延长。
日志写入也属于测量成本。改动C++客户端须保留旧二进制/共享库，避免“更新ready”使旧checkpoint
失去准确恢复入口；两模型小型旧/新输出/状态回归及4次新进程恢复逐字节通过，只验证本次改动，不能消除已有的独立数值差异。


[11-05隔离后继终态](evidence/llm-isolation-terminal-20261105.md)确认真实NPU厂商输出进入case工作目录，
固定源码仍干净、下载就绪后的延迟提交成功；这是隔离边界的真机补充，外层exit0仍不能覆盖内部容量失败。
Qwen MoE双卡B32/B128为23.912/53.138 token/s，四卡为21.312/43.458，跨层显式复制的容量用途
与加速是两件事；严格小检查数值/token失败，不能称等价加速。B128四卡allocated约17–18GiB，
reserved仍约60.9GiB，不能由参数分摊或allocated外推安全batch。
双卡B512是56.719GiB估算超过54.737GiB准入，不是OOM；先拆解KV/临时激活和allocator峰值，
再考虑显式时间分块/更多设备，不靠降低安全余量或重复原配置。逻辑B与prefill时间块仍分列。
llama B32两个完整流程在显式7200秒预算下结束：正式1332.666/1042.777秒，warmup另计；
原1200秒失败没有消失，较大B在当前ARM/BF16/无BLAS构建上没有形成可观吞吐，不能推广到量化/BLAS版本。
