# Dense/MoE 后台执行入口与交接（2026-11-02）

CPU/已安装wheel及条件后批固定`6bbc7e2240c2a20958c1bb5206030f4d2ade2ac4`；NPU队列后继固定`15dfabe55ea8f0608cf6fc795513c6487eb664c4`（锁等待修改）；资产bootstrap固定`3360eafd3f5f94dc993d5d8cefdc428186d50b02`。
固定数学未改。本文是启动/设施证据，不是全模型结果；[JSON](llm-background-execution-20261102.json)保存观测时刻、原终态、版本和资格身份。
当前队列状态只见[STATUS](../STATUS.md)。用户可继续开发主仓库，活跃任务仅使用仓库外固定源码和私有环境。

## 已验证与已启动

- 非editable wheel安装与独立安装入口通过；固定源码81项相关测试通过，涵盖资产完整性/续传、配置/预算、三架构随机tiny缓存、梯度累积/Adam状态恢复、队列路径、HTML及完整后代停止。不是预训练模型或NPU资格。
- Qwen1.5B全部3,098,973,447 bytes已按官方hash校验；OLMoE第一分片在1,045,278,078/4,997,744,872 bytes处提前断流，partial与exit1保留；不是模型数值失败。Qwen0.5B首次连接URLError保留，经Qwen1.5B完整传输证实路线可用后，显式提交一次恢复并已全部hash通过。其完成后才对OLMoE提交一次Range续传，与后批下载保持最多两路；原断流记录保留，不循环重试。
- CPU/NPU各有真实background.slice服务；项目计算flock避免正式计时互相重叠。NPU准入使用已固定的协作helper，真实物理设备映射保存在allocation.json。
- 首个Qwen1.5B CPU FP32 B2/L16/输出4独立检查完成：token一致，55个logits/KV张量检查超固定阈值，最大绝对差0.00018310546875。参考为eager完整前缀，候选为SDPA在线缓存。该差异不改写为通过；后续大点仍独立标not_checked，不推断严格等价。
- 首批包含68个清单流程：59测量/观察及9小检查。后批12个主推理点在首批下载终态且Qwen1.5B B1校准完成后获取资产；下载完成再提交独立24小时CPU/NPU后批服务。
- 总计71个可执行测量/观察加9小检查；另10个优化引擎/TP、EP点未排队，不映射成其他实现冒充完成。

## 保留的启动失败

`9cdf7d9`已通过79项设施检查，但实际CPU/NPU启动在`dict(**plan, cases=...)`重复字段处失败，未执行模型。
原`cpu-a/npu-a` exit1保留。`assets2-a`仍在等待，已停止，实际MainPID=0、ControlGroup为空、remaining_live_pids=[]。
后继`6bbc7e2`改用显式字段覆盖、分离不可变manifest与生效case、新增队列完整路径/内部失败测试；81项固定检查通过。
使用新的`*-b`单位和输出目录，不覆盖原失败。活跃bootstrap下载未中断或修改源码。
随后观察npu-b的非阻塞flock轮询反复错过CPU交接；确认尚未启动模型子进程后取消，实际MainPID=0、ControlGroup为空、remaining_live_pids=[]。
15dfabe改为有界内核阻塞等待，固定19项队列/清单检查通过，另提交npu-c；CPU-b继续原固定源码，活跃环境/wheel未改。

CPU BF16 Qwen1.5B B8/L512/输出128线程校准：4/16/64线程完整窗分别161.754/67.664/184.702秒。
后续CPU选16线程。B1执行与新进程精确续接通过；B8随后完整窗64.569秒。
这些是当前共享主机上的单次观察、独立数值未逐大点检查，不外推所有模型/线程，也不把自身恢复当严格资格。

## 范围与接续

[操作手册](../external-lm.md)给出安装CLI、JSON示例、plan/check/run/resume/results、有限矩阵和停止方式。
结果使用独立LM schema与token计量，复用基座持久监督器和离线HTML；不把token/s除以core-5样例窗/s。
所有运行保留展开配置、源/包/输入身份、精度、参数位置、实际批次、构造/warmup/计时/恢复及峰值。
小型独立检查、自身精确续接和有限性能观察是不同事实。profile单列；无直接模型效果或所有规模资格。

当前资源上限：项目80/160GiB、账户300GiB停止线（硬限800GiB）、计算服务请求384GiB主存并静态准入；
每配置计算/恢复1200秒、设备/计算锁等待6小时、计算service24小时、下载service30小时。
本机v1 memory controller仍共用user service，MemoryCurrent未设置；不把MemoryMax属性当已证明的逐unit硬隔离。
公共Torch/CANN未改，gpu3090未连接。没有自动唤醒AI或跨重启续跑承诺；终态后按配置与记录审阅再决定后继。
