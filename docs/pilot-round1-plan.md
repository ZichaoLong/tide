# 第一轮真实文本机制与成本试运行

用户于2026-11-19同意执行。正式任务表为40项（32 NPU、8 CPU），最长3600秒，提前300秒停止派发。一次性环境、数据与训练器准备另记；正式窗口包括排队、初始化、训练、验证、推理、保存和恢复。

## 固定输入

数学 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，实现 `bef26abb2458cfff003c84f376f93bc18a0b4106`，生产tree `a9d9634a2a7a36e6f1bc194ee0edb34f47e22465`。外部仓库只读，固定源码建wheel并安装到 `.venv-pilot`。实际包版本、实验源码SHA、计划SHA及数据manifest身份在 `.local/pilot-r1/frozen/identity.json`。

公开WikiText103 raw训练分片 revision `b08601e04326c79dfdd32d625aee71d232d685c3` 已校验。完整一级标题文章按内容哈希加固定盐选2%留出，重复文章保持同侧。前约16MiB完整训练文章拟合byte BPE8192，编码全部数据，每文章追加EOS。训练28,868篇/133,224,044 token，留出576篇/2,725,949 token。先均匀选足够长的文章，再均匀选文内起点；这是试运行留出探针，不是官方perplexity。

## 模型与训练

[任务表](../configs/pilot-round1/plan.json)覆盖FC/Lattice/Exchange/Multiscale、K/S/X、三档共享、四种接收、quad/Shannon/pool、Settle/Timed、三种分区，以及反向混合/窄KV/仅记忆共享扩展。Transformer参照为d128×4、d256×6、d384×8，采用RMSNorm、RoPE、SwiGLU、滑动窗口SDPA及真实KV缓存；没有宣称精确参数匹配或完成质量调参。

共同learned embedding、输出RMSNorm与tied LM head。Settle每位置读出；Timed取下一次外部注入前最后tick，连续窗口不插入flush；记录空输出数，不宣称当前token全部传播已经完成。

FP32 AdamW，lr3e-4，10更新线性预热后常数，weight_decay0.01，梯度裁剪1。每次更新detach，4个连续chunk在同一文章内接续，然后换文档块并清空状态。默认B8/L64，另有B16/L128和CPU B2/L32；CPU覆盖8/16/40线程。

NPU训练软预算420秒，CPU240秒；单任务实际工作最长900秒并服从轮次截止。排队单列；最多16个NPU请求、4个CPU任务。软训练截止遇文档块尾时，在总时限内多做一次更新，以非空历史作为恢复锚点。

## 测量与验收

检查因果性、初末固定小样本留出NLL、实际参数更新/None梯度、完整训练步吞吐、准备/Full/消息计数、驻留prefill与带历史decode、应用checkpoint和新解释器下一次AdamW更新。

保存数据、下一批输入及RNG要求精确；logits、梯度、参数、优化器和私有状态使用固定atol1e-5/rtol1e-4，整数/时钟等严格比较。有限超差保留原比较，执行完成不能认证等价；因果、非有限、执行或恢复进程错误使受影响任务失败。

训练计时包括数据采样/传输、CE前向、反向、裁剪、AdamW及同步，排除JSON日志写入；前三更新作为测量预热但仍计token曝光。构造、验证、保存和恢复单列。推理包含LM head，排除tokenizer和输入传输，报告实际样本数。

本轮只记录观察到的共存与吞吐；单独运行和并发降速的配对测量留到下一轮。父/子进程allocator峰值分别报告，不冒充同卡总HBM峰值。小留出样本和短曝光不支持最终模型质量排名。

## 资源和接续

全部长任务在background.slice，固定输入；NPU使用共享队列FIFO、两次空闲检查及锁，CPU按亲和性分配。输出同时受硬配额、软限宽限和文件系统空间限制，预留16GiB，每活动任务另预留2GiB写入，轮次输出上限640GiB仍受实际可用quota约束。

保留每项最新checkpoint；恢复通过后删除大型比较临时包，失败保留参考/实际载荷和日志。准备阶段首次SSD状态序列化失败及修复后的接入均保留。

服务 `tide-pilot-r1-round-20261119.service`；状态 `.local/pilot-r1/round/scheduler.json`。停止用 `systemctl --user stop tide-pilot-r1-round-20261119.service` 并核验cgroup及后代。轮末汇报与用户对齐，后续轮次和正式队列不自动启动。
