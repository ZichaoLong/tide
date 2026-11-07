# 首批预训练起点：Qwen2.5-0.5B-Instruct

本批选择公开的 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)，
固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`。已有完整本地缓存，实验只读使用，
不修改原缓存，也不新增权重下载。缓存名 `qwen05` 不代表 base 模型；README 明确记载它经过预训练和指令后训练。

选择理由是已有材料、公开出处、较小规模和当前栈可直接接入。所检查的 Hugging Face 默认缓存没有小模型权重；
既有实验资产中有这个完整的 0.5B 模型。SmolLM2-135M 会更小，但本机没有检查到对应缓存；
这轮先避免新下载和新的架构适配变量。没有把随机 tiny 测试权重当作可靠预训练模型。

| 项目 | 固定内容 |
| --- | --- |
| 参数 | safetensors 290 个存储张量，494,032,768 个参数，BF16 权重文件 988,097,824 字节 |
| 架构 | Qwen2；24 层，hidden 896，FFN 4864，14 个 Q heads / 2 个 KV heads |
| 词表与位置 | 原词表 151,936，原 tokenizer；RoPE；配置最大长度 32,768，本批仅极短序列 |
| 来源背景 | 官方模型卡：预训练加指令后训练；Apache-2.0；不是从零初始化 |
| 执行 | 私有 Transformers 4.57.1；eager attention；FP32 参数、loss、梯度和 optimizer 槽 |
| 训练范围 | 首批默认全部骨干参数和随机任务分类头可训练，旁路候选再增加旁路参数；不做多层替换 |

实际包版本、安装内容与本次 CPU/NPU 资格以依赖锁和运行检查为准。公共 CANN9.2 基线的随机 Qwen2
小检查只说明接入起点；其中默认 NPU 精确恢复曾失败，显式确定性算法后的后继检查才通过。
这些旧证据不认证本批 checkpoint、数据或完整训练。本批显式启用确定性算法并独立检查。

本次重新计算的文件身份如下；本地 hash 固定所用材料，不额外声称有厂商签名验证：

| 文件 | SHA256 |
| --- | --- |
| model.safetensors | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| config.json | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| tokenizer.json | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| tokenizer_config.json | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| vocab.json | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| merges.txt | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| README.md | `b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15` |
| LICENSE | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |

## 与从零路线共用任务

两线使用同一个延迟关联检索生成器、关联映射 hash 划分及目标值类别。默认 8 个 key、8 个 value、
4 个关联、2 个延迟 token 和最后一个查询，共 7 个 token。一个 `(key,value)` 是一个原子符号，
查询是一个 key 符号。目标类别不在查询后输入。训练、验证、测试按整个关联映射分开；
同一映射的不同排列和查询不会泄漏到另一个集合。

从零路线学习 73 个符号的 embedding。预训练路线把这 73 个符号按固定顺序映射到原 tokenizer
独立编码后互异的 ASCII 单 token，不增词、不重新初始化原 embedding。代码表进入每次运行的 model facts。
这个任意代码表保留 token 数和样本语义，但不是自然语言描述，也不保证匹配原预训练分布。
不能把任务收益解释为语言能力改善，也不能把预训练背景和大骨干计算成本忽略。

## 候选与检查范围

`GrowthClassifier` 具有同一份原骨干和同 seed 初始化的 `Linear(896,8)` 任务头。
`none` 是原骨干同数据曝光、更新次数和 batch 的继续训练对照。`tide` 将完整最后一层 hidden 序列
经 `Linear(896,16)`、tanh 输入小型 FC/KV core-5，取最后查询位置，再通过 `Linear(16,896)`
加回原最后 hidden；`mlp` 用最后 hidden 的简单瓶颈 MLP 作为必要时的架构对照。
三者参数数目和额外计算不同，运行单独记录，不声称同参数或同实际 FLOPs。

输出投影从零开始且可训练。初态任务 logits 和原 LM 最后位置 logits 应严格保持；
首步内部旁路梯度为零是预期，输出投影首步应有梯度，更新后第二步内部梯度应实际进入。
训练后关闭分支是依赖性干预；其影响不单独证明 TIDE 优于重新训练的简单适配器。

[`check_growth.py`](../scripts/check_growth.py) 用实际 checkpoint 分开检查原 LM 的有限 logits、
原生 KV cache 续接、短贪心推理，以及新任务初始中性、每个前缀的续接和两步 SGD 梯度进入。
缓存续接和整段计算沿用基座 external-LM 的 `atol=1e-5, rtol=1e-4`，初态中性要求逐值相等；失败保留原阈值。
旁路的检查续接采用“骨干 KV cache＋重放累计 hidden 前缀”，没有声称联合旁路 streaming 已优化。
TIDE 独立数学根/梯度检查和正式 AdamW 应用恢复由各自接入检查负责。
短程检查不承担任务泛化、完整语言能力保持或架构价值结论；这些须由有验证集的实际研究回答。
