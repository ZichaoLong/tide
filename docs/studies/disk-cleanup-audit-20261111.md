# 两个实验目录的磁盘清理只读审计

2026-11-11。用户本轮仅授权清理前只读调查；**没有删除、移动、去重、压缩或改写任何原实验产物，未清理Trackio Lens。**
新材料仅是本报告及[路径级清单](../evidence/disk-audit-20261111.json)。需求提交另见[TL-REQ-005](../requests/TL-REQ-005.md)。

## 占用与口径

| 范围 | 实际分配空间 |
| --- | ---: |
| `tide-learning-lab`工作目录 | 38.985 GiB |
| `gef-lm-experiments` | 122.585 GiB |
| 账户当前使用 / 软配额 / 硬配额 | 291.888 / 300 / 800 GiB |
| 账户硬配额余量 | 508.112 GiB |
| 所在文件系统可用 | 899.284 GiB |

使用`du -x -B1`与lstat分配块，文件清单保留size、mtime_ns、link count；不跟随符号链接。
目录总量包含目录块，不与仅普通文件的合计混算。共享Git对象库/公共环境及符号链接目标不在清理范围。
未读入大型tensor或反复hash约百GiB权重；已有hash明确标为原记录引用，不冒称本次重新验证字节。
下面是候选的当前分配空间，不是已回收量或未来文件系统必然释放量。

## 候选分组

| 分组 | 文件数 | 当前空间 |
| --- | ---: | ---: |
| `tide-completed-recovery-probes` | 603 | 17.241 GiB |
| `tide-rebuildable-b-features` | 3 | 2.125 GiB |
| `gef-terminal-qwen05-benchmark-checkpoints` | 2 | 11.043 GiB |
| `gef-rebuildable-bf16-gguf` | 2 | 15.775 GiB |
| `gef-unused-public-large-weight-shards` | 26 | 92.570 GiB |
| `tide-previous-training-checkpoints` | 200 | 4.136 GiB |

优先四组共 **46.185 GiB**；若后续也退役五个旧公开模型的权重shard，
再增加 **92.570 GiB**，两者不重叠、合计
**138.755 GiB**。
所有条目目前只是候选，不是删除授权或执行结果。

### 优先四组的依据与代价

1. **603个v6恢复检查checkpoint，17.241 GiB。** 每项都有`recovery.json: passed`与各probe结果/来源。
   `campaign_stage.py`明确拒绝把`probe=True`的checkpoint作为正式训练祖先；正式训练另起或明确恢复。
   这批检查的状态、配置、hash与结果应保留，删除checkpoint会失去立即重新加载旧字节的能力，重建需新身份。
   三项未解决下溢案例的9个probe额外保留（约0.189 GiB），没有列入603个候选。
2. **3份B语言feature缓存，2.125 GiB。** `lm-b-language-v2/cache-L32`、`lm-b-language-v4/cache-confirmation`、
   `lm-b-fc8-v5/cache`下的`features.pt`。模型、数据、tokenizer、提取源码及manifest保留后可重建。
   旧作业记录仍引用这些路径，退役凭据应明确字节已不可用；不将历史引用自动解释成活动作业依赖。
3. **旧GEF两个Qwen05训练benchmark checkpoint，11.043 GiB。** CPU/NPU各一个，两个result均completed，
   终态/恢复小记录仍在。它们不是当前B路线的公开原模型权重；后者独立保留。
   清理前保存checkpoint身份/hash、result/resume/progress/resolved及必要失败信息；执行完成不升级为独立正确性通过。
4. **两个BF16 GGUF转换文件，15.775 GiB。** OLMoE和Qwen1.5B的`model-bf16.gguf`；固定模型与转换入口可重建。
   保留转换来源、版本和日志；若之后连原safetensors一起退役，重新转换还需要下载相应固定权重。

### 可选的大模型权重退役

仅列26个safetensors shard；保留模型配置、tokenizer、index、license、asset-state和固定下载revision。

| 模型标识 | 固定revision | 权重空间 |
| --- | --- | ---: |
| olmoe | `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` | 12.888 GiB |
| qwen15 | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 2.875 GiB |
| qwen3 | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 5.748 GiB |
| qwen7 | `a09a35458c702b33eeacc393d103063234e8bc28` | 14.185 GiB |
| qwenmoe | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` | 56.873 GiB |

没有发现当前`tide-learning-lab`作业把这些五个模型作为asset输入。旧GEF历史报告仍引用它们，
历史身份及下载/验证记录必须保留；本次未检查未来其他项目的所有未登记使用意图。
它们是可重新获取的公开权重，但当时的下载曾有重试，退役后的网络/下载成本是真实取舍。

## 保护与暂缓

- **跨目录Qwen05基线必须保留。** `gef-lm-experiments/20261102/models/qwen05-7ae557604adf67be50417f59c2c2f167def9a775`
  约0.931 GiB，被本实验33份作业的`--asset-path`直接引用，支持B路线及旧缓存重建。
  不能整体删除`gef-lm-experiments`或整个models目录。
- **TL-REQ-005证据链保留。** 三项原失败运行、被诊断明确引用的latest.pt、四个`v6-positive-*`、
  实际failure-boundary/input、数据、固定源码/安装/native及原始JSON/日志。完整路径和输入hash在清单`protected`。
- **科学终态审阅尚未完成。** 当前训练latest、token里程碑、最终模型和数据/tokenizer均保留；
  200份非失败案例的previous.pt约4.136 GiB暂缓，逐一确认恢复/共同曝光分析用途后再决定。
- **性能原始trace暂缓。** 四个大诊断目录约2.413 GiB；
  这只是目录占用上界，不能整目录删除，里面还有结果/配置/摘要。待TL-REQ-004复核需求明确后可选择性退役原始trace。
- 环境、构建和源码快照较小，当前优先保留可复现入口；不追求为了多省几百MiB扩大破坏面。
- 本轮未调查或清理私有Lens存储，不因项目记录可退役就自动删除其投影。

## 活动核对及限制

`tide-learning-lab`247份job终态：204 completed、39 failed、3 timed_out、1 stopped；
其中v6训练200配置为184 completed、11 failed、5 budget_truncated，当前training/checking均0。
`gef-lm-experiments`47份job终态：33 passed、12 failed、2 cancelled；最后记录结束于2026-11-05。
外层passed不等于内部严格数值或模型学习通过。

294个相关systemd单元当前均无active/MainPID；57个仍loaded，237个已被回收为not-found。
另按PID/start-tick核对本项目2266条、旧GEF94条已记录进程身份，无匹配存活项（条目可重复，不是2360个独立进程）。
只读/proc检查未见相关训练，观察到目录中的交互shell；部分进程的fd/maps读取受权限限制。
因此这是一份带边界的调查，正式删除前仍需重新核对活动、文件身份和外部引用。
部分旧内层state仍写running，只保留历史事实，本次没有为清理而改写这些记录。

## 后续实际清理的最小要求

本轮不执行。若后续授权，按清单逐文件限定范围，复查新活动和mtime/size，补齐必要hash/小记录后原子写退役凭据，
再删除明确候选；不跟随symlink、不删除共享对象或只凭同文件名认定重复。保护项优先于任何通配模式。
清理后核对权威结果/失败/图链接、必要复现输入与账户quota，分别报告实际释放量和仍保留的候选。
