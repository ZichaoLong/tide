# 优先清理与公开模型整理结果

**后续变更：用户已仅把huggingface移至`/mi/data2T/models/huggingface`。六个旧别名已更新，[当前路径清单](../../dependencies/public-model-paths.json)。以下为原整理时点记录；原sudo命令已停用，无需再次迁移。**

2026-11-11，用户明确授权执行上一轮优先候选，并保留、整理全部发现的公开模型。
**已删除610个文件，原分配空间46.185GiB；六个公开原模型全部保留并整理完成，约93.553GiB。**
用户sudo迁入共享目录尚未执行；本轮没有新训练、基座安装变更或Lens清理。

## 删除与保护

| 已清理组 | 文件数 | 原分配 GiB |
| --- | ---: | ---: |
| `tide-completed-recovery-probes` | 603 | 17.241 |
| `tide-rebuildable-b-features` | 3 | 2.125 |
| `gef-terminal-qwen05-benchmark-checkpoints` | 2 | 11.043 |
| `gef-rebuildable-bf16-gguf` | 2 | 15.775 |


[逐文件退役凭据](../evidence/artifact-retirement-20261111.json)记录审计SHA256、每个删除文件的实际SHA256、
inode、大小、原结果/恢复凭据及删除时间。全部文件在删除前完成hash，全部身份先持久化，再逐一删除。
两次检查294个原unit及2360条PID/start-tick记录均无匹配存活；也拒绝新出现的相关活动服务。
原[只读审计](disk-cleanup-audit-20261111.md)保持历史，不将当时的候选文字冒充执行结果。

[保留/整理复核](../evidence/housekeeping-retention-20261111.json)：610个路径确实消失；
67323个原保留文件的inode/device/size/mtime在删除及模型搬迁后全部一致，11个显式保护输入SHA256再次匹配。
Qwen05基线、三项真实下溢失败及其probe/诊断、科学latest/final/里程碑、数据、配置、指标和原生命周期记录保留。
200份previous.pt、原性能trace和其他延后候选未删除。Lens未触及。
研究图没有直接指向已退役字节的链接；科学结论和检查阈值不因清理而改变。

退役的probe/benchmark checkpoint字节已经不可用；原恢复通过记录保留，但即时重新加载需重建或重新运行。
三份B feature缓存与两个GGUF可按原模型/数据/配置/版本重建，需新运行身份，不保证新字节与旧文件完全相同。

## 公开模型集合

本账户home中发现的六个原始HF checkpoint全部保留；另两个可重建BF16 GGUF属于已授权清理项。
扫描未发现额外公开权重；`~/.cache/huggingface/hub`只有数据缓存。
既有`/mi/data2T/models`其他内容不属于本次整理范围。

| 模型仓库 | 固定revision | 权重分片 | 全目录文件 GiB |
| --- | --- | ---: | ---: |
| `allenai/OLMoE-1B-7B-0125-Instruct` | `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` | 3 | 12.892 |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | 1 | 0.931 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1 | 2.886 |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 2 | 5.759 |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 4 | 14.196 |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` | 16 | 56.889 |


整理路径：`/mi/data2T/zlong/home/llm/public-models-staging-20261111/huggingface/<org>/<model>/<revision>/`。
[整理凭据](../evidence/public-model-staging-20261111.json)；集合根另附`README.md`、`CATALOG.json`、
`UPSTREAM-CATALOG.json`与`SHA256SUMS`。完整模型、tokenizer、配置、index和license一起保留。
27个权重分片全部匹配固定来源SHA256；84个来源文件按catalog的SHA256/Git blob与长度验证；
全部102个原文件均记录自身SHA256。旧六目录通过链接访问相同inode，没有复制权重。
从旧路径解析config、tokenizer、所有权重header；存在index的模型，index与header键/分片逐一对应。
这是文件身份和包完整性检查，不是追加模型推理资格或新科学结果。

## 用户sudo命令

先审阅[发布脚本](../../scripts/publish_public_models_20261111.sh)，再执行：

```bash
cd /mi/data2T/zlong/home/llm/tide-learning-lab
bash scripts/publish_public_models_20261111.sh --check
sudo bash scripts/publish_public_models_20261111.sh --publish
```

脚本只发布这一集合到`/mi/data2T/models/public-checkpoints-20261111`：
同文件系统rename、旧集合路径补链接、目标root所有及公共只读权限。拒绝覆盖/合并已有其他目标，
支持rename后中断的恢复和完成后的幂等重跑。原六个模型路径通过两层链接继续工作。
发布前检查使用保存的hash与当前inode/size/mtime；本轮全量hash已在整理作业中完成。
**真正转出zlong配额的是root所有权变更，单纯移动到共享路径没有这个效果。**
当前账号约245.744/800GiB，硬余量约554.256GiB；发布并chown后预计约152.19GiB，最终以`quota -s`为准。
本机原目录当前约：实验分支19.653GiB、gef-lm-experiments2.214GiB、模型集合93.553GiB；不跟随链接计数。

## 固定执行与检查

源码`c8285e5d090477653bb09120745fcde2ada4ce4b`，冻结于`.local/source-housekeeping-20261111`。
两个作业顺序执行，各2 CPU线程、2GiB RAM、3600秒上限；实际都在background.slice，无NPU分配。

| 作业 | 状态 | 实际wall秒 |
| --- | --- | ---: |
| `retire-priority-20261111` | completed；无存活后代 | 56.93 |
| `stage-public-models-20261111` | completed；无存活后代 | 120.24 |


状态/日志：`python scripts/job.py status artifacts/jobs/<作业ID>`与`python scripts/job.py logs artifacts/jobs/<作业ID>`。
停止入口仍为`python scripts/job.py stop artifacts/jobs/<作业ID>`；当前两者已经终态，不需要停止或恢复。
删除不是训练恢复入口，不能重复执行退役清单；中途失败应根据退役凭据按剩余项另建身份，不覆盖原报告。
模型发布可使用上面的幂等命令。已检查路径越界、symlink/变更/hash不符拒绝、inode保持和目标碰撞，4项测试通过。
发布脚本语法及当前集合的只读preflight通过；sudo实际动作留给用户，未声称已执行。

数学0770593与已选基座74e7bbd不变；TL-REQ-005已提交，待基座交付及下游验收。
v6科学终态审阅和密封test仍是独立待办，清理不会把执行完成升级为质量、scaling或严格数值通过。
