# v6后台任务：查看、固定输入、停止与恢复

观察时点及逐项映射见[交接JSON](../evidence/language-v6-handoff.json)。当前200个科学配置通过116个独立worker接收：96个CPU覆盖worker，加2个覆盖失败修正、2个NPU小Transformer和16条规模轨迹。一个覆盖worker顺序负责至多两个固定配置；不同worker并行。旧18个目录冲突和2个native正支持拒绝是独立失败尝试，计费和记录保留；没有训练曝光冒充新种子。

## 运行与查看

```sh
python scripts/campaign_status.py --brief
python scripts/campaign_status.py
python scripts/job.py status artifacts/jobs/v6-cover-worker-000
python scripts/job.py status artifacts/jobs/v6-scale-fc16-kv-quad-retry1
tail -f runs/v6-cover-fc-anchor/train/metrics.jsonl
tail -f runs/v6-scale-fc16-kv-quad-retry1/train/metrics.jsonl
cat runs/v6-budget-v2/state.json
```

`campaign_status.py`只读。逐项`record`字段给出实际结果路径，`previous_attempts`保留旧尝试。训练目录含manifest、metrics、progress、result、validation-*.json、checkpoint.json、latest.pt/previous.pt与里程碑checkpoint；每个case还保存check、probe-full/split/resumed、recovery及阶段日志。worker退出0不意味着它的所有case通过。

Lens项目`tide-learning-lab`，`study_id`为TL-COVER-v6/TL-SCALE-v6；分别搜索`tl-v6-cover-`、`tl-v6-scale-`。实际显示名带`-projection-`后缀，目标ID/名字在交接JSON及各训练目录projection.json；录制中为recording，结束后再核对verified/readback_matches。私有地址和SSH转发方法只在本机`.local/LENS.md`，服务和访问策略均未改动。检查失败的候选尚无学习曲线，原因在case检查记录。

## 固定身份与预算

- 首64覆盖worker：`.local/source-v6-launch` / `c4585715b3ffe3aa7e7bd88043ad26897ed76df5`。
- 其余32覆盖worker、18个独立作业修正及2个正支持修正：`.local/source-v6-corrected` / `4087310162338e4f4e90102989cd2c9b98db62b9`。目录冲突只改提交输出位置；Shannon显式选Python，不改变正支持语义。
- 实际配置来自各source下的`configs/studies/language/v6/resolved/`；主checkout可继续写审阅文档，活动source不可编辑。
- foundation `74e7bbd62341d2ccd382194a2c52ad4fea033665`；数学 `0770593838a37c790a568226930567ed60f99b83`；环境`.venv-positive-floor-v1`，锁`dependencies/positive-floor-v1/foundation.lock.json`。native为独立同源码/同Torch/ABI构建，manifest与binary hash在交接JSON，每个相关job固定这两个输入。
- 数据`runs/v6-data-final`，锁SHA256 `8977a3636b7111db3ace7b00849e12a0d18da700a9c40feb1dbcdf142b0aa0f6`；制作源码`3786de02ec47f0b86b8f7b80a2c9ce2a8ae02867`。测试密封，训练不访问。
- 总截止**2026-11-11 20:31:31北京时间**。1536 CPU核时/384 NPU卡时/新增120GiB，至少512GiB可用RAM及160GiB账号硬配额余量；最多192 host线程、128 CPU worker、12 NPU worker，累计RSS≤768GiB。原预算监督继续运行，v2增加实际进程RSS监视以适配本机v1 cgroup。人工主动停止预算监督会请求停止本批其余活动服务，不会停止别人的作业。

96覆盖worker是在首64个已运行、实际RSS/内存及CPU余量核对后追加。执行选择保留多个卡上的实测：当前小算子/host控制使TIDE NPU未超过CPU；不是把CPU FP64或严格数值全通过当作NPU学习门槛。两个NPU TIDE轨迹与CPU同模型/seed/全局batch/token日程，其差异属于实现/设备轨迹；其他候选差异属于架构/配方，分别解释。

## 停止与恢复

停止某个worker（它负责的后续case也随之停止）：

```sh
python scripts/job.py stop artifacts/jobs/v6-cover-worker-000
python scripts/job.py status artifacts/jobs/v6-cover-worker-000
```

整体停止使用 `systemctl --user stop tide-lab-v6-budget-v2.service`，随后逐项核对`campaign_status.py`和job状态/实际cgroup无后代。不要删除或编辑活动输入。每600秒在完整optimizer边界发布rolling checkpoint；单步很长或强制停止可能丢失未发布的进度，恢复只从已发布边界。

恢复必须创建**新job和新输出目录**，保持原source、配置、数据、安装、后端/线程和native身份，不延长共享期限/累计预算。在对应冻结source中，应用命令是：

```sh
python scripts/campaign_stage.py train   --config ORIGINAL_FROZEN_CONFIG   --data /absolute/path/to/runs/v6-data-final   --admission ORIGINAL_CASE/check/result.json   --recovery ORIGINAL_CASE/recovery.json   --resume ORIGINAL_CASE/train/latest.pt   --output NEW_OUTPUT   --deadline-epoch 1794400291.0601735
```

将该argv交给现有`scripts/job.py submit`，复用原job.json的source/env/固定依赖及NPU分配方式，timeout取剩余上限。普通`--resume`恢复模型、optimizer、样本游标、token日程进度及CPU/Python/NPU RNG；拒绝不同身份与微型probe checkpoint。三步同栈新进程恢复已逐项检查，完整长轨迹独立重放仍是未检查。跨后端或改配方不是精确恢复。恢复输出保留同一科学run ID，若再投影Lens，需用独立投影身份关联原run，避免覆盖已有录制；当前不自动恢复或延长任何轨迹。

初始validation只用2048目标；覆盖/规模的过程与终点评价分别固定8192/32768目标。每次实际tokens及文档记录均保存，首点不是与终点完全同集的损失基线；后续科学审阅需按共同集合重评。

## 交接边界

已完成的小接入/控制作业不是本批科学结论。先按实际共同曝光审阅验证曲线、独立检查、恢复、活动/Full/消息和内存，再冻结最终比较与test评价；不按作业退出码自动增加支持边。历史B预训练承接保留，不新增多层Qwen替换。本批由用户验收后再选择下一轮。
