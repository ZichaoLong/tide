# 首批启动审计

启动实际HEAD为`942c329`且干净；未回退后续提交、未覆盖用户修改。用户2026-11-07指令明确替代旧“仅图形原型”限制。实现已固定并普通推送为`0d7ec5bdaa34556df9c93886e95f55fa5c9876ab`；后台只读`.local/source-first-a`，内容树SHA256为`476b748a778741c9f99ecbfba11aff1507538e4d6a59f9abb7ad7439fd1c0c77`。后续检查脚本/配置作为独立固定输入另记hash，不修改活动源。

基座只读`85d161d5c9148eac2b164f89a0ba4204b4331898`，数学固定`0770593838a37c790a568226930567ed60f99b83`。已读该提交downstream-contract、公共消费者接入示例、当前STATUS及严格失败；没有导入基座benchmark或复制执行核心。公共eager Python/packed/单worker是本轮实际客户端，native/resident/多卡不在本批。

## 安装与材料身份

从固定git archive在实验目录构建`tidegraph 0.3.0` wheel，安装至私有`.venv`，未使用基座editable环境。`lock verify`确认实际安装与wheel一致；[来源锁](../dependencies/foundation.lock.json)、[安装锁](../dependencies/installation.lock.json)、[环境身份](../dependencies/environment.lock.json)、[下游安装](../dependencies/consumer.lock.json)分别保存材料身份。公共CANN9.2.0 r1、Python3.12.13、Torch2.12.0+cpu、TorchNPU2.12.0.post3；私有Transformers4.57.1；pip check无冲突。

使用缓存[Qwen2.5-0.5B-Instruct](checkpoint-selection.md)，494,032,768骨干参数，保持只读并逐文件hash核对，无新权重/数据下载。任务为本项目确定性生成的关联检索；全部查询/顺序变体按关联映射散列进入互斥划分。validation由同一完整样本集合分块，batch改变不改变验证集合。最终test不参与先导筛选。

## 实际检查与保留失败

完整可审计记录见[startup-checks.json](evidence/startup-checks.json)，该文件保留各原记录hash。检查范围均为列出的微型配置，不能外推所有语义、dtype或规模。

| 检查 | 实际结果与边界 |
| --- | --- |
| 安装CPU基础related | 通过；这不是原CANN综合矩阵的整体解除 |
| 下游FC CPU FP64 | 独立完整前向/状态/路由、输入与参数梯度、共享参数求和、两步AdamW、因果、prefill/streaming/分段续接通过 |
| CPU完整应用恢复 | 全模型、optimizer、步号、CPU/Python RNG和episode边界状态，新进程逐值相同 |
| CPU FP32 FC/HB | 保留稠密AdamW参数更新严格失败；eps1e-8原尝试和显式eps1e-6分别记录。后者小型quad稀疏检查通过，不替代dense资格 |
| NPU基础related | 复现原综合根`input.0.1.0.1`坐标4差2.861022949e-6，原atol1e-6/rtol1e-5失败保留 |
| NPU下游FC FP32 | dense AdamW参数最大差5.334615707e-6，后继descriptor亦有有限超差；共享梯度、因果/streaming及对应输出/loss/梯度检查通过，整体严格失败 |
| 原checkpoint CPU | 旁路初态任务与LM logits逐值保持；原LM全序列/cache最大差4.386901855e-5，既有atol1e-5/rtol1e-4严格失败 |
| 原checkpoint NPU | 初态任务与LM logits逐值保持；全序列/cache与更新后续接通过；两步SGD检查首步内层梯度0、输出投影梯度非0，次步内层梯度0.00285665；关闭分支logits最大改变0.00226879 |
| NPU完整应用恢复 | Qwen+TIDE模型、SGD、数据步号、CPU/Python/NPU RNG和源身份新进程逐值相同；大检查点经比较后按预声明策略删除，hash/日志/结果保留 |
| CPU显式有限探索 | 另一个检查输入允许有限cache/prefix超差后继续梯度与续接观察；保留strict_numeric=failed。非有限、初态不等和梯度不进入仍硬失败；不是原严格检查的修复 |

[TL-REQ-001](requests/TL-REQ-001.md)记录下游浮点差异需求；未改基座或向外发布issue。A研究选CPU FP64；B的NPU任务按有限数值探索记录，不能称为与独立CPU根严格等价。模型学习效果、执行完成、独立数值/离散检查、恢复分别解释。

## 先导资源与选择

A校准同一FC宽16、8节点/2区域、68,722可训练参数。两步训练（同步、前向/反向/clip/optimizer，不含验证/序列构造/checkpoint）墙钟：B4为1.142/0.967秒，B16为3.015/3.233秒，B32为5.922/5.674秒。实际max_full_batch分别28/112/224，包含样本和时刻，不把它当独立样本数。B32单位样本更快，内存余裕，故选B32，80updates，独立validation128。此校准不是CPU/NPU加速比较，首步含启动效应。

后台入口已实际验证`background.slice`、CPU/RAM/时间限制、内层退出记录和最终无后代。队列双采样后NPU审计取得物理1→逻辑0；这是一次观测，不写入模型代码或后续固定选卡。A目前仅CPU，B仅单NPU。账号约165GiB，软300/硬800GiB；首批新增上限20GiB，使用共享只读模型缓存与环境，避免重复大材料。

Lens采用安装实例的独立`record`进程，训练写权威JSONL。原因是Transformers4.57.1与Trackio0.44.0的huggingface-hub要求冲突；没有强装或升级公共包。单事件实际进入`tide-learning-lab`，读回一致、`verified`、SDK0.44.0；实际先导逐run也单独读回。`projection.json`连接项目run ID与Lens目的run ID。私有目的地/访问方式保存在`.local/LENS.md`，公开仓库只含项目名和运行身份。

实际宽16/8键8值/4关联/延迟2的B2 CPU64完整独立检查补验通过；正式训练使用FP32初始化后转FP64执行（固定源码可复现），不把小检查当每步正式轨迹认证。CPU/NPU新进程恢复分别覆盖实际声明的完整应用状态。

B校准B8两步5.603/5.470秒，B32两步20.252/19.502秒；allocated峰值约3.8GiB。选B32因单位样本吞吐略高，数据/step预算三候选一致。小TIDE在NPU路径仍有大量主机调度和小算子开销；这不是稀疏加速证据，也没有根据逻辑batch推断高效矩阵计算。

A先导已有学习与历史利用信号，因此按原表进入三seed400步确认；未达到50% HB门槛，HB继续待验。没有启用小任务降阶，未改变原研究问题。
