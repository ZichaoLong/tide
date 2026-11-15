# v12操作、固定身份与记录

研究表：[HB机制对照](hb-mechanisms-v12.md)。19配置×seed11/22，全部目标128Mi，
没有基于短期排名的淘汰或自动256Mi延伸。统一截止2026-11-17 23:08:07+08。
共享12 NPU slot，最多200M参数；新增64GiB目标/96GiB停止，硬配额余量至少200GiB。

校准源码`8e085be3`，`.local/source-hb-v12-cal`；主源码由
`artifacts/jobs/v12-main-<配置>/job.json`固定commit/tree/config/data/env身份，
源码目录`.local/source-hb-v12-main`。不得编辑任何这些快照或活动配置。
训练继续用`.venv-positive-floor-v1`及Torch2.12/TorchNPU2.12/CANN9.2，
基座74e7bbd、数学0770593，当前独立直接模型不借用基座执行核心。

8项实际NPU校准全部完成，global64固定；物理batch实测选择：d64/R64为16，
R28为32，独立d128为4，共享d128为8，共享d192为4。
更大的batch不总是更快；d192的batch8和独立d128的batch8实际更慢，
超出峰值预测的档不尝试OOM。实测约1.8K–12.9K目标/s；保守规划10–12个worker约36–42h，
包含1.35系数和一小时额外开销，不是无竞争吞吐保证。数值、恢复、数据准备时间另有原记录。

8校准Lens投影全部verified/readback_matches，项目`tide-learning-lab`。
训练环境没有Trackio，沿用已安装独立JSONL记录器；不安装或修改公共环境。
数据首次因训练环境缺tokenizers失败（`v12-data`）；显式`v12-data-r2`用原私有`.venv`
完成262144目标/文档排除。原失败保留；数据在`runs/v12-data`，原v11数据未改。

## 运行、日志与Lens

统一配置表：`configs/studies/language/hb-v12/table.json`，逐配置`configs/`目录。
每条服务`v12-main-<variant>-s11`或`s22`独立取得设备；actual mapping见queue-state.json，
设备不是模型代码配置。每条依次check→小型恢复→8Mi pilot→完整batch受训恢复→128Mi→新确认→诊断。
所有阶段都在background.slice的同一服务控制组内。失败停止对应链，其余独立继续。

- 状态：`python scripts/review_hb_v12.py`；只读，不启动模型。
- 单作业：`python scripts/job.py status artifacts/jobs/v12-main-base-q3-s11`。
- 阶段日志：`runs/v12-main-base-q3-s11/check.log`、`pilot.log`、`train.log`；外层`artifacts/jobs/<JOB>/job.log`。
- 进度：`runs/<JOB>/<pilot或train>/progress.json`；终态`result.json`，不能把进度等同可恢复checkpoint。
- 停止：`python scripts/job.py stop artifacts/jobs/<JOB>`，再用status核对所有后代已退出。
- 硬件：`runs/v12-hardware/observations.jsonl`每60秒按实际设备记录AICore/AIVector/AICPU及HBM；共享负载、短时遗漏明确。

Lens组为`TL-HB-SHARING-v12`、`WIDTH`、`ROUTING`、`INTERACTION`、`READOUT`、
`INFORMATION`、`RECIPE`、`CONTROL`（均同前后缀），校准`TL-HB-CAL-v12`，
诊断`TL-HB-DIAG-v12`、数值`TL-HB-NUMERICAL-v12`、确认`TL-HB-CONFIRM-v12`。
run名`tl-v12-pilot-<配置>`与`tl-v12-main-<配置>`共用配置/seed；
`tl-v12-confirm-<配置>`和`tl-v12-diag-<配置>`关联128Mi终点。
私有查看地址只在本机`.local/LENS.md`，现有服务策略未修改。

20个已训练的合理配方Transformer/MoE及HB稠密/旧低活动控制在
`configs/studies/language/hb-v12/controls.json`，通过`v12-controls`重评相同新确认集。
它们保留原训练源码/配置/曝光，不计算新训练或新增seed。

## 恢复与数值排查

已有main源码的显式恢复使用同config、安装、后端、训练data-lock和source身份，
新job/output调用`campaign_stage.py train --resume <latest.pt> --admission <check/result.json>
--recovery <trained-compare/recovery.json> --until-tokens 134217728`；完整基础启动参数取原job.json。
不覆盖旧run，不把8Mi验证用checkpoint作为科学父轨迹。
只有具名正式latest.pt带Adam可续训；tokens-*.pt是weights-only。
数值/离散/恢复失败先定位，不自动放行或重启；原2e-5可行性阈值不改。

`v12-quad-stress`固定CPU合成压力已完成，原16条件2048行中32行超阈，
减有效评分行最大值后该范围0行超阈；这是有限机制证据，不是任意范围稳定性证明。
`v12-quad-replay`用原159190c3源码、失败seed22原checkpoint/RNG/Adam/游标，
至多1024个内存内更新或1h，捕获失败行，不产生科学训练后代或改变旧失败。

成功恢复比较后的临时权重按receipt退役；正式128Mi完整状态与8Mi科学权重确认存在后，
才退役被其替代的pilot optimizer。失败父checkpoint保留；其他历史记录不清理。
