# 当前交接 — LH节点/区域合批、REQ-006诊断与resident重写（2026-11-12）

用户已授权继续完成[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
本轮起点a8a4ee028a5d79696d49bc47242a29c4c39aee4d；当前HEAD bab9408e；固定数学0770593838a37c790a568226930567ed60f99b83。
19份教材快照通过。下游及参考仓库只读；三份用户新增TL-REQ-004/005/006 handoff保留原样、不纳入提交。

## 已终态与证据

- 129bbbdc aligned列：native CPU99通过/9原数值xfail；half15通过及6安装训练/新进程恢复通过。FC16 CPU/NPU完整步与范围见[aligned证据](evidence/aligned-columns-20261112.md)。
- c83a2820 LH三层：CPU/NPU有限矩阵完成，代表176节点/d64/B4/L16、61输入位置。完整训练NPU直接0.6075s、公共native7.1231s、默认native7.4958s；数学/观察/实际Full工作差异见[LH证据](evidence/lh-three-layer-20261112.md)。
- c83a2820 REQ006：四measure成功、四trace已导出；原profile任务因CANN字符串时间戳失败保留。2927ba40离线恢复分析成功，无需重跑设备profile。报告待集成。
- 78fc90b4节点prepare_many：gef-rewrite-groups构建成功，但测试文件名错误使外层failed；后继gef-rewrite-groups-check CPU106通过/13原严格数值xfail，原失败不改写。
- gef-rewrite-groups-device因180秒内无空闲NPU准入超时，未开始设备检查；原state=failed保留，不作设备数值失败或通过。wheel已构建/安装，记录artifacts/jobs/rewrite-groups-device/。
- 上述unit已终止、无运行后代；原始结果artifacts/rewrite-20261112/，生命周期artifacts/jobs/rewrite-*/。

## 当前动作

区域按实际候选数分组、真实成员int64索引、独立None/零及分支VJP已写Python/C++，尚未固定资格。
开发42项Python检查通过；4个改动C++翻译单元语法通过，准备提交并冻结新native/SDK构建。
修正了含空节点的对齐发布不得丢失空列；默认Settle按新区域布局执行，FC相同成员保留直接张量路径。

resident新独立任务块调度和KV Read/Describe/区域数值组合已写，17项CPU元数据/调度检查通过、5个数值TU语法通过；
尚未完成设备构建/数值真机资格或完整图owner，不能继承旧WeightedFlow资格。契约docs/resident-block-execution.md。
REQ006已将HB88 352次慢转换定位到浮点clamp_min标量反向；正准备同dtype Tensor阈值独立算子见证。
三项分边界开发，主代理统一集成/提交/资源调度，不存在活跃长任务。

## 下一步与资源

1. 固定区域实现，独立background.slice新构建并运行native各发送/独立根、默认端口及恢复检查。
2. 补resident数值组合CMake，固定SDK/设备源码构建，CPU独立公式与设备资格分列。
3. 空闲卡准入后检查新节点/区域/half/安装恢复和有限FC/LH回归；无空闲卡时继续CPU工作。
4. 整理REQ006归因及算子诊断，记录采用/暂缓；同步ROADMAP、语义/架构/复现及性能经验。

当前16张NPU均有进程，共享NPU未获本轮授权；不绕过准入、不停止其他任务。
磁盘配额286GiB/软300GiB/硬800GiB；主存可用约1.6TiB、shm余308GiB。源码/build优先有界shm，持久记录留磁盘。
CPU可用320，长构建使用jobs all或按实际并行资源分池，正式计时避开重型构建。
普通异步跨NPU反向EE1017、旧resident FC16 EE1019/507002均保持原失败；新原语不自动解除。

```sh
python scripts/status.py
git diff --stat
cat artifacts/jobs/rewrite-groups-check/status.json
cat artifacts/jobs/rewrite-groups-device/status.json
```

当前状态只在本页；未完成范围、停止及重启条件只在ROADMAP。实现、严格数值、恢复和性能分别报告。
