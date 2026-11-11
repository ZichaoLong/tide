# 当前交接 — 张量窗口公共消费者与精确续接（2026-11-11）

授权继续[ROADMAP P1—P6](ROADMAP.md#当前计划批量张量贯穿与架构验证已授权实施)。
分支graph-execution-foundation；固定数学0770593838a37c790a568226930567ed60f99b83，19快照已核验。
当前HEAD b993f38；origin为9453c9e。用户两个未跟踪TL-REQ-004/005交接文件原样保留、不提交。

## 已核验

- 01cde3b/9fa852c张量整链、FC成本和KV位置证据已记录。
- b860e6a native五记忆/卷积：CPU306 passed、13原数值xfail；24个安装消费者和standalone CPU通过，证据9453c9e已推送。
- b993f38公共实验输入/loss整列默认：固定资格133 passed，wheel已安装install-consumer。
  CPU代表成本任务passed；NPU代表任务failed：SSD/GDN/混合LH下一窗参数更新出现逐位尾差。
  原失败保留，不能作精确续接通过。8安装消费者恢复全部通过；独立numerical检查仅Python float loss被错作离散字段，待按原dtype容差离线分列复核。
- NPU followup已终态passed：补齐12层链成本；probe显示普通重复gather反向非逐位确定，
  诊断用全局deterministic=True时重复VJP及SSD/GDN双新owner相等。未采用该全局开关为默认修复。
  原记录artifacts/columns-20261111/consumer-resume-probe.json与jobs/column-consumer-npu-followup。

## 当前开发与下一动作

局部卷积tap gather一阶转置改为按bank行/lag收集后归约，避免碰撞浮点scatter；Python/C++、容量和独立公式检查正在补齐。
Python独立关联矩阵、方向导数、None/零/版本/阶数和循环公式16 passed。
开发快照tap-dev1及gef-column-tap-dev1已passed：all320 CPU、640GiB、1200秒，
Python-owned NPU和独立SDK实际构建，74项CPU相关检查及standalone五记忆通过。
新可重复检查脚本CPU小图两个独立进程精确恢复通过；随后提交固定源码，做NPU合法tap重复VJP
及原SSD/GDN/混合LH失败定向复验。未通过前不宣称修复完成。

后续：提交固定实现后资格/证据分开，整理公共消费者、层链和宽/多节点结果，运行已准备的有限约1B/9B推理。
新列split placement/resident、多设备及异步边界仍按ROADMAP推进；GPU延后。全仓审阅未完成。

## 现场与资源

环境artifacts/column-env.sh；私有env artifacts/additive-work/env；CANN9.2/Torch2.12。
旧固定worktree qualified-consumer，安装install-consumer，构建recurrent-dev2/{build,build-sdk}均在artifacts/columns-20261111/。
当前无旧活跃作业；新作业状态以python scripts/status.py和实际unit为准。
CPU/内存/空闲NPU可用，共享NPU须当前授权；不与重构建重叠正式计时。
配额226G/soft300G/hard800G，磁盘余1.1T，shm余398G，主机可用1.8TiB，没有存储阻塞。

修复固定源码8c72ca94；gef-column-tap-qualify（CPU80–111/64GiB/600秒）正在固定资格，
worktree qualified-taps，hash核验复用tap-dev1两类构建，见tap-build-reuse.json。
后继gef-column-tap-npu（空闲一卡/host0–15/128GiB/2400秒），六个原规模/客户端新进程恢复，
四个安装GDN/SSD消费者独立三步AdamW/恢复。配置生成首次误用了系统Python3.9已记录并用私有环境修正。

8c72ca94固定74项通过；gef-column-tap-npu终态passed：六个原代表/客户端各两个新进程精确恢复、
四个安装GDN/SSD独立三步AdamW/恢复全部通过。六个重复性检查deterministic=false；四安装例沿用true，证据分列。原b993失败保持。
准备gef-column-consumer-scale：固定b993f38/install-consumer，CPU64与NPU2依次计时约1B B32/L16、9B B4/L16推理；
用taskset实际限制CPU80–143，512GiB/3600秒，空闲一卡最多等待120秒。单点720秒，停机条件由公共matrix执行。
临时大checkpoint在/dev/shm/gef-column-scale-20261111，容量400GiB且计入主机预算；这是volatile成本产物，
小型结果由公共results archive写入durable artifacts，不能冒称重启可续接。脚本consumer-scale.sh，日志jobs/column-consumer-scale。

规模四点全部completed，精确恢复未测，archive已持久化。当前mixed-a/b列迁移正在开发：
Read/区域control/selection按能力整列复制，保留raw owners、共享VJP和原始时间；Python10项定向通过。
准备gef-column-placement-dev1，冻结placement-dev1/source；all320 CPU/640GiB/1200秒，
Python-owned NPU及SDK重建、相关CPU测试、standalone独立分调度/评分检查。日志jobs/column-placement-dev1。

新放置开发首轮207 passed/4原数值xfail及SDK通过。已把新检查统一为既有FP32容差（18项复验通过），
正在补standalone实际设备入口与安装示例显式deterministic策略；需固定新实现再资格。
