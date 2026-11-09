# 当前交接 — TL-REQ-004 实现提交，准备固定资格与性能（2026-11-09）

用户授权的显式`query_execution=dense-v1`已完成首个实现增量。数学固定0770593，
分支graph-execution-foundation；默认legacy及原严格阈值不变。接受范围见[ROADMAP](ROADMAP.md)，
代码/域/独立根契约见[weighted-dense-query](weighted-dense-query.md)。下游来源0439efd只读，
两个用户未跟踪handoff文件保留原字节、不纳入提交；不更新下游依赖或运行。

## 已观察的开发结果

- gef-tl-req-004-build-a：新C++ native构建，CPU FP64新路径8项与相关默认回归192项通过。
- gef-tl-req-004-check-b：CPU27项中的旧版本26项：24 passed/2 FP32数值xfail；NPU14项通过。
  FC/HB/反馈PDG三步AdamW、完整记录/根、pending/保留cut和新owner精确恢复已覆盖。
  native公共消费者CPU/NPU独立三步严格通过，新进程3步vs2+1逐值恢复通过。
- 后续主工作区导航/配置拒绝/FP64方向差分检查：45 passed/2已知数值xfail。
- CPU FP32小见证的evidence_weight/bias约1.044e-6严格失败原样保留；其他字段、None/连接零通过。
  这是密集GEMM次序的具名数值路径，不宣称全部FP32严格等价。NPU该局部见证通过不消除CPU失败。

以上两个开发服务均终态passed、MainPID=0、ControlGroup为空；日志分别在
artifacts/jobs/tl-req-004-{build-a,check-b}/task.log。dev-a/dev-b是冻结脏源码，
build-a构建来源保留其实际身份，不冒称固定提交资格。没有活动作业。

## 下一动作（本轮继续，不在提交后停机）

1. 提交本实现增量；以该提交创建冻结source-c，新建wheel/私有venv和build-c。
2. `bash scripts/qualify_dense_query.sh NEW_OUTPUT cpu|npu BUILD`：安装包的定向完整资格，
   NPU按单卡队列maxwait600s；CPU FP64/FP32、NPU FP32，数值失败和精确恢复分别记录。
3. `scripts/benchmark_dense_query.py`从下游固定Git对象导出任务到新输出，
   串行测CPU/native与NPU/native的legacy/dense-v1各B8/L128一预热一步；
   另以B2/L8分别作同材料host算子与NPU设备profile，profile与正式计时分开。
   不改任务头/loss/精度/完整返回，报告实际candidate/Full/messages及完整墙钟。
4. 审阅实测收益/剩余瓶颈，同步契约、经验和证据；独立证据提交后普通推送。

长作业独立background.slice，构建全部affinity CPU、MemoryMax512G；测试/计时MemoryMax96G，
每服务≤3600s。本轮产物预算10GiB，现有约201GiB/硬800GiB，系统空间约1.1TiB。
固定环境ascend/dev-workspace-9.2.0；不改公共工具链。
下次接续先`python scripts/status.py`，运行前将精确unit/源码/日志写回本页。
