# 当前交接 — 联合prefill反向装配与代表回归（2026-11-10）

授权分支graph-execution-foundation；数学固定0770593838a37c790a568226930567ed60f99b83。
用户批准持续实施/推送；全部320 CPU及空闲/有余量共享NPU可使用，以尽快得结果为准。
数值超差有限诊断后分列推进，不放宽阈值；参考/下游只读，不使用子代理。
两份未跟踪用户handoff原样保留，不纳入提交。

已交付分量窗口、联合KV准备和批量守卫；整序列VJP实现413faf6的固定资格见
[evidence/kv-sequence-vjp-20261110.md](evidence/kv-sequence-vjp-20261110.md)：
CPU143通过、NPU49通过/2原GDN-HB数值xfail；四个安装消费者训练/新进程恢复通过。
有界FP64见证支持该NPU差异为舍入，不阻塞后继；FC16最小返回CPU25.407/NPU36.841秒。
完整返回另列32.564/57.795秒，原任务数值/离散/学习未认证，设备间活动数不同。

活动：gef-joint-prefill-roots-a，413faf6+roots-a-snapshot冻结overlay，
源码artifacts/joint-prefill/source-roots-a，Python build-roots-a与独立build-roots-sdk，
日志artifacts/jobs/joint-prefill-roots-a/task.log。320核/512G/7200秒background.slice。
构建及SDK三项CTest通过；SDK CPU FP64完整/最小返回的三步/新进程恢复通过，
Python/native定向检查仍运行，不能把外层running写成通过。
检查/停止：systemctl --user status/stop gef-joint-prefill-roots-a。

工作区改动：query完整连续view分区伴随合并、公共CLI输出任务省去内部trace、
实际工作计数和按实际选定路径精确恢复、代表配置生成器；尚未提交。
下一步审阅roots开发终态→提交实现→固定安装CPU/NPU/SDK定向资格及FC16有限成本，
再跑大宽度FC、多节点HB/LH、sum、循环层链B×L代表，CPU/mixed/resident/多卡分列。
新长任务固定源码、独立服务有界资源，构建不与重型计时重叠；不无限排队。

下一命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
唯一待办与停止条件见[ROADMAP](ROADMAP.md)，GPU按用户授权延后。
