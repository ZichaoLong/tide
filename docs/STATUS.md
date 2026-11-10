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

开发终态：gef-joint-prefill-roots-a通过，210项Python/native通过/2原数值xfail；
SDK三项CTest与CPU FP64完整/最小返回三步/新进程恢复通过。
新增旧greedy记录解码及冻结配置内部trace兼容修复，80项相关检查通过；
原14项短检查与误写测试文件名的零测试错误不累计成资格。

本提交交付query完整连续view分区伴随合并、公共CLI按需内部记录、实际工作计数与
真实选定执行的精确恢复；执行身份v4，旧冻结配置保留完整内部trace。
下一步在本提交干净worktree安装wheel，按内容hash复用两个已冻结core构建，
重建独立SDK客户端并跑固定CPU/NPU/SDK资格及FC16有限成本；然后有限代表矩阵。
源/构建/测试身份分列，不将开发overlay当固定提交资格。两份用户handoff不提交。
新长任务固定源码、独立background.slice有界资源，构建不与重型计时重叠。

下一命令：git status --short --branch；python scripts/status.py；python scripts/check_upstream.py。
唯一待办与停止条件见[ROADMAP](ROADMAP.md)，GPU按用户授权延后。
