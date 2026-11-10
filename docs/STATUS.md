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

活动已准备：gef-joint-prefill-roots-fixed，固定a553bea/fixed-roots及独立env-roots；
core复用内容hash核对结果roots-fixed-reuse.json，SDK客户端新建roots-fixed-sdk。
320CPU/512G/7200秒background.slice，构建后固定CPU/NPU资格、安装消费者与原FC16两种返回成本。
日志artifacts/jobs/joint-prefill-roots-fixed/task.log；检查/停止systemctl --user status/stop gef-joint-prefill-roots-fixed。
队列最多60秒，已核对共享PID身份/余量，不等待独占硬件；仍记录争用。

已准备gef-joint-prefill-resident-build：等待roots-fixed终态（依赖及避免计时争用，最多3600秒），
随后固定a553bea构建CANN9.2的weighted resident Python库，320CPU/512G/7200秒；
日志artifacts/jobs/joint-prefill-resident-build/task.log。构建不是设备资格，先小真机再扩大对照。

有限代表矩阵已准备：a553bea固定配置/安装，CPU16项/NPU18项（默认与greedy、
宽FC、多节点HB/LH、sum、qry、SSD/GDN层链；另PyTorch/standalone及mixed-a/b）。
B4/P16、一次预热/测量，CPU1线程/NPU2host线程；实际数值与恢复逐项记录。
gef-joint-prefill-representatives等待本项目重型构建/成本终态后运行，最多7200秒等待；
总RuntimeMaxSec14400、512G、320CPU。日志artifacts/jobs/joint-prefill-representatives/task.log。
原配置生成器可复现，不需本轮临时驱动；未执行不记通过。

固定a553bea CPU249通过/2原xfail，安装CPU消费者及SDK通过。
NPU门禁73通过/2原数值xfail/1失败：新增query_roots在设备反向线程丢失观测ContextVar，
不是数值断言；原门禁终态failed保留。有限同材料诊断已完成，合并前后梯度逐位相等，
对FP64最大9.61643e-8；后继补统计上下文传播与该真实NPU回归。
gef-query-root-diagnosis：冻结单脚本hash/query-diagnosis-source.json，实际安装仍a553bea，
60秒准入/600秒总时限/32G/8CPU，不与性能计时混作收益样本。
工作区另有KV前缀伴随K/V按owner一次拼接增量，Python8项通过；C++尚未构建，不纳入a553资格。
