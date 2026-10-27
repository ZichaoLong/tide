# Core-5 新进程启动与分阶段内存补充 — 2026-10-28

固定消费者源码 `f72a6cc2f78befdcda1f93a9ed00d774a54030f7`，固定上游 `0770593838a37c790a568226930567ed60f99b83`。
[观测契约](../weighted-performance.md#启动与内存观测)规定边界；[同名JSON](core5-weighted-observation-20261028.json)
保存完整四阶段、请求/生效配置、时间/工作量、输入与所有source/binary/loader/log/result身份。
本报告补[ceae7bf有限性能](core5-weighted-performance-20261028.md)的观测缺口，不重跑24项矩阵或解除其中严格失败。

四个新进程均完成一个warmup及一个连续测量窗，KV/HB small、FP32、SGD，每窗detach。
测量后独立CPU调度对照全部记录、loss和最终参数均通过。CPU记录器5项、CPU/SDK/CUDA三个
standalone消费者构建与无Python依赖loader/help均通过；CUDA没有真机测量。
使用原静态fixture，宽4/1样本/每窗2位置。启动首个1×1矩阵乘法不执行图，不提供路由提示。

| 路径 | 启动至ready秒 | 内部初始化秒 | 测量期allocated峰值MiB | reserved峰值MiB | 此时进程RSS峰值MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu/python | 1.447520 | 0.053238 | — | — | 318.547 |
| cpu/standalone | 0.312901 | 0.056585 | — | — | 139.707 |
| npu-python/npu-python | 15.529839 | 14.101790 | 83.102 | 102.000 | 2557.043 |
| npu-standalone/npu-standalone | 7.965241 | 7.404423 | 75.100 | 98.000 | 2325.852 |

启动至ready包含exec、loader/Python导入、参数/源码检查、库验证、首个设备算子与同步及管道送达。
内部初始化排除最初Python导入和C++ main之前链接，不能跨语言当作同一分解项。
没有清空OS文件缓存；不同CANN算子缓存/驱动初始化也未隔离。本组只说明新进程准备成本
不能忽略，不据一次观测估计统计稳定冷启动或把Python/SDK差异归因于语言。

Python/NPU使用Torch2.12/CANN9.1，standalone使用LibTorch-NPU2.10/CANN9.0。
两NPU进程都获物理9→逻辑0，项目四项计时串行，CUDA编译在计时全部结束后启动；
其他用户争用没有排除。加载/构造/warmup与全部时间在JSON单列；单个测量窗不更新此前吞吐选型结论。

allocator在runtime_ready、construction、warmup读取后重置峰值；measurement采样在最后参数导出
及独立参考之前。峰值包括阶段起点仍存活的分配，不等于阶段增长，也不包括所有vendor/driver分配。
NPU Python构造后的allocated约23.34MiB，SDK约19.34MiB；测量期间峰值约83.10/75.10MiB。
因此仅按构造常驻tensor估计完整训练峰值会漏掉伴随/工作区成本；本组不足以推定大规模容量。
CPU无设备allocator指标，以“—”明确缺失；RSS是进程生命周期峰值，包含初始化，不是阶段独占内存。

运行从只读冻结checkout `artifacts/worktrees/core5-weighted-observation`发起，build/{cpu,npu,cuda}独立。
jobs为`core5-weighted-observation-{build-cpu,build-npu,build-cuda,cpu,npu-python,npu-standalone}`；
units为`tide-JOB-20261028.service`，记录`artifacts/jobs/JOB/{status.json,task.log}`。
各case还有`*-process/{result.json,process.log}`和`*-run/result.json`。
所有job持久终态passed/exit0及结果已审阅；提交时确认live unit在background.slice，随后成功瞬态unit已回收，
最终show为inactive/not-found，其默认Result/ExecMainStatus不作退出证据；两个构建worker、ATen/BLAS1、Nice10/background.slice，
构建600秒，每候选180秒、NPU queue180秒、测量job420秒。冻结源码全程无修改。

复现：按项目开发规范选择匹配core/可选resident库，构建consumer；用
`measure_weighted_process.py --output-dir NEW --timeout-seconds 180 -- COMMAND...`
包装文档中的Python/standalone小型训练命令并指定`--steps 1`。命令argv与所有组件身份见JSON。
本资格不涉及尚在原工作树开发的单样本节点放置；既有默认异步两卡与严格数值限制保持。
