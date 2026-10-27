# 当前交接 — 2026-10-28

本轮采用固定上游`0770593838a37c790a568226930567ed60f99b83`的本机可闭环目标已完成。
授权分支`graph-execution-foundation`；工作路径`/mi/data2T/zlong/home/llm/graph-execution-foundation`。
最后运行库源码`0155b0f679caab089cbabccf7986e479fe5493a5`、检查器修正
`220052abb592bc9a9aab6f27e4e042a55415edeb`已提交推送；本页与审阅证据/导航作为独立收尾提交。
没有活动实验、构建或测试任务。当前交付入口为[ROADMAP](ROADMAP.md)、
[能力范围](execution-capabilities.md)、[语义对应](semantic-map.md)与[证据索引](evidence/README.md)。

## 完成范围

- 功能：一般SettleGraph同秩区域/端口与精确嵌入、TimedDAG、PDG；带权结构化表示、五记忆、
  原始逻辑时间短卷积、十种发送及正式评分/证据/key、共享参数与异构绑定。
- 正确性：独立公式/求解/方向导数与独立调度各司其职；完整字段、支持、control、逐边/端口、
  pending、独立梯度根和None/零连接；在线greedy、实际packed与批量VJP都有固定源码证据。
- 训练与恢复：CPU、mixed及resident完整声明一阶训练、SGD/AdamW/master、累计、保留窗口、
  原始状态/optimizer checkpoint；样本副本与单样本跨节点放置、共同拒绝/发布及跨放置恢复。
- 后端：Python参考、Python调用native和standalone分别验收；NPU FP32/FP16范围分列；
  CUDA源码/三架构构建和安装loader完成，真机未验，x86_64不能继承本机aarch64证据。
- 性能：24配置22通过、两项FP32 PDG严格失败；本组小中规模CPU成本最低。
  medium HB standalone prefill约350ms、streaming456ms、mixed-c约1.41–1.61s、resident约2.60–2.71s。
  完整计时含准备/调度/packing/反向/通信/optimizer/同步；启动/内存四流程补充完成，不外推大模型或多卡吞吐。
- 追溯：固定数学→配置/参数/快照→代码→独立测试→不可变源码证据；仓库可独立导航和重新构建，
  不依赖外部交接文档或旧运行目录。参考仓库和公共环境未修改。

## 最后资格与终态

[放置窗口反向与公共训练报告](evidence/core5-weighted-node-training-20261028.md)及同名JSON
保存全部源码/构建/命令/日志哈希、物理到逻辑映射和独立终态。

| 任务 | 终态与范围 |
| --- | --- |
| `core5-weighted-node-services-python-qualify` | passed/exit0，0155b0f放置训练8项；37922eb原30项另行通过 |
| `core5-weighted-node-services-cuda-qualify` | passed/exit0，0155b0f sm80/90/100及安装消费者；无真机 |
| `core5-weighted-node-services-npu-qualify` | failed/exit1，CPU10及FP32四组合×12独立根/两个跨窗桥通过；五训练轨迹后被错误检查器拒绝 |
| `core5-weighted-node-refusal-npu-v2-qualify` | passed/exit0，220052a检查器/0155b0f运行库，两dtype各20更新及拒绝/续接、各6副本共同更新、两单设备窗口及两安装消费者 |
| `core5-weighted-node-refusal-cuda-v2-qualify` | passed/exit0，修正检查器链接已核验安装库及help；没有新GPU结论 |

units均为`tide-JOB-20261028.service`，记录`artifacts/jobs/JOB/{status.json,task.log}`；
SDK补验另有`check-build/result.json`、`corrected-training.json`、`checks/result.json`、
`single-window.json`与`consumer-checks.json`，Python为`tests.xml`。
只读源码在`artifacts/worktrees/core5-weighted-node-{window,training,services,refusal}`。

原f84a007全远端反向的CANN任务缓冲区满由0155b0f每节点嵌套服务修复，原反例未改并通过。
37922eb/0155b0f原SDK失败来自检查器误写拒绝码21；220052a纠正为数值拒绝码20，生产optimizer未变。
首次补验CUDA漏装training组件失败、SDK等待中取消，v2独立补齐通过。原失败/取消均保留；
GDB诊断exit0只表示抓栈完成，不是设备通过。详情和最小复现指纹在上述报告。

## 保留边界与下一步

没有待续的本机任务，不恢复旧队列，不重跑旧性能矩阵。新任务前运行：

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
```

若有GPU或新目标机器，按[ROADMAP](ROADMAP.md)及各模块门禁开始新的固定源码资格；
以全新build/output目录运行，不修改保留的失败材料。检查历史任务可用
`python scripts/status.py --all-jobs`；unit检查为
`systemctl --user show UNIT -p ActiveState -p MainPID -p Result -p ExecMainStatus`。
已回收unit的默认属性不能替代持久退出码与实际结果。

严格限制继续保留：eager FP16 GDN默认AdamW约.001984；默认异步双NPU eager VJP CANN invalid event；
SDK2.10 FP32 fc/affine第三轮evidence差3.09944e-6/mixed-a同见证；ceae7bf FP32反馈PDG两项
maxabs4.11272e-6/2.71201e-6。显式同步、resident/master或FP64提升的通过不解除原严格失败。
各域/容量/正支持下溢明确拒绝，不静默替代算法；全部CUDA真机范围待验。

本机规范为`/opt/software/ascend-locks/dev-workspace-9.1.0/README.md`和`/etc/modulefiles`。
Python用ascend/dev-workspace-9.1.0及项目.local/venv-core5；standalone用libtorch-npu/2.10.0-cann9.0.0；
CUDA用账户activate-cuda.sh；不要混用二进制。完整入口见[development](development.md)。
长任务继续固定源码、独立background.slice、Nice10、两个build workers、ATen/BLAS1和有限预算。
STATUS写入仍用durable_records的fsync/atomic replace并回读；不新增并行状态账本。
