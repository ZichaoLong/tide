# 当前交接 — 2026-10-29

授权分支 `graph-execution-foundation`；数学固定 `0770593838a37c790a568226930567ed60f99b83`。
本轮配置、自然规模、CPU/NPU mixed与resident有限测试、分阶段提交及普通推送均已授权。
不使用子代理；参考仓库及公共工具链只读；GPU真机延后。构建默认all，也可half；测试使用空闲资源。

## 当前结论

本轮实现、固定源码资格和有限测量均已终态，收尾证据随本页提交；没有未提交实现、活跃作业或自动重试队列。
实现固定到 `d6d2fa741db32c15993cbec80dd0fafaa8ff5ce3`；本次收尾只更新文档、证据和支持声明。
实现与资格范围、原失败及停止/重启条件分别记录，不宣称全部数值、异步、设备及规模限制已解除。

| 实现提交 | 已交付内容 |
| --- | --- |
| `d841450` | 取消历史两线程构建限制；全部可用CPU/half/显式线程入口 |
| `8f51ba6` | 公共配置解析、四种身份、FC/HB/LH启发拓扑与自然参数规模；固定CPU336项通过 |
| `372eb87` / `e50b6d8` | 有限目录/运行、三客户端配置及实际训练/恢复；独立half存储/FP32算术参考 |
| `b255e27` | native显式正int64文件预算，默认4GiB保留，codec格式不变 |
| `d6d2fa7` | 直接查询仓库内审阅证据，无需历史运行目录；固定5项相关检查通过 |

## 结果入口

- [配置与三个消费者资格](evidence/core5-study-clients-20261029.md)：372 CPU365项；e50相关CPU43项、standalone CPU6/NPU8配置、Python NPU3配置通过。检查有重叠，不相加为全矩阵资格。
- [自然规模有限测量](evidence/core5-natural-study-20261029.md)：25配置中7项完整通过；两个中型FC各三个CPU FP64客户端，以及LH启发1.051B CPU FP64。独立NPU profile确认区域批B4；失败计时不作合格吞吐。8B/16B仅静态计划。
- [checkpoint预算资格与原1B复验](evidence/core5-checkpoint-budget-20261029.md)：b255 CPU32项及CPU6/NPU8消费者通过；5,253,356,896字节文件保存/新owner加载返回。原owner下一训练窗出现非有限梯度，完整续接与独立CPU核对未通过。
- [导航与修改维护表](core5-study.md)连接配置、实现、运行、结果和数学对应；[performance-lessons](performance-lessons.md)保留LH/core-3及本轮正、负经验。

## 作业终态与本地产物

下表名称同时是 `artifacts/jobs/<名称>/`；日志为其中的 `task.log`，阶段结果/日志在同目录。
对应服务为 `<名称>-20261029.service`。新checkout无需这些本地产物即可按证据报告重新构造输入。

| 名称 | 固定源码 | 终态及含义 |
| --- | --- | --- |
| `core5-study-v1-cpu` | 372eb87 | passed/0；CPU365与早期消费者 |
| `core5-study-v2-cpu`、`core5-study-v2-sdk`、`core5-study-v2-python` | e50b6d8 | 均passed/0；相关资格与真实消费者 |
| `core5-study-v2-performance`、`core5-study-v2-supplement` | e50b6d8 | launcher均passed/0；study均completed_with_failures，合计25配置7通过，另有独立profile |
| `core5-checkpoint-cpu`、`core5-checkpoint-sdk-v3` | b255e27 | 均passed/0；CPU/SDK独立资格，SDK正确9392目标 |
| `core5-checkpoint-sdk`、`core5-checkpoint-sdk-v2` | b255e27 | cancelled/143、failed/1；错误SoC目标和已有目录拒绝，不改写 |
| `core5-checkpoint-large` | b255e27 | launcher passed/0；原FC1B配置failed，文件阻断解除而下一窗梯度非有限 |
| `core5-evidence-query` | d6d2fa7 | passed/0；5项相关检查，未改执行器/扩大性能资格 |

只读源码树：`artifacts/worktrees/core5-study-v2-qualified` @e50b6d8、
`artifacts/worktrees/core5-checkpoint-qualified` @b255e27；其余固定身份见对应机器证据。
新鲜checkout复现命令、实际构建/二进制/日志hash均在报告；原失败与large文件保留。
本机Python入口为`ascend/dev-workspace-9.1.0`和项目venv；standalone SDK独立使用
`libtorch-npu/2.10.0-cann9.0.0`。不得跨ABI复用库，详见[development](development.md)。

## 接续动作

[ROADMAP](ROADMAP.md)是唯一未解除边界及停止/重启表。数值、异步peer、resident流/在途任务、
局部分项预算及CUDA/x86待验继续保留。没有新算术/有界服务方案及对应独立证据时，不重跑已停止的同配置大规模候选。
后续先按下列命令核对状态和查询仓库证据，再按具体新方案启动有限任务；不恢复旧队列。

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
python -S scripts/index_core5_study.py docs/evidence/core5-natural-study-20261029.json --state passed --format tsv
python -S scripts/index_core5_study.py docs/evidence/core5-checkpoint-budget-20261029.json --format tsv
```

收尾检查已核对固定上游Git对象、本轮文档链接、26条运行记录与7组资格的源码/结果/日志hash及通过标志；
支持矩阵schema通过。它们是文档/追溯检查，不新增运行资格，不重复已经通过的构建和相关测试。
