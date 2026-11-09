# 当前交接 — TL-REQ-003 基座交付完成（2026-11-09）

用户授权的最小正支持数值模式已实施并完成本次CPU/NPU有限资格；下游学习验收未运行。
交付与复现：[证据/回复](evidence/positive-support-tl-req-003-20261109.md)、[JSON](evidence/positive-support-tl-req-003-20261109.json)、
[公开契约](positive-support-policy.md)。固定数学0770593838a37c790a568226930567ed60f99b83不变。

## 实际源码、结果与边界

- 实现/wheel：2c11695ca66a5dcd028f7d777ec26b177add0378。
- 零接收补测：2502fa4a3df3096e3f5773a7c8586b517b96da89；资格筛选：b4879ffa6bceb989f052484f18cfb480f7d1a68f。
  后两项Python包源码与2c11695完全相同，复用同一已验wheel，未声称重新构建。
- CPU新策略30项、相关默认路径314项通过；NPU流程13项实际设备目标及17项CPU/配置辅助通过。
  含实际行/独立高精度、完整两区域记录/根/共享参数、AdamW、pending/保留跨窗和新进程精确下一窗。
- 默认strict仍拒绝原FP32行；新模式显式拒绝弱总量实质扰动、不可表示公开伴随及非法配置。
  原历史失败、旧checkpoint跨模式拒绝保持；不等同原HB16训练恢复或学习效果通过。
- 单设备Python eager/KV/FP32或CPU FP64；新策略其余后端/记忆/精度限制及重启条件仅在[ROADMAP](ROADMAP.md)。

## 作业终态

全部正式作业MainPID=0、ControlGroup为空且对应cgroup不存在；没有活动计算/排队。
固定源只读：artifacts/tl-req-003/source-{a,b,c}；独立资格venv和wheel在同目录，均非下游依赖。

| 服务 / artifacts/jobs目录 | 终态 |
| --- | --- |
| gef-tl-req-003-cpu-a / tl-req-003-cpu-a | failed exit1；新策略29通过，回归314通过/8个native缺库失败；筛选遗漏，不重标通过 |
| gef-tl-req-003-npu-a / tl-req-003-npu-a | passed exit0；29定向及安装消费者 |
| gef-tl-req-003-boundary-b / tl-req-003-boundary-b | passed exit0；CPU/NPU零接收各1 |
| gef-tl-req-003-cpu-c / tl-req-003-cpu-c | passed exit0；修正筛选后的8个Python拒绝快照/CPU安装消费者/上游校验 |

各目录保留status.json/task.log；审阅证据已抽取实际来源、安装hash、计数、失败、队列/资源和终态。
没有重复314项已通过回归；修正资格脚本后未整体重跑门禁，分段证据不冒称旧failed任务变passed。
采用公共ascend/dev-workspace-9.2.0及独立私有环境，未改公共环境或下游安装。
本次材料约百MiB量级、预算≤2GiB；账号约169GiB/硬800GiB，无需tmpfs或扩盘。

## 下一步与干净接续

本任务已收尾；若继续实验，由tide-learning-lab在新的私有安装接入、记录验收，再显式更新依赖锁。
新窗口无需旧产物目录：干净检出交付源码，按[development](development.md)选择环境、构建/安装wheel，
按[交付回复](evidence/positive-support-tl-req-003-20261109.md#下游接入和复验)执行公共行和小型检查。

```sh
python examples/consumer_core5/positive_support.py --row tests/data/tl_req_003_shannon_row.json
python -m pytest -q tests/test_positive_support.py tests/test_positive_support_config.py tests/test_positive_support_graph.py
# 固定源码/独立wheel的持久资格：bash scripts/qualify_positive_support.sh NEW_OUTPUT cpu|npu
# NPU由站点队列提供ASCEND_RT_VISIBLE_DEVICES，程序使用npu:0。
```

不自动重启原HB16训练，不改其cat/ind、loss或配方；学习效果和工作量由下游拥有。
用户原未跟踪文件docs/tide-learning-lab-TL-REQ-003-handoff.md保持原字节、未纳入提交；
唯一请求仍在实验分支。其余本轮修改均进入实现/证据提交，普通推送授权分支。
既有CANN9.2三客户端迁移与严格失败见[迁移证据](evidence/cann92-project-migration-20261106.md)，未扩大其资格。
