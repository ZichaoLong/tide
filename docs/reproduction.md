# 从 Git 复现问题与维护失败经验

本页是诊断入口导航，不是第二份状态表。当前限制、停止/重启条件仅见[ROADMAP](ROADMAP.md)；
数学定义见[semantic-map](semantic-map.md)，已有正负经验见[performance-lessons](performance-lessons.md)。
历史源码/环境身份及观察值以各审阅证据为准。清理不重跑旧队列，不改变任何失败或容差。

## 接续时先做什么

1. 按[development](development.md)在固定提交重建私有环境及所需库；不假设旧venv、build或run目录仍存在。
2. 从下面找到问题、Git内输入或生成器、诊断代码和原证据。先读证据中的范围/资源及ROADMAP重启条件。
3. 只有出现具体新方案或新任务时，才在新目录运行有界检查。记录实际源码、环境、输入hash、完整命令和终态。
4. 诊断通过、原反例通过、完整功能资格和性能结论分别表述；小反例不能替代完整图、独立梯度根及恢复。

需要重演旧结论时使用报告所记提交及匹配环境；当前源码运行的是新检查，不继承旧失败或旧通过。
命令中的NEW路径必须不存在，CPU_CORE必须是对应环境和源码的构建。长任务遵守[verification](verification.md)。

## 复现入口

| 调查问题 | Git内材料和入口 | 原证据与范围 |
| --- | --- | --- |
| 分配中的大数相减及残差拒绝 | [两份实际失败行](../tests/fixtures/allocation/provenance.json)、[Decimal独立求解](../scripts/diagnose_allocation_failure.py)、[局部回归](../tests/test_retired_allocation_witnesses.py) | [局部坐标修复](evidence/core5-local-allocation-factorized-kv-20261028.md)；仅局部分配，不能解除D2048全图keys失败 |
| D64 FP32完整训练、局部精度和反向累加 | [计划生成](../scripts/prepare_weighted_workload.py)、[完整诊断](../scripts/diagnose_weighted_workload.py)、[汇合点探针](../scripts/diagnose_weighted_joins.py) | [算术诊断契约](weighted-arithmetic-diagnostics.md)、[eager边界](evidence/core5-eager-boundary-20261029.md)；参数/梯度/槽及每调度自身恢复分别核对 |
| 原始逻辑时间反馈PDG | [四窗诊断](../scripts/diagnose_weighted_feedback.py) | [非GPU边界](evidence/core5-nongpu-boundaries-20261029.md)；W32/B4/seed7，FP32与显式FP64分开 |
| FP16 GDN/默认AdamW | [完整精度诊断](../scripts/diagnose_weighted_precision.py)、[干预检查](../tests/test_weighted_half_probe.py)、[原训练用例](../tests/test_weighted_precision.py) | [数值隔离](evidence/core5-numerical-isolation-20261028.md)、[后续半精度干预](evidence/core5-boundary-isolation-20261029.md)；原strict xfail/epsilon/loss scale保持 |
| 默认异步双NPU | [纯Torch小图](../scripts/diagnose_torch_peer.py)、[完整图诊断](../scripts/diagnose_weighted_async.py)、[桥接独立根检查](../tests/test_weighted_peer_bridge.py) | [数值/异步隔离](evidence/core5-numerical-isolation-20261028.md)、[完整CPU桥接失败](evidence/core5-boundary-isolation-20261029.md)；12轮小图不能认证完整图 |
| CPU autograd之后晚注册NPU | [完整小程序和退出码](evidence/core5-resident-boundary-20261029.json)的bootstrap.program/相关结果 | [初始化边界](evidence/core5-resident-boundary-20261029.md)；early/late必须两个新进程，原程序已在Git中，无需旧私有启动器 |
| native workers2跨进程恢复 | [安装消费者](../examples/consumer_core5/README.md)的check.py，显式FP32/native/workers2 | [原失败与命令](evidence/downstream-contract-20261030.md)；默认worker1不代表workers2修复，单次通过也不证明并发顺序确定 |
| D2048首次写入的真实局部算术 | [有界读取诊断](../scripts/diagnose_weighted_first_write.py)、[参数匹配及导出](weighted-workloads.md) | [首次写入隔离](evidence/core5-boundary-isolation-20261029.md)；原输入binary及witness已退役，需重建相同种子/配置材料；约37GB静态材料的成本仍存在 |
| 较大FC/HB数值、NPU流/任务资源及局部预算 | [有限目录](../benchmarks/core5-study-v1.json)、[单配置入口](../scripts/run_core5_study.py)、[配置/证据查询](core5-study.md) | [自然规模测量](evidence/core5-natural-study-20261029.md)、[流资源](evidence/core5-compact-fibers-20261029.md)、[文件预算](evidence/core5-checkpoint-budget-20261029.md)；停止条件保持，8B/16B仍无实测吞吐 |

SDK affine、normalization、源变换及query舍入的旧临时变体不作为第二套实现保留。
后续从上述完整诊断、[设备用例入口](../scripts/check_weighted_device.py)、相关独立测试及原证据进入；
这是承接调查问题，不宣称旧临时脚本与现行入口逐值等价。

## 小型实际分配行

两份JSON共不足6KiB，保留原FP32输入、控制、响应、乘子、时间/节点和拒绝阈值，
[provenance](../tests/fixtures/allocation/provenance.json)记录生产源码的dirty事实及原文件hash。
它们是经过挑选的回归输入，不是完整模型快照。独立Decimal根据捕获的二进制浮点输入求解；
新测试分别检查旧返回确实越界及当前Python/native标量/批局部前向，不复写旧返回。

```sh
python scripts/diagnose_allocation_failure.py --failure tests/fixtures/allocation/wide2048.json --output NEW_WIDE.json
python scripts/diagnose_allocation_failure.py --failure tests/fixtures/allocation/matched2048.json --output NEW_MATCHED.json
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_retired_allocation_witnesses.py
```

其他诊断的完整命令、原始形状、精度及预算已有[算术诊断](weighted-arithmetic-diagnostics.md)和各证据，
本页不再复制另一套默认配置。原始大输入已退役时，hash只能核对重新生成的内容，不能恢复丢失的字节。

## 一次调查结束后维护什么

| 产生的内容 | 应维护的位置 |
| --- | --- |
| 当前结论、停止/重启条件 | ROADMAP已有行；STATUS只保留当前交接与活跃任务 |
| 已审阅结果、原失败、源码/环境/命令/输入身份、原始材料是否在线 | docs/evidence的小报告及索引；历史报告不随新结果改写 |
| 仍需执行的通用诊断 | scripts中的有界入口；相关测试及本页导航，输入参数显式传入 |
| 必要的小型实际输入 | tests/fixtures，保留来源/hash、适用范围，并以独立公式或原严格断言验证 |
| 性能与工程经验 | performance-lessons，记录正负收益、工作量、资源争用及适用边界 |
| build、环境、原始日志/张量、完整源码快照、checkpoint | 临时ignored目录；终态审阅并有清理授权后退役，不成为下次接续的隐含依赖 |

不要把所有失败脚本都升级为公共API，也不要为了清理把未采用补丁全部合并。
正式交付的代码必须已提交；实验差异被放弃时明确记录，不用“其HEAD已合并”代替这项判断。
已安装的下游应持有自己的包/库及依赖锁；不要依赖本仓库临时build目录长期存在。

[退役任务摘要](evidence/retired-jobs-20261030.json)保留520份旧任务的终态/来源/原记录hash，
外层passed不表示每个内部case通过。原始日志不再在线；判断具体语义/性能结论仍需相应审阅报告。
