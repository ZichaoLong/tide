# 容量与区域执行阶段的集成检查

本页收束配置目录、CPU 套件、集成检查和源码身份；不增加设备性能或所有组合的资格。
[JSON](capacity-validation-20261118.json)保留各自源提交、原日志身份、环境、目录统计、
分支映射及终态检查。数学固定为 `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`，
`python scripts/check_upstream.py` 的 108 文件核验通过。

## 冻结 CPU 套件与配置目录

`artifacts/capacity-integrated-cpu-v2` 使用冻结提交
`8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee`，**544 项 CPU 测试通过，285.99s**。
其 315 个归档文件在收口时逐字节匹配该提交；实际 Tide 导入来自冻结目录。
作业位于 `background.slice`，CPU affinity64–71，测试 conftest 设置 ATen线程1，
vendor autoload关闭，不占NPU。监督器总时间288.525s、采样进程树RSS峰值约1.113GiB，
界限32GiB/1800s；共享主机身份记录，采样界限不是cgroup硬内存限制。
完整suite不是性能测量，也不将单独设备检查中的数值失败重标为通过。

`artifacts/capacity-support-catalog-v1` 固定 `7b77457d`，308个归档文件逐字节一致。
独立菜单核对41空间/55时序绑定、475几何设置、225,400参数配置、676,200分配行，
5,510项预算检查通过；没有导入Torch或分配模型。可选固定通过Hub、显式本地时钟、
terminal Hubs的行为检查由有限测试及[新安装包](capacity-installed-20261118.md)承担。
元数据枚举不认证全部组合的执行、数值、性能或所有上游探索绑定。

## 集成后的必要复核

源身份 `5ae060db940e44d447ce541eeb8fada865893b4c`：

- `tests/test_installed_validation.py` 和 `tests/test_region_batch.py` 共 **17项通过，8.89s**。
  前者9项检查有限值、support、元数据、非有限、None/梯度连接和恢复失败的分类；
  后者8项保持已有合批检查。本页没有声称重新运行了553项完整套件。
- `scripts.check_installed_capacity`、`scripts.support_catalog`、`benchmarks.search`、
  `benchmarks.variants`、`benchmarks.restore`、`benchmarks.memory_ledger`、
  `scripts.check_placement` 七个 `--help` 入口全部返回0。
- 当前21个`src/tide`文件与`8a22530d`的AST全部一致；20个文件字节相同，
  `region_batch.py`只修正偏置/Linear舍入的说明注释。AST相同不是字节相同，
  新wheel及完整套件仍保留其原冻结源身份，没有追认为当前提交的新硬件实验。

## 可从主分支取得的实现文件

子工作树的原始提交身份保留在证据里。下面主分支提交中，原提交改动的文件已经逐一
核对字节相同；JSON给出具体文件及SHA256。映射只覆盖这些文件，不声称两提交整树相同。
复现脚本或native切片时可从右列取文件，无需取得未推送的工作分支。

| 原工作树提交 | 主分支集成提交 | 文件范围 |
|---|---|---|
| `9b220bb5` | `8fa6690e` | 原安装harness，1文件 |
| `cd131fc2` | `22bc86af` | 修复分类的harness与测试，2文件 |
| `4f7ea32a` | `bf7233f6` | 区域并发实施，9文件 |
| `89ce08ff` | `062a8ab6` | LibTorch区域切片/检查/文档，13文件 |
| `a4509760` | `8f98ffbc` | 舍入诊断入口，1文件 |
| `e45e0d0e` | `fd54d67e` | 新进程恢复计时，3文件 |
| `5d96f8ee` | `36c42a7a` | 执行变体驱动，3文件 |
| `3f0075ea` | `3543825a` | 账本runtime scope标签，1文件 |

## 终态与保留

另核对已集成区域/安装证据引用的130个小型在线产物SHA和32个Git blob；
9个较大checkpoint/张量文件仅保留原报告身份，没有重复读取其payload。
这些核对不重跑实验，也不替代报告中的全量数值/实际设备检查。

本阶段主树及安装子任务的45份状态均终态：40 passed、4 failed、1 terminated。
外层passed与内部数值在阈值内分开；4份原失败及停止记录保留。所登记PID/start身份
未发现存活后代，systemd当次没有running/activating的`tide-*`服务。
四份原用户未跟踪文档SHA256与阶段前记录相同；没有改动公共环境、参考或下游。
本次未清理工作树、冻结源、wheel、私有环境或checkpoint；后续清理仍需依赖审查。

复现环境见[开发](../development.md)，菜单入口为
`python -m scripts.support_catalog --output NEW-support.json`。
完整CPU任务按上述冻结提交、原launch命令和新的输出目录，在重新检查资源后按
[执行契约](../execution-flows.md)启动；安装和硬件复现分别采用各自报告命令。
