# 仅保留 Git 内容：清理与全新开发链核验

实现固定`e99733ddb0ff885cc05e287b8b28ccc15f105458`；数学仍为`0770593`。
用户授权在补齐失败知识与复现导航后，删除本地环境、构建、原始运行材料和临时副本。
[机器凭据](artifact-cleanup-tracked-only-20261030.json)记录实际源码、配置、命令、环境、输入身份和结果。

## 从新环境及源码验证

独立`background.slice/tracked-checkout-verify-20261030.service`，固定只读工作树，
新建项目私有环境和CPU构建，没有使用旧`.local`、旧build或旧运行输入。
公共模块`ascend/dev-workspace-9.1.0`，Python3.12.13/Torch2.12.0+cpu，
pytest9.1.1/ninja1.13.2，aarch64/GCC10.3.1；C++构建`--jobs half`实际160/320 CPU，测试ATen/BLAS=1。
请求时限2700s/CPUQuota16000%/MemoryMax512G/Nice10；不将legacy cgroup或slice称为独占资源。

- 全新CPU库、CPU算子探针、FP32/FP64原生smoke通过。
- 132项定向检查通过，无跳过；覆盖新增真实失败行、Decimal、持久记录、基本语义和native Settle。
- 安装消费者4项单元、8组CPU调用/独立两步训练/新进程精确恢复、4项错误请求拒绝通过。
  组合包含PyTorch/native、FP32/FP64、FC/HB、streaming/prefill；逐配置事实见JSON。
- 之前39项开发检查包含在上述132项范围内，不累加成另一份资格。
- 静态工具给出265处warning/0 error；已检查运行时/设备门禁、测试和后端边界，
  CUDA语言启用有条件门禁。字符串匹配不代表缺陷或完整可移植性认证。

这是起步与安装链路的定向资格，不是新全矩阵、性能、NPU/GPU或standalone SDK资格。
复现从[开发指南](../development.md)的新环境与新build开始；使用以下受版本控制的入口：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir NEW_BUILD --jobs half
NEW_BUILD/tidegraph-smoke --device=cpu --dtype=float32
NEW_BUILD/tidegraph-smoke --device=cpu --dtype=float64
TIDE_BUILD_DIR=NEW_BUILD python -m pytest -q tests/test_retired_allocation_witnesses.py tests/test_allocation_decimal_diagnostic.py tests/test_durable_records.py tests/test_semantics.py tests/test_native_settle.py
python scripts/check_downstream_consumer.py --build-dir NEW_BUILD --output-dir NEW_CONSUMER
```

这里的NEW_BUILD应替换为绝对路径；匹配的依赖与完整环境变量见JSON。

## 保留与退役边界

[复现导航](../reproduction.md)连接正式生成器、诊断、小型输入、独立检查和审阅证据。
两份实际分配失败行连同来源不足7KiB，保留原始错误返回，不改写成当前成功输出。
[520份旧任务摘要](retired-jobs-20261030.json)逐项核验原记录hash：317 passed、192 failed、11 cancelled；
外层passed不保证内部case全部通过。既有失败和ROADMAP停止条件不变。

旧环境、预编译库、源码tar、临时诊断变体和原始见证均在本轮范围内；
28份未采用实验差异的本地归档退役，不宣称这些差异已合并。
历史报告中的路径成为来源标签。精确checkpoint、完整日志/profile及已退役大输入不能从hash恢复；
重演完整大型问题需重新生成材料，预算与当前停止条件见[ROADMAP](../ROADMAP.md)。
已有随Git保留的20261012历史记录归档约38.5MiB继续保留，不追加新的原始运行包。

## 清理执行

已审阅10项dry run并通过独立`tracked-checkout-retire-20261030.service`删除；exit0。
删除前重新比较候选元数据、受保护Git内容和可见活动用户；Git clean限定为明确路径，不跟随符号链接。

| 退役范围 | 删除路径分配空间 |
| --- | ---: |
| artifacts：旧库、源码快照、原始运行/失败材料、临时脚本归档 | 23,516,815,360 bytes |
| .local：私有环境、旧启动器/调查变体 | 508,153,856 bytes |
| pytest及Python缓存 | 4,685,824 bytes |
| 合计 | **24,029,655,040 bytes = 22.379 GiB** |

207,573个目录项中包含146,102个普通文件、480个符号链接和4个历史测试FIFO。
外链只涉及公共Python解释器，删除链接本身，不修改其目标。
2,584个Git跟踪文件的模式/内容hash、`.git/HEAD`、index、config在删除前后不变，Git连通性检查通过；
`git clean -ndx`为空。刚完成删除时目录为132,288,512 bytes（约126.16MiB，含Git和临时worktree登记），
后续小型证据提交/删除工作树登记会略有变化。上述空间是仓库路径的分配空间，不是共享盘df净变化承诺。

两次只读预检先行退出，均未删除文件，原失败保存在JSON：第一次发现测试生成的FIFO，
确认对应`test_standalone_persists_phase_before_blocking_fixture_read`后仅放行4个明确路径，不打开管道；
第二次遇到4个SSH会话、1个PAM辅助进程和1个僵尸的proc可见性限制。
只读核实其身份并记录限制后，继续独立检查可见的普通进程/子进程及520份终态jobs；
没有观察到候选文件使用者，不宣称读取了不可见的fd/maps，没有改变权限或终止无关进程。

本报告先以`ef8de6135770f14477a9ee4beafee33b1ec13fd1`提交，随后移除临时工作树及整个验证根。
本轮新建环境/build/source/原始控制记录共292,491,264 bytes也已删除，没有另存旧22GiB副本。
最终只登记主工作区，ignored/untracked预览为空；目录约126MiB（含Git），Git跟踪文件逻辑总大小约62.16MiB。
精确测量时点见JSON；提交会使`.git`占用有小幅变化。当前开发不依赖这些退役目录。
