# 张量窗口混合放置、共享伴随与有限成本

放置实现固定 `540ccbd86a1d90d8d395acee97d9326667c7d1f8`；共享伴随修复固定
`2ca0045828dd920effb0551a2dd079693e6cb74c`。数学固定 `0770593`。
[机器记录](column-placement-20261111.json)保存源码、原构建身份/hash、作业终态、实际配置、
逐项比较、原失败和测量；[审阅清单](column-placement-20261111-reviews.json)只认证8个文件的实际审阅。

## 实现与独立验证

Python/C++窗口现在按声明的 mixed-a/b/c 计算Read、区域控制和选择，payload/raw状态及参数owner
保持原位。Read只传需要的参数和bank字段，Full/更新大矩阵不随Read迁移。参数、int64元数据、
实际正支持和声明一阶VJP保留。实际位置及评分精度进入窗口执行记录。
这是单payload设备的主机调度路径，不是resident或多payload设备新资格。

| 固定范围 | 实际结果 |
| --- | --- |
| 540ccbd8 CPU相关公式/调度/共享/训练/恢复 | 213 passed，4个原数值xfail |
| 2ca00458 CPU扩大到共享使用与一般端口 | 239 passed，13个原数值xfail；与上行重叠，不相加 |
| 540ccbd8 NPU安装客户端 | 11项中10项独立三步AdamW/新进程恢复通过；Python KV mixed-a恢复失败 |
| 540ccbd8独立SDK NPU | 五记忆×mixed-a/b/c共15项：独立streaming、raw、VJP、两个SGD窗通过 |
| 2ca00458 NPU安装客户端 | 原Python KV mixed-a两个独立流程、KV mixed-b/c、SSD mixed-a，共5项独立三步/新进程精确恢复通过 |
| 2ca00458同图反向 | Python KV mixed-a固定一次前向、六次VJP，含内部列边界，逐位相等 |

两套C++构建与固定源码的完整C++/CMake hash相等，所有manifest中的binary hash逐项验证；
原开发manifest仍保留原 `ee37cdac` 身份，未伪造重建。修复只改Python共享交接，未改C++字节。
安装客户端全部显式 `--no-deterministic-algorithms`，未以全局确定性策略使恢复通过。
三个child分别运行whole/prefix/resumed；比较真实输出、raw状态、输入/参数梯度、参数与AdamW槽。
SDK运行使用独立LibTorch2.12/CANN9.2模块，没有加载Python-wheel运行库。

## 新失败、定位与局部修复

原 `column-placement-npu` 任务保持failed：Python KV mixed-a的新进程恢复有72个浮点字段不等，
离散比较通过。仅查看脚本退出或小型独立数值通过不足以验收精确恢复。

有界调查固定同一前向图，排除了输入、前向和optimizer改变：默认并行反向有12个参数/输入根
不等，最大记录差异为输入约1.49e-8；增加中间根后，最后区域两个节点共享的content梯度先不同。
关闭反向多线程、或者仅取消Aggregate计算共享的两个诊断均通过。独立native没有该Aggregate缓存，
该小见证亦通过；这不证明任意native跨设备图都可精确恢复。

采用 `column_sharing.shared_uses`：数值Aggregate仍只算一次，各节点得到只读整列使用视图，
VJP按稳定节点顺序合并全部使用者，避免跨设备分支到达顺序选择共享输入的求和次序。
没有移除计算共享，没有关闭生产autograd并行，没有改全局精度或确定性开关。
独立消去反例检查 `1e8 + (-1e8) + 1` 的指定使用顺序；另验None/零、原始版本和一阶边界。
新版本的同图反向及原安装恢复见证均通过，原失败保留。

## 有限性能观察

三个原代表，各做mixed-a/b/c一次独立测量流程：一窗warmup、一窗完整训练，随后精确下一窗检查。
均为FP32、B4/L16、staggered实际61输入位置，Python调用native、单空闲NPU物理11、ATen2线程，
实际16 CPU affinity；无本项目重构建或其他计时重叠。共享主机的外部争用未排除。
使用公共实验CLI `input_interface=auto`，全部实际命中 `settle-columns-v1`。
参数与局部候选来自固定 `prefill_regression_config.py` 的默认recipe，完整请求在机器记录。

| 用例 | mixed-a秒 | mixed-b秒 | mixed-c秒 | 候选/Full/边消息 |
| --- | ---: | ---: | ---: | --- |
| wide-fc，4节点/d512，KV sum | 0.183364 | 0.183649 | 0.196920 | 214/122/122 |
| sum-fc，16节点/d64，KV sum | 0.304023 | 0.334819 | 0.371230 | 915/382/1228 |
| 四层SSD，d128 | 0.100534 | 0.091375 | 0.100355 | 244/244/183 |

9项均completed且自身精确恢复通过，独立规模数值为not_checked；小图资格另列。
完整成本含准备、copy、调度、packing、forward/loss、finite检查、backward、optimizer和同步；
不含构造、磁盘checkpoint发布及独立恢复检查。吞吐按61除以上表秒数，不能用B或候选数冒充输入量。
这些小计算中mixed-a有实际价值；不据一次测量修改所有模型的默认放置或宣称普遍最快。

对受共享修复影响的PyTorch默认窗口另测主FC16：B8/L128/d128、KV qry、H2/FFN512，
原下游 `0439efd` 的参数、head、loss与AdamW；逐步同步完整训练。

| 设备 | 完整训练秒 | 输入位置/秒 | 候选/Full/边消息 |
| --- | ---: | ---: | --- |
| CPU1，affinity80–95 | 1.244159 | 823.05 | 16384/8344/25060 |
| NPU，host2，affinity144–159 | 0.736821 | 1389.75 | 16384/8345/25064 |

分子为1024。原01cde3b的PyTorch成本为1.2824/0.7571秒，只作历史参照；未观测明显默认路径退化，
不同进程/资源条件不能给出统计加速结论。实际Full/消息不同，不能称两设备严格等价吞吐。
本轮未测TTFT、推理三模式或完整尺度独立数值；原严格失败不解除。

## 复现与边界

从固定提交构建/安装，参考[开发入口](../development.md)，然后在合法设备分配内运行：

```sh
python scripts/check_column_adjoint_order.py --device npu:0 --placement mixed-a --include-columns --output-dir NEW
python examples/consumer_core5/tensor_window.py --device npu:0 --dtype float32 --client python --placement mixed-a --no-deterministic-algorithms --output-dir NEW_CONSUMER
tidegraph-column-check --device=npu:0 --dtype=float32
python scripts/prefill_regression_config.py --case sum-fc --device npu:0 --placement mixed-a --threads 2 --output NEW_CONFIG.json
```

性能请求、原始结果hash、真正affinity和环境在机器记录；本地raw目录不构成复现前置。
诊断脚本可用 `--autograd-threads serial` 对照，但不是库默认或新的恢复保证。
新列多payload设备、resident、FP16及其他资源/异步问题仍由ROADMAP管理；此处不扩大资格。
