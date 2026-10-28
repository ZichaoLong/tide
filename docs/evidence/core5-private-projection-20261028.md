# Quad 数值响应与私有 KV 矩阵批 — 2026-10-28

实现固定为 `07e56ca03b72e177f2e7ee5cbcb47994710e04ac`；数学仍为
`0770593838a37c790a568226930567ed60f99b83`，图 v13/checkpoint v5 不变。
[同名 JSON](core5-private-projection-20261028.json)保存来源、开发失败、固定资格、材料身份和有限测量。
这是[前一轮经验迁移](core5-performance-transfer-20261028.md)后的具体改进；旧证据、失败和原容差均不改写。

## 数学对应与独立验收

Quad 原来虽在居中坐标求解，却用原坐标中已舍入的乘子重建响应进行残差和温度反向，
共同偏移可造成相消。现在 scalar、batch、resident 使用实际居中响应点；公开 multiplier
仍是近似报告。保存原参数版本，原正支持下溢拒绝、活动集合及精确解 VJP 身份保持。
饱和点的温度斜率避免无意义的 `0*inf`。定义/实现/测试见[分配契约](../weighted-allocation.md)。

独立闭式例子 `scores=[10000,9999], caps=[1,1], lower=upper=.7, tau=1` 的解 `[.7,0]` 通过。
另一个多自由坐标例子 `scores=[10000.5,10000,10002,9997], caps=[1,1,.25,1], bound=.95,
tau=1, shift=64` 的解为 `[.6,.1,.25,0]`。对根 `[2,-1,4,7]`，独立手算给出
score 梯度 `[1.5,-1.5,0,0]`、cap 梯度 `[0,0,3.5,0]`、bound `.5`、tau `-.75`、shift `0`。
FP32/FP64 Python/native scalar/batch、resident CPU 适配及 NPU 组件分别核对；不只对照两个求解器。
这个共同偏移修复没有证明所有大分差或 D2048 数值问题均已解决。

在已证明相同 base Parameter、相同 Content Tensor 输入的准备组中，再按私有矩阵形状、dtype、device 分组，
通过三次真实 BMM 计算 `Vx`、`U(Vx)` 和 `summary@x`。单行/异形保留原路径，私有缓存与时间不合并。
独立行一阶反向按 `dx=Wᵀg, dW=gxᵀ` 汇总有根的行；未使用行仍为 None，连接零、重复参数/输入身份
及保存版本均保留。普通 stack/BMM 自动反向不能代替这个连接关系契约。
准备结果是候选实际计算的临时值，不是新状态或参考补算；一般 StateKernel 默认不启用此能力。
定义、符号、独立 Jacobian 与完整训练检查见[KV 共享契约](../weighted-kv-sharing.md)。

## 固定源码资格

| 范围 | 终态与能支持的结论 |
| --- | --- |
| CPU | 574 passed、2 strict xfail；独立公式/字段根/版本、共享与未用行、完整执行、optimizer/恢复及通用自定义 State。standalone FP32/FP64 各 24 图×4更新含精确 checkpoint；两个原 eager FP16 GDN/AdamW 见证保持 xfail |
| standalone NPU | 物理1/11；allocation FP32，加 flow/config/training/node-training 各 FP32/FP16，共9项通过；含完整单/双卡训练和恢复。额外 SDK eager `fc/kv` FP32 通过 |
| 纯 Python NPU | Torch2.12/CANN9.1，`mixed-c/kv-fc` FP32，通过完整根、4次训练更新、streaming/prefill及设备→CPU恢复；不是 standalone 结果的转述 |
| CUDA | 新 core 和 resident，sm80/90/100，loader/help 和 core CPU FP32/FP64 检查通过；没有 GPU 真机结论 |

CPU core 复用终态开发产物前，固定 checkout 重新核验全部 C++ 内容 hash 及每个 binary hash；
其开发 manifest 的 base commit 为 b7c5173，不能把它当资格来源。独立 benchmark 从 07e56ca 重建。
NPU/CUDA core 与 resident 均从固定源码构建，组件 hash 相同。三个资格 job 均 exit0。
开发 quad 332 项、private 修正后244项通过；首次 private 开发因新测试括号及 grad_mode include 缺失而 failed，
该 job 的原日志与 source.tar.gz 保留，未重标通过。

## 有限对照与工作量

每配置一个新进程、一次 warmup 加一次状态连续的测量窗；没有为取有利结果重试。
完整训练计时包含输入准备、调度/packing、完整返回、loss、反向、finite checks、SGD、detach和同步。
构造、warmup、事后独立参考核对分列；候选不消费参考轨迹。profile 另开进程。
当前 CPU baseline 为2da3f2c，纯 Python mixed baseline 为0503593，均与07e56ca比较；
这次同时包含数值修复和私有 BMM，不能逐项归因。

| CPU 同一材料 | 结果 | 实际工作 |
| --- | --- | --- |
| 原 HB medium，D32/B4/4位置，102604参数 | baseline 388.921 ms，当前387.704 ms；两者完整对照通过；约0.3%差异不足以证明稳定收益 | 同128候选/128 Full/192消息；新增私有BMM最大2行、144调用/288总行；State最大批仍4 |
| ranked64，480节点/2208边，9076975参数 | 3.183 s，但完整对照严格失败；不计等价加速 | 479候选/344 Full/1604消息；私有BMM最大32行、6调用/102总行，只覆盖34次State；State最大批仍1 |
| wide2048，4节点/4边，201463846参数 | 分配残差拒绝，metrics空 | 宽算子探针，无吞吐结论 |
| matched2048，480节点/2208边，9262006511参数 | 构造195.640 s、warmup161.020 s后，测量窗仍分配残差拒绝，metrics空 | 匹配旧core-3 Add主体9262008512，差−2001；真实FFN宽319/320，B1/1位置推理 |

matched2048复用校验过的37048052160字节二进制材料，生命周期RSS峰值50596757504字节（约47.12 GiB）。
这不是训练峰值，也不是历史B512/12位置训练的复刻。共同偏移手算修复没有解除原大型执行边界，
仍不能给出大模型吞吐胜者；五族大预算静态计划不等于十个大模型运行资格。

独立 ranked64 CPU profile 记录22501次 `aten::mm`、12次 `aten::bmm`、100523次 `aten::copy_`，
约9.094秒CPU self事件中mm约.261秒、bmm约.000684秒、forward作用域self约1.502秒。
前一轮同材料观察为22905次mm和100590次copy；算术修复同时影响求解成本，不作单因素提速归因。
仅34/479次State进入这次私有矩阵批，调度、Query、装配和反向仍明显；不能仅凭最大批32就称整图充分打包。
profile完整对照同样严格失败，不进入通过的吞吐对照。trace为864037170字节，进程RSS峰值17403076608字节；
profile的采集/事件分析约4分钟，开销独立于非profile训练窗口。

单卡NPU物理1、纯Python mixed-c的同一HB medium fixture（哈希两侧相同）为1874.480→1988.877ms，
前后完整返回/参数更新对照均通过；同128候选/128 Full/192消息、最大私有批2行。
这是单次约6.1%的变慢，没有获得设备端到端提速。构造allocated两侧同为538624字节；
测量peak allocated由38882304增至39332864字节，reserved峰值同为41943040字节。
测量结束的live allocated略降不代表峰值下降。不得与前一轮resident存储收益混写为同一路径的结论。
追加SDK `fc/affine` FP32在step2再次出现同一evidence最大差3.09944e-6，原严格限制仍成立。
CPU和NPU总体study均为failed，各内部通过配置单列；没有把故意保留的失败诊断改成成功作业。

## 保留的数值和硬件边界

ranked64 的完整记录仍在 Python/native FP32 下失败：native 首个见证为 node86 KV keys、最大差
4.11272049e-6。两步诊断中离散结构相同，但输出 loss 的首步 base 梯度最大差约
.00188947（Python）/.00204515（native），SGD 参数也严格不同。显式提升同一FP32材料到FP64后，
两条独立调度的完整返回、梯度和两步参数均通过。它仅支持该例的有限精度敏感性诊断，不能解除FP32失败。

原 eager FP16 GDN/AdamW、默认异步双NPU纯Torch反向、SDK FP32 affine及反馈PDG严格失败分别保留；
详见[数值隔离](core5-numerical-isolation-20261028.md)和前一轮报告。
CUDA真机、x86_64、大规模设备完整训练/跨卡吞吐仍待独立资格。

新增 BMM 仅覆盖合法共享 base/输入组中的同形私有矩阵；不同 base/输入、私有状态写入、
全图物理行和逐窗口反向构造仍有优化空间。真实合批及局部手算通过不等于所有低成本优化已完成，
也不解除大图的严格正确性门禁。后续先取得失败分配的实际输入见证，再决定数值修复；不根据共同错误字串
断言所有拒绝同因，不放宽容差或删除拒绝。

## 公共复现入口与资源

按[开发入口](../development.md)在固定提交构建对应 core，再运行：

```sh
TIDE_BUILD_DIR=CPU_CORE python -m pytest -q tests/test_weighted_quad_offset.py \
  tests/test_weighted_private_projection.py tests/test_weighted_kv_sharing.py \
  tests/test_weighted_compute_sharing.py tests/test_cpp_weighted_training.py --dtype both
python scripts/build_weighted_benchmark.py --core-build CPU_CORE --output-dir NEW_BUILD
python scripts/benchmark_weighted.py --device npu:0 --implementation python \
  --placement mixed-c --topology hb --size medium --steps 1 --output-dir NEW_MIXED
python scripts/verify_device_control.py --build-dir DEVICE_BUILD --output-dir NEW_CHECKS \
  --device npu:0 --checks weighted-allocation weighted-flow weighted-flow-config weighted-training weighted-node-training
```

完整574项测试清单在同名JSON。可扩展材料的生成、二进制导出、standalone及两步dtype诊断命令见
[workload契约](../weighted-workloads.md)和[前一轮复现](core5-performance-transfer-20261028.md#复现与边界)。
baseline/current各固定干净源码，读取同一输入或相同种子fixture；不必保留旧绝对运行目录。

所有长任务 background.slice/Nice10、算子单线程、build2；CPU测量总70min，逐进程通常32GiB/300—360秒，
matched2048为192GiB/1200秒；RLIMIT_AS实际应用于CPU子进程。该大上限还覆盖事后独立参考/参数副本，
不能当候选自身的需求。NPU测量总20min、队列180秒、设备工作900秒、每case240秒、MemoryMax32GiB。
项目重型测量串行且开始于资格构建终态之后；其他用户争用未排除，systemd属性不表示物理资源独占。
CPU/Python为Torch2.12；standalone NPU为LibTorch-NPU2.10/CANN9.0，mixed为Torch-NPU2.12/CANN9.1。
