# TL-REQ-004 交付：显式密集查询与完整训练成本

2026-11-10。接受并交付首个限定范围的 `query_execution="dense-v1"`：
把Read/Describe及Full/FFN的密集投影从KV缓存长度分组中拆出，扩大实际矩阵批。
原默认legacy和严格阈值保持。小图独立训练/恢复、原任务有限成本、数值失败分别报告；
本次不是原语言训练或下游验收。权威[请求与来源](https://github.com/ZichaoLong/tide/blob/0439efd2af31cf10b07ce9e6423b0195f7449968/docs/requests/TL-REQ-004.md)
仍在提出方，基座回复为本页及[机器可读证据](dense-query-tl-req-004-20261110.json)。

## 来源、兼容与安装物

| 项目 | 固定身份 |
| --- | --- |
| 数学 | `0770593838a37c790a568226930567ed60f99b83`，19份core-5快照核验通过 |
| 下游任务/配置 | `0439efd2af31cf10b07ce9e6423b0195f7449968`，`configs/studies/language/v6/scale-fc16-kv-quad.json` |
| 请求SHA256 | `04b9d022a3efe2a01ae058d571033a77b1c74cae94269f4d5a2be6a451a85ae9` |
| 基座起点 | `74e7bbd62341d2ccd382194a2c52ad4fea033665` |
| 实现/构建/安装/固定资格 | `75a502790985044b12125bee83d7f0dddfa39079` |
| 成本/舍入诊断驱动 | `26ca1b6b32c43adfb78434b6402498dc4c53e40b`；与75a5027的包和C++源码相同 |
| wheel | tidegraph0.3.0，SHA256 `b4d11023ccfb3d27f9ae2675c236b376f30966cb519c21223bbaa3b8392e7526` |
| 已安装Python文件集合 | `f825c6288f6e4352b06ed54af6f0c6d9ca8fde0fdb530e7a059abec3cad102cb` |
| C++源码集合 | `9e055f0a652bc73bf6f37e7e8295619314a52daf9ff967a17bb24758e84edfce` |
| Python-owned NPU adapter | `_tide_native.so` SHA256 `2bf972e3661706eb5f129fb0d381ab94ab067e01c37b7ca3f2d910b870243af8` |
| 安装锁 | `2a585cfe668c7376e373c124c32b3ed844f280f45bc586c07fcb71ebf2865336`；wheel内容匹配及lock verify通过 |
| 环境 | aarch64，Python3.12.13，Torch2.12.0+cpu，TorchNPU2.12.0.post3，CANN9.2.0，GCC10.3.1/C++17 |

公共入口是`weighted.execution.eager_runner(..., query_execution="dense-v1")`、`Native(...,
query_execution="dense-v1")`及`ExecutionSpec.query_execution`；配置驱动eager接入同一选择器。
[域、数学对应与实现符号](../weighted-dense-query.md)：单payload设备、FP32/FP64、packed prefill、
一个host worker；KV/qry、cat/ind、regional_total、context/cat、quad、strict正支持。
不合域显式拒绝。Python执行视图保留原参数owner；C++在placement前选择底层kernel。

新执行身份区别于legacy，旧默认身份不变；模型/图v13、checkpoint v5及固定数学不变。
checkpoint仍精确保存实际状态，不用有效投影或参考重算补写。
应用必须固定执行身份；跨模式续接未认证，不能只因图身份相同便视为精确恢复。
应用存储介质、保存间隔和发布策略不变。本消费者使用磁盘小文件，恢复于更新边界；
没有新增大文件codec、流式checkpoint或resident任务接入。

本次新构建是Python-owned NPU库，同时执行CPU张量；不是standalone SDK资格。
C++公共选择函数已实现、核心不依赖Python调度；standalone benchmark和旧study命令生成器
显式拒绝该非默认选项。CUDA、resident、FP16、跨设备、其他语义组合不在本增量资格中。
未修改下游安装、运行、依赖锁或任何参考仓库。

## 独立正确性、训练和恢复

| 实际范围 | 结果 |
| --- | --- |
| CPU固定资格 | 27项：25 passed，2项保留的FP32严格数值xfail；FP64/FP32 × Python/native |
| NPU固定资格 | 15 passed：13项实际NPU FP32，2项CPU数学/配置辅助；不能把15全算设备测试 |
| 局部独立根 | scalar公式对比ragged Read/Full，逐字段/行根、共享参数/输入、None/连接零、未使用AdamW参数、版本守卫和一阶限制 |
| 独立数学锚点 | CPU FP64的evidence方向差分；只验证该方向与局部映射，不冒称任意图解析证明 |
| 完整小图 | FC/HB/反馈PDG、B2、宽2、三步AdamW；独立时间主序参考与候选各自调度，核对完整返回/状态/控制/history/pending/消息、独立descriptor根、参数和optimizer |
| 非空图续接 | 保留autograd图的cut独立参考；磁盘pending/raw int64状态、新owner下一窗/梯度/optimizer逐值恢复 |
| 四个安装消费者 | CPU/NPU FP32 × Python/native：独立三步完整训练严格通过；whole3 vs prefix2+resume1三个新进程的最后完整观察/梯度/参数/optimizer逐值一致 |
| 默认路径相关回归 | 冻结开发源dev-a的192项通过；另8项CPU FP64新路径通过，保留开发源身份，不冒称固定提交或全矩阵 |

消费者每次更新从空图continuation开始，验证更新边界的跨进程恢复；非空pending图恢复由上表
另外的完整图测试验证。这两种证据不合并成“原FC16跨进程长窗已通过”。
固定资格脚本允许显式记录有限浮点超差；离散/结构/None、非有限、执行异常及逐值恢复失败
仍终止。外层服务passed表示声明流程完成，**不把内部两个strict失败改为通过**。

CPU FP32见证是`0.7 * row1.evidence.square().sum()`的独立根，
evidence_weight/bias最大差分别为`1.0435469448566437e-6` / `1.0436633601784706e-6`，
原`atol=1e-6, rtol=1e-5`保持；其余字段及None结构通过。Python/native各保留一个xfail。
NPU同一局部见证通过不消除CPU失败。

有界三方诊断将**相同FP32权重/输入**提升至FP64 scalar；evidence_bias梯度相对该锚点的最大差：
scalar FP32为`9.059958794290987e-7`，dense FP32为`1.376674807493719e-7`。
该见证支持近零归一化伴随的舍入放大解释，说明scalar FP32本身也有误差；
不放宽原检查，也不能解释或认证下面的大用例梯度/离散差异。

## 原任务成本与实际工作量

同一新安装/构建分别执行legacy/dense-v1；所有成本流程都是Python调用native，CPU或单NPU mixed-c。
FC16为4区域×4节点，宽128、summary/adapter32、2 heads、FFN512；
KV/qry、read_mode+、cat/ind/context/cat、regional_total/quad、温度.25、k=2、gamma_max=2、
按region共享。history512、stride6，非终端KV窗口3072，终端区域窗口0。
core可训练4,564,404，总5,092,788；原4096词表embedding/head、全位置CE、AdamW不改。
FP32，确定性策略显式开启；CPU ATen1线程/NPU2 host线程，scheduler1。
优化器eps1e-6、weight_decay.01、clip_norm1；沿用原lr warmup，synthetic seed991+step，模型seed11。

每配置一个独立进程，一预热更新后一步正式计时，B8/L128、1024 token。
每阶段同步的phase_sum与原诊断边界一致；完整墙钟另含输入生成/上传、有限值检查、清理与同步。
core前向含public runner构造与完整native返回。无trace关闭/紧凑返回优化，未改变目标位置。
一次性模型构造和预热单列；周期checkpoint、验证集、日志I/O不在本次训练步计时内。
因此token/s是该有界诊断步成本，不能直接当36小时持续训练吞吐。

| 路径 | core前向(s) | 反向(s) | phase sum(s) | 完整墙钟(s) | token/s | Full / 内部消息 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU legacy | 26.009 | 36.099 | 62.274 | 62.839 | 16.30 | 8352 / 25068 |
| CPU dense-v1 | 18.999 | 25.463 | 44.582 | 45.162 | 22.67 | 8355 / 25072 |
| NPU legacy | 128.096 | 123.658 | 252.143 | 252.922 | 4.05 | 8350 / 25108 |
| NPU dense-v1 | 98.911 | 83.393 | 182.644 | 183.549 | 5.58 | 8351 / 25112 |

每项candidate均16,384。完整墙钟单次减少CPU 28.1%、NPU 27.4%；仍未达到约32.4 token/s。
NPU新路径仍慢于本CPU配置。CPU峰值RSS9456→8744MiB，NPU峰值allocated6096→5994MiB、
reserved6184→6116MiB，NPU主机峰值RSS4910→4109MiB；一次流程中的峰值，不作通用容量承诺。

严格数值、大用例完整离散核对和恢复均未检查。实际Full/消息计数不同；CPU第二步
裁剪前梯度范数legacy约3267.39、dense约26741.89，NPU对应为6569.64→1391.59，也有明显差异；不能用相近loss掩盖这种差异。
这些是有限成本观察，不能叫严格等价加速，也不能据此批准原语言训练切换。
warmup参数更新之后的测量步不保证两条路径仍具有相同权重值；JSON保留每步loss、范数和实际工作量。
下游学习效果与接入验收仍未执行。

## 小型NPU profile与剩余瓶颈

B2/L8同模型；前两个更新不插桩，第三步另加host/device profiler。
无插桩测量步legacy/dense完整墙钟5.053/3.226秒，candidate256、Full128、消息384。
插桩第三步两者candidate256、Full130、消息388；不能把两步混算。

| 实际第三步计数 | legacy | dense-v1 |
| --- | ---: | ---: |
| device kernel记录 | 168589 | 135504 |
| `aten::matmul` / `matmul_backward` | 1823 / 1873 | 596 / 646 |
| `aclnnMatmul` API记录 | 5569 | 1888 |
| `_local_scalar_dense` | 7308 | 6666 |
| `aten::index` / `_index_put_impl_` | 6875 / 3603 | 6233 / 3603 |
| `nonzero` / `bmm` | 804 / 320 | 804 / 320 |
| device AICPU Index / IndexPutV3 | 1170 / 1170 | 1170 / 1170 |

原FFN up/gate矩阵`[512,128]×[128,1/2]`分别120/70次，down单独60/35次；
新up/gate共32次、down16次，实际批为2、4、5、7、8、9、10、11、15行。
这验证密集阶段确实越过缓存长度桶，不只是调度报告最大batch变大。
NPU二维矩阵操作记录为`aten::matmul`，只搜`aten::mm`会只看到两个任务head反向操作。

厂商设备Index与IndexPutV3累计约131/119ms→131/120ms，未消除；AICPU是设备处理单元，
不是host CPU fallback。索引反向、nonzero、拆装和剩余host判断是具体后继候选。
保留私有VJP分支、实际长度/正支持分桶；本次没有统一padding、attention索引重写或专用核。

插桩timeline的Computing/Free/Stage为.681/8.203/8.883秒与.606/4.248/4.855秒；
只能解释本次trace，不能充当正常利用率、可相加的排他耗时或无插桩吞吐。
weighted projection计数927→468也只覆盖该helper：新Describe进入它，旧scalar头不在其中，
不能据总数推断全部矩阵工作量。实际形状以原CSV及JSON摘录为准。

## 作业、资源与材料

| 服务 | 实际源码 | 终态 |
| --- | --- | --- |
| `gef-tl-req-004-build-a` | 冻结dev-a，74e7bbd+记录的开发差异 | passed/exit0；开发构建、8新项和192回归 |
| `gef-tl-req-004-check-b` | 冻结dev-b，74e7bbd+记录的开发差异 | passed/exit0；数值xfail单列 |
| `gef-tl-req-004-qualify-c` | 干净75a5027 | passed/exit0；上述固定资格和四个消费者 |
| `gef-tl-req-004-cost-d` | 干净26ca1b6驱动，75a5027包/native | passed/exit0；六个子流程completed |

最终四服务MainPID=0、ControlGroup为空且cgroup不存在；两份NPU队列记录completed/return_code0。
最后成本服务于2026-11-09 16:28:08 UTC终止。日志/状态hash及每个子结果独立保留于JSON。
TorchNPU profiler提示stop时处于RECORD；两个窗口均结束并解析出CSV/trace，审阅记录了实际范围。
这些profile计数是诊断观察，不作为严格正确性或正常吞吐资格。

固定资格新构建使用全部320 affinity CPU；MemoryMax512GiB、≤3600秒。
性能独立background.slice、KillMode=control-group、Nice10、MemoryMax96GiB、≤3600秒，
CPU/NPU六流程串行；单NPU队列maxwait600秒，资格physical1、性能physical3映射logical0。
实际设备Ascend910、64GiB/chip，npu-smi/driver25.3.rc1；无跨卡通信。
共享主机约99/102/106 load、320CPU且有其他设备作业，未排除外部争用；未停止其他项目。
原始产物约2.1GiB，低于本轮10GiB预算；记录/CSV/trace/checkpoint留ignored artifacts，
小型审阅结果、来源hash和可再生成入口入Git。旧目录只是位置提示，不是接续前提。

## 下游接入与复验

**基座首个增量已交付，tide-learning-lab尚未验收。** 在新的私有安装使用26ca1b6或包含它的
交付提交；先按[开发入口](../development.md)选择匹配公共栈和固定干净checkout。
以下是workload命令，长构建/测试仍由本站background.slice/设备队列运行：

```sh
# 调用者设置新的环境、wheel、native库与输出目录，不覆盖活动实验安装。
: "${TIDE_VENV:?新的私有环境}"
: "${TIDE_WHEELS:?新的wheel目录}"
: "${TIDE_CORE:?新的native构建目录}"
: "${TIDE_OUTPUT:?源码checkout之外的新资格与锁目录}"
mkdir -p "$TIDE_OUTPUT"
python -m venv --system-site-packages "$TIDE_VENV"
source "$TIDE_VENV/bin/activate"
python -m pip wheel --no-deps --no-build-isolation . -w "$TIDE_WHEELS"
python -m pip install --no-deps "$TIDE_WHEELS/tidegraph-0.3.0-py3-none-any.whl"
python scripts/build.py --backend npu --npu-runtime python --jobs all --build-dir "$TIDE_CORE"
python -c 'import json,sys; from pathlib import Path; print(json.dumps({"paths":{"native_library":str(Path(sys.argv[1]).resolve())}}))' "$TIDE_CORE" > "$TIDE_OUTPUT/native-site.json"
tidegraph lock create --site "$TIDE_OUTPUT/native-site.json" --wheel "$TIDE_WHEELS/tidegraph-0.3.0-py3-none-any.whl" --output "$TIDE_OUTPUT/installation.lock.json"
tidegraph lock verify "$TIDE_OUTPUT/installation.lock.json"
bash scripts/qualify_dense_query.sh "$TIDE_OUTPUT/cpu" cpu "$TIDE_CORE"
# 在分配到单NPU的队列进程内：
bash scripts/qualify_dense_query.sh "$TIDE_OUTPUT/npu" npu "$TIDE_CORE"
```

安装环境和锁/检查输出须位于资格checkout之外；构建与wheel也选外部或Git忽略目录，
不要让生成文件使资格源码变脏。新构建wheel/binary具有自己的hash，保留构建来源关系。
纯CPU安装可改为`build.py --backend cpu`，但本轮不是这套独立CPU构建的资格。
公开任务接入只需在已有eager adapter传入`query_execution="dense-v1"`，或配置
`execution.query_execution`，保持任务数据/head/loss/优化器。无需导入benchmark内部代码。
原任务尚未暴露该配置字段；本次成本脚本只在自己的只读Git导出副本用调用adapter传入，没有修改下游。

不依赖下游仓库的小型公共复验命令：

```sh
python examples/consumer_core5/dense_query.py --device cpu --client native \
  --native-library "$TIDE_CORE" --output-dir NEW_CONSUMER
python scripts/diagnose_dense_rounding.py > NEW_ROUNDING.json
```

成本复现需要只读的固定下游Git对象，不需语料或旧run目录；输出目录必须全新。
在一次独立有界流程内分别执行legacy/dense-v1，CPU与NPU串行。正式测量不用profile：

```sh
python scripts/benchmark_dense_query.py --downstream-repo "$TL_REPO" --device cpu \
  --query-execution dense-v1 --native-library "$TIDE_CORE" --output-dir NEW_CPU_COST
# 单NPU队列内，将device换为npu:0；小profile另加 --batch 2 --length 8 --profile npu。
```

先核对安装锁→独立基座消费者→原任务小型接入/恢复→有限真实任务，再由实验方决定新锁及训练。
数值/离散/精确恢复与学习指标分别验收；不凭服务退出码或本页成本替代。
剩余工作和重启条件仅维护在[ROADMAP](../ROADMAP.md)，通用经验见
[performance-lessons](../performance-lessons.md#tl-req-004密集查询与异形记忆分离)。
