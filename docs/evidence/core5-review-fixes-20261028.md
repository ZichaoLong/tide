# Core-5 审查修复资格 — 2026-10-28

实现及检查源码固定为 `1d19c5998b96689b3620ecb69c7f712a9dd84ad0`，
修复前基线 `2c46aa7d0b77afe7a7b82ed428466e36f5162de4`；数学仍固定
`0770593838a37c790a568226930567ed60f99b83`。图 v13、checkpoint v5及指定一阶VJP身份不变。
[同名JSON](core5-review-fixes-20261028.json)保存源码、构建、二进制、loader、
命令、测试XML和日志哈希、任务终态及NPU映射。
本报告补验三项审查修复；主体能力及其他严格限制从[路线图](../ROADMAP.md)进入。

## 修复与独立证据

| 缺口及影响 | 修复入口 | 核验方式及边界 |
| --- | --- | --- |
| ClockedState包装使带权参数未被批量VJP捕获，且未透传局部时间分支/准备数据 | `weighted/parameter_tree.py::supported`递归捕获；`ClockedState.batch_autograd_key`按局部时钟分组；`state_batch.py`透传prepared | `test_weighted_clocked_vjp.py`：五记忆、独立状态/输入/参数根、None/零、包装内参数版本、KV及共享delta三步SGD；参考独立scalar调度。局部数学锚点沿[State VJP](../weighted-state-vjp.md)进入；该新增组合检查是Python范围，不单独认证native时钟组合 |
| quad的已下溢响应与残差同时为零，会丢失数学正支持 | Python/C++ scalar/batch在正响应分子且正上限时显式拒绝零控制 | `test_weighted_quad_support.py`：dyadic精确输入的独立闭式解`[1,tiny/2]`；另外验证最小可表示正数、真实零、零上限和全零单点。不是人为最小发送阈值；[分配契约](../weighted-allocation.md) |
| resident参数bank可能先展开/转换/设备拷贝，随后才发现预算不足 | `tide::copy_parameter_bank`分配前检查并逐角色扣减FP32额度；来源表/填充和Full输出先预留 | `cpp/test/parameter_bank.cpp`：手算字节、巨大广播视图、不可用目标、非连续切片、half拓宽和独立存储；`weighted_flow_boundary_check.cpp`：实际设备上的104字节Full参数组、差一字节拒绝、巨大视图提前拒绝及全图预算；[图预算契约](../resident-weighted-flow.md) |

修复前在2c46aa7上运行新增Python测试，19失败、4通过、8个native用例未执行；
测试文件哈希与修复后相同。首次baseline命令被pytest的pythonpath配置导向当前源码，
不作为旧版证据；修正后显式指定基线pythonpath的命令/日志另存于JSON。
修复后的新增定向测试为31项；完整相关CPU门禁为442项，不将二者重复相加。

## 实际通过范围

| 后端/工具链 | 资格结果 |
| --- | --- |
| aarch64 CPU，Torch2.12/Python3.12.13，FP64/FP32 | 442项，失败/错误/跳过均0；包含Python/native分配、时钟/状态VJP、熵分配回归和两dtype C++自定义kernel检查 |
| standalone LibTorch-NPU2.10/CANN9.0，Python3.11.15，Ascend910_9392 | 四个组件×FP32/FP16，共8组真机门禁通过；SDK核心的CPU FP32/FP64检查另通过 |
| CUDA工具链，Torch2.10+cu128/Python3.11.15，aarch64 | CPU定向31项；sm80/90/100的四组件构建、DSO/loader及help通过；没有GPU真机结果 |

NPU每个dtype分别验证21个完整图案例、15个正式配置、单设备五配置20次更新，
以及双卡节点放置五配置20次更新；另含累计、数值拒绝及恢复检查。
训练配置为FC/KV/SGD、HB异构/AdamW、非对齐delta/SGD、反馈异构/AdamW、全冻结FC/affine。
每次更新包含三个窗口（一个为空）；输出、连接零、全部字段分别种根。
原始状态、pending、逐边/端口、控制、共享参数/别名、optimizer及换放置恢复分别比较。
参考使用独立CPU Greedy/autograd及NamedOptimizer；候选不接收参考实际轨迹。

FP32完整记录沿用rtol=1e-5、atol=1e-6；FP16记录对照显式FP16存储/FP32算术参考，
rtol=.02、atol=.001。训练梯度/master为3e-5/3e-6，named参数为1e-5/1e-6。
离散身份、int64、存在/连接位和None/零保持严格比较。
恢复时有效字段逐位相等，保留过期raw条目；日志中的未用journal/padding差异不属于
`present/valid`定义的逻辑字段，见[公共训练契约](../resident-weighted-training.md)。
没有通过调整断言、容差或输入来取得本次通过。

## 源码、失败与运行边界

核心构建来自冻结的2c46aa7加修复补丁，完整C++源码哈希
`b6be97fcafa20f4f3db84c0cacbf247ced9ac6b12ec017f2ffc31401c8ea7bd8`
与1d19c59一致；复用前后逐一核验二进制哈希。原归档身份保留。
该开发快照没有最后的Full输出预留，因此NPU/CUDA组件从干净1d19c59资格快照重新构建，
组件哈希`9ef5b749d42a11041df7175aa6d6a60f6f15014d5eb66354c081510b8a70b6c2`。
全部正式检查读取同一只读资格快照，不把dirty开发记录伪称干净源码资格。

首次NPU资格任务完成构建后，在第一项CANN初始化失败（E40001、exit1），没有通过设备用例。
启动器把9.1私有venv放在PATH前部，却使用SDK的3.11解释器启动；CANN调用的
`python3-config`未选到SDK配套版本。新任务仅修正PATH，将SDK配套Python bin置首，
复用完全相同的源码/二进制，完成上述8组。原失败记录、日志及哈希保留；公共环境未修改。
三项早期组件dev任务在等待依赖时取消（exit143），没有编译/设备资格，不改称通过。

任务使用独立background.slice、Nice10、最多两个构建线程和单线程ATen/BLAS。
CPU资格900秒；原NPU/CUDA资格各5400秒；修正后NPU任务3300秒，队列至多180秒、
设备检查3000秒，unit上限3600秒，CPUQuota200%、MemoryMax24GiB。
NPU队列实际分配物理1/13为逻辑0/1；这些限制不证明设备或资源独占，也不是性能测量。
任务名称、时间和结果见JSON；仅在任务/队列终态、全部用例及哈希核验后发布本报告。
新NPU unit在终止后已回收，`systemctl show`的未加载默认属性不作为退出证据；
JSON区分会话中观察到的live slice/资源属性与终止后属性，完成依据为持久任务/队列退出码及实际验收输出。

## 从干净源码复现

在干净1d19c59检出上，按[开发](../development.md)激活匹配环境，
设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
`MKL_NUM_THREADS=1`。以下是workload；长任务须按[资格规范](../verification.md)包装为
独立持久服务，使用新的构建/输出目录，无需历史worktree、日志或二进制。

```sh
python scripts/build.py --backend cpu --build-dir build/review-cpu --jobs 2
TIDE_BUILD_DIR="$PWD/build/review-cpu" python -m pytest -q --dtype both \
  tests/test_weighted_clocked_vjp.py tests/test_weighted_quad_support.py \
  tests/test_weighted_state_vjp.py tests/test_state_clock_contract.py \
  tests/test_state_clock_schedules.py tests/test_state_clock_edges.py \
  tests/test_weighted_allocation.py tests/test_weighted_allocation_batch.py \
  tests/test_weighted_entropy_allocation.py tests/test_weighted_entropy_native.py \
  tests/test_cli.py::test_cpp_custom_state_kernel
```

standalone NPU另起SDK环境，确认`python3-config --prefix`与Python的`sys.prefix`匹配，
设置`ACL_OP_INIT_MODE=0`，构建后通过账户队列映射两张可用卡到逻辑0/1：

```sh
python scripts/build.py --backend npu --npu-runtime standalone --build-dir build/review-npu-core --jobs 2
python scripts/build_device_control.py --core-build build/review-npu-core \
  --build-dir build/review-npu --jobs 2 --ascendc-soc Ascend910_9392 \
  --checks weighted-flow weighted-flow-config weighted-training weighted-node-training
python scripts/verify_device_control.py --build-dir build/review-npu --device npu:0 \
  --output-dir artifacts/review-npu-checks \
  --checks weighted-flow weighted-flow-config weighted-training weighted-node-training
```

CUDA另用匹配工具链构建`--backend cuda`核心，组件选择
`--runtime standalone --cuda-architectures '80;90;100'`，不传Ascend SoC。
相同两份新增pytest在CPU运行；四个检查器的`--help`只检验加载/入口，不是GPU执行。

本次未新增性能测量或Python-owned NPU资格；不重跑历史完整矩阵。
CUDA真机、x86_64及既有GDN FP16 AdamW、异步双NPU eager、SDK affine、FP32反馈PDG
严格见证继续保留，详见[路线图](../ROADMAP.md)与[精度契约](../weighted-precision.md)。
