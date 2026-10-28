# FP32 GDN/SSD 归一化反向资格 — 2026-10-28

实现固定`ad6d1e71d8798a68765e0efa6f8192c00bfc5a94`；数学固定
`0770593838a37c790a568226930567ed60f99b83`。图v13、checkpoint v5、输入、容差及optimizer默认值不变。
[同名JSON](core5-fp32-normalization-20261028.json)记录精确命令、来源、构建/日志hash、终态及设备分配。
本次是受影响函数的定向资格，没有重跑全部历史矩阵。

## 修复、对应和独立依据

固定教材GDN §1 epsilon-L2及GDN/SSD §3 scaled-RMS，由
[精度契约](../weighted-precision.md)、[循环契约](../weighted-recurrent.md)和
[语义索引](../semantic-map.md)定位到`python/tidegraph/weighted/normalization.py`、
`cpp/src/weighted_normalization.cpp`。
FP32现在使用已用于FP16的解析一阶收缩：令`n=x/d`，
L2伴随为`(b-n*sum(b*n))/d`，scaled-RMS把`b`换成`b*rho`并用mean；
rho根为`b*n`，归约其所有广播使用者。它与实数Jacobian相同，不是近似求解、HST或二阶承诺。
FP32前向表达式逐值不变；没有提升到FP64、调整AdamW epsilon或放宽检查。
原FP16/FP64策略、int64元数据、版本检查和None/连接零保持。

`tests/test_weighted_half_normalization.py`及`cpp/test/weighted_normalization.cpp`
直接按实数Jacobian逐元素构造期望，覆盖零/小量/一般向量、广播scale和FP32逐值前向。
Python另验版本、None/连接零与拒绝二阶；循环公式/方向导数提供不同的局部锚点。
完整记录、独立梯度根及共享参数由State/Read/Full批量VJP和独立调度测试核验，
局部Jacobian通过不替代完整训练或恢复。

## 固定源码结果

| 入口 | 结果和范围 |
| --- | --- |
| CPU Torch2.12/Python3.12，aarch64 | **453 passed、2 strict xfail、0 failed**；FP32/FP64相关循环、packed、State/Read/Full VJP、完整记录、每切面恢复、optimizer；包含两个原FP32 GDN失败用例 |
| 独立C++ CPU训练 | 上行包含每dtype 24图×4次更新及真实checkpoint/optimizer恢复；FP16显式master/SGD和限定AdamW用例另列，不能推为全部half AdamW通过 |
| LibTorch-NPU2.10/CANN9.0 | 增量core构建及CPU FP32/FP64核通过；真机FC/HB×GDN/SSD四组FP32，每组4次AdamW、完整记录/梯度/槽、原始状态精确解码、同设备续接及CPU交接通过 |
| 纯Python Torch-NPU2.12.post2/CANN9.1 | mixed-c的GDN/HB与SSD非对齐DAG两组FP32通过；各7种返回根、60/30个独立浮点状态根、四次AdamW、streaming/prefill及实际恢复/CPU交接 |
| CUDA2.10/cu128 | 同源core构建、绑定链接及CPU FP32/FP64核通过；**无GPU真机证据** |

CPU二进制复用已终态开发缓存。其manifest原基提交1850434保留不改；资格运行前后均核对
完整C++源码hash`8c942fa686ed276204b3bd5a04a026513a3ca21d0a480506c2ae3ba32211d7b2`
及全部23个二进制hash与本次固定源码一致。Python测试源来自只读ad6d1e7。
两套SDK从只读ad6d1e7独立增量构建。没有把旧静态链接的benchmark消费者或resident核记为本提交重验。
Python/native NPU适配矩阵未在本增量重跑；CPU native、纯Python NPU及standalone NPU分别记录。

四个服务`core5-fp32-normalization-{qualify,npu,cuda,python-npu}-20261028`均passed/0，
退出后systemd已卸载transient unit；终态以持久status/exit code及验收产物为准。
启动时已核对PID、Transient及background.slice/control group。前三者64GiB/20min，Python NPU64GiB/10min；background.slice/Nice10，
build2/算子1。NPU合作队列priority0/count1/wait180s，实际SDK物理1、Python物理9，各映射npu:0。
Python任务没有设置launch blocking；这只是单卡通过，不解除双卡异步失败。
日志保留厂商int64 ArgSort转AiCPU的提示；未将时间转成FP32，也不宣称全部算子在AiCore执行。

## 历史失败与剩余限制

1f1ec57的原资格仍是572 passed/2 failed，旧1850434复现及80位Decimal更新见证仍保留在
[上一阶段证据](core5-stable-preparation-precision-20261028.md)。原两项FP32 GDN/HB/AdamW
现在在ad6d1e7通过原断言与恢复检查；新结果不回写旧作业。FP16的两项严格预期失败仍在，
没有把旧FP32失败改成xfail。

D64/D2048 KV FP32完整记录/训练、SDK FP32 affine、FP32反馈PDG和默认异步双NPU见证均未解除。
CUDA真机、x86_64、大型设备训练及跨卡吞吐待验；有限用例不能证明任意数值域都通过。
本轮不新增性能测量。KV不调用这两个函数，故保留上一阶段同材料FP64训练/参数匹配推理资格，
不重复约92.62亿参数测量，也不从归一化修复推导性能收益。
更广私有合批、设备小算子提交及resident窗口反向构造仍有优化空间，见[经验记录](../performance-lessons.md)。

## 从干净源码复现

检出ad6d1e7，按[开发入口](../development.md)选择匹配工具链；长命令按
[固定资格流程](../verification.md)置于独立持久服务，所有输出目录使用新名称。

```sh
python scripts/check_upstream.py
python scripts/build.py --backend cpu --build-dir build/cpu --jobs 2
PYTHONPATH="$PWD/python:$PWD/build/cpu" TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q \
  tests/test_weighted_half_normalization.py tests/test_weighted_normalization.py \
  tests/test_weighted_recurrent_formulas.py tests/test_weighted_recurrent_execution.py \
  tests/test_weighted_packed.py tests/test_weighted_state_vjp.py tests/test_weighted_read_batch.py \
  tests/test_weighted_query_vjp.py tests/test_weighted_query_training.py \
  tests/test_weighted_precision.py tests/test_cpp_weighted_training.py \
  tests/test_cli.py::test_cpp_custom_state_kernel --dtype both
```

NPU SDK使用`build.py --backend npu --npu-runtime standalone`；CUDA使用`--backend cuda`，
两者分别运行对应build目录的`tidegraph-kernel-check --device cpu --dtype float32|float64`。
SDK真机执行`tidegraph-weighted-training-check --device npu:0 --dtype float32
--case fc/gated_delta --case hb/gated_delta --case fc/ssd --case hb/ssd --output-dir NEW`。
当前Python NPU执行`python scripts/check_weighted_device.py --device npu:0 --implementation python
--placement mixed-c --dtype float32 --case gated-hb --case ssd-dag --output-dir NEW`。
设备须通过本机合作队列选择并记录物理/逻辑映射；不固定本报告当时使用的物理卡号。
上述公开测试与命令可重新生成证据，无需旧运行目录或.local驱动。
