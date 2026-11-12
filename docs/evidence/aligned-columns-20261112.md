# 对齐消息/区域列：固定资格与FC16成本

实现：`129bbbdce4aea56367561f64ba5ec76f6e370a0c`，数学固定`0770593`。
它删除已对齐行域的补零/恒等index_copy，以及全正质量行的重复gather/scatter。
物理重复边仍逐条参与；只在实际行域相同时直接交接，稀疏行域保持真实索引。
本报告不包含后继节点组/区域数量分组，也不认证新resident执行器。

## 固定资格

在干净冻结源码上新构建Python-owned NPU native库：CANN9.2.0，Python3.12.13，
Torch2.12.0+cpu，TorchNPU2.12.0.post3，aarch64/GCC10.3.1，C++17/ABI1，320构建workers。
C++源码hash `37c39a9c0ca8ca320519c3be69c4c1daa2179ca8ebb025febbd2669efef94089`；
`_tide_native.so` SHA256 `ba08b9cd4a9a93a7bb99bfc54613479c4ec22a0b28230b89f262eb17e0dc7189`。

- CPU FP32/FP64定向：99通过、9项原严格数值xfail。范围为column assembly、一般Settle端口、
  公共window、共享、放置和实验接口。独立逐纤维公式检查同类重复原子、同值独立根、全零质量及None。
- 单空闲NPU物理11映射npu:0：half独立检查15通过。
- 安装消费者六项：Python/native × KV/SSD mixed-c，以及native KV位置 mixed-a/mixed-b。
  每项三步AdamW、独立数值/离散比较及新进程精确恢复通过。示例显式deterministic=true；
  不以此认证普通异步双卡、完整规模或全部dtype。

`gef-rewrite-aligned`和`gef-rewrite-aligned-device`均终态passed；MainPID=0、ControlGroup空。
两任务在独立background.slice，构建768GiB/3600秒；设备任务CPU144–159、256GiB/2400秒。
设备准入只使用空闲卡，未使用共享卡。正式计时与本任务重型构建错开；不声称排除整机外部争用。

## FC16：默认native张量窗口完整训练

同固定参数与下游`0439efd2`材料，B8/L128/d128，16节点/4区域，KV qry，
2头、summary32、adapter32、FFN512，regional_total/quad，cat/ind/context/cat，FP32。
核心参数4,564,404，含实验embedding/head共5,092,788。
初始/测量参数hash `7f41057f0196bc2b259902f8dadafc9c3f46d197e02de5ebbda9517a632f9a1a`。
入口`benchmark_dense_query.py --baseline window --client native`，生效tensor-window-v1/settle-columns-v1，
主机调度；NPU为mixed-c。CPU线程1，NPU主机线程2，interop1。

| 位置 | 完整步秒 | 输入位置/s | candidate | Full | 内部边消息 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1.272193 | 804.91 | 16384 | 8344 | 25060 |
| NPU | 0.617622 | 1657.97 | 16384 | 8345 | 25064 |

吞吐为1024/完整步耗时。一次独立测量进程、一warmup、一测量步，warmup后恢复参数/optimizer；
计时含输入创建/上传、默认forward与raw续接导出、loss、backward、finite、clip、optimizer、清理及同步。
`phase-fences=step`的阶段数只是主机提交区间，不能当NPU阶段耗时或相加归因。
没有测TTFT，不由prefill耗时推导首token。此性能点未做完整尺度独立数值资格，
CPU/NPU实际Full/消息不同，不能称严格等价加速；小型训练/恢复资格单列于上节。

## 复现与材料

从上述实现提交检出，按[开发入口](../development.md)构建匹配库，运行：

```sh
TIDE_BUILD_DIR=BUILD python -m pytest -q tests/test_column_assembly.py tests/test_settle_columns.py tests/test_window_execution.py tests/test_column_sharing.py tests/test_column_placement.py tests/test_experiment_columns.py --dtype both
python scripts/benchmark_dense_query.py --downstream-repo LAB --device cpu --client native --baseline window --trace-records minimal --mode training --phase-fences step --batch 8 --length 128 --native-library BUILD --output-dir NEW_CPU
```

NPU须合法准入；将device改为npu:0并固定threads2。安装后的`examples/consumer_core5/tensor_window.py`
用上述family/client/placement组合复现训练和恢复；详见[张量窗口契约](../weighted-tensor-windows.md)。
原始记录：`artifacts/jobs/rewrite-aligned*`、`artifacts/rewrite-20261112/aligned-{cpu.xml,npu-half.xml}`、
`aligned-installed-*`和`aligned-fc16-{cpu,npu}/result.json`。复现需要源码/声明工具链与固定下游Git对象，
不要求保留原build、运行目录或安装目录。旧数值、resident和异步双卡失败没有因本报告解除。
