# 局部合批与owner伴随汇合的固定资格

实现`f15c4a5b45ea8d74fde9e8cb21ce5fe04beb4b07`（owner一次装配）和
`3d88b0b3cd958fdbec02a801bc17d2a6e170bfff`（局部合批）；数学固定0770593。
[审阅记录](prefill-local-batches-20261110.json)保留源码、两套构建、测试、客户端、原配置、
原始hash和实际工作量。前一增量收益有限，后一增量减少了可观察的细碎算子。

3d88b0b将cat evidence改为共享GEMM后逐行归一化；结构化Read有限值按device/dtype和8MiB
目标分组检查，整组验证后才发布；分配逐行metadata校验保持，数值交给真实完整批求解器，
实际int64分支一次回读并复用于反向。没有删除有限值校验、更改求解算法或放宽阈值。
公式、状态及独立根对应见[语义导航](../semantic-map.md)、[KV查询](../weighted-dense-query.md)。

| 固定增量 | CPU流程 | NPU目标流程 | 安装消费者/独立SDK |
| --- | --- | --- | --- |
| f15c4a5 | 74通过 | 37通过 | Python/native各CPU/NPU三步独立检查与新进程恢复通过；SDK如下 |
| 3d88b0b | 281通过、4数值xfail | 68通过 | 同上，使用新包及新C++库 |

NPU流程包含CPU辅助公式/metadata测试，不能把流程总数当作全部在NPU上执行。
两增量均使用站点Torch2.12/TorchNPU2.12.post3/CANN9.2，分别新建Python-owned和独立SDK，
构建采用320个可用CPU，与计时隔开。独立SDK CTest3项及CPU FP64/FP32 kernel检查通过。
SDK的CPU FP64/NPU FP32 complete分别通过完整独立返回及16根、三步训练和新进程精确恢复；
outputs分别通过三步和精确恢复，独立数值为not_checked。SDK静态材料是FC4/d8/B2/P3、KV sum；
不能把它扩大为所有配置。Python/native独立局部qry、默认FC/HB/反馈PDG训练和切窗由上述定向门禁覆盖。

原CPU FP32指定evidence根最大差约1.04355e-6仍超过原atol1e-6/rtol1e-5。
逐行BMM候选也出现该差异，未采用复制权重批。原FP32材料提升到FP64的有界见证中，
标量最大误差9.06e-7、共享GEMM1.38e-7；这支持指定见证的舍入解释，不代表所有FP32误差已解释。
新增FP64字面公式、完整/部分/重复行导数和无关参数None检查，以及末行非法值/shape拒绝，
独立验证了此次合批。原4项严格数值失败保留为xfail，不使用它们认证等价吞吐。

## 原FC16同返回成本

固定下游0439efd、16节点/4区域、KV qry、cat/ind/context/cat、regional_total/quad，
B8/L128/d128、含原头5,092,788训练参数、原loss/AdamW；FP32，CPU1/NPU2主机线程。
每项一个进程，一次预热后重置参数和optimizer再测一步；初始/测量前参数hash相等。
这里全部为minimal返回，包含准备、前后向、packing、检查、clip、optimizer及同步；
不与旧complete返回混算。原任务完整独立数值、恢复及学习未验，本表是有限成本观察。

| 设备 | f15完整步秒 | 3d完整步秒 | 3d前向秒 | 3d反向秒 | 成本下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 22.301568 | 18.640405 | 6.124282 | 12.417434 | 16.4% |
| NPU | 31.023336 | 22.902820 | 9.347498 | 13.419598 | 26.2% |

同设备前后均candidate16384、区域批4；CPU Full8344/消息25060，NPU Full8345/消息25064。
设备间工作量不同；同设备梯度范数仍有有限差异（CPU1362.025→1361.972，NPU1359.880→1359.864），
不能将总数相等替代完整轨迹/梯度等价。NPU峰值allocated4210.89→4188.97MiB，reserved4308→4288MiB，
均含预热；共享主机无独占保证，不作单次测量之外的统计结论。

独立B2/L8 NPU profile：标量回读5549→4349，matmul640→400，kernel记录59897→54105；
cat1377→1393，index199、nonzero4保持。仍有大量where/copy和私有根装配，不能称低垂果实全部完成。
f15相对更早roots只将cat1601→1377，完整步没有明显改善，说明减少某类调用未必减少主要成本。
profile不计作B8/L128的计时，也不能把重叠task时间求和当作wall time。

## 干净复现

按[开发入口](../development.md)在固定提交建立私有安装和两套匹配库。
CPU在固定cwd、安装包环境运行（NPU按站点准入，设置TIDE_DENSE_DEVICE=npu:0）：

```sh
TIDE_BUILD_DIR=MATCHING_BUILD python -m pytest -q -o pythonpath= \
  tests/test_prefill_local_batches.py tests/test_prefill_guards.py tests/test_structured_records.py \
  tests/test_weighted_allocation_batch.py tests/test_weighted_region_batch.py tests/test_positive_support.py \
  tests/test_weighted_dense_query.py tests/test_weighted_query_layout.py tests/test_query_roots.py \
  tests/test_selected_boundary_execution.py --dtype both
python examples/consumer_core5/dense_query.py --device cpu --client native \
  --native-library MATCHING_BUILD --no-trace --output-dir NEW_CONSUMER
python scripts/benchmark_dense_query.py --downstream-repo READ_ONLY_DOWNSTREAM \
  --native-library MATCHING_BUILD --device cpu --trace-records minimal --output-dir NEW_COST
```

NPU不运行FP64；该流程使用上述列表中的test_prefill_local_batches、test_weighted_dense_query、
test_weighted_query_layout、test_query_roots和test_selected_boundary_execution，改为--dtype float32，
设置TIDE_DENSE_DEVICE=npu:0和TIDE_WINDOW_DEVICE=npu:0；CPU辅助明确保留。
SDK材料生成和`check_query_sdk.py`复用[完整复建步骤](prefill-default-20261110.md#可独立复现)，
分别指定`--return-mode complete|outputs`；NPU用float32。profile用B2/L8及`--profile npu`，
再用`summarize_query_profile.py`只读汇总。材料只含静态参数/输入，不向候选提供参考轨迹。
