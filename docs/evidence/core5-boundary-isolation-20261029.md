# 首次写入、半精度反向与完整双卡桥接隔离

固定诊断源码`bdd7f6c6621a641e41b9fdf07d55c4749b5c5e86`，固定上游0770593。
[审阅记录](core5-boundary-isolation-20261029.json)保存来源、实际操作数哈希、结果与失败终态。
本报告只认证这些诊断，不认证尚在实施的singleton/字段packing/resident优化。

## 原D2048首次写入

取原92.62亿主体参数材料的第一个仅有外部输入区域，32节点共享真实输入/来源参数及base，
私有适配器保持各自身份。仅读取101个实际操作数、337133572字节，逐张量hash全部核对；
没有重跑全图或分配全部模型。原JSON hash为`de26031cc637b6e0f126b5ae04d801baf5ff623a576ecccdcbd0eab67b3523b4`。
二进制声明hash见JSON；本次没有重算整个约37GB文件hash，不能称全文件再次校验。

| 比较（原atol=1e-6、rtol=1e-5） | 最大绝对差 | 不通过坐标数 |
| --- | ---: | ---: |
| 来源批量变换 vs 标量 | 8.58307e-6 | 3 |
| 相同标量内容的私有BMM vs 标量投影 | 3.81470e-6 | 3 |
| 来源批量+私有BMM vs 标量完整局部链 | 4.19617e-5 | 194 |
| 标量FP32完整局部链 vs 局部FP64公式 | 3.84631e-5 | 289 |
| 来源批量+私有BMM vs 局部FP64公式 | 3.55915e-5 | 234 |
| 连续局部FP64、仅最终cast到FP32 vs 局部FP64公式 | 9.15933e-7 | 0 |

私有BMM确实3次/96行、最大32。来源变换的不同舍入进入归一化/投影后扩大；
不是仅凭重复同一行推断大图根因。FP64公式是独立更高精度数值锚点，不是任意精度实数证明。
标量FP32本身也偏离它，不能把标量值当作精确真值，或保证仅改变候选就与标量严格一致。
连续局部FP64结果有改进，但没有对应的公共全图精度策略、完整VJP/路由/更新资格；
因此不据此再跑原9B FP32全图。重启需要明确、跨接口一致的整段算术与梯度绑定及小图资格。

## 原FP16 GDN/default AdamW

相同fixture、forward、loss scale16、FP32 masters、AdamW默认epsilon和原容差；
仅把RMS/scaled-RMS/L2解析VJP内部算术提升到CPU FP64，分别命中404/48/112次，
返回与梯度仍原dtype。两个forward通过，第二步`nodes.5.kernel.conv.weight[0,2]`
为0.0262603759765625 vs 0.0282440185546875，差0.001983642578125，仍严格失败。
当步未缩放梯度分别1.862645149230957e-8与4.842877388000488e-8。
这说明仅提升归一化反向不足，不证明所有宽精度局部伴随方案无效；停止该候选，保留原strict xfail。

## 默认异步双NPU完整CPU桥接

Torch2.12/Torch-NPU2.12.post2、CANN9.1、Ascend910_9392，物理1/9映射逻辑0/1，
HF32关闭，未设置launch blocking。`delta-pdg`完整训练检查在VJP仍报EE1017：
`rtStreamWaitEvent`的event无效/已销毁。实际桥接20次forward/240字节、2次backward/24字节后退出。
没有完成后续四次更新及换卡恢复，不能用小桥接反例通过宣称完整方案可用。
停止默认async下该桥接候选；重启条件是运行时修复或新的事件所有权方案通过独立反例与完整训练。
显式blocking作为另一配置保留；本次没有用它回填默认异步结果。

## 任务终态与复现

两个服务均在background.slice，单线程ATen/BLAS、MemoryMax=16G；不由systemd属性推断
硬隔离已生效。CPU总预算390秒（每stage180秒），桥接排队180秒/执行480秒。
CPU job failed/1：设施2项和首次写入诊断passed/0，half strict_mismatch/2；
桥接job failed/1。未重复同候选，原状态与最小见证保留在ignored artifacts。

干净固定源码、匹配私有环境下使用公共入口（NEW路径均须不存在）：

```sh
python -m pytest -q tests/test_weighted_peer_bridge.py tests/test_weighted_half_probe.py
python scripts/diagnose_weighted_first_write.py --fixture MATERIAL_V2.json --output-dir NEW_FIRST
python scripts/diagnose_weighted_precision.py --implementation python \
  --normalization-vjp float64 --output-dir NEW_HALF
python scripts/diagnose_weighted_async.py --device npu:0 --peer-device npu:1 \
  --copy-policy cpu_bridge --scope complete --case delta-pdg --output-dir NEW_BRIDGE
```

MATERIAL_V2由仓库[参数匹配设施](../weighted-workloads.md)生成同seed7、ranked15×32、
width2048、B1/1位置、core3-add主体匹配材料；其声明和各操作数哈希可与记录核对。
长任务按[执行契约](../execution-flows.md)冻结源码和设置有限预算，不依赖旧worktree或私有driver。
