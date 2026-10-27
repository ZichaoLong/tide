# Core-5 精确联合记忆 CPU 资格 — 2026-10-27

固定实现/测试 `26dad7d22d61468823a986fb67608fc8df3b74e0`，上游 `0770593`。
**378 passed / 0 failed / 0 skipped；CPU FP64/FP32、aarch64。**
干净源码、工作退出 0、服务终态成功已核对；[命令与哈希](core5-packed-20261027.json)。
数学→输入/状态→算子→解码→独立测试见 [weighted-packed](../weighted-packed.md)。

KV 正接收投影真正合批并构造实际 ragged 前缀缓存；affine/delta/GDN/SSD 采用批量系数
与有序仿射前缀复合。固定短卷积按 int64 逻辑时间 searchsorted/gather，保留带权历史 raw
与未加权当前 tap 的区别。按正写入长度分组，时间与 sample/head 参与批量张量计算。
无 scalar Upd 补写、无 CPU 真实轨迹预跑；在线调度仍由现有独立执行器负责。

独立非交换矩阵多项式及 VJP、基础循环手算、GDN/SSD conv1d 公式、稀疏时间与 >2^54
精确时间戳通过。全零接收保留完整 raw tuple 和断开的输入/参数根；观察计数溢出明确失败。
禁止调用 scalar Upd 的局部测试约束真实联合数值路径。C++ standalone 另调用 packed
接口，检查原始时间、所有槽和基础循环手算；它不依赖 Python 调度。

完整图回归覆盖独立调度、FC/HB、反馈 PDG、非对齐 DAG、五种记忆与异构组合、评分/发送，
完整返回、私有样本状态、各槽/独立输出根、Read0/-/+、多步 AdamW、每切点续接、
checkpoint/optimizer 恢复与 no_grad 推理。计数不与旧资格重复累计。

本提交的数值是联合候选计算，状态反向仍是声明的逐局部语义重放；
**不是批量 State VJP、完整 Read/Full/Aggregate/Region packing、常驻或跨卡资格**。
不是整个仓库的全量门禁。混合设备、GPU 真机、完整 standalone 训练/checkpoint、
容量切分与性能均须独立记录。

保留开发失败：packed-dev 构建退出 1（standalone 测试缺 NoGradGuard 头文件，未跑 pytest）；
packed-build **363 passed / 5 failed**（FP32 SSD 两 Read 的 Python/native input.2 梯度，
及 PDG Read− 的 key.out_bias 梯度）。独立矩阵求值顺序诊断中，行批/bmm 保留失败，
列批/单行列向量变体通过。正式统一权重矩阵乘输入列批；不改输入、容差、dtype 或裁剪梯度。
packed-order 开发 **57/0** 后固定本提交进行独立资格。原失败状态/源码/日志保持不变。

复现：固定提交，按 [development](../development.md) 构建后执行：

```sh
TIDE_BUILD_DIR="$PWD/build/cpu" python -m pytest -q --dtype both \
  tests/test_weighted_packed.py tests/test_weighted_execution.py \
  tests/test_weighted_native.py tests/test_weighted_recurrent_execution.py \
  tests/test_weighted_scoring_execution.py tests/test_weighted_sending_execution.py \
  tests/test_weighted_kv_formulas.py tests/test_weighted_recurrent_formulas.py \
  tests/test_cli.py::test_cpp_custom_state_kernel
```

Python 3.12.13、Torch 2.12.0+cpu、GCC 10.3.1。ATen/BLAS 单线程，build workers=2。
服务 tide-core5-packed-qualify-20261027，background.slice、Nice=10、15 分钟上限。
运行前核验复用 C++ 源码与全部二进制 hash；原始记录 artifacts/jobs/core5-packed-qualify，
复现不依赖该目录。
